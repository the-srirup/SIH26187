"""
Camera source lifecycle — registration, MP4 sources, and safe retirement.

This module exists because "add a camera" and "remove a camera" are not
database operations.  A registered source owns a capture thread, an analytics
thread, an OpenCV handle, a rule engine, a frame buffer, MJPEG subscribers,
possibly a file on disk, and a tail of sealed evidence.  Scattering that
knowledge across the API layer is what allowed the removal path to be wrong in
several ways at once.

Two kinds of source, one pipeline
--------------------------------
An MP4 file is a **first-class camera**, not a separate feature.  It is stored
under ``videos/sources/``, registered with ``source_kind="file"`` and handed to
the same :class:`~core.camera.CameraProcessor` as an RTSP stream — so it gets
the same detection, tracking, zones, tripwires, ANPR, events and evidence, and
appears in the same Live Camera grid.  ``LiveSource`` already paces a file to
its own frame rate and loops it at EOF; nothing about the analytics needs to
know which kind of source it is reading.

Why removal archives instead of deleting
----------------------------------------
The reported bug was that Remove Camera did not work.  The cause was concrete
and reproducible: ``analysis_sessions.camera_id`` is a foreign key to
``cameras.id`` with no ORM relationship and no cascade, and SQLite runs with
``PRAGMA foreign_keys=ON``.  Deleting a camera that had ever been used as the
rule source for an uploaded video therefore raised ``FOREIGN KEY constraint
failed``, the endpoint returned 500 — and because the processor had *already*
been stopped by then, the camera stopped streaming but survived in the
database and came back on the next refresh.  That is exactly the symptom.

Fixing only the foreign key would have exposed a worse problem.  The camera
relationship cascaded deletes to ``alerts``, and ``alerts`` is a SHA-256 hash
chain: every row's hash covers its predecessor's.  Deleting a camera's rows
from the middle of that chain invalidates every row after them, so
``/api/integrity/verify`` would fail for the life of the database — silent,
permanent evidence corruption triggered by a routine UI action.

So removal is split by what the camera owns:

* **no events, no analysis sessions** — the row is genuinely deleted;
* **otherwise** — the camera is *archived*: hidden from every listing, never
  auto-started, its stream 404s, its rules deleted, its own MP4 file removed if
  it owned one, while its sealed evidence stays in place and verifiable.

Either way the operator sees the camera disappear and it stays gone across a
restart, which is what "remove" means to them.  The operation is idempotent:
calling it twice is a no-op, never a crash.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from core.config import settings
from core.models import Alert, AnalysisSession, Camera, Rule
from core.timeutil import utc_iso

log = logging.getLogger("ibvap.sources")

#: Source kinds a camera may have.
KIND_LIVE = "live"
KIND_FILE = "file"
KIND_UPLOAD = "upload"


class SourceError(ValueError):
    """Raised when a source cannot be registered, with an operator-facing message."""


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def visible_cameras(db: Session, include_uploads: bool = False):
    """
    Cameras the operator should see.

    Archived cameras are excluded here, which is the single place that
    guarantees a removed camera cannot reappear on a dashboard refresh.
    """
    query = db.query(Camera).filter(Camera.is_deleted.is_(False))
    if not include_uploads:
        query = query.filter(Camera.source_kind != KIND_UPLOAD)
    return query.order_by(Camera.id.asc()).all()


def get_live_camera(db: Session, camera_id: int) -> Optional[Camera]:
    """A camera that exists and has not been archived."""
    return (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_deleted.is_(False))
        .first()
    )


def startup_cameras(db: Session):
    """Sources to auto-start on boot — active, not archived, not the upload stub."""
    return (
        db.query(Camera)
        .filter(
            Camera.is_active.is_(True),
            Camera.is_deleted.is_(False),
            Camera.source_kind != KIND_UPLOAD,
        )
        .all()
    )


# --------------------------------------------------------------------------- #
# MP4 sources
# --------------------------------------------------------------------------- #


def is_managed_source_file(path: str) -> bool:
    """
    True when ``path`` is an MP4 this platform stored for a camera source.

    Deleting a camera may remove its own video file, but must never touch an
    operator's original footage, an offline-analysis upload, or an evidence
    artefact.  Only files inside ``videos/sources/`` qualify.
    """
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(settings.SOURCES_DIR.resolve())
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def store_source_video(temp_path: Path, original_name: str) -> tuple[Path, dict]:
    """
    Validate an uploaded MP4 and move it into managed source storage.

    Validation is shared with the offline-analysis path (extension, size,
    ISO-BMFF container signature, and actual decodability), so a renamed
    executable is rejected regardless of what it claims to be.  The stored
    filename is derived from a server-generated id, so a hostile client has no
    influence over the path at all.
    """
    from core.analysis import UploadValidationError, sanitize_filename, validate_upload

    safe_name = sanitize_filename(original_name)
    try:
        probe = validate_upload(temp_path, safe_name)
    except UploadValidationError as exc:
        temp_path.unlink(missing_ok=True)
        raise SourceError(str(exc)) from exc

    settings.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    stored = settings.SOURCES_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    shutil.move(str(temp_path), str(stored))
    log.info(
        "Video source stored: %s (%.1f MB, %dx%d @ %.1f fps, %.1fs) -> %s",
        safe_name, probe["size_bytes"] / (1024 * 1024),
        probe.get("width", 0), probe.get("height", 0),
        probe.get("fps", 0.0), probe.get("duration_seconds", 0.0), stored.name,
    )
    return stored, probe


#: Network schemes a live camera source may use. A bare path to an existing
#: video file is also accepted (see :func:`validate_live_url`) because that is
#: the documented demo source; ``file://`` is not, since it buys nothing over a
#: plain path and only widens what a URL string can reach.
_ALLOWED_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")

#: Container types a local file source may use.
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mjpeg", ".mjpg"}


def validate_live_url(url: str) -> str:
    """
    Check a live camera URL before anything tries to open it.

    Rejecting a malformed URL here, synchronously, is worth doing precisely
    because the failure mode otherwise is so poor: the row is created, a capture
    thread starts, FFmpeg blocks on an unroutable host, and the operator sees a
    camera tile that never loads and a dashboard that has grown sluggish, with
    nothing anywhere saying the URL was wrong.

    The check that actually bites in practice is the bracket one. This project's
    own database contains::

        rtsp://[akulsharma]:[Akulsharma@17]@[192.168.1.2]:554/stream1

    — credentials pasted with the placeholder brackets left in. In RFC 3986
    square brackets delimit an IPv6 literal, so that host is not "192.168.1.2"
    and never resolves; it simply hangs.
    """
    text = (url or "").strip()
    if not text:
        raise SourceError("Camera source is required")

    # A webcam index is a first-class source.
    if text.isdigit():
        if not (0 <= int(text) <= 15):
            raise SourceError("Webcam index must be between 0 and 15")
        return text

    from core.youtube import available as youtube_available, is_youtube_url

    if is_youtube_url(text):
        if not youtube_available():
            raise SourceError(
                "YouTube sources need yt-dlp. Install it with: pip install yt-dlp"
            )
        return text

    low = text.lower()
    if not low.startswith(_ALLOWED_SCHEMES):
        # A local video file is a supported source — `manage.py camera-add --url
        # samples/sample_border_scenario.mp4` is the documented demo path, and
        # LiveSource already paces and loops a file correctly. Accept it when it
        # genuinely exists; a path that does not resolve is a typo, and saying so
        # now is far better than starting a capture thread that can never open
        # anything and reporting it as a camera that is merely "offline".
        candidate = Path(text)
        if candidate.exists():
            if candidate.is_dir():
                raise SourceError(f"'{text}' is a directory, not a video file")
            if candidate.suffix.lower() not in _VIDEO_SUFFIXES:
                raise SourceError(
                    f"'{candidate.suffix or candidate.name}' is not a supported "
                    f"video type ({', '.join(sorted(_VIDEO_SUFFIXES))})"
                )
            return text
        raise SourceError(
            f"'{text}' is not a reachable source. Use an RTSP/HTTP(S) URL, a "
            f"YouTube link, a webcam index (e.g. 0), or the path of an existing "
            f"video file. To upload a video, use 'Add video source' instead."
        )

    if "[" in text or "]" in text:
        raise SourceError(
            "Remove the square brackets from the URL — they are placeholders. "
            "Write it as rtsp://user:password@192.168.1.2:554/stream1 "
            "(brackets are reserved for IPv6 addresses, so the host never resolves)."
        )
    if " " in text:
        raise SourceError("Camera URL must not contain spaces")

    # Must have a host after the scheme.
    scheme, _, remainder = text.partition("://")
    host = remainder.split("/")[0].split("@")[-1]
    if not host or host.startswith(":"):
        raise SourceError(f"No host found in the URL after '{scheme}://'")
    return text


def find_duplicate(db: Session, url: str) -> Optional[Camera]:
    """An existing, non-archived camera already using this exact source."""
    return (
        db.query(Camera)
        .filter(Camera.url == url, Camera.is_deleted.is_(False))
        .first()
    )


def register_camera(
    db: Session,
    *,
    name: str,
    url: str,
    location: str = "",
    is_active: bool = True,
    source_kind: str = KIND_LIVE,
    validate: bool = True,
    allow_duplicate: bool = False,
) -> Camera:
    """
    Create and persist a camera row (of any kind).

    Duplicates are rejected rather than silently accepted.  Registering one URL
    twice used to produce two independent camera rows, each with its own capture
    thread, decoder, analytics thread, tracker and rule engine, all doing
    identical work on identical pixels — this database still holds four such
    rows (``CYCLE-1``..``CYCLE-4``, all pointed at the same sample clip).  On a
    machine already near its inference budget that is the difference between a
    responsive dashboard and a stalled one, and the operator has no way to tell
    the copies apart afterwards.
    """
    name = (name or "").strip()
    url = (url or "").strip()
    if not name:
        raise SourceError("Camera name is required")
    if not url:
        raise SourceError("Camera source is required")
    if validate and source_kind == KIND_LIVE:
        url = validate_live_url(url)

    if not allow_duplicate:
        existing = find_duplicate(db, url)
        if existing is not None:
            raise SourceError(
                f"This source is already registered as camera #{existing.id} "
                f"“{existing.name}”. Remove that camera first, or use it "
                f"directly — running two pipelines over one feed doubles the load "
                f"without adding coverage."
            )

    camera = Camera(
        name=name[:120],
        url=url[:500],
        location=(location or "").strip()[:200],
        is_active=bool(is_active),
        is_online=False,
        source_kind=source_kind,
        is_deleted=False,
        deleted_at="",
        created_at=utc_iso(),
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    log.info("Camera registered: #%d %s (%s, kind=%s)",
             camera.id, camera.name, camera.url, camera.source_kind)
    return camera


# --------------------------------------------------------------------------- #
# Retirement
# --------------------------------------------------------------------------- #


def retire_camera(db: Session, camera_id: int) -> dict:
    """
    Remove a camera completely and safely.

    Order matters.  The runtime is torn down **first**, so no thread can seal an
    event against a camera row that is about to disappear (which would raise a
    foreign-key error inside a camera thread), and no frame can be published
    after the stream has gone.  Only then is the database touched.

    Steps:

    1. stop the analytics thread, the capture thread and the OpenCV handle, and
       drop the shared frame buffer entry (all idempotent);
    2. delete the camera's rules — configuration, not evidence;
    3. count dependent evidence (alerts) and analysis sessions;
    4. delete the row outright when nothing depends on it, otherwise archive it;
    5. delete the camera's own managed MP4, if it had one.

    Returns a summary describing what actually happened, so the UI can tell the
    operator whether evidence was retained rather than guessing.
    """
    from core.camera import CameraManager, FrameBuffer

    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None or camera.is_deleted:
        # Idempotent: a second removal — a double-clicked button, a retried
        # request — is a no-op, not an error. Any runtime that somehow outlived
        # the row is still torn down, because that is the state we are asserting.
        CameraManager.get().remove_camera(camera_id)
        FrameBuffer.get().drop(camera_id)
        return {
            "ok": True,
            "removed": camera_id,
            "mode": "archived" if camera is not None else "absent",
            "already_removed": True,
            "alerts_retained": (
                db.query(Alert).filter(Alert.camera_id == camera_id).count()
                if camera is not None else 0
            ),
            "detail": "Camera was already removed.",
        }

    name = camera.name
    stored_url = camera.url
    was_file = camera.is_file_source

    # 1. Runtime teardown, before any schema change.
    CameraManager.get().remove_camera(camera_id)
    FrameBuffer.get().drop(camera_id)

    # 2. Rules are configuration.
    rules_removed = db.query(Rule).filter(Rule.camera_id == camera_id).delete()

    # 3. What depends on this camera?
    alert_count = db.query(Alert).filter(Alert.camera_id == camera_id).count()
    session_count = (
        db.query(AnalysisSession)
        .filter(AnalysisSession.camera_id == camera_id)
        .count()
    )

    # 4. Delete or archive.
    if alert_count == 0 and session_count == 0:
        db.delete(camera)
        mode = "deleted"
    else:
        camera.is_deleted = True
        camera.is_active = False
        camera.is_online = False
        camera.deleted_at = utc_iso()
        mode = "archived"

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        log.exception("Failed to retire camera %d (%s): %s", camera_id, name, exc)
        raise

    # 5. The camera's own video file, and only that.
    file_removed = False
    if was_file and is_managed_source_file(stored_url):
        try:
            Path(stored_url).unlink()
            file_removed = True
        except OSError as exc:
            log.warning("Could not delete source video %s: %s", stored_url, exc)

    log.info(
        "Camera %d (%s) %s — %d rule(s) removed, %d alert(s) retained, "
        "%d analysis session(s) retained%s",
        camera_id, name, mode, rules_removed, alert_count, session_count,
        ", source video deleted" if file_removed else "",
    )
    return {
        "ok": True,
        "removed": camera_id,
        "name": name,
        "mode": mode,
        "rules_removed": int(rules_removed or 0),
        "alerts_retained": alert_count,
        "sessions_retained": session_count,
        "source_file_removed": file_removed,
        # Explain the outcome so the dashboard can be honest about it rather
        # than claiming a purge that did not happen.
        "detail": (
            f"Camera removed. {alert_count} sealed event(s) kept in the audit "
            "log — deleting them would break the evidence hash chain."
            if mode == "archived" else
            "Camera and its configuration removed. It had no recorded events."
        ),
    }


__all__ = [
    "KIND_LIVE", "KIND_FILE", "KIND_UPLOAD", "SourceError",
    "validate_live_url", "find_duplicate",
    "visible_cameras", "get_live_camera", "startup_cameras",
    "store_source_video", "register_camera", "retire_camera",
    "is_managed_source_file",
]
