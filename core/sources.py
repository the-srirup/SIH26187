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

Hard reset
----------
:func:`hard_reset` is the deliberate opposite: the one operation allowed to
destroy the audit chain, so it does so *completely* — every pipeline stopped,
every frame dropped, every table emptied, tombstones cleared, optionally the
evidence tree wiped — leaving a system that is immediately usable again with
ids starting from 1.  A partial wipe would be strictly worse than either
extreme, because verification would read COMPROMISED forever with nothing left
to show for it.  It is exposed as ``POST /api/system/hard-reset`` and
``python manage.py hard-reset``.
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import delete, text, update
from sqlalchemy.orm import Session

from core.config import settings
from core.models import (
    Alert, AnalysisSession, ANPRDetection, Camera, Checkpoint, FaceDetection,
    Rule, WatchlistEntry,
)
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


def _runtime():
    """
    The camera runtime of *this* process, or ``None`` if it has none.

    ``core.camera`` pulls in the detector stack and costs about four seconds to
    import, and a process that has never imported it cannot be running a camera
    thread — so for every CLI invocation the honest answer is "there is nothing
    here to stop", reached without paying for the import to find out. In the
    API process the module is always loaded and this is a dict lookup.
    """
    return sys.modules.get("core.camera")


def _teardown_runtime(camera_id: int, *, tombstone: bool = True,
                     background: bool = False) -> bool:
    """Stop a camera's pipeline here and close its frame slot. Idempotent."""
    module = _runtime()
    if module is None:
        return False
    try:
        manager = module.CameraManager.get()
        stopped = (manager.retire(camera_id, background=background) if tombstone
                   else manager.remove_camera(camera_id, background=background))
        module.FrameBuffer.get().drop(camera_id)
        return bool(stopped)
    except Exception as exc:
        log.exception("Runtime teardown failed for camera %d: %s", camera_id, exc)
        return False


def _clear_runtime_tombstone(camera_id: int) -> None:
    """
    Let a newly registered camera use an id a removed one used to hold.

    SQLite hands out ``max(id) + 1``, so a fresh camera can legitimately
    inherit the id of one that was hard-deleted — and the machinery keeping
    that old camera dead (the manager's tombstone, the buffer's closed slot)
    would otherwise refuse the new one on sight.

    """
    module = _runtime()
    if module is None:
        return
    try:
        module.CameraManager.get().release(camera_id)
        module.FrameBuffer.get().open(camera_id)
    except Exception as exc:      # never fail a registration over bookkeeping
        log.debug("Could not clear runtime tombstone for camera %d: %s",
                  camera_id, exc)


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

    _clear_runtime_tombstone(camera.id)
    log.info("Camera registered: #%d %s (%s, kind=%s)",
             camera.id, camera.name, camera.url, camera.source_kind)
    return camera


# --------------------------------------------------------------------------- #
# Startup reconciliation
# --------------------------------------------------------------------------- #


def reconcile_on_start(db: Session) -> dict:
    """
    Make the database agree with reality before anything reads it.

    Run at boot, and this is the *only* place it can correctly run. A shutdown
    hook cannot do this job: it does not execute when the process is killed,
    when the terminal closes, when Python segfaults inside a native decoder, or
    when the machine loses power — and those are exactly the terminations that
    leave stale state behind.

    What goes stale, demonstrated by killing a live server with ``taskkill /F``:

        {'id': 1, 'name': 'KILLTEST', 'is_active': 1, 'is_online': 1}

    ``is_online`` is a *runtime* fact — "a processor in this process is
    receiving frames" — persisted so the dashboard can render a camera list
    without waiting for the first stats frame. After an ungraceful exit nothing
    is receiving anything, but the row still claims otherwise, so the next boot
    serves a green ONLINE tile for a camera that has no thread behind it. It
    looks like a working system; it is a lie told by a leftover flag.

    So: on every start, every camera is offline until a live processor says
    otherwise. A camera that really is reachable turns green a second later,
    which costs nothing and is honest.
    """
    stale = (
        db.query(Camera)
        .filter(Camera.is_online.is_(True))
        .update({Camera.is_online: False}, synchronize_session=False)
    )
    db.commit()
    if stale:
        log.warning(
            "STARTUP_RECONCILE cleared %d stale ONLINE flag(s) — the previous "
            "run did not shut down cleanly", stale,
        )
    return {"stale_online_cleared": int(stale or 0)}


def fresh_start(db: Session) -> dict:
    """
    Begin as though the software had just been installed.

    Retires every registered camera — and, because retirement archives anything
    holding sealed evidence, also clears those archived rows outright so they
    cannot accumulate across runs. The audit chain is untouched: alerts keep
    their ``camera_id``, which is a foreign key to a row that must therefore
    survive, so cameras that own events are archived rather than deleted and
    this function leaves them alone. Use :func:`hard_reset` to clear those too.

    Like :func:`reconcile_on_start`, this runs at *boot* rather than at
    shutdown, for the same reason: a termination that skips the shutdown path
    is precisely the one after which a clean slate matters most.
    """
    cameras = [c.id for c in db.query(Camera.id)
               .filter(Camera.source_kind != KIND_UPLOAD).all()]
    retired, archived = 0, 0
    for camera_id in cameras:
        outcome = retire_camera(db, camera_id)
        if outcome.get("mode") == "archived":
            archived += 1
        elif outcome.get("mode") == "deleted":
            retired += 1

    # A retired camera cannot be started, but the tombstones would refuse the
    # ids a freshly registered camera is about to be given.
    module = _runtime()
    if module is not None:
        try:
            module.CameraManager.get().clear_retired()
            module.FrameBuffer.get().clear()
        except Exception as exc:      # pragma: no cover - bookkeeping only
            log.debug("Could not clear runtime state on fresh start: %s", exc)

    log.warning(
        "FRESH_START — %d camera(s) deleted, %d archived (they own sealed "
        "evidence); the dashboard starts empty",
        retired, archived,
    )
    return {"deleted": retired, "archived": archived, "total": len(cameras)}


def prune_archived(db: Session, *, dry_run: bool = False) -> dict:
    """
    Reclaim archived camera rows that no longer hold anything.

    Removal archives rather than deletes a camera that owns sealed events,
    because ``alerts.camera_id`` is a foreign key and the alert log is a hash
    chain that must not lose rows. That is correct, and it means archived rows
    accumulate: this project's own database reached 49 of them.

    They are not free — every one is a row the startup query filters, a row in
    every join, and a name the event log resolves. Once an archived camera's
    last dependant is gone, the row is pure residue and can go. Anything still
    holding evidence is reported, not deleted, so the chain is never at risk.
    """
    archived = db.query(Camera).filter(Camera.is_deleted.is_(True)).all()
    removed, kept = [], []
    for camera in archived:
        deps = _dependants(db, camera.id)
        if any(deps.values()):
            kept.append({"id": camera.id, "name": camera.name, **deps})
            continue
        db.execute(delete(Rule).where(Rule.camera_id == camera.id))
        db.execute(delete(Camera).where(Camera.id == camera.id))
        removed.append({"id": camera.id, "name": camera.name})

    if dry_run:
        db.rollback()
    else:
        db.commit()
        if removed:
            log.info("Pruned %d archived camera row(s) holding no evidence",
                     len(removed))
    return {"removed": removed, "kept": kept, "dry_run": bool(dry_run),
            "archived_total": len(archived)}


# --------------------------------------------------------------------------- #
# Retirement
# --------------------------------------------------------------------------- #


def _dependants(db: Session, camera_id: int) -> dict:
    """
    Everything in the database that points at this camera.

    Every one of these columns is a foreign key to ``cameras.id`` and SQLite
    runs with ``PRAGMA foreign_keys=ON``, so anything missed here is not a
    cosmetic omission — it is a ``FOREIGN KEY constraint failed`` raised from
    inside the delete, a 500 on the endpoint, and a camera that stopped
    streaming but survived in the database and came back on the next refresh.
    That was the original Remove Camera bug, caused by ``analysis_sessions``;
    ``anpr_detections`` and ``face_detections`` were added to the schema later
    with the same shape and would have reproduced it exactly.
    """
    return {
        "alerts": db.query(Alert).filter(Alert.camera_id == camera_id).count(),
        "sessions": db.query(AnalysisSession)
                      .filter(AnalysisSession.camera_id == camera_id).count(),
        "anpr": db.query(ANPRDetection)
                  .filter(ANPRDetection.camera_id == camera_id).count(),
        "faces": db.query(FaceDetection)
                   .filter(FaceDetection.camera_id == camera_id).count(),
    }


def _already_removed(db: Session, camera_id: int, *, existed: bool,
                    background: bool = False) -> dict:
    """The idempotent answer, with the runtime asserted dead either way."""
    _teardown_runtime(camera_id, background=background)
    return {
        "ok": True,
        "removed": camera_id,
        "mode": "archived" if existed else "absent",
        "already_removed": True,
        "alerts_retained": (
            db.query(Alert).filter(Alert.camera_id == camera_id).count()
            if existed else 0
        ),
        "detail": "Camera was already removed.",
    }


def retire_camera(db: Session, camera_id: int, *,
                  background_join: bool = False) -> dict:
    """
    Remove a camera completely and safely.

    Order matters.  The runtime is torn down **first**, so no thread can seal an
    event against a camera row that is about to disappear (which would raise a
    foreign-key error inside a camera thread), and no frame can be published
    after the stream has gone.  Only then is the database touched.

    Steps:

    1. tombstone the id in the camera manager, then stop the analytics thread,
       the capture thread and the OpenCV handle, and close the shared frame
       buffer entry (all idempotent).  The tombstone is what stops a concurrent
       ``PUT``/``restart`` holding a stale read of this row from starting a
       fresh processor in the window between teardown and commit;
    2. delete the camera's rules — configuration, not evidence;
    3. count everything that depends on the camera;
    4. delete the row outright when nothing depends on it, otherwise archive it.
       Both transitions are conditional single statements, so two concurrent
       removals cannot both "win" and the loser reports success rather than
       raising;
    5. delete the camera's own managed MP4, if it had one.

    ``background_join`` finishes the *joining* of the stopped threads on a
    reaper instead of making the caller wait. Removal is already complete
    without it — step 1 is what makes the camera dead — so the HTTP endpoint
    uses it and returns in milliseconds even when a thread is parked in a
    native call. Callers that need every thread gone before they continue (the
    CLI, the hard reset, the tests) leave it off.

    Returns a summary describing what actually happened, so the UI can tell the
    operator whether evidence was retained rather than guessing.
    """
    camera_id = int(camera_id)
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None or camera.is_deleted:
        # Idempotent: a second removal — a double-clicked button, a retried
        # request — is a no-op, not an error. Any runtime that somehow outlived
        # the row is still torn down, because that is the state we are asserting.
        return _already_removed(db, camera_id, existed=camera is not None,
                                background=background_join)

    name = camera.name
    stored_url = camera.url
    was_file = camera.is_file_source

    # 1. Runtime teardown, before any schema change. This also refuses every
    #    later start for this id in this process.
    _teardown_runtime(camera_id, background=background_join)

    # The row may have been archived by a racing request while we were tearing
    # the runtime down; re-reading costs one indexed lookup and turns a
    # duplicate into the idempotent answer instead of a redundant write.
    db.expire_all()
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None or camera.is_deleted:
        return _already_removed(db, camera_id, existed=camera is not None,
                                background=background_join)

    try:
        # 2. Rules are configuration.
        rules_removed = db.execute(
            delete(Rule).where(Rule.camera_id == camera_id)
        ).rowcount or 0

        # 3. What depends on this camera?
        deps = _dependants(db, camera_id)
        keeps_evidence = deps["alerts"] > 0 or deps["sessions"] > 0

        # 4. Delete or archive — as one conditional statement either way, so a
        #    concurrent duplicate simply matches no row.
        if keeps_evidence:
            won = db.execute(
                update(Camera)
                .where(Camera.id == camera_id, Camera.is_deleted.is_(False))
                .values(is_deleted=True, is_active=False, is_online=False,
                        deleted_at=utc_iso())
            ).rowcount or 0
            mode = "archived"
        else:
            # No sealed evidence and no analysis run, so nothing here is part
            # of the hash chain: the detection *index* rows for this camera go
            # with it rather than dangling against a camera that no longer
            # exists (and blocking the delete on their foreign key).
            db.execute(delete(ANPRDetection)
                       .where(ANPRDetection.camera_id == camera_id))
            db.execute(delete(FaceDetection)
                       .where(FaceDetection.camera_id == camera_id))
            won = db.execute(
                delete(Camera).where(Camera.id == camera_id)
            ).rowcount or 0
            mode = "deleted"
        db.commit()
    except Exception as exc:
        db.rollback()
        log.exception("Failed to retire camera %d (%s): %s", camera_id, name, exc)
        raise
    finally:
        # Bulk statements bypass the identity map; expiring it keeps any object
        # the caller still holds from reporting the pre-removal state.
        db.expire_all()

    if not won:
        return _already_removed(db, camera_id, existed=(mode == "archived"),
                                background=background_join)

    # 5. The camera's own video file, and only that.
    file_removed = False
    if was_file and is_managed_source_file(stored_url):
        try:
            Path(stored_url).unlink()
            file_removed = True
        except OSError as exc:
            log.warning("Could not delete source video %s: %s", stored_url, exc)

    log.info(
        "CAMERA_RETIRED cam=%d (%s) %s — %d rule(s) removed, %d alert(s) "
        "retained, %d analysis session(s) retained%s",
        camera_id, name, mode, rules_removed, deps["alerts"], deps["sessions"],
        ", source video deleted" if file_removed else "",
    )
    return {
        "ok": True,
        "removed": camera_id,
        "name": name,
        "mode": mode,
        "rules_removed": int(rules_removed),
        "alerts_retained": deps["alerts"],
        "sessions_retained": deps["sessions"],
        "anpr_retained": deps["anpr"],
        "faces_retained": deps["faces"],
        "source_file_removed": file_removed,
        # Explain the outcome so the dashboard can be honest about it rather
        # than claiming a purge that did not happen.
        "detail": (
            f"Camera removed. {deps['alerts']} sealed event(s) kept in the audit "
            "log — deleting them would break the evidence hash chain."
            if mode == "archived" else
            "Camera and its configuration removed. It had no recorded events."
        ),
    }


# --------------------------------------------------------------------------- #
# Hard reset
# --------------------------------------------------------------------------- #

#: Deleted in this order — children before parents — because SQLite enforces
#: foreign keys and ``cameras`` is the parent of almost everything.
_RESET_TABLES: tuple = (
    ANPRDetection, FaceDetection, Alert, Rule, AnalysisSession,
    Checkpoint, WatchlistEntry, Camera,
)

def _evidence_directories() -> tuple[Path, ...]:
    """
    Directories whose *contents* a hard reset may delete.

    Deliberately an explicit list rather than anything derived: the bundled
    ``samples/`` clips, the model weights, the dashboard and the logs are not
    evidence, and a reset that removed them would break the documented demo
    with no way back short of a re-clone.
    """
    return (
        settings.SNAPSHOTS_DIR,
        settings.CLIPS_DIR,
        settings.EVIDENCE_DIR,
        settings.PROCESSED_DIR,
        settings.SOURCES_DIR,
        settings.VIDEOS_DIR,      # last: it is the parent of the two above
    )


def _wipe_directory(directory: Path) -> tuple[int, int]:
    """Delete everything *inside* ``directory``; keep the directory itself."""
    files = dirs = 0
    if not directory.exists():
        return 0, 0
    for entry in sorted(directory.iterdir(), key=lambda e: e.is_file(), reverse=True):
        try:
            if entry.is_dir() and not entry.is_symlink():
                # A nested managed directory (videos/sources) is wiped by its
                # own pass; removing the tree here is equivalent and cheaper.
                shutil.rmtree(entry)
                dirs += 1
            else:
                entry.unlink()
                files += 1
        except OSError as exc:
            log.warning("Hard reset could not remove %s: %s", entry, exc)
    return files, dirs


def hard_reset(db: Session, *, wipe_evidence: Optional[bool] = None,
               actor: str = "operator") -> dict:
    """
    Return the platform to a clean, immediately usable state.

    This is the deliberate "start the demo over" control, and it is the only
    operation in the system permitted to destroy the audit chain — so it does
    so completely and visibly rather than partially.  A half-wiped chain is
    worse than either extreme: verification would read COMPROMISED forever
    with nothing to show for it.

    Order is the whole point:

    1. **stop every camera first.**  Truncating ``cameras`` while a pipeline is
       running would have live threads sealing events against rows that no
       longer exist, and publishing frames for cameras that are gone;
    2. cancel in-flight offline analysis, for the same reason;
    3. clear the shared frame buffer, so no stale JPEG survives into the reset
       system and the dashboard genuinely goes empty;
    4. delete every row, children before parents;
    5. drop the in-memory event history and counters;
    6. optionally wipe the evidence tree;
    7. recreate the schema and clear the manager's tombstones, so ids start
       from 1 again and a new camera can be added immediately.

    ``wipe_evidence`` defaults to ``settings.HARD_RESET_WIPE_EVIDENCE``.
    """
    from core.database import init_db
    from core.events import EventManager
    from core.evidence import invalidate_evidence_usage

    if wipe_evidence is None:
        wipe_evidence = bool(settings.HARD_RESET_WIPE_EVIDENCE)

    started = time.time()
    log.warning("HARD_RESET_REQUESTED by=%s wipe_evidence=%s", actor, wipe_evidence)

    runtime = _runtime()
    manager = runtime.CameraManager.get() if runtime is not None else None

    # 1. Every pipeline goes down before a single row is touched.
    cameras_stopped = manager.stop_all() if manager is not None else 0

    # 2. Offline analysis writes alerts too.
    analyses_cancelled = 0
    try:
        from core.analysis import AnalysisManager

        analysis = AnalysisManager.get()  # cheap: analysis has no model imports
        for job in analysis.list_jobs():
            if job.get("status") in ("queued", "running") and analysis.cancel(
                job.get("session_uid", "")
            ):
                analyses_cancelled += 1
    except Exception as exc:          # analysis is optional; never fail the reset
        log.warning("Hard reset could not cancel analysis jobs: %s", exc)

    # 3. No stale frames may survive into the reset system.
    frames_cleared = runtime.FrameBuffer.get().clear() if runtime is not None else 0

    # 4. The database.
    deleted: dict[str, int] = {}
    try:
        for model in _RESET_TABLES:
            deleted[model.__tablename__] = int(
                db.execute(delete(model)).rowcount or 0
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        log.exception("Hard reset failed while clearing the database: %s", exc)
        raise
    finally:
        db.expire_all()

    # 5. The dashboard's live history is in memory, not in the database.
    EventManager.get().reset()
    invalidate_evidence_usage()
    try:
        from core.analytics import _AsyncStage

        _AsyncStage.reset_budgets()      # a reset should not inherit a back-off
    except Exception as exc:             # pragma: no cover - bookkeeping only
        log.debug("Could not reset stage budgets: %s", exc)

    # 6. Evidence on disk.
    evidence = {"files_removed": 0, "directories_removed": 0, "wiped": bool(wipe_evidence)}
    if wipe_evidence:
        for directory in _evidence_directories():
            files, dirs = _wipe_directory(directory)
            evidence["files_removed"] += files
            evidence["directories_removed"] += dirs

    # 7. A usable, empty system.
    settings.ensure_dirs()
    init_db()
    if manager is not None:
        manager.clear_retired()

    # Reclaim the file now rather than leaving a 2.6 MB database that reports
    # zero rows. Best-effort: VACUUM cannot run inside a transaction and is not
    # worth failing a reset over.
    try:
        db.commit()
        db.execute(text("VACUUM"))
        db.commit()
    except Exception as exc:
        db.rollback()
        log.debug("VACUUM after hard reset skipped: %s", exc)

    result = {
        "ok": True,
        "cameras_stopped": cameras_stopped,
        "analyses_cancelled": analyses_cancelled,
        "frames_cleared": frames_cleared,
        "rows_deleted": deleted,
        "rows_deleted_total": sum(deleted.values()),
        "evidence": evidence,
        "duration_ms": round((time.time() - started) * 1000.0, 1),
        "reset_at": utc_iso(),
        "detail": (
            "System reset. No cameras, no rules, no events; the integrity "
            "chain restarts from genesis. Add a camera to begin."
        ),
    }
    log.warning(
        "HARD_RESET_COMPLETE by=%s — %d camera(s) stopped, %d row(s) deleted, "
        "%d evidence file(s) removed, %.0f ms",
        actor, cameras_stopped, result["rows_deleted_total"],
        evidence["files_removed"], result["duration_ms"],
    )
    return result


__all__ = [
    "KIND_LIVE", "KIND_FILE", "KIND_UPLOAD", "SourceError",
    "validate_live_url", "find_duplicate",
    "visible_cameras", "get_live_camera", "startup_cameras",
    "store_source_video", "register_camera", "retire_camera",
    "is_managed_source_file", "hard_reset",
    "reconcile_on_start", "fresh_start", "prune_archived",
]
