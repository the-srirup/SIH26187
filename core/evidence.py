"""
Evidence capture and retention.

Every meaningful event is backed by artefacts an investigator can actually
use, stored on disk under ``alerts/snapshots`` and ``clips``:

* an **annotated snapshot** — what the operator saw, boxes and all;
* a **clean snapshot** — the unmarked frame, so evidence is not editorialised;
* a **contextual MP4 clip** covering a few seconds *before* and after the
  event, taken from a bounded ring buffer.

The ring buffer is the reason a clip can start before the trigger without
recording the entire feed: the last ``CLIP_PRE_SECONDS`` of frames are always
in memory, and only promoted to disk when something happens.

Retention is enforced (age and total size) so an unattended BOP deployment
cannot fill its own disk.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings
from core.timeutil import file_stamp

log = logging.getLogger("ibvap.evidence")

#: OpenCV 5 moved the helper onto the class; support both.
_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc


class ClipRecorder:
    """
    Rolling pre-roll buffer plus post-event writer for one source.

    Thread-affine by design: ``push`` and ``start_clip`` are called from the
    analytics thread that owns the source, so writing never blocks the API.
    Multiple concurrent clips are supported (two events seconds apart each get
    their own file) up to a small cap.
    """

    MAX_CONCURRENT = 3

    def __init__(
        self,
        source_id: str,
        clip_dir: Optional[Path] = None,
        fps: Optional[float] = None,
        pre_frames: Optional[int] = None,
        post_frames: Optional[int] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.source_id = str(source_id)
        self.clip_dir = Path(clip_dir or settings.CLIPS_DIR)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps or settings.TARGET_FPS)
        self.pre_frames = int(pre_frames or settings.clip_pre_frames)
        self.post_frames = int(post_frames or settings.clip_post_frames)
        self.frame_size = frame_size or settings.frame_size

        self._buffer: deque = deque(maxlen=max(1, self.pre_frames))
        self._active: list[dict] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def push(self, frame: np.ndarray) -> None:
        """
        Feed one annotated frame to the pre-roll buffer and any open clips.

        The copy here is deliberate and unavoidable: the frame buffer that
        feeds the MJPEG stream hands out references, and a clip must capture
        the frame as it was at this instant.
        """
        with self._lock:
            self._buffer.append(frame.copy())
            if not self._active:
                return
            finished = []
            for clip in self._active:
                try:
                    clip["writer"].write(frame)
                except Exception as exc:  # pragma: no cover - writer failure
                    log.warning("Clip write failed for %s: %s", clip["path"], exc)
                    finished.append(clip)
                    continue
                clip["remaining"] -= 1
                if clip["remaining"] <= 0:
                    finished.append(clip)
            for clip in finished:
                self._finalize(clip)

    def start_clip(self, alert_id: int) -> Optional[str]:
        """
        Open a new evidence clip seeded with the pre-roll buffer.

        Returns the clip path, or ``None`` when a clip could not be started.
        The file is written progressively; ``push`` closes it once the
        post-roll window has elapsed.
        """
        if not settings.EVIDENCE_ENABLED:
            return None

        with self._lock:
            if len(self._active) >= self.MAX_CONCURRENT:
                log.debug("Clip slots full for %s — reusing existing evidence", self.source_id)
                return None

            path = self.clip_dir / f"cam{self.source_id}_alert{alert_id}_{file_stamp()}.mp4"
            writer = cv2.VideoWriter(
                str(path), _FOURCC(*"mp4v"), self.fps, self.frame_size
            )
            if not writer.isOpened():
                log.error("Cannot open VideoWriter for %s", path)
                try:
                    writer.release()
                except Exception:
                    pass
                return None

            for frame in self._buffer:
                try:
                    writer.write(frame)
                except Exception:
                    break

            self._active.append({
                "writer": writer,
                "path": path,
                "remaining": self.post_frames,
                "started": time.time(),
            })
            return str(path)

    def _finalize(self, clip: dict) -> None:
        try:
            clip["writer"].release()
        except Exception:
            pass
        if clip in self._active:
            self._active.remove(clip)
        log.info("Evidence clip sealed: %s", Path(clip["path"]).name)

    def close(self) -> None:
        """Release every open writer — called on shutdown, never skipped."""
        with self._lock:
            for clip in list(self._active):
                self._finalize(clip)
            self._buffer.clear()

    @property
    def active_clips(self) -> int:
        return len(self._active)


def save_snapshot(
    frame: np.ndarray,
    source_id: str,
    alert_id: int,
    clean_frame: Optional[np.ndarray] = None,
) -> tuple[str, str]:
    """
    Write the annotated snapshot (and, when supplied, the unmarked original).

    Returns ``(annotated_path, clean_path)``; either may be ``""`` on failure —
    a disk problem must degrade the evidence, not kill the alert.
    """
    settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = file_stamp()
    base = f"cam{source_id}_alert{alert_id}_{stamp}"
    annotated_path = settings.SNAPSHOTS_DIR / f"{base}.jpg"
    clean_path = settings.SNAPSHOTS_DIR / f"{base}_clean.jpg"

    saved_annotated = saved_clean = ""
    try:
        if cv2.imwrite(str(annotated_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            saved_annotated = str(annotated_path)
    except Exception as exc:
        log.warning("Snapshot write failed: %s", exc)

    if clean_frame is not None:
        try:
            if cv2.imwrite(str(clean_path), clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                saved_clean = str(clean_path)
        except Exception:
            pass

    return saved_annotated, saved_clean


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #


def sweep_evidence(
    max_mb: Optional[int] = None,
    max_age_days: Optional[int] = None,
) -> dict:
    """
    Enforce evidence retention limits.

    Deletes by age first, then by total size (oldest first) until the store is
    back under budget.  Returns a summary so the sweep can be logged and
    surfaced in system info rather than happening invisibly.
    """
    max_mb = settings.MAX_EVIDENCE_MB if max_mb is None else max_mb
    max_age_days = settings.MAX_EVIDENCE_AGE_DAYS if max_age_days is None else max_age_days

    files: list[tuple[float, int, Path]] = []
    for directory in (settings.SNAPSHOTS_DIR, settings.CLIPS_DIR, settings.PROCESSED_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))

    files.sort(key=lambda item: item[0])
    total_bytes = sum(size for _, size, _ in files)
    removed, freed = 0, 0

    cutoff = time.time() - max_age_days * 86400
    survivors: list[tuple[float, int, Path]] = []
    for mtime, size, path in files:
        if mtime < cutoff:
            try:
                path.unlink()
                removed += 1
                freed += size
            except OSError:
                survivors.append((mtime, size, path))
        else:
            survivors.append((mtime, size, path))

    budget = max_mb * 1024 * 1024
    current = total_bytes - freed
    for mtime, size, path in survivors:
        if current <= budget:
            break
        try:
            path.unlink()
            removed += 1
            freed += size
            current -= size
        except OSError:
            continue

    if removed:
        log.info("Evidence sweep: removed %d file(s), freed %.1f MB",
                 removed, freed / (1024 * 1024))

    return {
        "files_before": len(files),
        "files_removed": removed,
        "bytes_freed": freed,
        "bytes_remaining": max(0, total_bytes - freed),
        "budget_mb": max_mb,
        "max_age_days": max_age_days,
    }


#: Last measured footprint, and when. See :func:`evidence_usage`.
_usage_cache: Optional[dict] = None
_usage_measured_at: float = 0.0
_usage_lock = threading.Lock()

#: How stale the footprint may be before a caller re-measures it. Generous on
#: purpose: this feeds a storage *indicator*, and the number moves slowly.
USAGE_MAX_AGE_SECONDS = 60.0


def measure_evidence_usage() -> dict:
    """
    Walk the evidence tree and total it up. Slow by nature — prefer the cache.

    Measured on a store of 2,449 files / 2.0 GB this takes 190-1685 ms, because
    it is one ``stat`` per file. That cost is fine once a minute in the
    background and not fine on a request: ``/api/stats`` and
    ``/api/system/info`` both call it, both are synchronous routes, and each
    call therefore held one of the server's shared threadpool workers for up to
    1.7 s — on a store that only grows, so the deployment got slower the longer
    it was used.
    """
    out = {"snapshots": 0, "clips": 0, "processed": 0, "bytes": 0}
    mapping = {
        "snapshots": settings.SNAPSHOTS_DIR,
        "clips": settings.CLIPS_DIR,
        "processed": settings.PROCESSED_DIR,
    }
    for key, directory in mapping.items():
        if not directory.exists():
            continue
        count = 0
        for path in directory.iterdir():
            if path.is_file():
                count += 1
                try:
                    out["bytes"] += path.stat().st_size
                except OSError:
                    pass
        out[key] = count
    out["megabytes"] = round(out["bytes"] / (1024 * 1024), 2)
    out["budget_mb"] = settings.MAX_EVIDENCE_MB
    _store_usage(out)
    return out


def _store_usage(usage: dict) -> None:
    global _usage_cache, _usage_measured_at
    with _usage_lock:
        _usage_cache = dict(usage)
        _usage_measured_at = time.monotonic()


def evidence_usage(max_age: Optional[float] = None) -> dict:
    """
    Current evidence footprint, for the dashboard's storage indicator.

    Served from the last measurement when it is recent enough. The evidence
    sweeper refreshes it on its own schedule, so in a running server this is a
    dictionary copy and the walk never lands on a request at all. ``max_age=0``
    forces a fresh measurement.
    """
    limit = USAGE_MAX_AGE_SECONDS if max_age is None else max_age
    with _usage_lock:
        cached, measured_at = _usage_cache, _usage_measured_at
    if cached is not None:
        age = time.monotonic() - measured_at
        if age <= limit:
            return {**cached, "measured_seconds_ago": round(age, 1)}
    return {**measure_evidence_usage(), "measured_seconds_ago": 0.0}


def invalidate_evidence_usage() -> None:
    """Forget the cached footprint — after a sweep, or a hard reset."""
    global _usage_cache, _usage_measured_at
    with _usage_lock:
        _usage_cache = None
        _usage_measured_at = 0.0


def is_safe_evidence_path(path: str | Path) -> bool:
    """
    Confirm a path really lives inside an approved evidence directory.

    Guards the file-serving endpoints against path traversal and against a
    tampered database row pointing at, say, ``C:\\Windows\\System32``.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return False
    for root in settings.evidence_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False
