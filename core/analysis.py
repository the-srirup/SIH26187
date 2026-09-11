"""
Offline analysis of uploaded video.

An uploaded MP4 is analysed by the **same** :class:`~core.analytics.FrameAnalyzer`
that drives a live camera — same detector, same tracker, same rules engine,
same event manager, same hash chain.  The only differences are structural and
deliberate:

* frames come from :class:`~core.video_source.FileSource` (never dropped),
* "now" is **media time**, so a person loitering for 30 s of footage triggers
  the loiter rule after 30 s of *video*, not 30 s of wall clock,
* an annotated MP4 render is written so the operator can replay exactly what
  the analytics saw.

Events land in the same ``alerts`` table, tagged ``source_type='upload'`` with
the session id, so uploaded-video findings are searchable next to live ones
and are sealed by the same tamper-evident chain.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

from core.analytics import FrameAnalyzer
from core.config import settings
from core.database import SessionLocal
from core.evidence import ClipRecorder
from core.events import EventManager
from core.models import AnalysisSession, Camera
from core.timeutil import fmt_ist, utc_iso
from core.video_source import FileSource, probe_video

log = logging.getLogger("ibvap.analysis")

_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc

#: Only these bytes may appear in a stored filename.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
#: ISO base media file format signature — 'ftyp' at offset 4.
_MP4_BRANDS = (b"ftyp",)


class UploadValidationError(ValueError):
    """Raised when an uploaded file is rejected before it is analysed."""


def sanitize_filename(name: str) -> str:
    """
    Reduce a client-supplied filename to a safe basename.

    Strips any directory component (defeating ``../../etc/passwd`` and
    ``C:\\Windows\\...`` style inputs), removes every character outside a
    conservative allowlist, and bounds the length.
    """
    base = Path(str(name or "")).name
    base = base.replace("\x00", "")
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "upload"
    stem, _, ext = cleaned.rpartition(".")
    if not stem:
        stem, ext = cleaned, ""
    return f"{stem[:80]}.{ext[:8]}" if ext else stem[:80]


def validate_upload(path: Path, original_name: str) -> dict:
    """
    Validate an uploaded file before it is accepted for analysis.

    Checks the extension, the size budget, the container signature (so a
    renamed executable is rejected regardless of its name) and finally that
    the file actually decodes.  Raises :class:`UploadValidationError` with a
    message intended for the operator.
    """
    suffix = Path(original_name).suffix.lower()
    allowed = {e.lower() for e in settings.UPLOAD_ALLOWED_EXTENSIONS}
    if suffix not in allowed:
        raise UploadValidationError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if not path.exists():
        raise UploadValidationError("Upload did not reach the server.")

    size = path.stat().st_size
    if size == 0:
        raise UploadValidationError("Uploaded file is empty.")
    if size > settings.UPLOAD_MAX_MB * 1024 * 1024:
        raise UploadValidationError(
            f"File is {size / (1024 * 1024):.0f} MB — the limit is "
            f"{settings.UPLOAD_MAX_MB} MB."
        )

    with path.open("rb") as handle:
        header = handle.read(32)
    if not any(brand in header for brand in _MP4_BRANDS):
        raise UploadValidationError(
            "File is not a valid MP4 container (missing ISO-BMFF signature)."
        )

    info = probe_video(path)
    if not info.get("valid"):
        raise UploadValidationError(info.get("error", "Video could not be decoded."))

    info["size_bytes"] = size
    return info


class AnalysisJob:
    """One uploaded-video analysis run, executed on a worker thread."""

    def __init__(
        self,
        session_uid: str,
        stored_path: Path,
        original_name: str,
        probe: dict,
        camera_id: Optional[int] = None,
        rules: Optional[list[dict]] = None,
    ) -> None:
        self.session_uid = session_uid
        self.stored_path = stored_path
        self.original_name = original_name
        self.probe = probe
        self.camera_id = camera_id
        self.rules = rules or []

        self.status = "queued"
        self.error = ""
        self.progress = 0.0
        self.processed_frames = 0
        self.analysed_frames = 0
        self.total_frames = int(probe.get("frame_count") or 0)
        self.current_time = 0.0
        self.processing_fps = 0.0
        self.detections_total = 0
        self.alerts_generated = 0
        self.persons_seen: set = set()
        self.vehicles_seen: set = set()
        self.output_path = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self._cancel = threading.Event()
        self._preview_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def preview(self) -> Optional[bytes]:
        with self._lock:
            return self._preview_jpeg

    def snapshot(self) -> dict:
        """Serialisable progress state for the API and WebSocket."""
        duration = float(self.probe.get("duration_seconds") or 0.0)
        return {
            "session_id": self.session_uid,
            "filename": self.original_name,
            "status": self.status,
            "error": self.error,
            "progress": round(self.progress * 100, 1),
            "processed_frames": self.processed_frames,
            "analysed_frames": self.analysed_frames,
            "total_frames": self.total_frames,
            "current_time_seconds": round(self.current_time, 2),
            "duration_seconds": round(duration, 2),
            "processing_fps": round(self.processing_fps, 1),
            "detections_total": self.detections_total,
            "persons": len(self.persons_seen),
            "vehicles": len(self.vehicles_seen),
            "alerts": self.alerts_generated,
            "resolution": f"{self.probe.get('width')}x{self.probe.get('height')}",
            "source_fps": round(float(self.probe.get("fps") or 0), 2),
            "size_mb": round(float(self.probe.get("size_bytes", 0)) / (1024 * 1024), 2),
            "has_output": bool(self.output_path),
            "output_url": f"/api/analysis/{self.session_uid}/video" if self.output_path else None,
            "camera_id": self.camera_id,
        }

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Execute the analysis. Runs on a worker thread, never on the loop."""
        events = EventManager.get()
        self.status = "running"
        self.started_at = time.time()
        self._persist()
        self._emit(events)

        analyzer = FrameAnalyzer(
            source_id=f"upload-{self.session_uid[:8]}",
            display_name=f"UPLOAD · {self.original_name[:28]}",
            frame_rate=int(self.probe.get("fps") or settings.TARGET_FPS),
        )
        analyzer.set_rules(self.rules)

        clips = ClipRecorder(
            source_id=f"upload{self.session_uid[:8]}",
            fps=float(self.probe.get("fps") or settings.TARGET_FPS)
            / max(1, settings.UPLOAD_FRAME_STRIDE),
        )
        writer = None
        source = None
        last_emit = 0.0

        try:
            source = FileSource(self.stored_path, stride=settings.UPLOAD_FRAME_STRIDE)

            if settings.UPLOAD_RENDER_OUTPUT:
                settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                out_path = settings.PROCESSED_DIR / f"{self.session_uid}_analysed.mp4"
                out_fps = max(1.0, source.fps / max(1, settings.UPLOAD_FRAME_STRIDE))
                writer = cv2.VideoWriter(
                    str(out_path), _FOURCC(*"mp4v"), out_fps, settings.frame_size
                )
                if writer.isOpened():
                    self.output_path = str(out_path)
                else:
                    log.warning("Could not open output writer for %s", out_path)
                    writer.release()
                    writer = None

            analysed = 0
            wall_start = time.perf_counter()

            for frame_index, media_time, frame in source:
                if self._cancel.is_set():
                    self.status = "cancelled"
                    break

                result = analyzer.analyse(
                    frame,
                    timestamp=media_time,      # media time, not wall clock
                    fps=self.processing_fps,
                    online=True,
                )
                analysed += 1
                self.analysed_frames = analysed
                self.processed_frames = frame_index + 1
                self.current_time = media_time
                self.detections_total += len(result.detections)
                for det in result.detections:
                    (self.persons_seen if det.is_person else self.vehicles_seen).add(det.track_id)

                if writer is not None:
                    writer.write(result.frame)
                if settings.EVIDENCE_ENABLED:
                    clips.push(result.frame)

                self._record_events(events, analyzer, result, clips, media_time)

                if self.total_frames:
                    self.progress = min(1.0, self.processed_frames / self.total_frames)
                elapsed = time.perf_counter() - wall_start
                if elapsed > 0:
                    self.processing_fps = analysed / elapsed

                now = time.time()
                if now - last_emit >= 0.4:
                    last_emit = now
                    ok, buf = cv2.imencode(
                        ".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
                    )
                    if ok:
                        with self._lock:
                            self._preview_jpeg = buf.tobytes()
                    self._emit(events)

            if self.status != "cancelled":
                self.status = "completed"
                self.progress = 1.0

        except Exception as exc:
            log.exception("Analysis %s failed: %s", self.session_uid, exc)
            self.status = "failed"
            self.error = str(exc)[:400]
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if source is not None:
                source.release()
            clips.close()
            self.finished_at = time.time()
            self._persist()
            self._emit(events)
            log.info(
                "Analysis %s %s — %d frames analysed, %d event(s), %.1f fps",
                self.session_uid, self.status, self.analysed_frames,
                self.alerts_generated, self.processing_fps,
            )

    # ------------------------------------------------------------------ #
    def _record_events(self, events, analyzer, result, clips, media_time: float) -> None:
        """Seal this frame's events, tagged to the upload session."""
        by_track = {d.track_id: d for d in result.detections}

        for rule_alert in result.alerts:
            det = by_track.get(rule_alert.track_id)
            payload = events.record(
                camera_id=self.camera_id or 0,
                rule_alert=rule_alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=clips,
                object_class=det.class_name if det else "",
                confidence=det.confidence if det else 0.0,
                camera_name=f"UPLOAD · {self.original_name[:24]}",
                source_type="upload",
                session_id=self.session_uid,
            )
            if payload:
                self.alerts_generated += 1

        recognizer = analyzer._face()
        if recognizer is not None:
            for match in result.faces:
                alert = recognizer.build_event(match, analyzer.source_id)
                if alert is None:
                    continue
                if events.record(
                    camera_id=self.camera_id or 0, rule_alert=alert,
                    frame=result.frame, clean_frame=result.raw_frame,
                    clip_recorder=clips, object_class="face",
                    confidence=float(match.similarity if match.matched else match.det_score),
                    camera_name=f"UPLOAD · {self.original_name[:24]}",
                    source_type="upload", session_id=self.session_uid,
                ):
                    self.alerts_generated += 1

        processor = analyzer._anpr_processor()
        if processor is not None:
            for plate in result.plates:
                alert = processor.build_event(plate, analyzer.source_id)
                if alert is None:
                    continue
                if events.record(
                    camera_id=self.camera_id or 0, rule_alert=alert,
                    frame=result.frame, clean_frame=result.raw_frame,
                    clip_recorder=clips,
                    object_class=plate.vehicle_class or "vehicle",
                    confidence=float(plate.text_confidence),
                    camera_name=f"UPLOAD · {self.original_name[:24]}",
                    source_type="upload", session_id=self.session_uid,
                ):
                    self.alerts_generated += 1

    def _emit(self, events: EventManager) -> None:
        events.broadcast({"type": "analysis", "data": self.snapshot()})

    def _persist(self) -> None:
        """Mirror progress into the database so results survive a restart."""
        db = SessionLocal()
        try:
            row = (
                db.query(AnalysisSession)
                .filter(AnalysisSession.session_uid == self.session_uid)
                .first()
            )
            if row is None:
                now = utc_iso()
                row = AnalysisSession(
                    session_uid=self.session_uid,
                    filename=self.original_name,
                    stored_path=str(self.stored_path),
                    camera_id=self.camera_id,
                    created_at=now,
                    created_at_ist=fmt_ist(now),
                )
                db.add(row)

            row.status = self.status
            row.error = self.error
            row.output_path = self.output_path
            row.total_frames = self.total_frames
            row.processed_frames = self.processed_frames
            row.analysed_frames = self.analysed_frames
            row.duration_seconds = float(self.probe.get("duration_seconds") or 0.0)
            row.source_fps = float(self.probe.get("fps") or 0.0)
            row.width = int(self.probe.get("width") or 0)
            row.height = int(self.probe.get("height") or 0)
            row.size_bytes = int(self.probe.get("size_bytes") or 0)
            row.detections_total = self.detections_total
            row.persons_seen = len(self.persons_seen)
            row.vehicles_seen = len(self.vehicles_seen)
            row.alerts_generated = self.alerts_generated
            row.processing_fps = round(self.processing_fps, 2)
            if self.status in ("completed", "failed", "cancelled"):
                done = utc_iso()
                row.completed_at = done
                row.completed_at_ist = fmt_ist(done)
            db.commit()
        except Exception as exc:
            log.warning("Could not persist analysis session %s: %s", self.session_uid, exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()


class AnalysisManager:
    """Owns upload storage and the analysis worker pool."""

    _instance: Optional["AnalysisManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max(1, settings.UPLOAD_MAX_CONCURRENT))

    @classmethod
    def get(cls) -> "AnalysisManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    def store_upload(self, temp_path: Path, original_name: str) -> tuple[str, Path, dict]:
        """
        Validate and move an uploaded file into managed storage.

        The stored name is derived from a server-generated session id, so a
        hostile client cannot influence the path at all.  A rejected upload is
        deleted immediately rather than left on disk.
        """
        safe_name = sanitize_filename(original_name)
        try:
            probe = validate_upload(temp_path, safe_name)
        except UploadValidationError:
            temp_path.unlink(missing_ok=True)
            raise

        session_uid = uuid.uuid4().hex
        settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        stored = settings.VIDEOS_DIR / f"{session_uid}_{safe_name}"
        shutil.move(str(temp_path), str(stored))
        log.info("Upload stored: %s (%.1f MB) -> %s",
                 safe_name, probe["size_bytes"] / (1024 * 1024), stored.name)
        return session_uid, stored, probe

    def submit(
        self,
        session_uid: str,
        stored_path: Path,
        original_name: str,
        probe: dict,
        camera_id: Optional[int] = None,
        rules: Optional[list[dict]] = None,
    ) -> AnalysisJob:
        """Queue an analysis job on a worker thread."""
        job = AnalysisJob(session_uid, stored_path, original_name, probe, camera_id, rules)
        with self._lock:
            self._jobs[session_uid] = job
            self._order.append(session_uid)
            self._prune_locked()

        def _worker() -> None:
            with self._semaphore:
                job.run()

        threading.Thread(target=_worker, name=f"analysis-{session_uid[:8]}",
                         daemon=True).start()
        return job

    def _prune_locked(self) -> None:
        keep = max(1, settings.UPLOAD_KEEP_SESSIONS)
        while len(self._order) > keep:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def get_job(self, session_uid: str) -> Optional[AnalysisJob]:
        return self._jobs.get(session_uid)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = [self._jobs[uid] for uid in reversed(self._order) if uid in self._jobs]
        return [job.snapshot() for job in jobs]

    def cancel(self, session_uid: str) -> bool:
        job = self._jobs.get(session_uid)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.cancel()
        return True


def rules_for_camera(camera_id: Optional[int]) -> list[dict]:
    """
    Load a camera's rules as plain dicts, for reuse on an uploaded video.

    Letting an upload inherit CAM-01's fence is what makes "analyse this
    recording against the same tripwire" work without redrawing anything.
    """
    if not camera_id:
        return []

    import json

    from core.models import Rule

    db = SessionLocal()
    try:
        rows = (
            db.query(Rule)
            .filter(Rule.camera_id == camera_id, Rule.is_active.is_(True))
            .all()
        )
        out = []
        for row in rows:
            try:
                out.append({
                    "id": row.id,
                    "rule_type": row.rule_type,
                    "geometry": json.loads(row.geometry) if row.geometry else [],
                    "params": json.loads(row.params) if row.params else {},
                    "name": row.name or f"{row.rule_type}-{row.id}",
                })
            except (json.JSONDecodeError, TypeError):
                continue
        return out
    finally:
        db.close()


def ensure_upload_camera() -> int:
    """
    Return the id of the pseudo-camera that owns uploaded-video events.

    Alerts carry a foreign key to ``cameras``; giving uploads a dedicated
    registered source keeps that constraint honest and makes upload findings
    filterable in the event log like any other source.
    """
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.source_kind == "upload").first()
        if cam:
            return cam.id
        now = utc_iso()
        cam = Camera(
            name="UPLOADED VIDEO",
            url="upload://analysis",
            location="Offline analysis",
            is_active=False,
            is_online=False,
            source_kind="upload",
            created_at=now,
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)
        return cam.id
    finally:
        db.close()
