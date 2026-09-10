"""
Live camera processing.

Threading model — one camera, two threads:

    [capture thread]  decode as fast as the source emits, keep newest frame
            |
            v  (latest-frame handoff, older frames dropped)
    [analytics thread]  analyse -> rules -> events -> encode -> publish

The capture thread never waits for inference, and the analytics thread never
waits for the decoder.  If inference falls behind, frames are *dropped*, not
queued, so end-to-end latency stays flat instead of growing without bound —
this is the fix for the original build's frame backlog, where a single
serial loop read, inferred and slept in lockstep at a hard-capped 5 FPS.

The analytics thread also encodes the JPEG **once** and publishes the bytes.
Every MJPEG viewer then shares that one encode, instead of each client
re-encoding the same frame on its own timer.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.analytics import AnalysisResult, FrameAnalyzer, LowLightEnhancer
from core.config import settings
from core.database import SessionLocal
from core.evidence import ClipRecorder
from core.events import EventManager
from core.models import Camera, Rule
from cv import overlay as ov
from cv.detector import Detector
from cv.rules import Alert as RuleAlert

log = logging.getLogger("ibvap.camera")

_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc


# --------------------------------------------------------------------------- #
# Shared frame buffer — annotated frame + pre-encoded JPEG per camera
# --------------------------------------------------------------------------- #


class FrameBuffer:
    """
    Latest annotated frame *and* its JPEG bytes, per camera.

    Storing the encoded bytes here is what removes per-client encoding: with
    four cameras and three dashboard tabs the old code performed ~400 JPEG
    encodes per second of largely identical frames.  Now it performs one per
    produced frame, regardless of viewer count.

    ``seq`` lets a streaming client block until a genuinely new frame exists
    rather than polling on a timer.
    """

    _instance: Optional["FrameBuffer"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._frames: dict[int, np.ndarray] = {}
        self._jpegs: dict[int, bytes] = {}
        self._seq: dict[int, int] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @classmethod
    def get(cls) -> "FrameBuffer":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def publish(self, camera_id: int, frame: np.ndarray, jpeg: Optional[bytes] = None) -> None:
        with self._cond:
            self._frames[camera_id] = frame
            if jpeg is not None:
                self._jpegs[camera_id] = jpeg
            self._seq[camera_id] = self._seq.get(camera_id, 0) + 1
            self._cond.notify_all()

    # Backwards-compatible alias used by older call sites / tests.
    def set(self, camera_id: int, frame: np.ndarray) -> None:
        self.publish(camera_id, frame)

    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(camera_id)

    def get_jpeg(self, camera_id: int) -> Optional[bytes]:
        with self._lock:
            return self._jpegs.get(camera_id)

    def sequence(self, camera_id: int) -> int:
        with self._lock:
            return self._seq.get(camera_id, 0)

    def wait_for_jpeg(self, camera_id: int, last_seq: int, timeout: float = 2.0):
        """Block until a frame newer than ``last_seq`` is published."""
        deadline = time.time() + timeout
        with self._cond:
            while self._seq.get(camera_id, 0) <= last_seq:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            return self._jpegs.get(camera_id), self._seq.get(camera_id, 0)

    def drop(self, camera_id: int) -> None:
        with self._cond:
            self._frames.pop(camera_id, None)
            self._jpegs.pop(camera_id, None)
            self._seq.pop(camera_id, None)
            self._cond.notify_all()


# --------------------------------------------------------------------------- #
# Camera processor
# --------------------------------------------------------------------------- #


class CameraProcessor:
    """Full live pipeline for a single camera."""

    def __init__(
        self,
        camera_id: int,
        url: str,
        name: str = "",
        location: str = "",
        detector: Optional[Detector] = None,
    ) -> None:
        from core.video_source import LiveSource

        self.camera_id = int(camera_id)
        self.url = url
        self.name = name or f"CAM-{camera_id:02d}"
        self.location = location

        self.source = LiveSource(url, name=self.name)
        self.analyzer = FrameAnalyzer(
            source_id=str(camera_id), display_name=self.name, detector=detector
        )
        self.clips = ClipRecorder(source_id=str(camera_id))
        self.events = EventManager.get()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_id = -1

        # Runtime metrics — all measured, none fabricated.
        self._fps = 0.0
        self._fps_window: list[float] = []
        self._frames_analysed = 0
        self._inference_ms = 0.0
        self._pipeline_ms = 0.0
        self._latency_ms = 0.0
        self._last_detections = 0
        self._person_count = 0
        self._vehicle_count = 0
        self._is_night = False
        self._online = False
        self._started_at = 0.0
        self._offline_announced = False
        self._last_status_write = 0.0

        self.reload_rules()

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #
    def reload_rules(self) -> None:
        """
        Load rules from the database into the analyzer's in-memory engine.

        Called on start and whenever a rule changes — never per frame.  The
        previous build queried SQLite once per frame *just to draw the fence*.
        """
        import json

        db = SessionLocal()
        try:
            rows = (
                db.query(Rule)
                .filter(Rule.camera_id == self.camera_id, Rule.is_active.is_(True))
                .all()
            )
            payload = []
            for row in rows:
                try:
                    geometry = json.loads(row.geometry) if row.geometry else []
                    params = json.loads(row.params) if row.params else {}
                except (json.JSONDecodeError, TypeError):
                    log.warning("Rule %s has malformed JSON — skipped", row.id)
                    continue
                payload.append({
                    "id": row.id,
                    "rule_type": row.rule_type,
                    "geometry": geometry,
                    "params": params,
                    "name": row.name or f"{row.rule_type}-{row.id}",
                })
            self.analyzer.set_rules(payload)
        except Exception as exc:
            log.exception("Failed to load rules for camera %d: %s", self.camera_id, exc)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._started_at = time.time()
        self.source.start()
        self._thread = threading.Thread(
            target=self._run, name=f"analytics-cam{self.camera_id}", daemon=True
        )
        self._thread.start()
        log.info("Camera %d (%s) started — source=%s", self.camera_id, self.name, self.url)

    def stop(self) -> None:
        self._running = False
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.clips.close()
        FrameBuffer.get().drop(self.camera_id)
        self._set_online(False, persist=True)
        log.info("Camera %d (%s) stopped", self.camera_id, self.name)

    # ------------------------------------------------------------------ #
    # Analytics loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        buffer = FrameBuffer.get()
        min_interval = 1.0 / max(1, settings.TARGET_FPS)
        next_deadline = time.time()

        while self._running:
            frame, frame_id, captured_at = self.source.read(
                timeout=1.0, last_id=self._last_frame_id
            )

            if frame is None:
                self._handle_no_frame(buffer)
                continue

            self._last_frame_id = frame_id
            self._set_online(True)

            # Cadence limiter. The capture thread keeps draining regardless, so
            # skipping here sheds load without ever building a backlog.
            now = time.time()
            if now < next_deadline:
                continue
            next_deadline = max(now, next_deadline) + min_interval

            try:
                result = self.analyzer.analyse(
                    frame, timestamp=now, fps=self._fps, online=True
                )
            except Exception as exc:
                log.exception("Camera %d analysis error: %s", self.camera_id, exc)
                time.sleep(0.25)
                continue

            self._publish(buffer, result, captured_at)
            self._handle_alerts(result)
            self._update_metrics(result, captured_at)

        log.info("Camera %d analytics loop exited", self.camera_id)

    def _handle_no_frame(self, buffer: "FrameBuffer") -> None:
        """No frame within the read timeout — decide whether we are offline."""
        if self.source.is_online:
            return

        # Starting up is not the same as going down. Opening an RTSP stream or
        # a large file can take a few seconds; announcing OFFLINE in that
        # window produced a false alert on every single restart.
        if not self.source.ever_connected:
            if time.time() - self._started_at < settings.CAMERA_TIMEOUT:
                time.sleep(0.2)
                return

        if self._online or not self._offline_announced:
            self._set_online(False, persist=True)
            self._offline_announced = True
            card = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), np.uint8)
            ov.draw_offline(card, self.name, self.source.stats.last_error)
            ok, encoded = cv2.imencode(
                ".jpg", card, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            buffer.publish(self.camera_id, card, encoded.tobytes() if ok else None)
            self._emit_system_event(
                "camera_offline",
                f"{self.name} lost signal: {self.source.stats.last_error or 'no frames received'}",
            )
        time.sleep(0.2)

    def _publish(self, buffer: "FrameBuffer", result: AnalysisResult, captured_at: float) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
        )
        buffer.publish(self.camera_id, result.frame, encoded.tobytes() if ok else None)
        if settings.EVIDENCE_ENABLED:
            self.clips.push(result.frame)

    def _handle_alerts(self, result: AnalysisResult) -> None:
        if not result.alerts:
            return
        by_track = {d.track_id: d for d in result.detections}
        for rule_alert in result.alerts:
            det = by_track.get(rule_alert.track_id)
            self.events.record(
                camera_id=self.camera_id,
                rule_alert=rule_alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class=det.class_name if det else "",
                confidence=det.confidence if det else 0.0,
                camera_name=self.name,
                source_type="live",
            )

        # Face and ANPR produce their own event types on their own cadence.
        self._handle_face_events(result)
        self._handle_anpr_events(result)

    def _handle_face_events(self, result: AnalysisResult) -> None:
        if not result.faces:
            return
        recognizer = self.analyzer._face()
        if recognizer is None:
            return
        for match in result.faces:
            alert = recognizer.build_event(match)
            if alert is None:
                continue
            self.events.record(
                camera_id=self.camera_id,
                rule_alert=alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class="face",
                confidence=float(match.similarity if match.matched else match.det_score),
                camera_name=self.name,
                source_type="live",
            )

    def _handle_anpr_events(self, result: AnalysisResult) -> None:
        if not result.plates:
            return
        processor = self.analyzer._anpr_processor()
        if processor is None:
            return
        for plate in result.plates:
            alert = processor.build_event(plate)
            if alert is None:
                continue
            self.events.record(
                camera_id=self.camera_id,
                rule_alert=alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class=plate.vehicle_class or "vehicle",
                confidence=float(plate.text_confidence),
                camera_name=self.name,
                source_type="live",
            )

    def _emit_system_event(self, alert_type: str, description: str) -> None:
        """Log a system-level condition (offline, error) into the audit chain."""
        self.events.record(
            camera_id=self.camera_id,
            rule_alert=RuleAlert(
                rule_name="system", rule_type="system", track_id=0,
                alert_type=alert_type,  # type: ignore[arg-type]
                description=description,
                details={"camera": self.name, "url": self.url},
            ),
            frame=None,
            camera_name=self.name,
            source_type="live",
            capture_evidence=False,
        )

    # ------------------------------------------------------------------ #
    # Status / metrics
    # ------------------------------------------------------------------ #
    def _update_metrics(self, result: AnalysisResult, captured_at: float) -> None:
        now = time.time()
        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self._fps = (len(self._fps_window) - 1) / span

        self._frames_analysed += 1
        self._inference_ms = result.inference_ms
        self._pipeline_ms = result.total_ms
        if captured_at:
            self._latency_ms = max(0.0, (now - captured_at) * 1000.0)
        self._last_detections = len(result.detections)
        self._person_count = result.person_count
        self._vehicle_count = result.vehicle_count
        self._is_night = result.is_night

    def _set_online(self, online: bool, persist: bool = False) -> None:
        changed = online != self._online
        self._online = online
        if online:
            self._offline_announced = False

        now = time.time()
        if not (changed or persist):
            return
        # Throttle DB writes — status changes are rare, but a flapping RTSP
        # link must not turn into a write storm.
        if not changed and now - self._last_status_write < 30.0:
            return
        self._last_status_write = now

        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if cam and cam.is_online != online:
                cam.is_online = online
                db.commit()
                log.info("Camera %d (%s) is now %s", self.camera_id, self.name,
                         "ONLINE" if online else "OFFLINE")
        except Exception as exc:
            log.warning("Camera status write failed: %s", exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    @property
    def is_online(self) -> bool:
        return self._online and self.source.is_online

    def stats(self) -> dict:
        """Live runtime metrics — every value measured from this process."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "url": self.url,
            "online": self.is_online,
            "fps": round(self._fps, 1),
            "frames_analysed": self._frames_analysed,
            "inference_ms": round(self._inference_ms, 1),
            "pipeline_ms": round(self._pipeline_ms, 1),
            "latency_ms": round(self._latency_ms, 1),
            "detections": self._last_detections,
            "persons": self._person_count,
            "vehicles": self._vehicle_count,
            "night_mode": self._is_night,
            "uptime_seconds": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "rules": len(self.analyzer.rule_shapes),
            "active_clips": self.clips.active_clips,
            "source": self.source.describe(),
        }


# --------------------------------------------------------------------------- #
# Camera manager
# --------------------------------------------------------------------------- #


class CameraManager:
    """
    Owns every :class:`CameraProcessor`.

    Cameras are fully independent — one failing source cannot stall another —
    while the expensive resource (the YOLO model) is shared.
    """

    _instance: Optional["CameraManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._cameras: dict[int, CameraProcessor] = {}
        self._lock = threading.Lock()
        self._detector: Optional[Detector] = None

    @classmethod
    def get(cls) -> "CameraManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector.get()
        return self._detector

    def add_camera(self, camera: Camera) -> Optional[CameraProcessor]:
        with self._lock:
            existing = self._cameras.get(camera.id)
            if existing:
                return existing
            try:
                proc = CameraProcessor(
                    camera_id=camera.id,
                    url=camera.url,
                    name=camera.name,
                    location=camera.location or "",
                    detector=self.detector,
                )
            except Exception as exc:
                log.exception("Cannot start camera %d (%s): %s", camera.id, camera.name, exc)
                return None
            self._cameras[camera.id] = proc
        proc.start()
        return proc

    def remove_camera(self, camera_id: int) -> None:
        with self._lock:
            proc = self._cameras.pop(camera_id, None)
        if proc:
            proc.stop()

    def get_camera(self, camera_id: int) -> Optional[CameraProcessor]:
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list[CameraProcessor]:
        return list(self._cameras.values())

    def reload_camera_rules(self, camera_id: int) -> None:
        proc = self._cameras.get(camera_id)
        if proc:
            proc.reload_rules()

    def reload_all_rules(self) -> None:
        for proc in self.list_cameras():
            proc.reload_rules()

    def stats(self) -> list[dict]:
        return [proc.stats() for proc in self.list_cameras()]

    def aggregate(self) -> dict:
        """System-wide live counters for the dashboard's stat row."""
        cameras = self.list_cameras()
        online = [c for c in cameras if c.is_online]
        return {
            "cameras_total": len(cameras),
            "cameras_online": len(online),
            "persons_live": sum(c._person_count for c in online),
            "vehicles_live": sum(c._vehicle_count for c in online),
            "detections_live": sum(c._last_detections for c in online),
            "system_fps": round(sum(c._fps for c in online), 1),
            "avg_inference_ms": round(
                sum(c._inference_ms for c in online) / len(online), 1
            ) if online else 0.0,
            "avg_latency_ms": round(
                sum(c._latency_ms for c in online) / len(online), 1
            ) if online else 0.0,
            "night_mode": any(c._is_night for c in online),
        }

    def stop_all(self) -> None:
        with self._lock:
            processors = list(self._cameras.values())
            self._cameras.clear()
        for proc in processors:
            try:
                proc.stop()
            except Exception as exc:
                log.warning("Error stopping camera %d: %s", proc.camera_id, exc)


# Backwards-compatible re-export: older imports expect these from core.camera.
ClipWriter = ClipRecorder
__all__ = [
    "FrameBuffer", "CameraProcessor", "CameraManager",
    "ClipRecorder", "ClipWriter", "LowLightEnhancer",
]
