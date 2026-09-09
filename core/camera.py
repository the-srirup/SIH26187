"""
Camera processing pipeline — the beating heart of IBVAP.

Each ``CameraProcessor`` runs in its own thread, continuously pulling frames
from a video source (HTTP stream, RTSP, file, or synthetic generator),
running YOLO detection + ByteTrack, applying the rules engine, enhancing
low-light frames with CLAHE, recording evidence clips on alert, and
persisting tamper-evident alerts with a hash chain.

An annotated frame is published to a shared, thread-safe buffer so the
FastAPI layer can serve MJPEG streams and WebSocket updates without
blocking on inference.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings
from core.database import SessionLocal
from core.hashchain import chain_hash, compute_alert_payload, latest_chain_hash
from core.models import Alert, Camera, Rule
from cv.detector import Detector
from cv.face import FaceRecognizer
from cv.rules import (
    Alert as RuleAlert,
    DirectionRule,
    FenceRule,
    LoiterRule,
    RuleEngine,
    ZoneRule,
)
from cv.anpr import get_anpr_processor, PlateDetection

log = logging.getLogger("ibvap.camera")

# --------------------------------------------------------------------------- #
# Shared frame buffer
# --------------------------------------------------------------------------- #


class FrameBuffer:
    """
    Thread-safe circular buffer of the latest annotated frame per camera.

    The MJPEG streamer reads from here with zero-copy semantics (numpy
    arrays are reference-counted, not copied, as long as we swap atomically).
    """

    _instance: Optional["FrameBuffer"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._frames: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "FrameBuffer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set(self, camera_id: int, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[camera_id] = frame

    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(camera_id)


# --------------------------------------------------------------------------- #
# CLAHE low-light enhancement
# --------------------------------------------------------------------------- #


class LowLightEnhancer:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) on L-channel.

    Applied conditionally when the mean frame luminance falls below a
    threshold (typical at night). Three lines, dramatic improvement on
    dark CCTV footage.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        luminance_threshold: float = 80.0,
    ) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.luminance_threshold = luminance_threshold

    def maybe_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE if frame is dark; otherwise return original."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = float(np.mean(l_channel))
        if mean_l < self.luminance_threshold:
            l_channel = self.clahe.apply(l_channel)
            lab[:, :, 0] = l_channel
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame


# --------------------------------------------------------------------------- #
# Evidence clip writer
# --------------------------------------------------------------------------- #


@dataclass
class ClipWriter:
    """
    Records pre- and post-alert frames to an MP4 file.

    Maintains a rolling buffer of the last N frames. When an alert fires,
    writes the buffer + M additional frames to disk. Runs in the camera
    thread so no extra queue is needed.
    """

    camera_id: int
    clip_dir: Path
    fps: float = settings.TARGET_FPS
    buffer_size: int = settings.BUFFER_SIZE
    post_frames: int = settings.POST_ALERT_FRAMES
    frame_size: tuple[int, int] = (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)

    _buffer: deque = field(default_factory=deque, init=False)
    _writing: bool = field(default=False, init=False)
    _writer: Optional[cv2.VideoWriter] = field(default=None, init=False)
    _frames_remaining: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = deque(maxlen=self.buffer_size)

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling buffer."""
        with self._lock:
            self._buffer.append(frame.copy())

    def start_clip(self, alert_id: int) -> Optional[Path]:
        """Begin writing a new evidence clip.

        Returns the clip's filesystem path, or ``None`` if a clip could not be
        started (another clip already in progress, or VideoWriter failure).
        """
        with self._lock:
            if self._writing:
                # Another clip in progress — ignore (shouldn't happen with debounce)
                return None

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.clip_dir / f"cam{self.camera_id}_alert{alert_id}_{ts}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(path), fourcc, self.fps, self.frame_size
            )
            if not self._writer.isOpened():
                log.error("Failed to open VideoWriter for %s", path)
                self._writing = False
                return None

            # Write pre-alert buffer
            for f in self._buffer:
                self._writer.write(f)

            self._writing = True
            self._frames_remaining = self.post_frames
            return path

    def write_post_frame(self, frame: np.ndarray) -> None:
        """Write one frame during the post-alert window."""
        with self._lock:
            if not self._writing or self._writer is None:
                return
            self._writer.write(frame)
            self._frames_remaining -= 1
            if self._frames_remaining <= 0:
                self._finalize()

    def _finalize(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        self._writing = False
        log.info("Evidence clip finalized for camera %d", self.camera_id)


# --------------------------------------------------------------------------- #
# Camera processor — one thread per camera
# --------------------------------------------------------------------------- #


@dataclass
class CameraProcessor:
    """
    Full processing pipeline for a single camera.

    Thread target: ``run()`` — loops forever (or until ``stop()``).
    """

    camera_id: int
    url: str
    detector: Detector
    face_recognizer: Optional[FaceRecognizer] = None
    claher: Optional[LowLightEnhancer] = None

    # Runtime state
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _rules_engine: Optional[RuleEngine] = field(default=None, init=False)
    _clip_writer: Optional[ClipWriter] = field(default=None, init=False)
    _cap: Optional[cv2.VideoCapture] = field(default=None, init=False)
    _frame_count: int = field(default=0, init=False)
    _anpr_results: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.claher = LowLightEnhancer()
        self._clip_writer = ClipWriter(
            camera_id=self.camera_id,
            clip_dir=settings.CLIPS_DIR,
        )
        self._anpr_processor = get_anpr_processor() if settings.ANPR_ENABLED else None
        self._load_rules()

    def _load_rules(self) -> None:
        """Load active rules from DB and build RuleEngine."""
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if not camera:
                raise ValueError(f"Camera {self.camera_id} not found")

            self._rules_engine = RuleEngine(camera_id=str(self.camera_id))

            rules = db.query(Rule).filter(Rule.camera_id == self.camera_id, Rule.is_active == True).all()  # noqa: E712
            import json

            for rule in rules:
                geom = json.loads(rule.geometry) if rule.geometry else []
                params = json.loads(rule.params) if rule.params else {}

                if rule.rule_type == "line":
                    if len(geom) >= 2:
                        r = FenceRule(rule.name or f"line_{rule.id}", *geom[0], *geom[1])
                        self._rules_engine.add_rule(r)
                elif rule.rule_type == "zone":
                    if len(geom) >= 3:
                        r = ZoneRule(rule.name or f"zone_{rule.id}", geom)
                        self._rules_engine.add_rule(r)
                elif rule.rule_type == "loiter":
                    if len(geom) >= 3:
                        dwell = params.get("dwell_seconds", settings.LOITER_SECONDS)
                        r = LoiterRule(rule.name or f"loiter_{rule.id}", geom, dwell)
                        self._rules_engine.add_rule(r)
                elif rule.rule_type == "direction":
                    if len(geom) >= 2:
                        allowed = params.get("allowed_direction", "entry")
                        r = DirectionRule(
                            rule.name or f"dir_{rule.id}",
                            *geom[0],
                            *geom[1],
                            allowed_direction=allowed,
                        )
                        self._rules_engine.add_rule(r)
        finally:
            db.close()

    def reload_rules(self) -> None:
        """Hot-reload rules without stopping the camera thread."""
        self._load_rules()
        log.info("Rules reloaded for camera %d", self.camera_id)

    # ------------------------------------------------------------------ #
    # Video capture
    # ------------------------------------------------------------------ #
    def _open_capture(self) -> cv2.VideoCapture:
        """Open video source with retries and fallback."""
        # If URL is a digit, treat as webcam index
        if self.url.isdigit():
            src = int(self.url)
        else:
            src = self.url

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            # Fallback: synthetic test pattern
            log.warning("Failed to open %s, using synthetic frames", self.url)
            cap = cv2.VideoCapture(self._synthetic_generator())
        return cap

    @staticmethod
    def _synthetic_generator() -> str:
        """Return a GStreamer pipeline for synthetic video (for demo)."""
        return (
            "videotestsrc pattern=ball ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! appsink"
        )

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read one frame, handling reconnection and looped playback."""
        if self._cap is None:
            self._cap = self._open_capture()

        ok, frame = self._cap.read()
        if not ok:
            # End of file or stream drop — try seek-back for local files
            log.warning("Frame read failed, attempting seek to frame 0...")
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total > 0 and pos >= total - 1:
                # Finite file reached EOF — loop back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok:
                    log.info("Looped video back to frame 0")
            if not ok:
                log.warning("Reconnect attempt...")
                self._cap.release()
                self._cap = self._open_capture()
                ok, frame = self._cap.read()
                if not ok:
                    return None

        # Resize for consistent processing
        frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        return frame

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Main processing loop — runs in background thread."""
        self._running = True
        log.info("Camera %d processor started", self.camera_id)

        frame_interval = 1.0 / settings.TARGET_FPS
        next_frame_time = time.time()

        while self._running:
            loop_start = time.time()

            frame = self._read_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Low-light enhancement
            frame = self.claher.maybe_enhance(frame)

            # Detect + track
            detections = self.detector.track(frame)

            # Face recognition (if enabled)
            if self.face_recognizer:
                face_results = self.face_recognizer.recognize(frame)
                # Could trigger face alerts here

            # ANPR processing (if enabled)
            self._anpr_results = []
            if self._anpr_processor and self._anpr_processor.is_available():
                # Preprocess for better Indian plate recognition if needed
                processed_frame = self._anpr_processor.preprocess_for_indian_plates(frame)
                self._anpr_results = self._anpr_processor.detect_and_recognize(processed_frame)
                # Could trigger ANPR-based alerts here (e.g., watchlist plate matching)

            # Run rules engine
            alerts = []
            if self._rules_engine:
                alerts = self._rules_engine.update(detections)

            # Persist alerts & write evidence clips
            for rule_alert in alerts:
                self._persist_alert(rule_alert, frame, detections)

            # Annotate frame for streaming
            annotated = self._annotate_frame(frame, detections, alerts)

            # Push to shared buffer
            FrameBuffer.get().set(self.camera_id, annotated)

            # Push to clip buffer
            self._clip_writer.push(annotated)

            self._frame_count += 1

            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("Camera %d processor stopped", self.camera_id)

    def _persist_alert(
        self, rule_alert: RuleAlert, frame: np.ndarray, detections: list
    ) -> None:
        """Save alert to DB with hash chain and start evidence clip."""
        db = SessionLocal()
        try:
            # Find object class for this track
            obj_class = ""
            confidence = 0.0
            for det in detections:
                if det.track_id == rule_alert.track_id:
                    obj_class = det.class_name
                    confidence = det.confidence
                    break

            # Hash chain - get previous hash
            prev_hash = latest_chain_hash(db)
            timestamp = datetime.now(timezone.utc).isoformat()
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Create a temporary alert to get an ID assigned FIRST
            temp_alert = Alert(
                camera_id=self.camera_id,
                alert_type=rule_alert.alert_type,
                object_class=obj_class,
                track_id=rule_alert.track_id,
                confidence=confidence,
                timestamp=timestamp,
                snapshot_path="",  # will be set after ID known
                clip_path="",      # will be set after ID known
                prev_hash=prev_hash,
                hash="",  # placeholder
            )
            db.add(temp_alert)
            db.flush()  # Get the real ID

            # NOW we know the real ID - compute final paths
            settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            real_snap_path = settings.SNAPSHOTS_DIR / f"cam{self.camera_id}_alert{temp_alert.id}_{ts}.jpg"

            # Save snapshot with final path
            cv2.imwrite(str(real_snap_path), frame)

            # Start clip with real alert ID
            clip_path = self._clip_writer.start_clip(temp_alert.id)
            real_clip_path = str(clip_path) if clip_path else ""

            # Now compute the CORRECT hash using the FINAL paths
            # This must match compute_alert_payload exactly
            payload = {
                "id": temp_alert.id,
                "camera_id": temp_alert.camera_id,
                "alert_type": temp_alert.alert_type,
                "object_class": temp_alert.object_class or "",
                "track_id": temp_alert.track_id or 0,
                "confidence": temp_alert.confidence or 0.0,
                "timestamp": temp_alert.timestamp,
                "snapshot_path": str(real_snap_path),
                "clip_path": real_clip_path,
            }
            alert_hash = chain_hash(payload, prev_hash)

            # Update the alert with the correct hash and final paths
            temp_alert.hash = alert_hash
            temp_alert.snapshot_path = str(real_snap_path)
            temp_alert.clip_path = real_clip_path

            db.commit()
            log.info("Alert persisted: %s (id=%d hash=%s...)", rule_alert.alert_type, temp_alert.id, alert_hash[:16])

        except Exception as e:
            log.exception("Failed to persist alert: %s", e)
            db.rollback()
        finally:
            db.close()

    def _annotate_frame(
        self, frame: np.ndarray, detections: list, alerts: list
    ) -> np.ndarray:
        """Draw boxes, tracks, rules, and alerts on frame for streaming."""
        out = frame.copy()

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{det.track_id} {det.class_name} {det.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Foot point
            cv2.circle(out, det.foot, 4, (0, 0, 255), -1)

        # Draw rules
        if self._rules_engine:
            import json
            db = SessionLocal()
            try:
                rules = db.query(Rule).filter(Rule.camera_id == self.camera_id, Rule.is_active == True).all()  # noqa: E712
                for rule in rules:
                    geom = json.loads(rule.geometry) if rule.geometry else []
                    if rule.rule_type == "line" and len(geom) >= 2:
                        p1 = tuple(map(int, geom[0]))
                        p2 = tuple(map(int, geom[1]))
                        cv2.line(out, p1, p2, (255, 255, 0), 3)
                        cv2.putText(
                            out, rule.name or "fence", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                        )
                    elif rule.rule_type in ("zone", "loiter") and len(geom) >= 3:
                        pts = np.array(geom, dtype=np.int32).reshape((-1, 1, 2))
                        color = (0, 255, 255) if rule.rule_type == "zone" else (255, 0, 255)
                        cv2.polylines(out, [pts], True, color, 2)
                        cv2.putText(out, rule.name or rule.rule_type, geom[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            finally:
                db.close()

        # Draw active alerts
        for alert in alerts:
            cv2.putText(
                out,
                f"ALERT: {alert.alert_type.upper()} ID:{alert.track_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        # Draw ANPR results (if available)
        if self._anpr_results:
            for i, plate in enumerate(self._anpr_results):
                x1, y1, x2, y2 = plate.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange for plates
                label = f"PLATE: {plate.plate_text} ({plate.text_confidence:.2f})"
                cv2.putText(out, label, (x1, y2 + 20 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # Camera ID overlay
        cv2.putText(
            out,
            f"CAM {self.camera_id} | {datetime.now().strftime('%H:%M:%S')}",
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return out

    def start(self) -> None:
        """Start the background processing thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the processing thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._cap:
            self._cap.release()


# --------------------------------------------------------------------------- #
# Camera manager — orchestrates all cameras
# --------------------------------------------------------------------------- #


class CameraManager:
    """Singleton that owns all CameraProcessor instances."""

    _instance: Optional["CameraManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cameras: dict[int, CameraProcessor] = {}
        self._detector = Detector()
        self._face_recognizer = FaceRecognizer()

    @classmethod
    def get(cls) -> "CameraManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def add_camera(self, camera: Camera) -> CameraProcessor:
        """Create and start processor for a camera."""
        if camera.id in self._cameras:
            return self._cameras[camera.id]

        proc = CameraProcessor(
            camera_id=camera.id,
            url=camera.url,
            detector=self._detector,
            face_recognizer=self._face_recognizer,
        )
        self._cameras[camera.id] = proc
        proc.start()
        return proc

    def remove_camera(self, camera_id: int) -> None:
        """Stop and remove a camera processor."""
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
        for proc in self._cameras.values():
            proc.reload_rules()

    def stop_all(self) -> None:
        for proc in list(self._cameras.values()):
            proc.stop()
        self._cameras.clear()

