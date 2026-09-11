"""
Object detection and multi-object tracking.

Design notes
------------
**One model, many streams.**  ``Detector`` loads YOLO exactly once and is
shared by every camera thread.  Inference is serialised behind a lock because
a single ultralytics predictor is not re-entrant.

**Tracking is per-stream.**  The previous implementation called
``model.track(persist=True)`` from every camera thread.  ByteTrack state lives
on the *shared* predictor, so two cameras fed one another's tracks — IDs
jumped between streams and reset constantly.  Here each video source owns a
private :class:`ObjectTracker`, so ``PERSON #12`` on CAM-01 is independent of
``PERSON #12`` on CAM-02 and IDs stay stable.

**Device selection is automatic.**  CUDA is used when torch reports it, with
FP16 where supported; otherwise the model runs on CPU with the torch thread
count tuned to the machine.  A missing GPU degrades performance, never
correctness.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.cv.detector")

#: COCO class ids that map onto border-surveillance categories.
PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 5, 6, 7, 8}

#: Display labels — uppercase, operator-facing.
CLASS_DISPLAY = {
    "person": "PERSON", "bicycle": "BICYCLE", "car": "CAR",
    "motorcycle": "MOTORCYCLE", "bus": "BUS", "train": "TRAIN",
    "truck": "TRUCK", "boat": "BOAT",
}


class ModelNotAvailable(RuntimeError):
    """Raised when the YOLO weights cannot be located or loaded."""


@dataclass
class Detection:
    """A single tracked detection in one frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2) raw detector output
    foot: tuple[int, int]                # bottom-centre — ground contact point
    #: Smoothed box used for rendering only; rules always use ``foot``.
    draw_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    age: int = 1                         # frames this track has been alive

    @property
    def label(self) -> str:
        return CLASS_DISPLAY.get(self.class_name, self.class_name.upper())

    @property
    def is_person(self) -> bool:
        return self.class_id in PERSON_CLASSES

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASSES

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class FrameResult:
    """All detections for one processed frame."""

    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    inference_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #


def resolve_device(requested: str = "auto") -> tuple[str, bool, dict]:
    """
    Pick an inference device.

    Returns ``(device, use_half, info)``.  ``info`` is surfaced by
    ``/api/system/info`` so the dashboard can prove which device is actually
    in use rather than assuming the GPU is being exercised.
    """
    info: dict = {"requested": requested, "cuda_available": False, "gpu_name": None}
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        info["error"] = "torch not installed"
        return "cpu", False, info

    info["torch_version"] = torch.__version__
    cuda_ok = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_ok
    if cuda_ok:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:  # pragma: no cover - driver quirks
            pass

    if requested and requested != "auto":
        device = requested
    else:
        device = "cuda:0" if cuda_ok else "cpu"

    if device.startswith("cuda") and not cuda_ok:
        log.warning(
            "DEVICE=%s requested but CUDA is unavailable — falling back to CPU. "
            "Install a CUDA build of torch to use the GPU.", requested
        )
        device = "cpu"

    use_half = bool(settings.USE_HALF and device.startswith("cuda"))

    if device == "cpu":
        # Leave a couple of cores for capture / encoding / the web server.
        threads = max(1, (os.cpu_count() or 4) - 2)
        try:
            torch.set_num_threads(threads)
            info["torch_threads"] = threads
        except Exception:  # pragma: no cover
            pass

    info["device"] = device
    info["half"] = use_half
    return device, use_half, info


# --------------------------------------------------------------------------- #
# Per-stream tracker
# --------------------------------------------------------------------------- #


class _TrackerInput:
    """
    Minimal ultralytics ``Results``-like view over plain numpy arrays.

    ``BYTETracker`` only needs ``len()``, boolean-mask indexing, ``conf``,
    ``cls`` and ``xywh``.  Building this directly avoids constructing a full
    ``Results`` object (and its torch tensors) once per frame.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_TrackerInput":
        return _TrackerInput(self.xywh[mask], self.conf[mask], self.cls[mask])

    @classmethod
    def from_xyxy(cls, xyxy: np.ndarray, conf: np.ndarray, class_ids: np.ndarray) -> "_TrackerInput":
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return cls(
            xywh,
            np.asarray(conf, dtype=np.float32).reshape(-1),
            np.asarray(class_ids, dtype=np.float32).reshape(-1),
        )


class ObjectTracker:
    """
    A private ByteTrack instance for one video source.

    Wraps ultralytics' ``BYTETracker`` and adds:

    * exponential box smoothing for rendering (kills the jitter that made
      overlays look unstable) while rules keep the raw foot point;
    * track age, so a rule can require an object to be established;
    * bounded state — dead tracks are evicted, so a 12-hour run does not grow.
    """

    #: ``BYTETracker.__init__`` calls ``reset_id()``, which zeroes the *global*
    #: STrack counter.  Creating a tracker for a newly added camera would
    #: therefore hand out IDs that collide with live tracks on an existing
    #: camera.  We snapshot and restore the counter around construction.
    _id_lock = threading.Lock()

    def __init__(self, frame_rate: Optional[int] = None) -> None:
        from types import SimpleNamespace

        self._frame_rate = int(frame_rate or settings.TARGET_FPS) or 1

        # ``track_buffer`` is how many *frames* a lost track survives before it
        # is discarded. Expressing that as a raw frame count makes its real
        # meaning depend on the analytics cadence: 45 frames is 3 s at 15 FPS
        # but 1.5 s at 30 FPS, so retuning TARGET_FPS silently halved how long a
        # track survived an occlusion. The intent is a *duration*, so it is
        # configured in seconds and converted per stream here.
        buffer_frames = max(
            int(settings.TRACK_BUFFER_MIN_FRAMES),
            int(round(self._frame_rate * settings.TRACK_LOST_SECONDS)),
        )
        self._args = SimpleNamespace(
            track_high_thresh=settings.TRACK_HIGH_THRESH,
            track_low_thresh=settings.TRACK_LOW_THRESH,
            new_track_thresh=settings.NEW_TRACK_THRESH,
            track_buffer=buffer_frames,
            match_thresh=settings.MATCH_THRESH,
            fuse_score=True,
        )
        self.track_buffer_frames = buffer_frames
        self._tracker = self._new_tracker()
        self._smoothed: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}

    def _new_tracker(self):
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        with self._id_lock:
            preserved = getattr(BaseTrack, "_count", 0)
            # Older ultralytics took ``frame_rate`` and derived
            # ``max_time_lost = frame_rate / 30 * track_buffer`` from it; current
            # versions dropped the parameter and use ``track_buffer`` directly as
            # a frame count. Support both rather than pinning a version — the
            # duration is already baked into track_buffer above either way.
            try:
                tracker = BYTETracker(self._args, frame_rate=self._frame_rate)
            except TypeError:
                tracker = BYTETracker(self._args)
            BaseTrack._count = preserved
        return tracker

    def reset(self) -> None:
        """Restart tracking (used when a source reconnects or a file loops)."""
        self._tracker = self._new_tracker()
        self._smoothed.clear()
        self._age.clear()

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        names: dict,
    ) -> list[Detection]:
        """Associate raw detections with persistent track IDs."""
        payload = _TrackerInput.from_xyxy(
            boxes_xyxy if boxes_xyxy is not None else np.zeros((0, 4), np.float32),
            scores if scores is not None else np.zeros((0,), np.float32),
            class_ids if class_ids is not None else np.zeros((0,), np.float32),
        )

        try:
            # An empty tick still matters: lost tracks must age out.
            tracks = self._tracker.update(payload)
        except Exception as exc:  # pragma: no cover - tracker edge cases
            log.warning("Tracker update failed (%s) — dropping frame's tracks", exc)
            tracks = np.empty((0, 8), dtype=np.float32)

        if tracks is None or len(tracks) == 0:
            self._gc(set())
            return []

        detections: list[Detection] = []
        alive: set[int] = set()
        alpha = float(np.clip(settings.BOX_SMOOTHING, 0.0, 1.0))

        for row in tracks:
            # ultralytics returns [x1, y1, x2, y2, track_id, score, cls, idx]
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            track_id = int(row[4])
            score = float(row[5])
            class_id = int(row[6])

            raw = np.array([x1, y1, x2, y2], dtype=np.float32)
            if (x2 - x1) * (y2 - y1) < settings.MIN_OBJECT_AREA:
                continue

            prev = self._smoothed.get(track_id)
            smooth = raw if prev is None else prev + alpha * (raw - prev)
            self._smoothed[track_id] = smooth
            self._age[track_id] = self._age.get(track_id, 0) + 1
            alive.add(track_id)

            class_name = names.get(class_id, str(class_id)) if names else str(class_id)
            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=score,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    foot=(int((x1 + x2) / 2), int(y2)),
                    draw_bbox=tuple(int(v) for v in smooth),  # type: ignore[arg-type]
                    age=self._age[track_id],
                )
            )

        self._gc(alive)
        return detections

    def _gc(self, alive: set[int]) -> None:
        """Evict smoothing/age state for tracks the tracker has dropped."""
        if len(self._smoothed) <= 256:
            return
        try:
            live_ids = {int(t.track_id) for t in self._tracker.tracked_stracks}
        except Exception:  # pragma: no cover
            live_ids = set()
        keep = alive | live_ids
        for tid in [t for t in self._smoothed if t not in keep]:
            self._smoothed.pop(tid, None)
            self._age.pop(tid, None)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


@dataclass
class _InferenceRequest:
    """One camera thread's pending frame, awaiting a batched forward pass."""

    frame: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[tuple] = None
    error: Optional[Exception] = None


def _precision_kwargs(use_half: bool) -> dict:
    """
    Express FP16 in whichever form the installed ultralytics understands.

    ultralytics 8.4 replaced ``half=`` with a unified ``quantize=`` scheme
    (16 = FP16, 8 = INT8, None = FP32) and warns on the old name. That warning
    is emitted **inside every predict call**, so on a 15 fps feed it printed
    fifteen identical deprecation lines per second per camera, which is exactly
    the per-frame log spam that makes a real fault invisible in an incident
    review. Detected once at import rather than probed per call.
    """
    if not use_half:
        return {}
    return {"quantize": 16} if _SUPPORTS_QUANTIZE else {"half": True}


def _detect_quantize_support() -> bool:
    try:
        from ultralytics.cfg import get_cfg  # noqa: F401
        import ultralytics.cfg as _cfg

        return "quantize" in getattr(_cfg, "CFG_INT_KEYS", set()) or _cfg_has_quantize()
    except Exception:
        return False


def _cfg_has_quantize() -> bool:
    """True when the packaged default config declares ``quantize``."""
    try:
        import ultralytics.cfg as _cfg
        from ultralytics.utils import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except Exception:
        return False


_SUPPORTS_QUANTIZE = _detect_quantize_support()


class Detector:
    """
    Shared YOLO model with adaptive batched inference.

    A single model serving N camera threads behind a plain lock serialises
    them: measured on an RTX 4060 with yolo11n at 640px, one frame costs
    17.5 ms but the GPU sits at ~25% — almost all of that is per-call launch
    overhead, not compute.  Four cameras therefore cost 4 x 17.5 ms while the
    GPU idles.

    Batching removes exactly that overhead.  Measured on the same machine:

        batch 1  17.49 ms total   17.49 ms/frame   1.00x
        batch 2  18.25 ms total    9.12 ms/frame   1.92x
        batch 4  20.99 ms total    5.25 ms/frame   3.33x
        batch 8  23.33 ms total    2.92 ms/frame   6.00x

    The batcher **never waits to fill a batch**.  It blocks for the first
    request, then drains whatever else is already queued.  With one camera the
    batch is 1 and latency is unchanged; with several cameras the queue is
    naturally non-empty by the time a pass completes, so batches form for free.
    """

    _instance: Optional["Detector"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or settings.MODEL_PATH
        self.device, self.half, self.device_info = resolve_device(settings.DEVICE)
        self._lock = threading.Lock()
        self._infer_times: list[float] = []
        self._infer_count = 0

        self._queue: "queue.Queue[_InferenceRequest]" = queue.Queue(maxsize=64)
        self._batch_thread: Optional[threading.Thread] = None
        self._batch_running = False
        self._batch_sizes: list[int] = []

        resolved = self._resolve_weights(self.model_path)
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(resolved))
            self.model.to(self.device)
        except Exception as exc:
            raise ModelNotAvailable(
                f"Failed to load YOLO weights from {resolved!s}: {exc}"
            ) from exc

        self.names: dict[int, str] = dict(self.model.names)
        self._warmup()
        if settings.INFERENCE_BATCHING:
            self._start_batch_worker()
        log.info(
            "YOLO loaded: %s | device=%s | half=%s | imgsz=%d | classes=%d",
            resolved.name, self.device, self.half, settings.INFERENCE_IMGSZ,
            len(self.names),
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls) -> "Detector":
        """Process-wide singleton — the model is loaded exactly once."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def _resolve_weights(model_path: str) -> Path:
        """
        Locate weights on disk.

        Search order: the configured path as given, then relative to the
        project root.  Ultralytics will download a known model name on first
        use; anything else that cannot be found raises a clear error rather
        than failing deep inside the camera thread.
        """
        candidate = Path(model_path)
        if candidate.is_file():
            return candidate.resolve()

        project_local = (settings.BASE_DIR / model_path).resolve()
        if project_local.is_file():
            return project_local

        known = {
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
            "yolov8n.pt", "yolov8s.pt", "yolo26n.pt",
        }
        if candidate.name in known:
            log.warning(
                "Model file %s not found locally — ultralytics will download it "
                "into %s on first use.", candidate.name, settings.BASE_DIR
            )
            return project_local

        raise ModelNotAvailable(
            f"Model weights '{model_path}' not found. Place the .pt file in "
            f"{settings.BASE_DIR} or set MODEL_PATH in .env."
        )

    def _warmup(self) -> None:
        """
        Run throwaway inferences so the first real frame is not 10x slow — and
        prove the selected device actually works end to end.

        A CUDA build of torch paired with a CPU-only torchvision, a driver
        mismatch, or an out-of-memory GPU all fail *here* rather than inside a
        camera thread at demo time.  On failure the detector falls back to CPU
        and says so, which is the difference between degraded performance and
        a dead surveillance platform.
        """
        blank = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
        try:
            for _ in range(3):
                self.detect(blank)
        except Exception as exc:
            if self.device == "cpu":
                log.warning("Model warm-up failed on CPU: %s", exc)
            else:
                log.error(
                    "Inference failed on %s (%s) — falling back to CPU. "
                    "Check that torch AND torchvision are both CUDA builds.",
                    self.device, exc,
                )
                self.device_info["fallback_reason"] = str(exc)[:200]
                self.device_info["fallback_from"] = self.device
                self.device, self.half = "cpu", False
                self.device_info["device"], self.device_info["half"] = "cpu", False
                try:
                    self.model.to("cpu")
                    self.detect(blank)
                except Exception as cpu_exc:  # pragma: no cover - unrecoverable
                    raise ModelNotAvailable(
                        f"Inference failed on both GPU and CPU: {cpu_exc}"
                    ) from cpu_exc
        finally:
            self._infer_times.clear()
            self._infer_count = 0

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    # -- batching service ------------------------------------------------ #
    def _start_batch_worker(self) -> None:
        if self._batch_thread and self._batch_thread.is_alive():
            return
        self._batch_running = True
        self._batch_thread = threading.Thread(
            target=self._batch_loop, name="yolo-batcher", daemon=True
        )
        self._batch_thread.start()
        log.info("Batched inference enabled (max batch %d)", settings.INFERENCE_BATCH_MAX)

    def stop_batch_worker(self) -> None:
        self._batch_running = False
        if self._batch_thread:
            self._batch_thread.join(timeout=3.0)
            self._batch_thread = None

    def _batch_loop(self) -> None:
        max_batch = max(1, int(settings.INFERENCE_BATCH_MAX))
        while self._batch_running:
            try:
                first = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue

            batch = [first]
            # Drain only what is *already* waiting — never stall to fill a
            # batch, so a single-camera deployment sees no added latency.
            while len(batch) < max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            self._batch_sizes.append(len(batch))
            if len(self._batch_sizes) > 200:
                del self._batch_sizes[:-200]

            try:
                outputs = self._forward([r.frame for r in batch])
                for request, output in zip(batch, outputs):
                    request.result = output
            except Exception as exc:
                log.warning("Batched inference failed: %s", exc)
                for request in batch:
                    request.error = exc
            finally:
                for request in batch:
                    request.done.set()

    def _forward(self, frames: list) -> list:
        """One forward pass over a list of frames; returns parsed arrays."""
        results = self.model.predict(
            frames,
            imgsz=settings.INFERENCE_IMGSZ,
            conf=settings.DEFAULT_CONFIDENCE,
            iou=settings.NMS_IOU,
            max_det=settings.MAX_DETECTIONS,
            classes=settings.DETECT_CLASSES or None,
            device=self.device,
            verbose=False,
            **_precision_kwargs(self.half),
        )
        return [self._parse(result) for result in results]

    @staticmethod
    def _parse(result):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep = areas >= settings.MIN_OBJECT_AREA
        return xyxy[keep], conf[keep], cls[keep]

    @staticmethod
    def _empty():
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32))

    def raw_detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Run detection and return ``(boxes_xyxy, scores, class_ids, elapsed_ms)``.

        Goes through the batching service when it is running, and falls back to
        a direct locked call otherwise — so the detector still works from a
        script, a test, or a one-shot API probe with no worker thread alive.
        """
        t0 = time.perf_counter()

        if self._batch_running and self._batch_thread and self._batch_thread.is_alive():
            request = _InferenceRequest(frame=frame)
            try:
                self._queue.put(request, timeout=2.0)
            except queue.Full:
                log.warning("Inference queue saturated — shedding frame")
                return (*self._empty(), (time.perf_counter() - t0) * 1000.0)

            if not request.done.wait(timeout=20.0):
                log.warning("Inference timed out after 20s")
                return (*self._empty(), (time.perf_counter() - t0) * 1000.0)
            if request.error is not None:
                raise request.error
            xyxy, conf, cls = request.result
        else:
            with self._lock:
                results = self.model.predict(
                    frame,
                    imgsz=settings.INFERENCE_IMGSZ,
                    conf=settings.DEFAULT_CONFIDENCE,
                    iou=settings.NMS_IOU,
                    max_det=settings.MAX_DETECTIONS,
                    classes=settings.DETECT_CLASSES or None,
                    device=self.device,
                    verbose=False,
                    **_precision_kwargs(self.half),
                )[0]
            xyxy, conf, cls = self._parse(results)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record(elapsed)
        return xyxy, conf, cls, elapsed

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detection without tracking — track ids are the box indices."""
        xyxy, conf, cls, _ = self.raw_detect(frame)
        out: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            class_id = int(cls[i])
            out.append(
                Detection(
                    track_id=i,
                    class_id=class_id,
                    class_name=self.names.get(class_id, str(class_id)),
                    confidence=float(conf[i]),
                    bbox=(x1, y1, x2, y2),
                    foot=((x1 + x2) // 2, y2),
                    draw_bbox=(x1, y1, x2, y2),
                )
            )
        return out

    def track(self, frame: np.ndarray, tracker: Optional[ObjectTracker] = None) -> list[Detection]:
        """
        Detect + track using a caller-supplied per-stream tracker.

        Passing ``tracker=None`` falls back to detection-only, which keeps the
        old call signature working for one-shot API endpoints.
        """
        xyxy, conf, cls, _ = self.raw_detect(frame)
        if tracker is None:
            # Reuse the boxes we already have. Calling self.detect() here ran a
            # second forward pass over the same frame for no benefit.
            out: list[Detection] = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                class_id = int(cls[i])
                out.append(
                    Detection(
                        track_id=i,
                        class_id=class_id,
                        class_name=self.names.get(class_id, str(class_id)),
                        confidence=float(conf[i]),
                        bbox=(x1, y1, x2, y2),
                        foot=((x1 + x2) // 2, y2),
                        draw_bbox=(x1, y1, x2, y2),
                    )
                )
            return out
        return tracker.update(xyxy, conf, cls, self.names)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _record(self, elapsed_ms: float) -> None:
        self._infer_count += 1
        self._infer_times.append(elapsed_ms)
        if len(self._infer_times) > 120:
            del self._infer_times[:-120]

    def metrics(self) -> dict:
        times = list(self._infer_times)
        return {
            "device": self.device,
            "half": self.half,
            "imgsz": settings.INFERENCE_IMGSZ,
            "model": Path(self.model_path).name,
            "inference_count": self._infer_count,
            "inference_ms_avg": round(sum(times) / len(times), 2) if times else 0.0,
            "inference_ms_last": round(times[-1], 2) if times else 0.0,
            "batching": bool(self._batch_running),
            "batch_size_avg": round(
                sum(self._batch_sizes) / len(self._batch_sizes), 2
            ) if self._batch_sizes else 0.0,
            "batch_max": settings.INFERENCE_BATCH_MAX,
            **{k: v for k, v in self.device_info.items() if k != "requested"},
        }
