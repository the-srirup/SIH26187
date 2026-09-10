"""
Application configuration.

Every tunable in IBVAP lives here — detection thresholds, inference
resolution, processing cadence, rule timings, evidence retention, night
hours, upload limits.  Values are read from environment variables / ``.env``
so an operator can retune a Border Out Post deployment without touching
code.  No magic numbers are scattered through the pipeline.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    PROJECT_NAME: str = "IBVAP"
    VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite:///./alerts.db"

    # ------------------------------------------------------------------ #
    # Media / storage
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    DASHBOARD_DIR: Path = BASE_DIR / "dashboard"
    ALERTS_DIR: Path = BASE_DIR / "alerts"
    CLIPS_DIR: Path = BASE_DIR / "clips"
    SNAPSHOTS_DIR: Path = ALERTS_DIR / "snapshots"
    VIDEOS_DIR: Path = BASE_DIR / "videos"                    # uploaded sources
    PROCESSED_DIR: Path = BASE_DIR / "videos" / "processed"    # annotated renders
    LOG_DIR: Path = BASE_DIR / "logs"

    #: Default camera feed used when seeding a fresh install.
    DEFAULT_CAMERA_URL: str = ""

    # ------------------------------------------------------------------ #
    # Detection model
    # ------------------------------------------------------------------ #
    MODEL_PATH: str = "yolo11n.pt"
    #: "auto" picks CUDA when available and falls back to CPU cleanly.
    DEVICE: str = "auto"
    #: FP16 on CUDA only; ignored on CPU (torch CPU has no fast fp16 path).
    USE_HALF: bool = True
    #: Inference letterbox size. Benchmarked on this pipeline (RTX 4060, yolo11n,
    #: 640x384 input): 640 gave 0.38 detections/frame vs 0.32 at 480 — double the
    #: vehicle recall — at a *lower* median latency (17.4 ms vs 19.3 ms), because
    #: the model is small enough that fixed overhead dominates and 640 needs no
    #: awkward letterbox padding. On a CPU-only host 640 costs ~1.8x the pixels;
    #: set INFERENCE_IMGSZ=480 in .env there.
    INFERENCE_IMGSZ: int = 640
    #: Below 0.30 the extra detections were marginal and added flicker; above it
    #: distant vehicles were dropped.
    DEFAULT_CONFIDENCE: float = 0.30
    NMS_IOU: float = 0.45
    MAX_DETECTIONS: int = 50
    #: Batch frames from concurrent cameras into one forward pass. Measured on
    #: an RTX 4060 with yolo11n@640: batch 4 costs 21.0 ms total (5.25 ms per
    #: frame) versus 4 x 17.5 ms served one at a time — a 3.3x throughput gain,
    #: because the per-call launch overhead dominates a model this small.
    #: The batcher never waits to fill a batch, so single-camera latency is
    #: unchanged. Set false to force strictly sequential inference.
    INFERENCE_BATCHING: bool = True
    INFERENCE_BATCH_MAX: int = 8
    #: COCO ids kept for border surveillance:
    #: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 6 train, 7 truck, 8 boat
    DETECT_CLASSES: list[int] = [0, 1, 2, 3, 5, 6, 7, 8]
    #: Minimum box area (px^2) below which a detection is ignored.
    MIN_OBJECT_AREA: int = 400

    # ------------------------------------------------------------------ #
    # Tracking (ByteTrack — one independent instance per video source)
    # ------------------------------------------------------------------ #
    TRACK_HIGH_THRESH: float = 0.45
    TRACK_LOW_THRESH: float = 0.10
    NEW_TRACK_THRESH: float = 0.50
    TRACK_BUFFER: int = 45          # frames a lost track survives before removal
    MATCH_THRESH: float = 0.85
    #: Exponential smoothing factor for drawn boxes (0 = off, 1 = no smoothing).
    #: Applied to rendering only — rule geometry uses the raw foot point.
    BOX_SMOOTHING: float = 0.55

    # ------------------------------------------------------------------ #
    # Video pipeline
    # ------------------------------------------------------------------ #
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 384
    #: Analytics cadence. The capture thread always runs at source speed and
    #: keeps only the newest frame, so raising this never builds a backlog.
    TARGET_FPS: int = 15
    #: Seconds without a decoded frame before a source is flagged OFFLINE.
    CAMERA_TIMEOUT: float = 6.0
    RECONNECT_INTERVAL: float = 4.0
    #: Loop finite video files (demo behaviour for the sample clip).
    LOOP_FILE_SOURCES: bool = True
    #: JPEG quality for the MJPEG stream (encoded once per frame, shared).
    JPEG_QUALITY: int = 72

    # ------------------------------------------------------------------ #
    # Low-light / night
    # ------------------------------------------------------------------ #
    NIGHT_START_HOUR: int = 19       # IST
    NIGHT_END_HOUR: int = 6          # IST
    #: Mean luma below which CLAHE enhancement kicks in.
    LOW_LIGHT_THRESHOLD: float = 80.0
    CLAHE_CLIP_LIMIT: float = 2.5
    #: Force night analytics regardless of clock — for demoing with day footage.
    FORCE_NIGHT_MODE: bool = False
    #: Treat a visually dark frame as night even during daylight hours.
    NIGHT_BY_LUMINANCE: bool = True
    NIGHT_MOVEMENT_MIN_TRAVEL: float = 45.0    # px of travel before alerting
    NIGHT_MOVEMENT_DEBOUNCE: float = 45.0      # seconds per track

    # ------------------------------------------------------------------ #
    # Rules / analytics
    # ------------------------------------------------------------------ #
    #: Global per-(rule, track, event) cooldown — the alert debouncer.
    DEBOUNCE_SECONDS: float = 12.0
    #: Consecutive confirmations on the new side/zone before a rule fires,
    #: in frames of the analytics cadence.
    ANCHOR_CONFIRMATION_FRAMES: int = 3
    LOITER_SECONDS: float = 15.0
    #: Restricted-zone presence threshold (seconds) before a dwell alert.
    ZONE_PRESENCE_SECONDS: float = 5.0
    #: Presence/first-sighting events for plain human & vehicle detections.
    PRESENCE_ALERTS_ENABLED: bool = True
    PRESENCE_MIN_CONFIDENCE: float = 0.55
    PRESENCE_DEBOUNCE_SECONDS: float = 60.0
    #: Drop rule state for tracks unseen for this long (prevents dict growth).
    TRACK_STATE_TTL: float = 120.0

    # ------------------------------------------------------------------ #
    # Evidence
    # ------------------------------------------------------------------ #
    EVIDENCE_ENABLED: bool = True
    #: Rolling pre-alert buffer, in seconds of analytics-rate frames.
    CLIP_PRE_SECONDS: float = 4.0
    CLIP_POST_SECONDS: float = 4.0
    #: Retention guard — oldest evidence is pruned past these caps.
    MAX_EVIDENCE_MB: int = 2048
    MAX_EVIDENCE_AGE_DAYS: int = 30
    EVIDENCE_SWEEP_INTERVAL: float = 900.0     # seconds between retention sweeps

    # ------------------------------------------------------------------ #
    # Face detection / watchlist recognition
    # ------------------------------------------------------------------ #
    FACE_ENABLED: bool = True
    FACE_RECOGNITION_EVERY_N_FRAMES: int = 12
    FACE_DET_SIZE: int = 320          # SCRFD input; 640 is 4x the pixels
    FACE_MIN_HEIGHT: int = 32
    FACE_MATCH_CACHE_SECONDS: float = 30.0
    FACE_SIMILARITY_THRESHOLD: float = 0.55
    FACE_ALERT_DEBOUNCE_SECONDS: float = 45.0
    #: Max person crops embedded per cadence tick — bounds worst-case latency.
    FACE_MAX_CROPS_PER_TICK: int = 3
    #: Emit a "face detected" event even when the face is not on the watchlist.
    FACE_DETECTION_ALERTS: bool = True
    FACE_DETECT_DEBOUNCE_SECONDS: float = 60.0

    # ------------------------------------------------------------------ #
    # ANPR
    # ------------------------------------------------------------------ #
    ANPR_ENABLED: bool = True
    #: OCR is expensive — only run it on this cadence, and only when a vehicle
    #: is actually present in the frame.
    ANPR_EVERY_N_FRAMES: int = 10
    ANPR_MAX_PLATES_PER_TICK: int = 2
    #: OCR confidence below which a plate is reported as PLATE UNCERTAIN
    #: instead of a fabricated-looking number.
    ANPR_CONFIDENCE_THRESHOLD: float = 0.45
    #: Confidence required before an ANPR event is written to the log.
    ANPR_ALERT_CONFIDENCE: float = 0.55
    ANPR_MIN_PLATE_AREA: int = 500
    ANPR_MIN_ASPECT_RATIO: float = 1.8
    ANPR_MAX_ASPECT_RATIO: float = 6.5
    ANPR_ALERT_DEBOUNCE_SECONDS: float = 30.0
    ANPR_LANGUAGES: str = "en"

    # ------------------------------------------------------------------ #
    # Video upload / offline analysis
    # ------------------------------------------------------------------ #
    UPLOAD_MAX_MB: int = 512
    UPLOAD_ALLOWED_EXTENSIONS: list[str] = [".mp4"]
    #: Analyse at most every Nth decoded frame of an uploaded file. 1 = all.
    UPLOAD_FRAME_STRIDE: int = 2
    #: Write an annotated MP4 render of the analysed upload.
    UPLOAD_RENDER_OUTPUT: bool = True
    UPLOAD_MAX_CONCURRENT: int = 1
    #: Uploaded-video sessions retained in memory.
    UPLOAD_KEEP_SESSIONS: int = 25

    # ------------------------------------------------------------------ #
    # Integrity (tamper-evident hash chain)
    # ------------------------------------------------------------------ #
    #: Seconds between Merkle checkpoints over the alert chain.
    CHECKPOINT_INTERVAL: float = 300.0

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    #: CORS origins. "*" is convenient on a closed LAN; set explicit origins
    #: for anything reachable beyond the BOP network.
    CORS_ORIGINS: list[str] = ["*"]
    #: Cap on concurrent MJPEG viewers per camera.
    MAX_STREAM_CLIENTS: int = 12

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.FRAME_WIDTH, self.FRAME_HEIGHT)

    @property
    def clip_pre_frames(self) -> int:
        return max(1, int(self.CLIP_PRE_SECONDS * self.TARGET_FPS))

    @property
    def clip_post_frames(self) -> int:
        return max(1, int(self.CLIP_POST_SECONDS * self.TARGET_FPS))

    @property
    def evidence_roots(self) -> tuple[Path, ...]:
        """Directories the API is permitted to serve files from."""
        return (
            self.SNAPSHOTS_DIR.resolve(),
            self.CLIPS_DIR.resolve(),
            self.VIDEOS_DIR.resolve(),
            self.PROCESSED_DIR.resolve(),
        )

    def ensure_dirs(self) -> None:
        """Create runtime directories that don't yet exist."""
        for d in (
            self.ALERTS_DIR, self.CLIPS_DIR, self.SNAPSHOTS_DIR,
            self.VIDEOS_DIR, self.PROCESSED_DIR, self.STATIC_DIR, self.LOG_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
