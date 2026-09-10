"""
Application configuration. Centralised settings loaded from environment
variables with sensible defaults for a LAN/hackathon deployment.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    PROJECT_NAME: str = "IBVAP"
    VERSION: str = "1.0.0"

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
    VIDEOS_DIR: Path = BASE_DIR / "videos"

    # Default camera feed used for the single-camera demo.
    DEFAULT_CAMERA_URL: str = ""

    # ------------------------------------------------------------------ #
    # CV
    # ------------------------------------------------------------------ #
    MODEL_PATH: str = "yolo11n.pt"
    TRACKER: str = "bytetrack.yaml"

    # Frame buffer for evidence clips — ~5 s at 30 FPS.
    BUFFER_SIZE: int = 150
    POST_ALERT_FRAMES: int = 50
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 5

    # ------------------------------------------------------------------ #
    # Rules / detection
    # ------------------------------------------------------------------ #
    # Minimum box area (in pixels) below which a detection is ignored.
    MIN_OBJECT_AREA: int = 300
    DEFAULT_CONFIDENCE: float = 0.25

    # Debounce window in seconds — prevents rule spam.
    DEBOUNCE_SECONDS: float = 10.0

    # Loitering threshold.
    LOITER_SECONDS: int = 60

    # Face recognition / watchlist matching.
    #
    # Running SCRFD + ArcFace on every frame is expensive on edge hardware.
    # We process face recognition on a sampling cadence and cache matches by
    # YOLO track-id, so a face that is already classified as a watchlist hit
    # is not re-embedded every frame.
    FACE_RECOGNITION_EVERY_N_FRAMES: int = 5
    FACE_MIN_HEIGHT: int = 40          # minimum face height in pixels
    FACE_MATCH_CACHE_SECONDS: float = 30.0
    FACE_SIMILARITY_THRESHOLD: float = 0.55  # cosine similarity threshold
    FACE_ALERT_DEBOUNCE_SECONDS: float = 30.0

    # ANPR.
    #
    # The ANPR module actively crops vehicle bounding boxes from YOLO, finds
    # high-contrast rectangular license-plate candidates, and runs OCR only on
    # those candidate regions.  This is much cheaper than OCR-ing every frame.
    ANPR_ENABLED: bool = True
    ANPR_CONFIDENCE_THRESHOLD: float = 0.45
    ANPR_MIN_PLATE_AREA: int = 700
    ANPR_MIN_ASPECT_RATIO: float = 1.6
    ANPR_MAX_ASPECT_RATIO: float = 6.5
    ANPR_ALERT_DEBOUNCE_SECONDS: float = 12.0

    # Foot-Point Anchor confirmation — number of consecutive frames
    # the foot point must remain in the alert zone before triggering.
    # This prevents false alarms from environmental shifts (swaying
    # branches, passing shadows, transient occlusions).
    ANCHOR_CONFIRMATION_FRAMES: int = 5  # ~1 second at 5 FPS

    # Maximum number of frames to wait before a foot point is considered
    # a confirmed anchor in the alert zone.
    ANCHOR_MAX_FRAMES: int = 30

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    def ensure_dirs(self) -> None:
        """Create runtime directories that don't yet exist."""
        for d in (self.ALERTS_DIR, self.CLIPS_DIR, self.SNAPSHOTS_DIR,
                  self.VIDEOS_DIR, self.STATIC_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
