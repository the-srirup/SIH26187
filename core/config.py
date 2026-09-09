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

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ANPR (Stretch Goal)
    ANPR_ENABLED: bool = False
    ANPR_CONFIDENCE_THRESHOLD: float = 0.5

    def ensure_dirs(self) -> None:
        """Create runtime directories that don't yet exist."""
        for d in (self.ALERTS_DIR, self.CLIPS_DIR, self.SNAPSHOTS_DIR,
                  self.VIDEOS_DIR, self.STATIC_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
