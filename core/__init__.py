"""
Core package for IBVAP — configuration, database, models, hash chain,
and camera processing pipeline.
"""
from core.config import settings
from core.database import SessionLocal, get_db, init_db, engine, Base
from core.models import Camera, Rule, Alert
from core.hashchain import (
    chain_hash,
    compute_alert_payload,
    verify_chain,
    latest_chain_hash,
    VerificationResult,
)
from core.camera import (
    FrameBuffer,
    CameraProcessor,
    CameraManager,
    ClipWriter,
    LowLightEnhancer,
)

__all__ = [
    # Config
    "settings",
    # Database
    "SessionLocal",
    "get_db",
    "init_db",
    "engine",
    "Base",
    # Models
    "Camera",
    "Rule",
    "Alert",
    # Hash chain
    "chain_hash",
    "compute_alert_payload",
    "verify_chain",
    "latest_chain_hash",
    "VerificationResult",
    # Camera pipeline
    "FrameBuffer",
    "CameraProcessor",
    "CameraManager",
    "ClipWriter",
    "LowLightEnhancer",
]