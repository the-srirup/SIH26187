"""
Core package for IBVAP — configuration, database, models, hash chain,
and the video-analytics pipeline.

Submodules are exposed lazily (PEP 562).  ``core.camera`` pulls in the CV
stack, which itself imports ``core.config``; importing it eagerly here made
``import cv.detector`` fail with a circular-import error whenever ``cv`` was
imported before ``core``.  Lazy attribute access removes that ordering trap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import settings
from core.database import SessionLocal, get_db, init_db, engine, Base
from core.models import Alert, Camera, Rule, WatchlistEntry
from core.timeutil import (
    IST,
    fmt_ist,
    now_ist,
    now_utc,
    to_ist,
    utc_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.camera import (
        CameraManager,
        CameraProcessor,
        ClipWriter,
        FrameBuffer,
        LowLightEnhancer,
    )
    from core.hashchain import (
        VerificationResult,
        chain_hash,
        compute_alert_payload,
        latest_chain_hash,
        verify_chain,
    )

_LAZY: dict[str, str] = {
    "FrameBuffer": "core.camera",
    "CameraProcessor": "core.camera",
    "CameraManager": "core.camera",
    "ClipWriter": "core.camera",
    "LowLightEnhancer": "core.camera",
    "chain_hash": "core.hashchain",
    "compute_alert_payload": "core.hashchain",
    "verify_chain": "core.hashchain",
    "latest_chain_hash": "core.hashchain",
    "VerificationResult": "core.hashchain",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "settings",
    "SessionLocal", "get_db", "init_db", "engine", "Base",
    "Camera", "Rule", "Alert", "WatchlistEntry",
    "IST", "now_ist", "now_utc", "to_ist", "fmt_ist", "utc_iso",
    "chain_hash", "compute_alert_payload", "verify_chain",
    "latest_chain_hash", "VerificationResult",
    "FrameBuffer", "CameraProcessor", "CameraManager",
    "ClipWriter", "LowLightEnhancer",
]
