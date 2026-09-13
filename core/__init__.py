
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
    from core.alarm import AlarmManager, get_alarm_manager
    from core.notify import NotificationChannel
    from core.sms import SMSNotifier, get_sms_notifier
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
    "AlarmManager": "core.alarm",
    "SMSNotifier": "core.sms",
    "NotificationChannel": "core.notify",
    "get_alarm_manager": "core.alarm",
    "get_sms_notifier": "core.sms",
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
    "AlarmManager", "SMSNotifier", "NotificationChannel",
    "get_alarm_manager", "get_sms_notifier",
]
