"""
Event manager — where an analytics outcome becomes a sealed, auditable record.

Responsibilities:

1. Persist the alert to SQLite **inside the tamper-evident hash chain**.
2. Capture evidence (snapshot + contextual clip) and record the paths.
3. Publish the alert to live subscribers (the WebSocket layer) immediately.

Ordering matters for the hash chain.  A row's hash covers its evidence paths
and the SHA-256 of the evidence bytes, so both must be final *before* the hash
is computed.  We therefore insert to obtain the primary key, write evidence
using that key, hash that evidence, then seal the row — all inside a single
transaction that holds the **database** write lock from before the chain tip is
read (see :func:`core.hashchain.begin_exclusive_append`).

A per-process ``threading.Lock`` is not sufficient on its own and this project
has the scar to prove it: appending a link is a read-modify-write, so a second
*process* touching the same database file read the same tip and produced two
rows claiming the same predecessor.  The process lock is kept as the cheap
fast path; the database lock is what makes the invariant actually hold.

Publication is decoupled from persistence: the pipeline never waits on an
asyncio loop, and a slow WebSocket client can never stall a camera thread.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np

from core.config import settings
from core.database import SessionLocal
from core.evidence import save_snapshot
from core.hashchain import (
    begin_exclusive_append, chain_hash, compute_alert_payload,
    evidence_digest, latest_chain_hash,
)
from core.models import Alert
from core.timeutil import fmt_ist, utc_iso
from cv.rules import Alert as RuleAlert, severity_for

log = logging.getLogger("ibvap.events")

#: Human-facing titles. Kept server-side so the API, the log and the UI agree.
ALERT_TITLES = {
    "entry": "INTRUSION — FENCE CROSSED (INBOUND)",
    "exit": "FENCE CROSSED (OUTBOUND)",
    "enter": "RESTRICTED ZONE INTRUSION",
    "zone_exit": "LEFT RESTRICTED ZONE",
    "zone_presence": "SUSPICIOUS ACTIVITY — SUSTAINED ZONE PRESENCE",
    "loiter": "SUSPICIOUS ACTIVITY — LOITERING",
    "wrong_direction": "SUSPICIOUS ACTIVITY — WRONG DIRECTION",
    "night_movement": "NIGHT MOVEMENT DETECTED",
    "human_detected": "HUMAN DETECTED",
    "vehicle_detected": "VEHICLE DETECTED",
    "watchlist_match": "WATCHLIST FACE MATCH",
    "face_detected": "FACE DETECTED",
    "anpr_detection": "ANPR — NUMBER PLATE READ",
    "camera_offline": "CAMERA OFFLINE",
    "source_restarted": "VIDEO RESTARTED — REPLAY",
    "system_error": "SYSTEM ERROR",
}

ALERT_ICONS = {
    "entry": "🚨", "exit": "🚨", "enter": "🚨", "zone_exit": "↩️",
    "zone_presence": "⚠️", "loiter": "⚠️", "wrong_direction": "⚠️",
    "night_movement": "🌙", "human_detected": "🚶", "vehicle_detected": "🚗",
    "watchlist_match": "👤", "face_detected": "👤", "anpr_detection": "🔢",
    "camera_offline": "📵", "system_error": "⚠️",
    "source_restarted": "🔁",
}

#: Which subsystem produced the event — the dashboard shows this so nobody
#: mistakes a geometry rule for a learned behaviour model.
DETECTOR_BY_TYPE = {
    "human_detected": "ai_detection",
    "vehicle_detected": "ai_detection",
    "face_detected": "ai_detection",
    "watchlist_match": "face_recognition",
    "anpr_detection": "anpr_ocr",
    "camera_offline": "system",
    "system_error": "system",
    "source_restarted": "system",
}


def title_for(alert_type: str) -> str:
    return ALERT_TITLES.get(alert_type, alert_type.replace("_", " ").upper())


def icon_for(alert_type: str) -> str:
    return ALERT_ICONS.get(alert_type, "🔔")


def detector_for(alert_type: str) -> str:
    return DETECTOR_BY_TYPE.get(alert_type, "rule_engine")


class EventManager:
    """
    Process-wide singleton that seals and publishes events.

    Subscribers are plain callables invoked synchronously on the publishing
    thread; the API layer's subscriber only drops the payload into a queue, so
    it returns in microseconds.
    """

    _instance: Optional["EventManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._chain_lock = threading.Lock()
        self._subscribers: list[Callable[[dict], None]] = []
        self._sub_lock = threading.Lock()
        self._recent: deque = deque(maxlen=200)
        self._counts: dict[str, int] = {}

    @classmethod
    def get(cls) -> "EventManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    # Subscription
    # ------------------------------------------------------------------ #
    def subscribe(self, callback: Callable[[dict], None]) -> None:
        with self._sub_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]) -> None:
        with self._sub_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _publish(self, payload: dict) -> None:
        self._recent.append(payload)
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(payload)
            except Exception as exc:  # a bad subscriber must not break the chain
                log.warning("Event subscriber failed: %s", exc)

    @property
    def recent(self) -> list[dict]:
        return list(self._recent)

    def reset(self) -> None:
        """
        Forget the in-memory event history and counters.

        The dashboard's event feed and per-type tallies live here, not in the
        database, so a hard reset that only truncated tables would leave the
        operator looking at events from the system they just wiped.
        Subscribers are deliberately left attached: they are the WebSocket
        fan-out and the alarm/SMS channels, which must survive a reset.
        """
        self._recent.clear()
        self._counts.clear()
        log.info("Event history cleared")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        camera_id: int,
        rule_alert: RuleAlert,
        frame: Optional[np.ndarray] = None,
        clean_frame: Optional[np.ndarray] = None,
        clip_recorder=None,
        object_class: str = "",
        confidence: float = 0.0,
        camera_name: str = "",
        source_type: str = "live",
        session_id: str = "",
        capture_evidence: bool = True,
    ) -> Optional[dict]:
        """
        Seal one event: persist it, attach evidence, publish it.

        Returns the serialised alert, or ``None`` if persistence failed (in
        which case the failure is logged and the pipeline keeps running — a
        database hiccup must never stop surveillance).
        """
        alert_type = rule_alert.alert_type
        severity = rule_alert.severity or severity_for(alert_type)
        details = dict(rule_alert.details or {})
        details.setdefault("rule_name", rule_alert.rule_name)

        timestamp_utc = utc_iso()
        timestamp_ist = fmt_ist(timestamp_utc)

        db = SessionLocal()
        try:
            with self._chain_lock:
                # The process lock above orders *our* threads. It cannot order a
                # second process — a CLI command, a benchmark, an extra worker —
                # writing the same file, and appending a link is a
                # read-modify-write, so two writers that both read the tip both
                # commit rows claiming the same predecessor. Taking SQLite's
                # RESERVED lock up front makes read-seal-commit atomic against
                # every writer, not just the ones sharing our interpreter.
                begin_exclusive_append(db)
                prev_hash = latest_chain_hash(db)

                row = Alert(
                    camera_id=camera_id,
                    alert_type=alert_type,
                    severity=severity,
                    object_class=object_class or details.get("object_class", ""),
                    track_id=int(rule_alert.track_id or 0),
                    confidence=float(confidence or 0.0),
                    timestamp=timestamp_utc,
                    timestamp_ist=timestamp_ist,
                    rule_name=rule_alert.rule_name,
                    rule_type=rule_alert.rule_type,
                    detector=detector_for(alert_type),
                    source_type=source_type,
                    session_id=session_id,
                    description=rule_alert.description or title_for(alert_type),
                    details_json=json.dumps(details, default=str),
                    snapshot_path="",
                    clip_path="",
                    prev_hash=prev_hash,
                    hash="",
                )
                db.add(row)
                db.flush()          # assigns row.id without committing

                snapshot_path = clip_path = ""
                if capture_evidence and settings.EVIDENCE_ENABLED and frame is not None:
                    snapshot_path, clean_path = save_snapshot(
                        frame, str(camera_id), row.id, clean_frame
                    )
                    if clean_path:
                        details["clean_snapshot"] = clean_path
                    if clip_recorder is not None:
                        clip_path = clip_recorder.start_clip(row.id) or ""

                row.snapshot_path = snapshot_path
                row.clip_path = clip_path

                # Seal the evidence *bytes*, not just its filename. Hashing the
                # path alone proves only which file was claimed — swap the JPEG
                # afterwards and the chain still verifies clean. The digests ride
                # in details, which the payload already covers, so the evidence
                # is bound into the chain without altering the sealed schema and
                # every previously sealed row still verifies unchanged.
                if snapshot_path or clip_path:
                    digests = evidence_digest(snapshot_path, clip_path)
                    if digests:
                        details["evidence_sha256"] = digests

                row.details_json = json.dumps(details, default=str)

                # Hash last: it covers the finalised evidence paths and digests.
                row.hash = chain_hash(compute_alert_payload(row), prev_hash)
                db.commit()

                payload = serialize_alert_row(row, camera_name=camera_name)

            self._counts[alert_type] = self._counts.get(alert_type, 0) + 1
            log.info(
                "EVENT #%d %s | cam=%s track=%s sev=%s | %s",
                payload["id"], alert_type, camera_name or camera_id,
                payload["track_id"], severity, payload["timestamp_ist"],
            )
            self._publish({"type": "alert", "data": payload})
            return payload

        except Exception as exc:
            log.exception("Failed to persist alert (%s): %s", alert_type, exc)
            try:
                db.rollback()
            except Exception:
                pass
            return None
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    def broadcast(self, message: dict) -> None:
        """Publish a non-alert message (status, stats, analysis progress)."""
        self._publish(message)

    def counts(self) -> dict:
        return dict(self._counts)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def _plate_of(details: dict) -> Optional[str]:
    """The registration this event recorded, formatted for display."""
    text = details.get("plate_text") or details.get("plate_raw") or ""
    text = str(text).strip().upper()
    return text or None


def serialize_alert_row(row: Alert, camera_name: str = "") -> dict:
    """
    Convert an ``Alert`` row into the canonical API/WebSocket payload.

    Every timestamp is exposed three ways: canonical UTC for machines, IST ISO
    for clients that want to reformat, and a ready-to-render IST display
    string so no consumer has to reimplement the timezone rule.
    """
    try:
        details = json.loads(row.details_json) if row.details_json else {}
    except (json.JSONDecodeError, TypeError):
        details = {}

    from core.timeutil import ist_iso

    return {
        "id": row.id,
        "camera_id": row.camera_id,
        "camera_name": camera_name or f"CAM-{row.camera_id:02d}",
        "alert_type": row.alert_type,
        "title": title_for(row.alert_type),
        "icon": icon_for(row.alert_type),
        "severity": row.severity or "MEDIUM",
        "object_class": row.object_class or "",
        "track_id": row.track_id or 0,
        "confidence": round(float(row.confidence or 0.0), 3),
        "timestamp": row.timestamp,
        "timestamp_ist": row.timestamp_ist or fmt_ist(row.timestamp),
        "timestamp_ist_iso": ist_iso(row.timestamp),
        "rule_name": row.rule_name or "",
        "rule_type": row.rule_type or "",
        "detector": row.detector or "rule_engine",
        "analysis_kind": (
            "AI DETECTION" if (row.detector or "").startswith(("ai_", "face_", "anpr"))
            else "RULE-BASED EVENT ANALYSIS"
        ),
        "source_type": row.source_type or "live",
        "session_id": row.session_id or "",
        "description": row.description or "",
        "details": details,
        # Recognition results promoted out of ``details`` so the event log,
        # the PDF and any C2 consumer can show a registration without having to
        # know the internal shape of an ANPR payload. A plate read is the point
        # of the event; it should not be buried one level down.
        "plate": _plate_of(details),
        "plate_confidence": (round(float(details.get("ocr_confidence") or 0.0), 3)
                             if details.get("plate_text") else None),
        "plate_verified": bool(details.get("format_verified")) or None,
        "watchlist_name": details.get("watchlist_name") or None,
        "has_snapshot": bool(row.snapshot_path),
        "has_clip": bool(row.clip_path),
        "snapshot_url": f"/api/alerts/{row.id}/snapshot" if row.snapshot_path else None,
        "clip_url": f"/api/alerts/{row.id}/clip" if row.clip_path else None,
        "hash": row.hash,
        "prev_hash": row.prev_hash,
    }
