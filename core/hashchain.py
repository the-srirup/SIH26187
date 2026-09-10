"""
Tamper-Evident Hash-Chained Event Log.

**Naming, honestly.** This is a cryptographic hash chain with periodic Merkle
checkpoints stored locally.  Nothing is broadcast to a public blockchain and
no distributed consensus is involved, so the code, the API and the dashboard
all call it what it is.  What it *does* provide is strong and easy to
demonstrate:

* Every alert is hashed together with its predecessor's hash.  Editing,
  reordering, inserting or deleting any row invalidates that row and every
  row after it — silent edits to the audit log are impossible to hide.
* Periodic **Merkle checkpoints** seal a contiguous range of the chain.  A
  checkpoint can be exported and handed to a third party, who can then verify
  a range of the log without receiving the whole database.
* An exportable **integrity certificate** re-runs verification at export time
  and states the actual measured result — it never asserts a clean bill of
  health it has not checked.

Layer 1: per-alert SHA-256 chain
Layer 2: periodic Merkle checkpoints (persisted in the ``checkpoints`` table)
Layer 3: exportable, self-describing integrity certificate
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.config import settings
from core.database import SessionLocal
from core.models import Alert, Checkpoint
from core.timeutil import fmt_ist, utc_iso

log = logging.getLogger("ibvap.integrity")

#: Explicit genesis value — SHA-256 of the empty string is *not* 64 zeros, so
#: a literal is used to make the chain's starting point unambiguous.
GENESIS_HASH = "0" * 64


def chain_hash(payload: dict, prev_hash: str) -> str:
    """
    SHA-256 of an alert payload bound to its predecessor's hash.

    ``sort_keys=True`` makes the digest deterministic across runs, processes
    and platforms — a requirement for verification to mean anything.
    """
    blob = json.dumps(payload, sort_keys=True, default=str) + prev_hash
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_alert_payload(alert: Alert) -> dict:
    """
    The canonical, hashed view of an alert.

    Every field here is sealed: changing the alert type, the timestamp, the
    confidence, the rule that fired, or the evidence path breaks the chain.
    Fields excluded from this dict are, by construction, *not* protected —
    which is why everything evidentially meaningful is included.
    """
    return {
        "id": alert.id,
        "camera_id": alert.camera_id,
        "alert_type": alert.alert_type,
        "severity": alert.severity or "",
        "object_class": alert.object_class or "",
        "track_id": alert.track_id or 0,
        "confidence": round(float(alert.confidence or 0.0), 6),
        "timestamp": alert.timestamp,
        "rule_name": alert.rule_name or "",
        "rule_type": alert.rule_type or "",
        "source_type": alert.source_type or "live",
        "session_id": alert.session_id or "",
        "details": alert.details_json or "{}",
        "snapshot_path": alert.snapshot_path or "",
        "clip_path": alert.clip_path or "",
    }


@dataclass
class VerificationResult:
    """Outcome of walking the hash chain."""

    valid: bool
    total_alerts: int
    broken_at: Optional[int] = None
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    message: str = ""
    chain_tip: str = GENESIS_HASH
    verified_at: str = ""
    verified_at_ist: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MerkleCheckpoint:
    """A sealed range of the chain."""

    checkpoint_uid: str
    merkle_root: str
    chain_tip_hash: str
    first_alert_id: int
    last_alert_id: int
    alert_count: int
    timestamp: str
    timestamp_ist: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegrityCertificate:
    """Exportable attestation covering a time range of the log."""

    certificate_id: str
    time_range_start: str
    time_range_end: str
    time_range_start_ist: str
    time_range_end_ist: str
    total_alerts: int
    chain_tip_hash: str
    merkle_root: str
    verification: dict
    checkpoints: list = field(default_factory=list)
    alert_digest: list = field(default_factory=list)
    scheme: str = "SHA-256 hash chain + Merkle checkpoints (local, non-blockchain)"
    issued_by: str = "IBVAP Integrity Subsystem"
    issued_to: str = "Evidentiary Review"
    issue_timestamp: str = ""
    issue_timestamp_ist: str = ""
    statement: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Layer 1 — chain verification
# --------------------------------------------------------------------------- #


def verify_chain(db: Optional[Session] = None) -> VerificationResult:
    """
    Recompute every link and report the first row that fails.

    Two independent checks per row: that its recorded ``prev_hash`` matches
    the running tip, and that its own ``hash`` matches a fresh digest of its
    payload.  The first catches reordering/deletion, the second catches edits.
    """
    import time as _time

    started = _time.perf_counter()
    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        alerts = db.query(Alert).order_by(Alert.id.asc()).all()
        now = utc_iso()

        if not alerts:
            return VerificationResult(
                valid=True, total_alerts=0,
                message="No events in the chain — trivially valid.",
                chain_tip=GENESIS_HASH, verified_at=now,
                verified_at_ist=fmt_ist(now),
                duration_ms=round((_time.perf_counter() - started) * 1000, 2),
            )

        prev_hash = GENESIS_HASH
        for alert in alerts:
            if alert.prev_hash != prev_hash:
                return VerificationResult(
                    valid=False, total_alerts=len(alerts), broken_at=alert.id,
                    expected_hash=prev_hash, actual_hash=alert.prev_hash,
                    message=(
                        f"Chain broken at event #{alert.id}: predecessor link "
                        f"mismatch. Expected {prev_hash[:16]}…, found "
                        f"{(alert.prev_hash or '')[:16]}…. An event was "
                        f"deleted, reordered or inserted."
                    ),
                    chain_tip=prev_hash, verified_at=now, verified_at_ist=fmt_ist(now),
                    duration_ms=round((_time.perf_counter() - started) * 1000, 2),
                )

            expected = chain_hash(compute_alert_payload(alert), prev_hash)
            if alert.hash != expected:
                return VerificationResult(
                    valid=False, total_alerts=len(alerts), broken_at=alert.id,
                    expected_hash=expected, actual_hash=alert.hash,
                    message=(
                        f"Chain broken at event #{alert.id}: payload digest "
                        f"mismatch. Expected {expected[:16]}…, found "
                        f"{(alert.hash or '')[:16]}…. This event was modified "
                        f"after it was sealed."
                    ),
                    chain_tip=prev_hash, verified_at=now, verified_at_ist=fmt_ist(now),
                    duration_ms=round((_time.perf_counter() - started) * 1000, 2),
                )

            prev_hash = alert.hash

        return VerificationResult(
            valid=True, total_alerts=len(alerts),
            message=f"All {len(alerts)} events verified — chain intact.",
            chain_tip=prev_hash, verified_at=now, verified_at_ist=fmt_ist(now),
            duration_ms=round((_time.perf_counter() - started) * 1000, 2),
        )
    finally:
        if own_session:
            db.close()


def latest_chain_hash(db: Optional[Session] = None) -> str:
    """The chain tip — the hash of the most recently sealed event."""
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        max_id = db.query(func.max(Alert.id)).scalar()
        if max_id is None:
            return GENESIS_HASH
        tip = db.query(Alert.hash).filter(Alert.id == max_id).scalar()
        return tip or GENESIS_HASH
    finally:
        if own_session:
            db.close()


# --------------------------------------------------------------------------- #
# Layer 2 — Merkle checkpoints
# --------------------------------------------------------------------------- #


def compute_merkle_root(hashes: list[str]) -> str:
    """
    Standard binary Merkle root over the given leaf hashes.

    An odd level duplicates its last node (Bitcoin convention).  Iterative
    rather than recursive so a long chain cannot blow the stack, and the input
    list is never mutated — the previous implementation appended to the
    caller's list as a side effect.
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()

    level = list(hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256((level[i] + level[i + 1]).encode("utf-8")).hexdigest()
            for i in range(0, len(level), 2)
        ]
    return level[0]


def create_checkpoint(db: Optional[Session] = None) -> Optional[MerkleCheckpoint]:
    """
    Seal every event added since the last checkpoint.

    Unlike the previous build — which generated anchor objects, logged them
    and threw them away — the checkpoint is **persisted**, so it can actually
    be presented later as evidence that the log looked a certain way at a
    certain time.  Returns ``None`` when there is nothing new to seal.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        last = db.query(Checkpoint).order_by(Checkpoint.id.desc()).first()
        start_after = last.last_alert_id if last else 0

        rows = (
            db.query(Alert.id, Alert.hash)
            .filter(Alert.id > start_after)
            .order_by(Alert.id.asc())
            .all()
        )
        if not rows:
            return None

        hashes = [r[1] for r in rows]
        now = utc_iso()
        checkpoint = Checkpoint(
            checkpoint_uid=uuid.uuid4().hex,
            first_alert_id=rows[0][0],
            last_alert_id=rows[-1][0],
            alert_count=len(rows),
            merkle_root=compute_merkle_root(hashes),
            chain_tip_hash=hashes[-1],
            timestamp=now,
            timestamp_ist=fmt_ist(now),
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)

        log.info(
            "Merkle checkpoint sealed: events #%d–#%d (%d) root=%s…",
            checkpoint.first_alert_id, checkpoint.last_alert_id,
            checkpoint.alert_count, checkpoint.merkle_root[:16],
        )
        return _to_dataclass(checkpoint)
    except Exception as exc:
        log.warning("Checkpoint creation failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return None
    finally:
        if own_session:
            db.close()


def _to_dataclass(row: Checkpoint) -> MerkleCheckpoint:
    return MerkleCheckpoint(
        checkpoint_uid=row.checkpoint_uid,
        merkle_root=row.merkle_root,
        chain_tip_hash=row.chain_tip_hash,
        first_alert_id=row.first_alert_id,
        last_alert_id=row.last_alert_id,
        alert_count=row.alert_count,
        timestamp=row.timestamp,
        timestamp_ist=row.timestamp_ist,
    )


def list_checkpoints(db: Optional[Session] = None, limit: int = 50) -> list[MerkleCheckpoint]:
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        rows = db.query(Checkpoint).order_by(Checkpoint.id.desc()).limit(limit).all()
        return [_to_dataclass(r) for r in rows]
    finally:
        if own_session:
            db.close()


def verify_checkpoint(checkpoint_uid: str, db: Optional[Session] = None) -> dict:
    """
    Re-derive a stored checkpoint's Merkle root from the live database.

    A mismatch proves the sealed range has been altered since the checkpoint
    was taken, and is reported with the specific range affected.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        row = (
            db.query(Checkpoint)
            .filter(Checkpoint.checkpoint_uid == checkpoint_uid)
            .first()
        )
        if row is None:
            return {"found": False, "valid": False, "message": "Checkpoint not found."}

        hashes = [
            h for (h,) in db.query(Alert.hash)
            .filter(Alert.id >= row.first_alert_id, Alert.id <= row.last_alert_id)
            .order_by(Alert.id.asc())
            .all()
        ]
        recomputed = compute_merkle_root(hashes)
        valid = recomputed == row.merkle_root and len(hashes) == row.alert_count
        return {
            "found": True,
            "valid": valid,
            "checkpoint_uid": checkpoint_uid,
            "range": [row.first_alert_id, row.last_alert_id],
            "expected_root": row.merkle_root,
            "recomputed_root": recomputed,
            "expected_count": row.alert_count,
            "actual_count": len(hashes),
            "timestamp_ist": row.timestamp_ist,
            "message": (
                "Checkpoint verified — sealed range is unchanged."
                if valid else
                f"Checkpoint FAILED: events #{row.first_alert_id}–#{row.last_alert_id} "
                f"have been altered since this checkpoint was sealed."
            ),
        }
    finally:
        if own_session:
            db.close()


# --------------------------------------------------------------------------- #
# Layer 3 — exportable integrity certificate
# --------------------------------------------------------------------------- #


def generate_integrity_certificate(
    db: Session,
    start_time: str,
    end_time: str,
    issued_to: str = "Evidentiary Review",
) -> IntegrityCertificate:
    """
    Build a self-contained attestation for a time range.

    Verification is **executed now**, and the certificate reports whatever it
    actually found.  The previous implementation printed an affidavit
    asserting "all integrity verification checks passed" without ever running
    one, and synthesised a fresh anchor per five-minute slot regardless of
    whether any checkpoint had been taken.
    """
    alerts = (
        db.query(Alert)
        .filter(Alert.timestamp >= start_time, Alert.timestamp <= end_time)
        .order_by(Alert.id.asc())
        .all()
    )
    verification = verify_chain(db)
    issued = utc_iso()

    hashes = [a.hash for a in alerts]
    merkle_root = compute_merkle_root(hashes)

    checkpoints = [
        c.to_dict() for c in list_checkpoints(db, limit=500)
        if start_time <= c.timestamp <= end_time
    ]

    digest = [
        {
            "id": a.id,
            "timestamp_ist": a.timestamp_ist or fmt_ist(a.timestamp),
            "alert_type": a.alert_type,
            "camera_id": a.camera_id,
            "track_id": a.track_id,
            "hash": a.hash,
        }
        for a in alerts[:500]
    ]

    if verification.valid:
        statement = (
            f"{len(alerts)} event(s) fall within this range. The complete event "
            f"log of {verification.total_alerts} record(s) was re-verified at "
            f"{fmt_ist(issued)} and the SHA-256 chain was found INTACT. Each "
            f"event is bound to its predecessor, so any deletion, reordering "
            f"or modification would have been detected. Merkle root over this "
            f"range: {merkle_root}."
        )
    else:
        statement = (
            f"INTEGRITY FAILURE. Re-verification at {fmt_ist(issued)} found the "
            f"chain broken at event #{verification.broken_at}. {verification.message} "
            f"This log must not be relied upon as unaltered."
        )

    return IntegrityCertificate(
        certificate_id=uuid.uuid4().hex,
        time_range_start=start_time,
        time_range_end=end_time,
        time_range_start_ist=fmt_ist(start_time),
        time_range_end_ist=fmt_ist(end_time),
        total_alerts=len(alerts),
        chain_tip_hash=verification.chain_tip,
        merkle_root=merkle_root,
        verification=verification.to_dict(),
        checkpoints=checkpoints,
        alert_digest=digest,
        issued_to=issued_to,
        issue_timestamp=issued,
        issue_timestamp_ist=fmt_ist(issued),
        statement=statement,
    )


def export_integrity_certificate_to_json(certificate: IntegrityCertificate) -> str:
    """Serialise a certificate for download / transmission."""
    return json.dumps(certificate.to_dict(), indent=2, default=str)


def chain_status(db: Optional[Session] = None) -> dict:
    """Cheap summary for the dashboard header (no full chain walk)."""
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        total = db.query(func.count(Alert.id)).scalar() or 0
        checkpoints = db.query(func.count(Checkpoint.id)).scalar() or 0
        return {
            "total_events": total,
            "chain_tip": latest_chain_hash(db),
            "checkpoints": checkpoints,
            "scheme": "SHA-256 hash chain + local Merkle checkpoints",
            "checkpoint_interval_seconds": settings.CHECKPOINT_INTERVAL,
        }
    finally:
        if own_session:
            db.close()
