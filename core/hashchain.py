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
        # Sealed as of payload v2. These were previously left out, which was a
        # real hole rather than an oversight of no consequence: `description` is
        # the human-readable account of the event — the sentence an operator
        # reads and a review board quotes — so with it unsealed, every narrative
        # in the log could be rewritten and verification would still report the
        # chain intact. `detector` attributes the event to a subsystem and
        # `timestamp_ist` is the operator-facing time; both are equally quotable.
        "description": alert.description or "",
        "detector": alert.detector or "",
        "timestamp_ist": alert.timestamp_ist or "",
    }


#: Version of the sealed payload above. A change here alters every digest, so
#: an existing database must be re-sealed explicitly (``manage.py chain-repair
#: --reseal``) rather than silently reporting itself as tampered with.
PAYLOAD_VERSION = 2


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
    #: Every failing link, not just the first. A single stop-at-first-error
    #: answer cannot distinguish "one row was edited" from "the log forked
    #: eleven times under concurrent writers", and those call for opposite
    #: responses from an operator.
    breaks: list = field(default_factory=list)
    #: ``"payload"`` (a row was modified after sealing), ``"fork"`` (two rows
    #: claim the same predecessor — concurrent append), ``"missing"`` (a row was
    #: deleted), or ``"mixed"``.
    break_kind: str = ""
    #: True when every fault is a fork and no payload digest failed: the
    #: recorded events are all individually authentic and only their linkage is
    #: wrong. Reported separately because it is a genuinely different finding
    #: from evidence tampering and must not be presented as the same thing.
    forks_only: bool = False

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

        # Walking the *whole* chain rather than returning at the first fault is
        # what makes the result diagnostic instead of merely alarming. Each row
        # is checked two independent ways, and the link check continues from the
        # row's own hash afterwards so one bad link does not cascade into
        # thousands of meaningless follow-on errors.
        seen_hashes = {GENESIS_HASH: 0}
        breaks: list[dict] = []
        prev_hash = GENESIS_HASH
        for alert in alerts:
            if alert.prev_hash != prev_hash:
                # A fork is specifically "this row's predecessor is a real row
                # in this chain, just not the one immediately before it" — the
                # fingerprint of two writers reading the same tip. A prev_hash
                # matching nothing at all means a row was removed.
                kind = "fork" if alert.prev_hash in seen_hashes else "missing"
                breaks.append({
                    "alert_id": alert.id, "kind": kind,
                    "expected_prev": prev_hash, "actual_prev": alert.prev_hash or "",
                    "forked_from_alert_id": seen_hashes.get(alert.prev_hash or ""),
                    "timestamp": alert.timestamp,
                    "detail": (
                        f"Event #{alert.id} claims the same predecessor as an "
                        f"earlier event — two writers appended concurrently."
                        if kind == "fork" else
                        f"Event #{alert.id} references a predecessor that is not "
                        f"in the log — a record was deleted."
                    ),
                })

            expected = chain_hash(compute_alert_payload(alert), alert.prev_hash or "")
            if alert.hash != expected:
                breaks.append({
                    "alert_id": alert.id, "kind": "payload",
                    "expected_prev": expected, "actual_prev": alert.hash or "",
                    "forked_from_alert_id": None,
                    "timestamp": alert.timestamp,
                    "detail": (
                        f"Event #{alert.id} does not match its own digest — the "
                        f"record was modified after it was sealed."
                    ),
                })

            seen_hashes.setdefault(alert.hash, alert.id)
            prev_hash = alert.hash

        duration = round((_time.perf_counter() - started) * 1000, 2)
        if not breaks:
            return VerificationResult(
                valid=True, total_alerts=len(alerts),
                message=f"All {len(alerts)} events verified — chain intact.",
                chain_tip=prev_hash, verified_at=now, verified_at_ist=fmt_ist(now),
                duration_ms=duration,
            )

        kinds = {b["kind"] for b in breaks}
        kind = kinds.pop() if len(kinds) == 1 else "mixed"
        forks_only = kind == "fork"
        first = breaks[0]
        if forks_only:
            summary = (
                f"{len(breaks)} fork(s) in {len(alerts)} events. Every event's "
                "own digest is valid, so no record was altered or deleted; the "
                "links are wrong because events were appended concurrently. "
                "Run 'python manage.py chain-repair' to re-link them."
            )
        else:
            summary = (
                f"{len(breaks)} integrity fault(s) in {len(alerts)} events "
                f"({', '.join(sorted(kinds | {kind})) if kind == 'mixed' else kind}). "
                f"First at event #{first['alert_id']}: {first['detail']}"
            )

        return VerificationResult(
            valid=False, total_alerts=len(alerts), broken_at=first["alert_id"],
            expected_hash=first["expected_prev"], actual_hash=first["actual_prev"],
            message=summary, chain_tip=prev_hash, verified_at=now,
            verified_at_ist=fmt_ist(now), duration_ms=duration,
            breaks=breaks[:200], break_kind=kind, forks_only=forks_only,
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


def begin_exclusive_append(db: Session) -> bool:
    """
    Take the database's write lock **before** the chain tip is read.

    This is the difference between a hash chain that holds and one that forks.
    Appending a link is a read-modify-write: read the tip, then insert a row
    whose ``prev_hash`` is that tip.  An in-process ``threading.Lock`` makes
    that atomic for threads *of one process* and does nothing at all for a
    second process — a CLI command, a benchmark run, a second worker — writing
    to the same file.  When two writers interleave, both read the same tip and
    both commit, producing two rows claiming the same predecessor.

    That is not hypothetical: this project's own database contains the
    signature.  Events #2068 and #2069 were sealed 44 ms apart by different
    camera threads and carry the identical ``prev_hash``, with no gap in the id
    sequence and no payload mismatch — a fork, not tampering.  Verification
    correctly reported the chain broken, and the flagship integrity feature
    read "COMPROMISED" for the rest of the database's life.

    ``BEGIN IMMEDIATE`` acquires SQLite's RESERVED lock at the *start* of the
    transaction rather than lazily at the first write, so a concurrent writer
    blocks here (up to ``busy_timeout``) instead of racing us to the tip.  The
    whole read-seal-commit sequence becomes atomic across threads *and*
    processes.

    Returns True when the lock was taken.  On any other backend — or if the
    driver has already opened a transaction — this is a no-op returning False
    and the caller proceeds; correctness then rests on the caller's lock alone,
    which is the behaviour we had before.
    """
    bind = db.get_bind()
    if bind is None or bind.dialect.name != "sqlite":
        return False
    try:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        return True
    except Exception as exc:  # already in a transaction, or a locked database
        log.debug("BEGIN IMMEDIATE unavailable (%s) — relying on process lock", exc)
        return False


def evidence_digest(*paths: str) -> dict:
    """
    SHA-256 of each evidence file that actually exists on disk.

    Sealing the *path* only proves which filename was claimed; it says nothing
    about the bytes, so a snapshot could be swapped for a different image and
    the chain would still verify clean.  Hashing the content closes that hole:
    the digest goes into the alert's ``details``, which
    :func:`compute_alert_payload` already seals, so the evidence file is bound
    into the chain without changing the payload schema — every previously
    sealed row still verifies exactly as before.

    A file that could not be read is reported honestly as unavailable rather
    than silently omitted, so a missing artefact is visible in the record.
    """
    from pathlib import Path

    out: dict[str, str] = {}
    for path in paths:
        if not path:
            continue
        try:
            data = Path(path).read_bytes()
        except OSError:
            out[Path(path).name] = "unavailable"
            continue
        out[Path(path).name] = hashlib.sha256(data).hexdigest()
    return out


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


# --------------------------------------------------------------------------- #
# Repair — for forks only, and never silently
# --------------------------------------------------------------------------- #


def repair_chain(db: Optional[Session] = None, *, dry_run: bool = False,
                 reseal: bool = False) -> dict:
    """
    Re-link a chain that forked under concurrent writers.

    This exists because of a real defect, now fixed at the writer (see
    :func:`begin_exclusive_append`), which left already-sealed databases
    permanently reporting COMPROMISED.  It is deliberately narrow:

    * It refuses to run if **any** payload digest fails.  A payload mismatch
      means a record was edited after sealing, and re-linking would erase the
      only evidence of that — the repair would become the cover-up.  Only forks,
      where every event's own digest still validates, are repairable.
    * It changes no event content whatsoever.  ``prev_hash`` and ``hash`` are
      recomputed in id order from the untouched payloads; every other column is
      left exactly as sealed.
    * It appends a ``chain_repaired`` system event recording when the repair ran
      and how many links it touched, so the repair is itself part of the audit
      trail rather than an invisible rewrite.

    Returns a summary; with ``dry_run`` nothing is written.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        before = verify_chain(db)
        if before.valid:
            return {"ok": True, "repaired": 0, "message": "Chain already intact.",
                    "before": before.to_dict()}
        if not before.forks_only and not reseal:
            return {
                "ok": False, "repaired": 0,
                "message": (
                    "Refusing to repair: the chain contains faults that are not "
                    "concurrent-append forks (" + before.break_kind + "). A payload "
                    "or deletion fault means a record was altered or removed, and "
                    "re-linking would destroy the evidence of it. Investigate "
                    "before repairing. If this database was sealed under an older "
                    "payload schema, re-seal it deliberately with --reseal."
                ),
                "before": before.to_dict(),
            }
        if dry_run:
            what = ("re-seal every link under payload schema "
                    f"v{PAYLOAD_VERSION}" if reseal
                    else f"re-link {len(before.breaks)} forked link(s)")
            return {"ok": True, "repaired": len(before.breaks), "dry_run": True,
                    "message": f"Would {what}.",
                    "before": before.to_dict()}

        begin_exclusive_append(db)
        alerts = db.query(Alert).order_by(Alert.id.asc()).all()
        prev_hash = GENESIS_HASH
        relinked = 0
        for alert in alerts:
            new_hash = chain_hash(compute_alert_payload(alert), prev_hash)
            if alert.prev_hash != prev_hash or alert.hash != new_hash:
                alert.prev_hash = prev_hash
                alert.hash = new_hash
                relinked += 1
            prev_hash = alert.hash
        db.commit()

        # Seal the repair itself into the chain it just repaired.
        now = utc_iso()
        marker = Alert(
            camera_id=alerts[-1].camera_id if alerts else 0,
            alert_type="chain_resealed" if reseal else "chain_repaired",
            severity="INFO", object_class="",
            track_id=0, confidence=0.0, timestamp=now, timestamp_ist=fmt_ist(now),
            rule_name="integrity", rule_type="system", detector="system",
            source_type="system", session_id="",
            description=(
                (f"Hash chain re-sealed under payload schema v{PAYLOAD_VERSION}: "
                 f"{relinked} link(s) rebuilt. No event content was modified — "
                 f"only the digests, which changed because additional fields "
                 f"(description, detector, timestamp_ist) are now sealed."
                 ) if reseal else
                (f"Hash chain re-linked: {relinked} link(s) rebuilt after "
                 f"{len(before.breaks)} concurrent-append fork(s). No event "
                 f"content was modified; all payload digests verified before repair.")
            ),
            details_json=json.dumps({
                "operation": "reseal" if reseal else "relink",
                "payload_version": PAYLOAD_VERSION,
                "faults_found": len(before.breaks),
                "fault_kind": before.break_kind,
                "links_rebuilt": relinked,
                "total_events": before.total_alerts,
                "affected_event_ids": [b["alert_id"] for b in before.breaks][:100],
            }),
            snapshot_path="", clip_path="", prev_hash=prev_hash, hash="",
        )
        db.add(marker)
        db.flush()
        marker.hash = chain_hash(compute_alert_payload(marker), prev_hash)
        db.commit()

        after = verify_chain(db)
        log.warning(
            "INTEGRITY chain repaired — %d link(s) rebuilt after %d fork(s); "
            "now valid=%s", relinked, len(before.breaks), after.valid,
        )
        return {
            "ok": after.valid, "repaired": relinked,
            "forks_found": len(before.breaks),
            "message": (
                (f"Re-sealed {relinked} link(s) under payload schema "
                 f"v{PAYLOAD_VERSION}. " if reseal else
                 f"Re-linked {relinked} link(s) after {len(before.breaks)} fork(s). ")
                + f"Chain now {'verifies clean' if after.valid else 'STILL BROKEN'}."
            ),
            "before": before.to_dict(), "after": after.to_dict(),
        }
    finally:
        if own_session:
            db.close()
