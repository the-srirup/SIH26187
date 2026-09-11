"""
Structured persistence for ANPR reads and face detections.

Why this exists alongside the alert log
---------------------------------------
Every plate and every face already produces a sealed ``alerts`` row — that is
the evidentiary record and it is not duplicated or replaced here.  But an alert
is a narrative row whose specifics live in a JSON blob, and the questions an
operator actually asks of these two subsystems are lookups::

    has this registration passed any camera this week?
    show every read of KA01F1234
    which faces went unidentified at BOP-03 last night?

Answering those with ``LIKE`` against ``details_json`` is slow and, worse,
unreliable — a substring match on JSON will happily match the wrong field.  So
each reading is *also* written to a typed, indexed table that links back to the
alert it came from.  The alert stays the source of truth; these tables are the
index over it.

Evidence layout
---------------
Crops are written under a predictable, dated tree rather than one flat folder::

    evidence/
      anpr/camera_7/2026/09/20260911T174233_MH12AB1234.jpg
      faces/camera_7/2026/09/20260911T174233_00412.jpg

Flat directories are a real operational problem at this scale: a single BOP
running four cameras produces tens of thousands of crops a month, and both
retention sweeps and an investigator looking for "that afternoon" need to narrow
by camera and date without listing everything.

Nothing here is allowed to break the pipeline.  A failed write is logged and the
row records honestly that no evidence file exists — it never stores a path to a
file that was not written.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings
from core.database import SessionLocal
from core.models import ANPRDetection, FaceDetection
from core.timeutil import fmt_ist, utc_iso

log = logging.getLogger("ibvap.detections")

#: Sub-tree of the evidence root for each kind of artefact.
KIND_ANPR = "anpr"
KIND_FACE = "faces"


def evidence_root() -> Path:
    """Root of the structured evidence tree."""
    return settings.EVIDENCE_DIR


def evidence_dir(kind: str, camera_id: int, when: Optional[str] = None) -> Path:
    """``evidence/<kind>/camera_<id>/<YYYY>/<MM>/`` for a UTC ISO timestamp."""
    stamp = when or utc_iso()
    year, month = stamp[0:4], stamp[5:7]
    if not (year.isdigit() and month.isdigit()):
        year, month = "0000", "00"
    return evidence_root() / kind / f"camera_{int(camera_id)}" / year / month


def _file_stamp(timestamp: str) -> str:
    """``2026-09-11T17:42:33.123+00:00`` -> ``20260911T174233``."""
    digits = "".join(ch for ch in timestamp if ch.isdigit())
    return f"{digits[:8]}T{digits[8:14]}" if len(digits) >= 14 else "unknown"


def _safe_slug(text: str, limit: int = 24) -> str:
    keep = "".join(ch for ch in str(text) if ch.isalnum())
    return keep[:limit] or "unknown"


def save_crop(
    frame: np.ndarray,
    bbox: tuple,
    kind: str,
    camera_id: int,
    timestamp: str,
    label: str,
    *,
    context: float = 0.35,
) -> tuple[str, str]:
    """
    Write the cropped region of interest and return ``(path, sha256)``.

    A little context around the box is included deliberately.  A plate crop cut
    exactly to its bounding box is nearly useless to a human reviewer: it proves
    nothing about which vehicle carried it.  Including the surrounding third
    makes the artefact self-evidencing.

    Returns ``("", "")`` if nothing could be written — the caller must record
    that honestly rather than storing a path to a non-existent file.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return "", ""
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return "", ""

    h, w = frame.shape[:2]
    pad_x = int(max(8, (x2 - x1) * context))
    pad_y = int(max(8, (y2 - y1) * context))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return "", ""

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "", ""

    directory = evidence_dir(kind, camera_id, timestamp)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_file_stamp(timestamp)}_{_safe_slug(label)}.jpg"
        # Quality 92: this is the artefact a reviewer zooms into, and JPEG
        # ringing on small glyphs is exactly what makes a plate unreadable.
        if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            log.warning("EVIDENCE_WRITE_FAILED %s", path)
            return "", ""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        log.debug("EVIDENCE_SAVED %s (%s)", path, digest[:12])
        return str(path), digest
    except OSError as exc:
        log.warning("EVIDENCE_WRITE_FAILED %s: %s", kind, exc)
        return "", ""


# --------------------------------------------------------------------------- #
# ANPR
# --------------------------------------------------------------------------- #


def record_plate(
    plate,
    *,
    camera_id: int,
    alert_id: Optional[int],
    frame: Optional[np.ndarray] = None,
    source_type: str = "live",
    session_id: str = "",
    raw_text: str = "",
) -> Optional[int]:
    """
    Persist one plate reading and its evidence crop.

    ``processing_status`` distinguishes a published registration from an
    uncertain one.  Both are stored — an uncertain read is still an observation
    worth reviewing — but only a published one is ever presented as an
    identified plate.
    """
    timestamp = utc_iso()
    normalised = str(plate.plate_text or "").replace(" ", "").upper()
    if not normalised:
        return None

    published = plate.text_confidence >= settings.ANPR_ALERT_CONFIDENCE
    path, digest = "", ""
    if frame is not None and settings.EVIDENCE_ENABLED:
        path, digest = save_crop(
            frame, plate.bbox, KIND_ANPR, camera_id, timestamp, normalised
        )

    db = SessionLocal()
    try:
        row = ANPRDetection(
            camera_id=camera_id,
            alert_id=alert_id,
            timestamp=timestamp,
            timestamp_ist=fmt_ist(timestamp),
            plate_text=normalised[:24],
            plate_display=str(plate.plate_text or "")[:32],
            plate_raw=str(raw_text or plate.plate_text or "")[:32],
            confidence=round(float(plate.text_confidence or 0.0), 4),
            format_verified=bool(plate.format_verified),
            votes=int(plate.votes or 1),
            consensus=bool(plate.consensus),
            vehicle_class=str(plate.vehicle_class or "")[:32],
            vehicle_track_id=int(plate.vehicle_track_id or 0),
            evidence_path=path,
            evidence_sha256=digest,
            source_type=source_type,
            session_id=session_id,
            processing_status="published" if published else "uncertain",
        )
        db.add(row)
        db.commit()
        log.info("ANPR_DETECTED cam=%d plate=%s conf=%.2f votes=%d status=%s",
                 camera_id, normalised, plate.text_confidence or 0.0,
                 plate.votes or 1, row.processing_status)
        return row.id
    except Exception as exc:
        log.warning("DATABASE_WRITE_FAILURE anpr_detections: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Faces
# --------------------------------------------------------------------------- #


def record_face(
    match,
    *,
    camera_id: int,
    alert_id: Optional[int],
    frame: Optional[np.ndarray] = None,
    source_type: str = "live",
    session_id: str = "",
    threshold: float = 0.0,
) -> Optional[int]:
    """
    Persist one face detection, and the identity only if one was established.

    ``identity_name`` is written **only** for a match above threshold.  A
    below-threshold nearest neighbour is kept in ``similarity`` for review, but
    is never recorded as a name: a border-security log that implies an identity
    the system did not establish is worse than one that records none.
    """
    timestamp = utc_iso()
    matched = bool(match.matched and match.watchlist_name)

    path, digest = "", ""
    if frame is not None and settings.EVIDENCE_ENABLED:
        label = _safe_slug(match.watchlist_name) if matched else f"t{match.track_id or 0}"
        path, digest = save_crop(
            frame, match.bbox, KIND_FACE, camera_id, timestamp, label, context=0.6
        )

    db = SessionLocal()
    try:
        row = FaceDetection(
            camera_id=camera_id,
            alert_id=alert_id,
            timestamp=timestamp,
            timestamp_ist=fmt_ist(timestamp),
            confidence=round(float(match.det_score or 0.0), 4),
            track_id=int(match.track_id or 0),
            recognition_status="matched" if matched else "unknown",
            identity_id=match.watchlist_id if matched else None,
            identity_name=(match.watchlist_name or "")[:120] if matched else "",
            similarity=round(float(match.similarity or 0.0), 4),
            similarity_threshold=round(float(threshold), 4),
            bbox_json=json.dumps([int(v) for v in match.bbox]),
            evidence_path=path,
            evidence_sha256=digest,
            source_type=source_type,
            session_id=session_id,
        )
        db.add(row)
        db.commit()
        log.info("FACE_DETECTED cam=%d status=%s score=%.2f sim=%.2f",
                 camera_id, row.recognition_status,
                 match.det_score or 0.0, match.similarity or 0.0)
        return row.id
    except Exception as exc:
        log.warning("DATABASE_WRITE_FAILURE face_detections: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return None
    finally:
        db.close()


__all__ = [
    "KIND_ANPR", "KIND_FACE", "evidence_root", "evidence_dir",
    "save_crop", "record_plate", "record_face",
]
