"""
Tests that a number plate reaches the event log as evidence.

A registration is the substance of an ANPR event, not a detail of it. It has to
appear on the event itself so the log, the PDF and any C2 consumer can show it
without knowing the internal shape of an ANPR payload — and the crop it was
read from has to be reachable from that event, because a plate with no pixels
behind it is an assertion rather than evidence.
"""
from __future__ import annotations

import json

import pytest


def _row(details: dict, alert_type: str = "anpr_detection"):
    """A minimal Alert row stand-in, as the serializer sees it."""
    return type("Row", (), {
        "id": 7, "camera_id": 2, "alert_type": alert_type, "severity": "LOW",
        "object_class": "car", "track_id": 42, "confidence": 0.9,
        "timestamp": "2026-09-13T05:00:00+00:00", "timestamp_ist": "",
        "rule_name": "anpr", "rule_type": "anpr", "detector": "anpr_ocr",
        "source_type": "live", "session_id": "", "description": "Number plate read",
        "details_json": json.dumps(details), "snapshot_path": "", "clip_path": "",
        "hash": "a" * 64, "prev_hash": "b" * 64,
    })()


def test_the_plate_appears_on_the_event_itself():
    from core.events import serialize_alert_row
    payload = serialize_alert_row(_row({
        "plate_text": "MH 12 DE 1433", "plate_raw": "MH12DE1433",
        "ocr_confidence": 0.93, "format_verified": True,
        "corroborating_reads": 4,
    }), "BOP-01")
    assert payload["plate"] == "MH 12 DE 1433"
    assert payload["plate_confidence"] == pytest.approx(0.93)
    assert payload["plate_verified"] is True
    # The full reading is still carried for anything that wants the detail.
    assert payload["details"]["corroborating_reads"] == 4


def test_a_raw_reading_is_still_surfaced():
    """An unverified read is an observation worth reviewing, not nothing."""
    from core.events import serialize_alert_row
    payload = serialize_alert_row(_row({
        "plate_raw": "KH12DE1433", "ocr_confidence": 0.36,
        "format_verified": False,
    }), "BOP-01")
    assert payload["plate"] == "KH12DE1433"
    assert payload["plate_verified"] is None      # falsy, and not claimed true


def test_events_without_a_plate_do_not_claim_one():
    from core.events import serialize_alert_row
    payload = serialize_alert_row(_row({"rule_name": "north fence"}, "entry"), "BOP-01")
    assert payload["plate"] is None
    assert payload["plate_confidence"] is None


def test_the_plate_crop_is_reachable_from_the_event():
    """The event has to lead to the pixels, or the plate is only an assertion."""
    import inspect
    import api.main as main

    source = inspect.getsource(main)
    assert '"/api/alerts/{alert_id}/plate"' in source, (
        "no endpoint serves the plate crop recorded for an event"
    )
    body = inspect.getsource(main.get_alert_plate)
    assert "ANPRDetection" in body and "alert_id" in body, (
        "the crop is not looked up against the alert that recorded it"
    )


def test_the_plate_reaches_the_printed_report():
    from core.report import build_event_log_pdf
    pypdf = pytest.importorskip("pypdf")
    import io as _io

    pdf = build_event_log_pdf([{
        "id": 7, "timestamp_ist": "13 Sep 2026, 10:00:00 AM IST",
        "camera_name": "BOP-01", "alert_type": "anpr_detection", "severity": "LOW",
        "object_class": "car", "track_id": 42, "plate": "MH12DE1433",
        "description": "Number plate read", "hash": "c" * 64,
    }], total_matching=1)
    text = "\n".join(p.extract_text() for p in pypdf.PdfReader(_io.BytesIO(pdf)).pages)
    assert "MH12DE1433" in text, "the registration is missing from the PDF report"
    assert "Plate" in text, "the report has no plate column"


def test_the_event_log_renders_a_plate_column():
    """The dashboard table has to show it, not bury it in the row detail."""
    import pathlib
    app = pathlib.Path("static/js/app.js").read_text(encoding="utf-8")
    html = pathlib.Path("dashboard/index.html").read_text(encoding="utf-8")
    assert "plateCell" in app, "the event log does not render a plate cell"
    assert "<th>Plate</th>" in html, "the event log has no plate column header"
    # Column count must match the header, or the empty-state row misaligns.
    assert 'colspan="11"' in app and 'colspan="11"' in html
