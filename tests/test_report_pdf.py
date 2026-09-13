"""
Tests for the printable event-log report.

A PDF of a sealed log is evidence, so what matters is not only that bytes come
out but that the page states its own provenance: which filters produced it, how
many events it covers against how many matched, and whether the hash chain
verified at the moment of printing. A table of rows with none of that proves
nothing about the log it came from.
"""
from __future__ import annotations

import pytest

report = pytest.importorskip("core.report")
pytest.importorskip("reportlab")


def _alert(i, severity="INFO", alert_type="human_detected"):
    return {
        "id": i,
        "timestamp_ist": f"13 Sep 2026, 10:0{i % 10}:00 AM IST",
        "camera_name": "BOP-01 NORTH",
        "alert_type": alert_type,
        "severity": severity,
        "object_class": "person",
        "track_id": 100 + i,
        "description": f"PERSON #{100 + i} detected in view",
        "hash": f"{i:064x}",
    }


def _text(pdf_bytes):
    pypdf = pytest.importorskip("pypdf")
    import io
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages), len(reader.pages)


def test_a_report_is_a_valid_pdf():
    pdf = report.build_event_log_pdf([_alert(i) for i in range(5)], total_matching=5)
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")


def test_every_event_appears_in_the_report():
    alerts = [_alert(i) for i in range(25)]
    text, _pages = _text(report.build_event_log_pdf(alerts, total_matching=25))
    for a in alerts:
        assert str(a["track_id"]) in text, f"track {a['track_id']} is missing"


def test_the_report_says_when_it_is_only_part_of_the_log():
    """A truncated page must never read as the whole story."""
    text, _ = _text(report.build_event_log_pdf(
        [_alert(i) for i in range(10)], total_matching=4306))
    assert "4,306" in text
    assert "most recent 10" in text


def test_the_report_records_the_filters_that_produced_it():
    text, _ = _text(report.build_event_log_pdf(
        [_alert(1, severity="HIGH")], total_matching=1,
        filters={"camera": "BOP-01 NORTH", "severity": "HIGH", "search": None}))
    assert "BOP-01 NORTH" in text
    assert "HIGH" in text
    # A filter that was not applied must not be claimed.
    assert "search" not in text.lower().split("hash-chain")[0].replace("search…", "")


def test_the_report_states_the_chain_status_it_measured():
    ok, _ = _text(report.build_event_log_pdf(
        [_alert(1)], total_matching=1,
        integrity={"valid": True, "verified": 4306, "breaks": []}))
    assert "VERIFIED" in ok

    bad, _ = _text(report.build_event_log_pdf(
        [_alert(1)], total_matching=1,
        integrity={"valid": False, "verified": 10, "breaks": [{"id": 4}]}))
    assert "BROKEN" in bad, "a tampered log must say so on the printed page"


def test_an_empty_result_still_produces_a_report():
    """'Nothing matched' is a finding, and an operator may need it on paper."""
    text, pages = _text(report.build_event_log_pdf([], total_matching=0))
    assert pages >= 1
    assert "No events matched" in text


def test_a_long_log_paginates():
    alerts = [_alert(i) for i in range(300)]
    _text_out, pages = _text(report.build_event_log_pdf(alerts, total_matching=300))
    assert pages > 1, "300 events were crammed onto one page"


def test_awkward_content_does_not_break_the_build():
    """Real rows carry missing fields and very long descriptions."""
    rough = [
        {"id": 1},                                     # almost everything absent
        {**_alert(2), "description": "x" * 4000},      # pathologically long
        {**_alert(3), "camera_name": None, "severity": "NOPE"},
        {**_alert(4), "description": "unicode — em-dash, ₹, नमस्ते"},
    ]
    pdf = report.build_event_log_pdf(rough, total_matching=4)
    assert pdf.startswith(b"%PDF-")
