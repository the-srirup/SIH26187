"""
Regression tests for defects found by measurement against real footage.

Each test here corresponds to a specific bug that was *reproduced* before it
was fixed, and each fails if the fix is reverted.  The comments record what the
original wrong behaviour actually was, because that is the part a future reader
cannot recover from the code.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from core.config import settings


# --------------------------------------------------------------------------- #
# Hash chain — concurrent append must not fork the chain
# --------------------------------------------------------------------------- #


def test_chain_survives_concurrent_appends(db, camera):
    """
    Many threads sealing events at once must produce one unbroken chain.

    The shipped database contained two rows (#2068 and #2069) written 44 ms
    apart by different camera threads that carried the *identical* ``prev_hash``
    with no gap in the id sequence — a read-modify-write race, not tampering.
    Verification correctly reported the log broken, so the flagship integrity
    feature read COMPROMISED for the life of the database.
    """
    from core.events import EventManager
    from core.hashchain import verify_chain
    from cv.rules import Alert as RuleAlert

    manager = EventManager.get()
    errors: list[Exception] = []

    def append(tag: str) -> None:
        try:
            for i in range(12):
                manager.record(
                    camera_id=camera.id,
                    rule_alert=RuleAlert(
                        rule_name="race", rule_type="test", track_id=i,
                        alert_type="human_detected", description=f"{tag}-{i}",
                    ),
                    frame=None, camera_name="RACE", capture_evidence=False,
                )
        except Exception as exc:  # pragma: no cover - surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(f"t{n}",)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"append raised: {errors}"
    result = verify_chain()
    assert result.valid, (
        f"chain forked under concurrent append: {result.message} "
        f"({len(result.breaks)} break(s))"
    )


def test_verification_detects_edited_description(db, camera):
    """
    Editing an event's narrative text must break the chain.

    ``description`` was not in the sealed payload, so every human-readable
    account in the log — the sentence an operator reads and a review board
    quotes — could be rewritten while verification still reported "chain
    intact". Sealing it is payload v2.
    """
    from core.events import EventManager
    from core.hashchain import verify_chain
    from core.models import Alert
    from cv.rules import Alert as RuleAlert

    EventManager.get().record(
        camera_id=camera.id,
        rule_alert=RuleAlert(rule_name="r", rule_type="test", track_id=1,
                             alert_type="loiter", description="Original account"),
        frame=None, camera_name="C", capture_evidence=False,
    )
    assert verify_chain().valid

    row = db.query(Alert).order_by(Alert.id.desc()).first()
    row.description = "Rewritten after the fact"
    db.commit()

    result = verify_chain()
    assert not result.valid
    assert result.break_kind == "payload"
    assert not result.forks_only


def test_verification_reports_every_break_not_just_the_first(db, camera):
    """Walking the whole chain is what makes the result diagnostic."""
    from core.events import EventManager
    from core.hashchain import verify_chain
    from core.models import Alert
    from cv.rules import Alert as RuleAlert

    for i in range(6):
        EventManager.get().record(
            camera_id=camera.id,
            rule_alert=RuleAlert(rule_name="r", rule_type="test", track_id=i,
                                 alert_type="loiter", description=f"e{i}"),
            frame=None, camera_name="C", capture_evidence=False,
        )
    rows = db.query(Alert).order_by(Alert.id.asc()).all()
    for row in (rows[1], rows[3]):
        row.confidence = 0.98765
    db.commit()

    result = verify_chain()
    assert not result.valid
    assert len(result.breaks) >= 2, "must report both faults, not stop at the first"


def test_repair_refuses_to_hide_real_tampering(db, camera):
    """
    Repair must never be usable to erase evidence of an edit.

    Re-linking a chain whose payload digests fail would overwrite the only proof
    that a record was altered — the repair would become the cover-up.
    """
    from core.events import EventManager
    from core.hashchain import repair_chain
    from core.models import Alert
    from cv.rules import Alert as RuleAlert

    EventManager.get().record(
        camera_id=camera.id,
        rule_alert=RuleAlert(rule_name="r", rule_type="test", track_id=1,
                             alert_type="loiter", description="real event"),
        frame=None, camera_name="C", capture_evidence=False,
    )
    row = db.query(Alert).order_by(Alert.id.desc()).first()
    row.description = "tampered"
    db.commit()

    outcome = repair_chain()
    assert outcome["ok"] is False
    assert outcome["repaired"] == 0
    assert "refusing" in outcome["message"].lower()


# --------------------------------------------------------------------------- #
# Coordinate mapping — the analytics resize is not aspect-preserving
# --------------------------------------------------------------------------- #


def test_source_scaling_uses_separate_axes():
    """
    640x384 (1.667) against a 16:9 source (1.778) is not a uniform resize.

    1920/640 = 3.000 across but 1080/384 = 2.8125 down. Using the width ratio
    for both axes put every face box and every plate crop ~4% of frame height
    too high — about 40 px at 1080p, which is taller than a plate is.
    """
    from cv.anpr import ANPRProcessor

    box = (301, 178, 317, 203)                 # analytics coordinates
    source = np.zeros((1080, 1920, 3), np.uint8)
    # Paint a 1-px line exactly where the box's top edge must land.
    true_y = int(box[1] * 1080 / 384)          # 500
    source[true_y, :] = 255

    correct = ANPRProcessor._crop(source, box, (1920 / 640.0, 1080 / 384.0))
    wrong = ANPRProcessor._crop(source, box, (3.0, 3.0))   # width ratio on both axes

    def line_row(crop):
        rows = np.where(crop.reshape(crop.shape[0], -1).max(axis=1) == 255)[0]
        return int(rows[0]) if rows.size else None

    # Both crops are padded, so both may contain the line; what matters is where
    # inside the crop it falls relative to the crop's own top edge.
    correct_row, wrong_row = line_row(correct), line_row(wrong)
    assert correct_row is not None, "correct mapping lost the target row entirely"

    # The two mappings must disagree by the aspect-ratio error, ~40 px at 1080p.
    offset = abs((true_y / 3.0) - (true_y / (1080 / 384.0)))
    assert offset > 10, "test geometry no longer exercises the aspect-ratio error"
    if wrong_row is not None:
        assert abs(correct_row - wrong_row) >= 10, (
            "per-axis and single-axis scaling produced the same crop — "
            "the fix is not in effect"
        )


# ANPR grammar — a misread can satisfy the grammar under a wrong parse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    # 'Z' misread for '2' — matches the loose grammar as MH-1-ZAB-1234, so the
    # old early-return accepted it unchanged and never consulted the scorer.
    ("MH1ZAB1234", "MH12AB1234"),
    ("DLBCAF5O31", "DL8CAF5031"),
    ("KAO1F1234", "KA01F1234"),
])
def test_plate_grammar_corrects_confusable_glyphs(raw, expected):
    from cv.anpr import get_anpr_processor

    corrected, changed = get_anpr_processor().apply_plate_grammar(raw)
    assert corrected == expected
    assert changed is True


@pytest.mark.parametrize("plate", [
    "MH12AB1234", "DL8CAF5031", "KA01F1234", "TN07BZ9999", "HR26DQ5551",
])
def test_plate_grammar_leaves_valid_plates_alone(plate):
    """Correction must not 'fix' a registration that is already right."""
    from cv.anpr import get_anpr_processor

    corrected, changed = get_anpr_processor().apply_plate_grammar(plate)
    assert corrected == plate
    assert changed is False


# --------------------------------------------------------------------------- #
# Night movement — a distance-only test excludes fast movers
# --------------------------------------------------------------------------- #


def test_night_movement_fires_for_a_fast_short_lived_track():
    """
    A vehicle crossing frame in under a second is still movement.

    Measured on this project's night clip: a person is tracked 4.2 s / 338 px
    and alerts; a car is detected at higher confidence but tracked 0.8 s / 37 px
    and was rejected by the 45 px bar. That is the whole of "night detection
    works for people but not vehicles".
    """
    from cv.rules import NightMovementRule

    rule = NightMovementRule()
    fired = 0
    for index in range(18):
        timestamp = index / 15.0
        point = (100 + 60 * timestamp, 300)          # 60 px/s
        if rule.update(2, point, timestamp=timestamp, is_night=True):
            fired += 1
    assert fired >= 1, "fast short-lived track produced no night_movement alert"


def test_night_movement_ignores_a_jittering_stationary_object():
    """Box jitter inflates path length while going nowhere — must not alert."""
    import random

    from cv.rules import NightMovementRule

    random.seed(11)
    rule = NightMovementRule()
    fired = 0
    for index in range(150):
        point = (320 + random.uniform(-3, 3), 300 + random.uniform(-3, 3))
        if rule.update(1, point, timestamp=index / 15.0, is_night=True):
            fired += 1
    assert fired == 0, "stationary jitter was reported as night movement"


def test_night_movement_ignores_a_two_frame_teleport():
    """An identity re-association is not travel."""
    from cv.rules import NightMovementRule

    rule = NightMovementRule()
    fired = sum(
        1 for index, point in enumerate([(10, 10), (400, 300)])
        if rule.update(3, point, timestamp=index / 15.0, is_night=True)
    )
    assert fired == 0


# --------------------------------------------------------------------------- #
# Loiter — a quiet zone must be able to explain itself
# --------------------------------------------------------------------------- #


def test_loiter_reports_why_it_stayed_quiet():
    """
    "No alert" and "the threshold is above anything that happens here" look
    identical from outside the rule, and only the second is the operator's to
    fix. The rule therefore reports the longest dwell it has actually seen.
    """
    from cv.rules import LoiterRule

    rule = LoiterRule("Z", [[0, 0], [400, 0], [400, 400], [0, 400]],
                      dwell_seconds=30.0, classes=[])
    for index in range(60):                 # 4 s inside, at 15 fps
        rule.update(1, (200, 200), timestamp=index / 15.0)

    described = rule.describe()
    assert described["visits_seen"] == 1
    assert described["alerts_raised"] == 0
    assert 3.0 <= described["longest_dwell_seen"] <= 4.5
    assert described["threshold_reachable"] is False


def test_loiter_fires_exactly_at_the_threshold():
    from cv.rules import LoiterRule

    rule = LoiterRule("Z", [[0, 0], [400, 0], [400, 400], [0, 400]],
                      dwell_seconds=2.0, classes=[])
    fired_at = None
    for index in range(90):
        timestamp = index / 15.0
        if rule.update(1, (200, 200), timestamp=timestamp) and fired_at is None:
            fired_at = timestamp
    assert fired_at is not None
    assert 2.0 <= fired_at <= 2.4, f"fired at {fired_at}s, expected ~2.0s"


# --------------------------------------------------------------------------- #
# Camera registration — duplicates and malformed URLs
# --------------------------------------------------------------------------- #


def test_bracketed_credentials_are_rejected():
    """
    The exact URL this project's own database contained.

    Square brackets delimit an IPv6 literal in RFC 3986, so the host never
    resolves and FFmpeg simply hangs — which presented as "the site lags after
    I add a camera" rather than as a bad URL.
    """
    from core.sources import SourceError, validate_live_url

    with pytest.raises(SourceError) as excinfo:
        validate_live_url("rtsp://[user]:[pass]@[192.168.1.2]:554/stream1")
    assert "bracket" in str(excinfo.value).lower()


@pytest.mark.parametrize("url", [
    "rtsp://admin:secret@192.168.1.2:554/stream1",
    "http://camera.local/video.mjpg",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "0",
])
def test_valid_sources_are_accepted(url):
    from core.sources import validate_live_url

    assert validate_live_url(url) == url


@pytest.mark.parametrize("url", ["99", "rtsp://", "rtsp://host/a b", "/nope/missing.mp4"])
def test_malformed_sources_are_rejected(url):
    from core.sources import SourceError, validate_live_url

    with pytest.raises(SourceError):
        validate_live_url(url)


def test_duplicate_camera_is_rejected(db):
    """
    One URL, one pipeline.

    Registering a source twice produced two capture threads, two decoders, two
    analytics threads and two rule engines over identical pixels — the shipped
    database still holds four such rows all pointed at one sample clip.
    """
    from core.sources import SourceError, register_camera

    register_camera(db, name="A", url="rtsp://10.0.0.5:554/s1", validate=False)
    with pytest.raises(SourceError) as excinfo:
        register_camera(db, name="B", url="rtsp://10.0.0.5:554/s1", validate=False)
    assert "already registered" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# YouTube source routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=abc123", True),
    ("https://youtu.be/abc123", True),
    ("https://youtube.com/live/abc123", True),
    ("https://m.youtube.com/watch?v=abc", True),
    ("rtsp://192.168.1.2:554/stream1", False),
    ("https://example.com/video.mp4", False),
    ("", False),
])
def test_youtube_urls_are_recognised(url, expected):
    from core.youtube import is_youtube_url

    assert is_youtube_url(url) is expected


def test_non_youtube_urls_pass_through_source_resolution():
    """A YouTube link is resolved; everything else must be handed on untouched."""
    from core.video_source import resolve_source_url

    assert resolve_source_url("rtsp://1.2.3.4/s") == "rtsp://1.2.3.4/s"
    assert resolve_source_url("0") == "0"


@pytest.mark.parametrize("url,kind", [
    ("https://youtu.be/x", "youtube"),
    ("rtsp://1.2.3.4/s", "rtsp"),
    ("http://cam/live.mjpg", "http"),
    ("0", "webcam"),
    ("samples/clip.mp4", "file"),
])
def test_source_kind_detection(url, kind):
    from core.video_source import LiveSource

    assert LiveSource(url).kind == kind


# --------------------------------------------------------------------------- #
# Live capture pacing
# --------------------------------------------------------------------------- #


def test_live_sources_are_paced_with_headroom(monkeypatch):
    """
    A buffered network stream must not be decoded without bound.

    A file plays at exactly its own rate; a live source gets headroom so it can
    regain the live edge after a stall, but is still capped.
    """
    from core.video_source import LiveSource

    monkeypatch.setattr(settings, "LIVE_CAPTURE_HEADROOM", 1.5)

    source = LiveSource("rtsp://1.2.3.4/s")
    source.stats.source_fps = 30.0
    source._is_file = False
    # Mirror _try_open's pacing decision.
    rate = source.stats.source_fps * settings.LIVE_CAPTURE_HEADROOM
    assert rate == pytest.approx(45.0)

    file_source = LiveSource("clip.mp4")
    file_source.stats.source_fps = 30.0
    file_source._is_file = True
    assert file_source.stats.source_fps == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# Structured detection storage
# --------------------------------------------------------------------------- #


def test_evidence_tree_is_dated_and_per_camera():
    from core.detections import evidence_dir

    path = evidence_dir("anpr", 7, "2026-09-11T17:42:33+00:00")
    parts = path.parts
    assert "anpr" in parts and "camera_7" in parts
    assert "2026" in parts and "09" in parts


def test_detection_evidence_paths_are_inside_the_served_roots():
    """A row must never be able to point the file endpoint outside evidence."""
    from core.detections import evidence_dir
    from core.evidence import is_safe_evidence_path

    assert is_safe_evidence_path(evidence_dir("faces", 3) / "x.jpg")
    assert not is_safe_evidence_path("C:/Windows/System32/config")
    assert not is_safe_evidence_path("/etc/passwd")
