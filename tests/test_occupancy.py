"""
Zone occupancy as a *continuous* signal.

Entering and leaving a polygon are moments, and the event log already records
them. Being inside one is a condition — and a condition reported only at its
edges cannot be shown continuously, so an operator who looks up thirty seconds
after the entry event sees an empty screen and concludes the area is clear.

These tests pin both halves of the fix: occupancy readable as live state on
every frame, and a sustained-presence event that re-announces itself for as
long as the subject stays rather than falling silent after one row.
"""
from __future__ import annotations

import pytest

from core.config import settings

SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


@pytest.fixture
def zone():
    from cv.rules import ZoneRule

    return ZoneRule("SECTOR-4", SQUARE, presence_seconds=5.0,
                    enter_frames=1, exit_grace=2.0)


def _walk_in(rule, track_id, t, frames=3, point=(200, 200)):
    """Put a track convincingly inside, returning the last alert seen."""
    alert = None
    for n in range(frames):
        alert = rule.update(track_id, point, timestamp=t + n * 0.1) or alert
    return alert


# --------------------------------------------------------------------------- #
# Occupancy is live state
# --------------------------------------------------------------------------- #


def test_an_empty_zone_reports_itself_empty(zone):
    state = zone.occupancy(now=1000.0)
    assert state["occupied"] is False
    assert state["count"] == 0
    assert state["tracks"] == []
    assert state["breached"] is False
    # ZoneGeometry normalises to floats and closes the ring, so compare shape
    # rather than the literal list we passed in.
    assert [[int(x), int(y)] for x, y in state["geometry"]][:4] == SQUARE


def test_occupancy_is_reported_every_frame_not_only_on_entry(zone):
    """
    The whole point: the signal must persist between events.

    Entry raises exactly one alert. Every frame after it raises nothing — and
    it is precisely those frames during which the operator must still be able
    to see that someone is in the zone.
    """
    entry = _walk_in(zone, track_id=7, t=1000.0)
    assert entry is not None and entry.alert_type == "enter"

    silent_frames = 0
    for n in range(1, 40):                    # ~4 s of frames, no dwell breach
        now = 1000.0 + n * 0.1
        if zone.update(7, (200, 200), timestamp=now) is None:
            silent_frames += 1
        state = zone.occupancy(now=now)
        assert state["occupied"] is True, f"lost the signal at frame {n}"
        assert state["tracks"] == [7]
        assert state["count"] == 1

    assert silent_frames > 30, "expected the event stream to be quiet"


def test_occupancy_counts_every_object_inside(zone):
    for track_id in (1, 2, 3):
        _walk_in(zone, track_id, t=2000.0)
    state = zone.occupancy(now=2000.5)
    assert state["count"] == 3
    assert state["tracks"] == [1, 2, 3]


def test_occupancy_reports_how_long_the_zone_has_been_occupied(zone):
    _walk_in(zone, 9, t=3000.0)
    # Keep the track alive: occupancy deliberately ignores a track that has not
    # been seen for longer than the exit grace, so that a detector which stops
    # reporting cannot leave the banner up forever.
    for n in range(1, 121):
        zone.update(9, (200, 200), timestamp=3000.0 + n * 0.1)
    state = zone.occupancy(now=3012.0)
    assert state["occupied"] is True
    assert 11.5 <= state["seconds"] <= 12.5


def test_breached_turns_on_at_the_dwell_threshold(zone):
    """``breached`` is the difference between traffic and an intrusion."""
    _walk_in(zone, 4, t=4000.0)
    for n in range(1, 21):                    # still there at t+2.0
        zone.update(4, (200, 200), timestamp=4000.0 + n * 0.1)
    assert zone.occupancy(now=4002.0)["breached"] is False   # inside, < 5 s

    for n in range(21, 61):                   # still there at t+6.0
        zone.update(4, (200, 200), timestamp=4000.0 + n * 0.1)
    assert zone.occupancy(now=4006.0)["breached"] is True    # inside, > 5 s


def test_occupancy_clears_once_the_subject_leaves(zone):
    _walk_in(zone, 5, t=5000.0)
    assert zone.occupancy(now=5000.5)["occupied"] is True

    # Outside, sustained past the exit grace.
    for n in range(30):
        zone.update(5, (10, 10), timestamp=5001.0 + n * 0.2)
    state = zone.occupancy(now=5008.0)
    assert state["occupied"] is False
    assert state["count"] == 0


def test_a_dropped_detection_does_not_empty_the_zone(zone):
    """
    Same reasoning as the exit grace: one missed frame is not a departure.

    Without this the banner would flicker off and on every time the detector
    lost the box for a frame, which is worse than no signal at all.
    """
    _walk_in(zone, 6, t=6000.0)
    # The track simply stops being reported — no update calls at all.
    assert zone.occupancy(now=6001.0)["occupied"] is True    # within grace
    assert zone.occupancy(now=6010.0)["occupied"] is False   # long gone


# --------------------------------------------------------------------------- #
# The log keeps saying so
# --------------------------------------------------------------------------- #


def test_sustained_presence_re_announces_while_the_subject_stays(zone, monkeypatch):
    """
    A single ``zone_presence`` row cannot distinguish someone who left from
    someone still standing there an hour later.
    """
    monkeypatch.setattr(settings, "ZONE_PRESENCE_REPEAT_SECONDS", 10.0)
    _walk_in(zone, 11, t=7000.0)

    announcements = []
    for n in range(1, 400):                   # 40 s at 10 Hz
        now = 7000.0 + n * 0.1
        alert = zone.update(11, (200, 200), timestamp=now)
        if alert is not None and alert.alert_type == "zone_presence":
            announcements.append(now)

    assert len(announcements) >= 4, announcements
    # First at the dwell threshold, then one per repeat interval.
    assert announcements[0] == pytest.approx(7005.0, abs=0.3)
    gaps = [b - a for a, b in zip(announcements, announcements[1:])]
    assert all(9.0 <= gap <= 11.0 for gap in gaps), gaps


def test_repeat_can_be_switched_off(zone, monkeypatch):
    """Zero restores the old behaviour: announce sustained presence once."""
    monkeypatch.setattr(settings, "ZONE_PRESENCE_REPEAT_SECONDS", 0.0)
    _walk_in(zone, 12, t=8000.0)

    count = 0
    for n in range(1, 400):
        alert = zone.update(12, (200, 200), timestamp=8000.0 + n * 0.1)
        if alert is not None and alert.alert_type == "zone_presence":
            count += 1
    assert count == 1


def test_leaving_and_returning_starts_a_new_announcement_cycle(zone, monkeypatch):
    monkeypatch.setattr(settings, "ZONE_PRESENCE_REPEAT_SECONDS", 10.0)
    _walk_in(zone, 13, t=9000.0)
    for n in range(1, 80):                    # dwell past the threshold
        zone.update(13, (200, 200), timestamp=9000.0 + n * 0.1)
    for n in range(40):                       # leave, past the exit grace
        zone.update(13, (10, 10), timestamp=9010.0 + n * 0.2)
    assert zone.occupancy(now=9020.0)["occupied"] is False

    re_entry = _walk_in(zone, 13, t=9030.0)
    assert re_entry is not None and re_entry.alert_type == "enter"
    assert zone.occupancy(now=9030.5)["occupied"] is True


# --------------------------------------------------------------------------- #
# It reaches the pipeline and the dashboard
# --------------------------------------------------------------------------- #


def test_the_engine_rolls_occupancy_up_across_rules():
    from cv.rules import FenceRule, LoiterRule, RuleEngine, ZoneRule

    engine = RuleEngine(camera_id="1")
    engine.add_rule(ZoneRule("ZONE-A", SQUARE, presence_seconds=5.0, enter_frames=1))
    engine.add_rule(LoiterRule("LOITER-B", SQUARE, dwell_seconds=5.0))
    engine.add_rule(FenceRule("WIRE-C", 0, 400, 640, 400))

    occupancy = engine.occupancy(now=100.0)
    names = {entry["rule"] for entry in occupancy}
    assert names == {"ZONE-A", "LOITER-B"}, "a tripwire has no inside to report"
    assert all(entry["occupied"] is False for entry in occupancy)


def test_the_analysis_result_carries_occupancy(db):
    """The pipeline must hand occupancy out, or nothing downstream can show it."""
    import numpy as np

    from core.analytics import FrameAnalyzer

    class _Detector:
        names = {0: "person"}

        def raw_detect(self, frame):
            return (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
                    np.empty((0,), np.int32), 0.0)

    analyzer = FrameAnalyzer(source_id="occ", display_name="OCC",
                             detector=_Detector(), enable_face=False,
                             enable_anpr=False)
    analyzer.set_rules([{
        "id": 1, "rule_type": "zone", "geometry": SQUARE,
        "params": {}, "name": "SECTOR-4",
    }])

    frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 90, np.uint8)
    result = analyzer.analyse(frame, timestamp=10.0)

    assert isinstance(result.zones, list) and len(result.zones) == 1
    assert result.zones[0]["rule"] == "SECTOR-4"
    assert result.zones[0]["occupied"] is False


def test_camera_stats_expose_occupancy_for_the_dashboard(db):
    """It has to ride the 1 Hz stats frame — that is what holds the UI signal."""
    from core.camera import CameraProcessor

    class _Detector:
        names = {0: "person"}

    proc = CameraProcessor(camera_id=8801, url="0", name="OCC-CAM",
                           detector=_Detector())
    try:
        stats = proc.stats()
        assert "zones" in stats
        assert stats["zones_occupied"] == 0
        assert stats["zones_breached"] == 0
    finally:
        proc.stop()
