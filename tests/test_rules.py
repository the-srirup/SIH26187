"""
Rules engine behaviour.

Each rule is exercised against a synthetic trajectory so the assertions state
what an operator would actually observe: a crossing fires once, jitter fires
never, a short visit is not loitering.

Three of these tests are regressions for rules that could *never* fire in the
previous implementation.
"""
import pytest

from core.config import settings
from cv.rules import (
    DirectionRule, FenceRule, LoiterRule, NightMovementRule, RuleEngine,
    ZoneRule, severity_for,
)

SQUARE = [(50, 50), (200, 50), (200, 200), (50, 200)]


def walk(rule, track_id, points, start=0.0, step=0.1, **kwargs):
    """Feed a trajectory through a rule, returning every alert it produced."""
    fired = []
    for i, point in enumerate(points):
        alert = rule.update(track_id, point, timestamp=start + i * step, **kwargs)
        if alert:
            fired.append(alert)
    return fired


# --------------------------------------------------------------------- fence


def test_fence_fires_once_per_crossing():
    fence = FenceRule("f", 100, 0, 100, 400)
    fired = walk(fence, 1, [(x, 200) for x in range(60, 160, 5)])
    assert len(fired) == 1
    assert fired[0].alert_type in ("entry", "exit")


def test_fence_reports_opposite_direction_on_reverse_crossing():
    forward = walk(FenceRule("a", 100, 0, 100, 400), 1, [(x, 200) for x in range(60, 160, 5)])
    reverse = walk(FenceRule("b", 100, 0, 100, 400), 1, [(x, 200) for x in range(160, 55, -5)])
    assert forward and reverse
    assert forward[0].alert_type != reverse[0].alert_type


def test_fence_ignores_single_frame_jitter():
    """A shadow or branch oscillating over the line must never alert."""
    fence = FenceRule("f", 100, 0, 100, 400)
    jitter = [(99 if i % 2 else 101, 200) for i in range(40)]
    assert walk(fence, 1, jitter) == []


def test_fence_requires_anchor_confirmation():
    fence = FenceRule("f", 100, 0, 100, 400)
    # Cross, but hold the new side for fewer frames than the anchor requires.
    frames = [(90, 200)] + [(110, 200)] * (settings.ANCHOR_CONFIRMATION_FRAMES - 1)
    assert walk(fence, 1, frames) == []


def test_fence_uses_foot_point_not_box_centre():
    """The rule must evaluate the point it is handed — the ground contact."""
    fence = FenceRule("f", 0, 200, 640, 200)     # horizontal line at y=200
    above = walk(fence, 1, [(300, y) for y in range(150, 260, 5)])
    assert len(above) == 1


# ---------------------------------------------------------------------- zone


def test_zone_enter_fires():
    """Regression: the old ZoneRule stored state before comparing it, so the
    transition test was always false and this event could never fire."""
    zone = ZoneRule("z", SQUARE, presence_seconds=999)
    fired = walk(zone, 1, [(100, 100)] * 10, step=0.2)
    assert [a.alert_type for a in fired] == ["enter"]


def test_zone_exit_fires():
    zone = ZoneRule("z", SQUARE, presence_seconds=999)
    fired = walk(zone, 1, [(100, 100)] * 6 + [(400, 400)] * 3, step=0.2)
    assert [a.alert_type for a in fired] == ["enter", "zone_exit"]


def test_zone_presence_alerts_after_threshold():
    zone = ZoneRule("z", SQUARE, presence_seconds=2.0)
    fired = walk(zone, 1, [(100, 100)] * 12, step=0.5)
    assert "zone_presence" in [a.alert_type for a in fired]


def test_zone_outside_never_alerts():
    zone = ZoneRule("z", SQUARE, presence_seconds=1.0)
    assert walk(zone, 1, [(400, 400)] * 20) == []


# -------------------------------------------------------------------- loiter


def test_loiter_fires_after_dwell():
    """Regression: the old LoiterRule never recorded an entry timestamp, so
    elapsed dwell was always ~0 and loitering could never be detected."""
    loiter = LoiterRule("l", SQUARE, dwell_seconds=5.0)
    fired = walk(loiter, 7, [(100, 100)] * 20, step=1.0)
    assert len(fired) == 1
    assert fired[0].details["dwell_seconds"] >= 5.0


def test_loiter_does_not_fire_for_brief_visit():
    loiter = LoiterRule("l", SQUARE, dwell_seconds=10.0)
    assert walk(loiter, 1, [(100, 100)] * 4, step=1.0) == []


def test_loiter_resets_when_subject_leaves():
    loiter = LoiterRule("l", SQUARE, dwell_seconds=5.0)
    frames = [(100, 100)] * 4 + [(400, 400)] * 2 + [(100, 100)] * 4
    assert walk(loiter, 1, frames, step=1.0) == []


def test_loiter_works_with_media_time_starting_at_zero():
    """Offline analysis passes media time, which legitimately starts at 0.0.
    Treating 0.0 as 'no timestamp' silently substituted wall-clock time."""
    loiter = LoiterRule("l", SQUARE, dwell_seconds=3.0)
    fired = walk(loiter, 1, [(100, 100)] * 10, start=0.0, step=1.0)
    assert len(fired) == 1


# ----------------------------------------------------------------- direction


def test_direction_silent_when_travel_is_permitted():
    rule = DirectionRule("d", 100, 0, 100, 400, allowed_direction="exit")
    assert walk(rule, 1, [(x, 200) for x in range(60, 160, 5)]) == []


def test_direction_alerts_on_wrong_way():
    rule = DirectionRule("d", 100, 0, 100, 400, allowed_direction="entry")
    fired = walk(rule, 1, [(x, 200) for x in range(60, 160, 5)])
    assert len(fired) == 1
    assert fired[0].alert_type == "wrong_direction"


def test_direction_does_not_refire_every_frame():
    """Regression: the old rule never cleared its crossing state and re-fired
    on every subsequent frame, relying on the debouncer to hide it."""
    rule = DirectionRule("d", 100, 0, 100, 400, allowed_direction="entry")
    path = [(x, 200) for x in range(60, 160, 5)] + [(200, 200)] * 40
    assert len(walk(rule, 1, path)) == 1


# --------------------------------------------------------------------- night


def test_night_movement_silent_during_day():
    rule = NightMovementRule(min_travel=40)
    assert walk(rule, 1, [(i * 10, 100) for i in range(12)], is_night=False) == []


def test_night_movement_fires_at_night():
    rule = NightMovementRule(min_travel=40)
    fired = walk(rule, 1, [(i * 10, 100) for i in range(12)], is_night=True)
    assert len(fired) == 1
    assert fired[0].alert_type == "night_movement"


def test_night_movement_ignores_stationary_object():
    """A parked vehicle or a bush must not be reported as movement."""
    rule = NightMovementRule(min_travel=40)
    assert walk(rule, 1, [(50, 100)] * 40, is_night=True) == []


# -------------------------------------------------------------------- engine


def test_engine_debounces_repeated_crossings(det_factory):
    engine = RuleEngine("1", debounce_seconds=10.0)
    engine.add_rule(FenceRule("fence", 100, 0, 100, 400))

    count, t = 0, 0.0
    for _ in range(6):                       # six round trips over the line
        for x in list(range(60, 160, 5)) + list(range(160, 55, -5)):
            t += 0.1
            count += len(engine.update([det_factory(1, x, 200)], timestamp=t))
    assert count <= 3, f"{count} alerts from 12 crossings in {t:.0f}s"


def test_engine_isolates_tracks(det_factory):
    engine = RuleEngine("1", debounce_seconds=0.0)
    engine.add_rule(FenceRule("fence", 100, 0, 100, 400))
    fired = []
    for i, x in enumerate(range(60, 160, 5)):
        fired += engine.update(
            [det_factory(1, x, 200), det_factory(2, x, 300)], timestamp=i * 0.1
        )
    assert {a.track_id for a in fired} == {1, 2}


def test_engine_releases_state_for_departed_tracks(det_factory):
    """A multi-hour run must not accumulate per-track dictionaries."""
    engine = RuleEngine("1")
    engine.add_rule(FenceRule("f", 100, 0, 100, 400))
    for tid in range(300):
        engine.update([det_factory(tid, 50, 200)], timestamp=1.0)
    engine.update([det_factory(9999, 50, 200)],
                  timestamp=1.0 + settings.TRACK_STATE_TTL + 120)
    assert len(engine.get_rules()[0]._state) < 20


def test_engine_survives_a_raising_rule(det_factory):
    """One broken rule must not stop the others from being evaluated."""
    class Exploding(FenceRule):
        def update(self, *a, **k):
            raise RuntimeError("boom")

    engine = RuleEngine("1", debounce_seconds=0.0)
    engine.add_rule(Exploding("bad", 0, 0, 1, 1))
    engine.add_rule(FenceRule("good", 100, 0, 100, 400))
    fired = []
    for i, x in enumerate(range(60, 160, 5)):
        fired += engine.update([det_factory(1, x, 200)], timestamp=i * 0.1)
    assert len(fired) == 1


def test_engine_with_no_rules_is_a_no_op(det_factory):
    engine = RuleEngine("1")
    assert engine.update([det_factory(1, 10, 10)], timestamp=1.0) == []


# ------------------------------------------------------------------ severity


@pytest.mark.parametrize("alert_type,expected", [
    ("entry", "CRITICAL"),
    ("loiter", "HIGH"),
    ("night_movement", "HIGH"),
    ("zone_exit", "LOW"),
    ("watchlist_match", "CRITICAL"),
])
def test_severity_mapping(alert_type, expected):
    assert severity_for(alert_type) == expected
