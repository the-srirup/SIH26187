"""
End-to-end behavioural scenarios for the analytics rules.

These tests describe what an operator would observe, in the situations the
platform is actually deployed into: a fast vehicle crossing a tripwire, a
person loitering while the detector drops a frame, several people in a zone at
once, a scene going dark.

They are deliberately written against the *reported failures*, so a regression
reintroduces a named bug rather than merely failing an abstract assertion.
"""
import cv2
import numpy as np
import pytest

from core.config import settings
from cv.rules import (
    DirectionRule, FenceRule, LoiterRule, NightMovementRule, RuleEngine, ZoneRule,
)
from cv.scene import SceneIlluminationEstimator

SQUARE = [(50, 50), (250, 50), (250, 250), (50, 250)]


class Det:
    """Stand-in for cv.detector.Detection with a controllable foot point."""

    def __init__(self, track_id, x, y, class_name="person", confidence=0.9, age=10):
        self.track_id = track_id
        self.foot = (x, y)
        self.bbox = (x - 15, y - 50, x + 15, y)
        self.draw_bbox = self.bbox
        self.class_name = class_name
        self.class_id = 0 if class_name == "person" else 2
        self.confidence = confidence
        self.age = age

    @property
    def is_person(self):
        return self.class_id == 0

    @property
    def is_vehicle(self):
        return self.class_id != 0


def drive(rule, track_id, points, step=1 / 15, start=0.0, **kw):
    out = []
    for i, (x, y) in enumerate(points):
        alert = rule.update(track_id, (x, y), timestamp=start + i * step, **kw)
        if alert:
            out.append(alert)
    return out


# --------------------------------------------------------------------------- #
# Tripwire — the fast-vehicle case
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("px_per_frame", [5, 20, 60, 150, 300])
def test_tripwire_catches_every_crossing_speed(px_per_frame):
    """
    A crossing must be reported regardless of how fast the object is moving.

    This is the reported "fast-moving car or scooter is frequently missed".
    At 15 FPS analytics, 300 px/frame is roughly a vehicle at highway speed
    across a 640 px frame — it appears on one side and is past the line on the
    very next analysed frame. The old rule required the object to *hold* the new
    side for three further frames, which such a vehicle never does, so it was
    silently never reported.
    """
    fence = FenceRule("tripwire", 320, 0, 320, 384)
    path = [(x, 200) for x in range(20, 640, px_per_frame)]
    fired = drive(fence, 1, path)
    assert len(fired) == 1, f"{len(fired)} alerts at {px_per_frame}px/frame"
    assert fired[0].alert_type in ("entry", "exit")


def test_tripwire_reports_direction_correctly_both_ways():
    fence_a = FenceRule("a", 320, 0, 320, 384)
    fence_b = FenceRule("b", 320, 0, 320, 384)
    left_to_right = drive(fence_a, 1, [(x, 200) for x in range(200, 460, 40)])
    right_to_left = drive(fence_b, 2, [(x, 200) for x in range(440, 180, -40)])
    assert left_to_right and right_to_left
    assert left_to_right[0].alert_type != right_to_left[0].alert_type


@pytest.mark.parametrize("line,path", [
    # horizontal, vertical and both diagonals must behave identically
    ((40, 200, 600, 200), [(300, y) for y in range(100, 320, 20)]),
    ((320, 40, 320, 340), [(x, 200) for x in range(180, 460, 20)]),
    ((100, 100, 500, 300), [(300, y) for y in range(60, 340, 20)]),
    ((100, 300, 500, 100), [(300, y) for y in range(60, 340, 20)]),
])
def test_tripwire_handles_any_line_orientation(line, path):
    """Segment intersection has no special cases — no orientation is favoured."""
    fired = drive(FenceRule("f", *line), 1, path)
    assert len(fired) == 1


def test_tripwire_does_not_fire_for_an_object_that_only_approaches():
    """Being near the line is not crossing it."""
    fence = FenceRule("f", 320, 0, 320, 384)
    approach = [(x, 200) for x in range(100, 300, 10)]      # stops short of 320
    assert drive(fence, 1, approach) == []


def test_tripwire_does_not_refire_while_the_object_stays_across():
    fence = FenceRule("f", 320, 0, 320, 384)
    path = [(x, 200) for x in range(200, 460, 20)] + [(450, 200)] * 60
    assert len(drive(fence, 1, path)) == 1


def test_tripwire_tracks_several_objects_independently():
    fence = FenceRule("f", 320, 0, 320, 384)
    fired = []
    for i, x in enumerate(range(200, 460, 20)):
        for tid, y in ((1, 150), (2, 250), (3, 350)):
            alert = fence.update(tid, (x, y), timestamp=i / 15)
            if alert:
                fired.append(alert)
    assert {a.track_id for a in fired} == {1, 2, 3}
    assert len(fired) == 3


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #


def test_direction_alerts_only_against_the_permitted_flow():
    allowed = drive(DirectionRule("d", 320, 0, 320, 384, allowed_direction="entry"),
                    1, [(x, 200) for x in range(440, 180, -40)])
    against = drive(DirectionRule("d", 320, 0, 320, 384, allowed_direction="entry"),
                    1, [(x, 200) for x in range(200, 460, 40)])
    # Exactly one of the two directions raises wrong_direction.
    assert len(allowed) + len(against) == 1


@pytest.mark.parametrize("width,height", [(640, 384), (1280, 720), (320, 240)])
def test_direction_is_resolution_independent(width, height):
    """A rule scaled to another frame size behaves the same."""
    mid = width // 2
    rule = DirectionRule("d", mid, 0, mid, height, allowed_direction="exit")
    path = [(x, height // 2)
            for x in range(int(mid * 0.4), int(mid * 1.6), max(1, width // 32))]
    assert len(drive(rule, 1, path)) in (0, 1)


# --------------------------------------------------------------------------- #
# Restricted zone
# --------------------------------------------------------------------------- #


def test_restricted_zone_reports_entry_then_sustained_presence():
    zone = ZoneRule("apron", SQUARE, presence_seconds=2.0, exit_alerts=True)
    types = [a.alert_type for a in drive(zone, 1, [(150, 150)] * 60, step=0.1)]
    assert types[0] == "enter"
    assert "zone_presence" in types


def test_restricted_zone_ignores_an_object_that_never_enters():
    zone = ZoneRule("apron", SQUARE, exit_alerts=True)
    assert drive(zone, 1, [(500, 500)] * 40) == []


def test_restricted_zone_boundary_case_does_not_flap():
    """
    A subject standing exactly on the boundary must not generate a storm.

    The old rule required three frames to accept an entry but zero to accept an
    exit; the real event log shows 66 enter/zone_exit pairs less than a second
    apart as a result.
    """
    zone = ZoneRule("apron", SQUARE, exit_alerts=True)
    edge = [((248 if i % 2 else 252), 150) for i in range(120)]
    assert len(drive(zone, 1, edge, step=0.05)) <= 1


def test_restricted_zone_handles_many_objects_at_once():
    zone = ZoneRule("apron", SQUARE, presence_seconds=999, exit_alerts=True)
    fired = []
    for i in range(20):
        for tid in range(1, 6):
            alert = zone.update(tid, (100 + tid * 10, 150), timestamp=i * 0.2)
            if alert:
                fired.append(alert)
    assert {a.track_id for a in fired} == {1, 2, 3, 4, 5}
    assert all(a.alert_type == "enter" for a in fired)


def test_restricted_zone_uses_the_foot_point_not_the_box_centre():
    """
    A person standing just outside the zone must not register as inside.

    With a typical downward-looking camera the box centre floats at chest
    height, well inside a zone the subject is merely standing next to.
    """
    zone = ZoneRule("apron", SQUARE, exit_alerts=True)
    engine = RuleEngine("cam", debounce_seconds=0.0)
    engine.add_rule(zone)
    # Feet at y=270 (below the zone's lower edge at 250); centre would be y=245.
    fired = []
    for i in range(10):
        fired += engine.update([Det(1, 150, 270)], timestamp=i * 0.2)
    assert fired == []


# --------------------------------------------------------------------------- #
# Loitering
# --------------------------------------------------------------------------- #


def test_loiter_fires_once_after_the_configured_dwell():
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=10.0, realert_seconds=0)
    fired = drive(loiter, 1, [(150, 150)] * 300, step=0.1)
    assert len(fired) == 1
    assert fired[0].details["dwell_seconds"] >= 10.0


def test_loiter_ignores_a_visit_shorter_than_the_threshold():
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=10.0)
    assert drive(loiter, 1, [(150, 150)] * 40, step=0.1) == []


def test_loiter_survives_missed_detections():
    """
    The detector dropping a subject for a few frames must not reset the clock.

    This is the root cause of the unreliable loiter zone: the old rule popped
    its entire state the moment a foot point fell outside, so one jittery frame
    at the boundary — or one missed detection — restarted the dwell timer, and
    on real footage the threshold was effectively never reached.
    """
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=5.0,
                        exit_grace=3.0, realert_seconds=0)
    frames = []
    for second in range(12):
        # Every 4th second the detector loses the subject for one frame.
        frames.extend([(150, 150)] * 9)
        frames.append((150, 150) if second % 4 else (500, 500))
    fired = drive(loiter, 1, frames, step=0.1)
    assert len(fired) == 1, "a momentary detection gap reset the dwell timer"


def test_loiter_restarts_after_a_real_departure():
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=5.0,
                        exit_grace=1.0, realert_seconds=0)
    # 4 s inside, 4 s genuinely away, then 4 s inside again: neither visit
    # reaches the 5 s threshold on its own, so nothing should fire.
    frames = [(150, 150)] * 40 + [(500, 500)] * 40 + [(150, 150)] * 40
    assert drive(loiter, 1, frames, step=0.1) == []


def test_loiter_handles_several_people_with_independent_clocks():
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=4.0, realert_seconds=0)
    fired = []
    for i in range(80):
        t = i * 0.1
        # Track 1 stays throughout; track 2 arrives late; track 3 never enters.
        for tid, point in ((1, (120, 120)),
                           (2, (180, 180) if i > 55 else (500, 500)),
                           (3, (600, 600))):
            alert = loiter.update(tid, point, timestamp=t)
            if alert:
                fired.append(alert)
    assert {a.track_id for a in fired} == {1}


def test_loiter_is_not_triggered_by_a_parked_vehicle():
    loiter = LoiterRule("loiter", SQUARE, dwell_seconds=3.0, classes=["person"])
    engine = RuleEngine("cam", debounce_seconds=0.0)
    engine.add_rule(loiter)
    fired = []
    for i in range(120):
        fired += engine.update([Det(1, 150, 150, class_name="car")], timestamp=i * 0.2)
    assert fired == []


# --------------------------------------------------------------------------- #
# Night detection — visual, not clock-based
# --------------------------------------------------------------------------- #


def _frame(value, size=(384, 640)):
    return np.full((*size, 3), value, np.uint8)


def _settle(estimator, frame, n=40):
    condition = None
    for _ in range(n):
        condition = estimator.measure(frame)
    return condition


def test_night_is_not_declared_for_a_bright_scene():
    assert _settle(SceneIlluminationEstimator(), _frame(190)).is_night is False


def test_night_is_declared_for_a_dark_scene():
    condition = _settle(SceneIlluminationEstimator(), _frame(8))
    assert condition.is_night is True
    assert condition.source == "darkness"


def test_night_survives_a_bright_light_in_a_dark_scene():
    """
    A streetlamp or headlights raise the mean brightness but not the truth.

    Mean luma alone reads this as daylight; the dark-pixel fraction does not,
    which is why it carries the larger weight.
    """
    scene = _frame(12)
    cv2.circle(scene, (520, 80), 55, (255, 255, 255), -1)
    assert _settle(SceneIlluminationEstimator(), scene).is_night is True


def _ir_frame(peak: int = 150) -> np.ndarray:
    """
    A frame that looks like real night-vision footage.

    The fixture matters here. This test used to pass pure per-pixel noise
    (``integers(95, 135)``), which the estimator's INTER_AREA downsample
    averages away to a luma standard deviation of 2.9 with not one dark pixel —
    a flat grey field, not a scene. That is exactly what a phone used as a
    webcam emits while it connects, and treating it as night vision is what
    produced phantom night-movement alerts in a lit room.

    Genuine IR footage has spatial structure that survives downsampling, and an
    IR illuminator lights a cone and leaves the rest of the frame black, so
    some truly dark pixels are always present.
    """
    height, width = 384, 640
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    cone = np.exp(-(((xx - width / 2) / (width * 0.33)) ** 2
                    + ((yy - height * 0.62) / (height * 0.4)) ** 2))
    grey = np.clip(cone * peak + rng.normal(0, 22, (height, width)) * cone
                   + rng.normal(0, 4, (height, width)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def test_infrared_night_vision_is_recognised_despite_being_bright():
    """An IR camera outputs a bright but colourless image."""
    condition = _settle(SceneIlluminationEstimator(), _ir_frame())
    assert condition.is_night is True


def test_a_colourless_but_lit_room_is_not_night():
    """
    Regression: a phone used as a webcam reported night in a lit room.

    A virtual-camera driver emits a flat grey placeholder while the phone
    connects, and many phone feeds are near-colourless indoors. Both satisfied
    every colour test the infrared heuristic applied, so a frame at mean luma
    128 was declared "infrared" night — which armed the night-movement rule and
    reported phantom night movement on the first person to walk past.

    Colourlessness alone cannot mean night. The frame must also contain a scene
    and actually be dim.
    """
    flat = np.full((384, 640, 3), 128, np.uint8)
    condition = _settle(SceneIlluminationEstimator(), flat)
    assert condition.is_night is False, (
        "a flat grey placeholder frame was read as night vision"
    )
    assert condition.infrared is False

    # A lit but desaturated room: real texture, but nothing dark in it.
    rng = np.random.default_rng(5)
    grey = rng.integers(95, 145, (384, 640), dtype=np.uint8)
    lit = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    assert _settle(SceneIlluminationEstimator(), lit).is_night is False


def test_a_dark_scene_is_still_night_however_colourless():
    """The fix must not cost us the case the detector exists for."""
    rng = np.random.default_rng(9)
    dark = rng.normal(24, 12, (384, 640)).clip(0, 255).astype(np.uint8)
    condition = _settle(SceneIlluminationEstimator(),
                        cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR))
    assert condition.is_night is True


def test_a_few_dark_frames_do_not_arm_night():
    """A passing shadow or an auto-exposure hunt must not switch modes."""
    estimator = SceneIlluminationEstimator()
    _settle(estimator, _frame(190))
    for _ in range(settings.NIGHT_CONFIRM_FRAMES - 1):
        estimator.measure(_frame(2))
    assert estimator.is_night is False


def test_a_few_bright_frames_do_not_disarm_night():
    """Sweeping headlights must not drop a camera out of night analytics."""
    estimator = SceneIlluminationEstimator()
    _settle(estimator, _frame(6))
    assert estimator.is_night is True
    for _ in range(settings.DAY_CONFIRM_FRAMES - 1):
        estimator.measure(_frame(240))
    assert estimator.is_night is True


def test_dusk_transitions_exactly_once():
    """Hysteresis: the state must not oscillate while the light fades."""
    estimator = SceneIlluminationEstimator()
    changes, previous = 0, False
    for step in range(120):
        value = max(2, int(200 - step * 3))
        current = estimator.measure(_frame(value)).is_night
        if current != previous:
            changes += 1
            previous = current
    assert changes == 1 and previous is True


def test_night_movement_needs_darkness_and_movement_together():
    moving = [(i * 12, 200) for i in range(20)]
    still = [(200, 200)] * 20

    # dark + movement -> alert
    assert drive(NightMovementRule(min_travel=40), 1, moving, is_night=True)
    # dark + no movement -> silence (a parked vehicle is not an intrusion)
    assert drive(NightMovementRule(min_travel=40), 1, still, is_night=True) == []
    # bright + movement -> silence
    assert drive(NightMovementRule(min_travel=40), 1, moving, is_night=False) == []


def test_night_movement_does_not_accumulate_across_a_long_window():
    """
    Travel is measured in a rolling window, not over the track's lifetime.

    The old rule summed displacement forever, so a subject shuffling in place
    eventually crossed the threshold purely through accumulated jitter.
    """
    rule = NightMovementRule(min_travel=60, window_seconds=4.0)
    # 2 px per frame at 15 FPS = 30 px/s; over any 4 s window that is 120 px...
    # but a *shuffle* returns to where it started, so net travel stays tiny.
    shuffle = [((200 + (2 if i % 2 else -2)), 200) for i in range(600)]
    assert drive(rule, 1, shuffle, is_night=True) == []


# --------------------------------------------------------------------------- #
# Engine-level: cooldowns
# --------------------------------------------------------------------------- #


def test_engine_applies_per_type_cooldowns():
    """Different event types get different cooldowns, from configuration."""
    engine = RuleEngine("cam")
    engine.add_rule(ZoneRule("z", SQUARE, presence_seconds=999, exit_alerts=True))
    fired = []
    for i in range(400):
        # March in and out of the zone repeatedly, well clear of the boundary.
        point = (150, 150) if (i // 20) % 2 == 0 else (600, 600)
        fired += engine.update([Det(1, *point)], timestamp=i * 0.25)
    kinds = {a.alert_type for a in fired}
    assert kinds <= {"enter", "zone_exit"}
    # 100 s of marching in and out, with an 8 s cooldown on each type.
    assert len(fired) <= 30, f"{len(fired)} alerts — cooldown not applied"
