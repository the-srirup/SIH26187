"""
Tests for how a looping video file behaves at the seam.

A file source that loops replays identical footage, which raises two distinct
problems. The events recorded on the first pass occur again on every later
pass, so the log fills with copies of itself and the first, real occurrence is
buried. And the restart itself is invisible, so a reader of the log cannot tell
a second pass over the same footage from a second incident.

Both are addressed here: each loop announces itself with a ``source_restarted``
event, and events already sealed on an earlier pass are recognised as repeats.
"""
from __future__ import annotations

import pytest

from core.config import settings


class _FakeSource:
    def __init__(self, is_file=True, loop_files=True):
        self.is_file = is_file
        self.loop_files = loop_files


class _Alert:
    def __init__(self, alert_type="entry", rule_name="north fence"):
        self.alert_type = alert_type
        self.rule_name = rule_name


@pytest.fixture
def processor():
    """A CameraProcessor with only the replay bookkeeping initialised."""
    from core.camera import CameraProcessor
    proc = CameraProcessor.__new__(CameraProcessor)
    proc._lap = 1                      # first pass under way
    proc._replay_seen = set()
    proc._replay_suppressed = 0
    proc.source = _FakeSource()
    return proc


def test_the_first_pass_records_everything(processor):
    for _ in range(3):
        assert processor._should_seal(_Alert(), "person") is True
    assert processor._replay_suppressed == 0


def test_a_second_pass_does_not_duplicate_the_first(processor):
    alerts = [_Alert("entry", "north fence"),
              _Alert("loiter", "gate zone"),
              _Alert("enter", "gate zone")]
    for a in alerts:
        assert processor._should_seal(a, "person") is True

    processor._lap = 2                 # the file has looped
    for a in alerts:
        assert processor._should_seal(a, "person") is False, (
            f"{a.alert_type} was sealed again on the replay"
        )
    assert processor._replay_suppressed == 3


def test_something_genuinely_new_on_a_later_pass_is_still_sealed(processor):
    processor._should_seal(_Alert("entry", "north fence"), "person")
    processor._lap = 2
    assert processor._should_seal(_Alert("entry", "north fence"), "car") is True, (
        "a different object class is a different event and must be recorded"
    )
    assert processor._should_seal(_Alert("loiter", "gate zone"), "person") is True


def test_track_ids_are_not_part_of_the_signature(processor):
    """Track ids are reissued at every loop, so they cannot identify a repeat."""
    from core.camera import CameraProcessor
    first = CameraProcessor._replay_signature(_Alert(), "person")
    second = CameraProcessor._replay_signature(_Alert(), "person")
    assert first == second


def test_a_live_camera_is_never_treated_as_a_replay(processor):
    """Real footage is never the same twice, however long the camera runs."""
    processor.source = _FakeSource(is_file=False, loop_files=False)
    processor._lap = 9
    for _ in range(5):
        assert processor._should_seal(_Alert(), "person") is True
    assert processor._replay_suppressed == 0


def test_a_non_looping_file_is_never_treated_as_a_replay(processor):
    processor.source = _FakeSource(is_file=True, loop_files=False)
    processor._lap = 3
    assert processor._should_seal(_Alert(), "person") is True


def test_suppression_can_be_switched_off(processor, monkeypatch):
    """A looping file standing in for a live feed may want every pass fresh."""
    processor._should_seal(_Alert(), "person")
    processor._lap = 2
    monkeypatch.setattr(settings, "FILE_LOOP_SUPPRESS_REPEATS", False)
    assert processor._should_seal(_Alert(), "person") is True


def test_the_restart_is_announced_as_an_event():
    """The seam has to be visible in the log, not only in the server console."""
    import inspect
    from core.camera import CameraProcessor

    body = inspect.getsource(CameraProcessor._handle_source_restart)
    assert "_emit_system_event" in body and "source_restarted" in body, (
        "a looping file restarts silently; a second pass over the same footage "
        "then reads as a second incident"
    )


def test_the_restart_event_type_is_registered():
    from core.events import icon_for, title_for
    from cv.rules import severity_for

    assert severity_for("source_restarted") == "INFO", (
        "a file looping is not a security event and must not escalate"
    )
    assert title_for("source_restarted") != "SOURCE RESTARTED"   # has a real title
    assert icon_for("source_restarted") != "🔔"                   # has its own icon


def test_the_first_connection_is_not_announced_as_a_replay():
    """Starting a camera is not a restart, and must not emit one."""
    from core.camera import CameraProcessor
    proc = CameraProcessor.__new__(CameraProcessor)
    proc._lap = 0
    proc._last_generation = 0
    proc._replay_seen = set()
    proc._replay_suppressed = 0
    proc.source = _FakeSource()
    proc.name = "LOOP CAM"
    proc.camera_id = 1
    emitted = []
    proc._emit_system_event = lambda t, d: emitted.append(t)
    proc.analyzer = type("A", (), {"reset_tracking": lambda self: None})()

    proc._handle_source_restart(1)
    assert emitted == [], "the initial connection was announced as a restart"

    proc._handle_source_restart(2)
    assert emitted == ["source_restarted"]


def test_the_alert_path_actually_consults_the_replay_guard():
    """Guards the wiring, not just the guard.

    The tests above drive ``_should_seal`` directly, so they pass against code
    where nothing calls it. This checks the path that seals events routes
    through it — which is where the duplicates were being written.
    """
    import inspect
    from core.camera import CameraProcessor

    body = inspect.getsource(CameraProcessor._handle_alerts)
    assert "_should_seal" in body, (
        "_handle_alerts no longer consults the replay guard, so every pass "
        "over a looping file will seal the same events again"
    )
    # It must skip the event, not merely evaluate the guard.
    assert "continue" in body.split("_should_seal")[1][:120]
