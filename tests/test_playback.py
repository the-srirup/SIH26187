"""
Tests for the operator's playback controls (pause / resume / speed).

Pause means two different things on purpose, and the difference is the whole
design:

* a **recording** pauses at the decoder, so the footage waits where the
  operator stopped it instead of racing on behind a frozen picture;
* a **live camera** pauses only at the publish. Capture, analytics and event
  sealing keep running, because freezing a real camera's analysis to look at a
  frame would blind the post at exactly the wrong moment.

Speed only applies where the source is paced exactly — a recording. A webcam
emits in real time by definition and cannot be asked for 2x.
"""
from __future__ import annotations

import pytest

from core.config import settings
from core.video_source import (
    PLAYBACK_MAX_SPEED, PLAYBACK_MIN_SPEED, LiveSource,
)


@pytest.fixture
def file_source():
    src = LiveSource.__new__(LiveSource)
    LiveSource.__init__(src, "clip.mp4")
    src._is_file = True
    src._is_local_device = False
    src._frame_interval = 1.0 / 12.0          # a 12 fps clip
    src._paced_rate = 12.0
    return src


@pytest.fixture
def webcam_source():
    src = LiveSource.__new__(LiveSource)
    LiveSource.__init__(src, "0")
    src._is_file = False
    src._is_local_device = True
    src._frame_interval = 0.0
    src._paced_rate = 30.0
    return src


def test_a_recording_honours_speed(file_source):
    assert file_source.playback_applies is True
    base = file_source._effective_interval()

    file_source.set_playback(speed=2.0)
    assert file_source._effective_interval() == pytest.approx(base / 2.0)

    file_source.set_playback(speed=0.5)
    assert file_source._effective_interval() == pytest.approx(base * 2.0)


def test_a_live_camera_reports_speed_as_unavailable(webcam_source):
    """Better to say so than to accept a setting that does nothing."""
    assert webcam_source.playback_applies is False
    state = webcam_source.set_playback(speed=2.0)
    assert state["speed_supported"] is False
    # The interval is untouched: a device emits at its own rate regardless.
    assert webcam_source._effective_interval() == webcam_source._frame_interval


@pytest.mark.parametrize("requested,expected", [
    (10.0, PLAYBACK_MAX_SPEED),
    (0.01, PLAYBACK_MIN_SPEED),
    (-4.0, PLAYBACK_MIN_SPEED),
])
def test_speed_is_clamped_to_the_supported_range(file_source, requested, expected):
    assert file_source.set_playback(speed=requested)["speed"] == pytest.approx(expected)


def test_pause_and_resume_round_trip(file_source):
    assert file_source.set_playback(paused=True)["paused"] is True
    assert file_source._paused is True
    assert file_source.set_playback(paused=False)["paused"] is False
    assert file_source._paused is False


def test_resuming_does_not_replay_the_pause_as_a_burst(file_source):
    """The schedule restarts from now, rather than repaying the whole gap."""
    import time
    file_source.set_playback(paused=True)
    file_source._next_frame_due = time.time() - 30.0      # a long pause
    file_source.set_playback(paused=False)
    assert file_source._next_frame_due == pytest.approx(time.time(), abs=1.0)


def test_speed_and_pause_are_independent(file_source):
    file_source.set_playback(speed=1.5)
    state = file_source.set_playback(paused=True)
    assert state["paused"] is True
    assert state["speed"] == pytest.approx(1.5), "pausing reset the review speed"


def test_effective_fps_is_reported_for_the_operator(file_source):
    state = file_source.set_playback(speed=2.0)
    assert state["effective_fps"] == pytest.approx(24.0)


def test_a_paused_camera_is_not_an_offline_camera():
    """
    Pausing must not seal a camera_offline event.

    A paused recording stops decoding on purpose, so the analytics read times
    out — which, before this was handled, announced the camera OFFLINE and
    wrote a camera_offline alert into the hash-chained log every time an
    operator pressed pause.
    """
    import inspect
    from core.camera import CameraProcessor

    body = inspect.getsource(CameraProcessor._handle_no_frame)
    assert "self._paused" in body, (
        "_handle_no_frame no longer recognises a paused source, so pausing "
        "will be reported as the camera going offline"
    )


def test_a_paused_live_camera_keeps_analysing():
    """Only the publish is withheld — surveillance must not stop."""
    import inspect
    from core.camera import CameraProcessor

    body = inspect.getsource(CameraProcessor._run)
    assert "if not self._paused:" in body and "_publish" in body
    # The alert handling must sit outside the pause gate.
    gated = body.split("if not self._paused:")[1]
    assert "_handle_alerts" in gated
    for line in gated.splitlines():
        if "_handle_alerts" in line:
            indent = len(line) - len(line.lstrip())
            assert indent <= 12, (
                "alert handling was moved inside the pause gate — a paused "
                "live camera would stop reporting intrusions"
            )
            break
