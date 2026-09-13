"""
Tests for the outbound escalation channels (alarm siren, SMS).

The webhook tests run against a real ``ThreadingHTTPServer`` on localhost
rather than a mocked ``requests`` — the things most likely to be wrong here are
the bytes on the wire and the signature computed over them, and a mock would
assert the code agrees with itself rather than that a controller could verify
the request.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.alarm import AlarmManager
from core.config import settings
from core.notify import gsm_safe, severity_at_least, unwrap_alert
from core.sms import SMS_SINGLE_SEGMENT, SMSNotifier


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_alert(**overrides) -> dict:
    """A sealed-alert payload shaped exactly like ``serialize_alert_row``."""
    alert = {
        "id": 41,
        "camera_id": 1,
        "camera_name": "BOP-NORTH-01",
        "alert_type": "entry",
        "title": "INTRUSION — FENCE CROSSED (INBOUND)",
        "icon": "🚨",
        "severity": "CRITICAL",
        "track_id": 7,
        "timestamp": "2026-09-12T04:30:00+00:00",
        "timestamp_ist": "12 Sep 2026 10:00:00 IST",
        "description": "Track 7 crossed the fence line heading inbound.",
        "details": {"rule_name": "north fence"},
    }
    alert.update(overrides)
    return alert


class _Recorder(BaseHTTPRequestHandler):
    """Alarm controller stand-in: records the request, replies as configured."""

    received: list = []
    reply_status = 200

    def do_POST(self):  # noqa: N802 - stdlib API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).received.append({
            "body": body,
            "headers": {k.lower(): v for k, v in self.headers.items()},
        })
        self.send_response(type(self).reply_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args):  # keep the pytest output readable
        pass


@pytest.fixture
def webhook():
    """A live local alarm controller. Yields ``(url, handler_class)``."""
    _Recorder.received = []
    _Recorder.reply_status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/alarm", _Recorder
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def alarm(monkeypatch, webhook):
    """A fresh, enabled AlarmManager pointed at the local controller."""
    url, _ = webhook
    monkeypatch.setattr(settings, "ALARM_ENABLED", True)
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_URL", url)
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(settings, "ALARM_GPIO_PIN", -1)
    monkeypatch.setattr(settings, "ALARM_COOLDOWN_SECONDS", 60.0)
    channel = AlarmManager()          # not .get(): tests must not share state
    yield channel
    channel.shutdown(timeout=2.0)


def drain(channel, timeout: float = 5.0) -> None:
    """Wait for the channel's worker to finish what it has queued."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = channel.get_status()
        if status["queued"] == 0 and (status["sent"] or status["failed"]):
            return
        time.sleep(0.02)
    raise AssertionError(f"channel did not drain: {channel.get_status()}")


# --------------------------------------------------------------------------- #
# Envelope handling
# --------------------------------------------------------------------------- #


def test_unwraps_the_event_manager_envelope():
    alert = make_alert()
    assert unwrap_alert({"type": "alert", "data": alert}) == alert


@pytest.mark.parametrize("message", [
    {"type": "stats", "data": {"fps": 15}},
    {"type": "checkpoint", "data": {}},
    {"type": "alert", "data": None},
    {"anything": "else"},
    "not a dict",
    None,
])
def test_ignores_everything_that_is_not_an_alert(message):
    """``broadcast()`` shares the subscriber list with ``record()``."""
    assert unwrap_alert(message) is None


def test_accepts_a_bare_alert_for_direct_calls():
    alert = make_alert()
    assert unwrap_alert(alert) == alert


# --------------------------------------------------------------------------- #
# Severity gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("severity,escalates", [
    ("CRITICAL", True),
    ("HIGH", True),
    ("MEDIUM", False),
    ("LOW", False),
    ("INFO", False),
])
def test_only_high_and_critical_escalate(severity, escalates):
    assert severity_at_least(severity, "HIGH") is escalates


def test_unknown_severity_does_not_escalate():
    """A future alert type must not reach a siren by accident."""
    assert severity_at_least("SPICY", "HIGH") is False


def test_severity_floor_is_configurable():
    assert severity_at_least("MEDIUM", "MEDIUM") is True


# --------------------------------------------------------------------------- #
# Alarm webhook
# --------------------------------------------------------------------------- #


def test_webhook_fires_on_a_critical_alert(alarm, webhook):
    url, recorder = webhook
    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)

    assert len(recorder.received) == 1
    payload = json.loads(recorder.received[0]["body"])
    assert payload["action"] == "trigger_alarm"
    assert payload["severity"] == "CRITICAL"
    assert payload["alert_type"] == "entry"
    assert payload["camera_name"] == "BOP-NORTH-01"
    assert payload["track_id"] == 7
    assert payload["timestamp_ist"] == "12 Sep 2026 10:00:00 IST"
    assert payload["details"] == {"rule_name": "north fence"}
    assert payload["test"] is False
    assert alarm.get_status()["sent"] == 1


def test_medium_severity_does_not_reach_the_siren(alarm, webhook):
    _, recorder = webhook
    alarm.trigger({"type": "alert", "data": make_alert(
        alert_type="vehicle_detected", severity="MEDIUM")})
    time.sleep(0.3)
    assert recorder.received == []


def test_stats_broadcast_does_not_reach_the_siren(alarm, webhook):
    _, recorder = webhook
    alarm.trigger({"type": "stats", "data": {"system_fps": 15}})
    time.sleep(0.3)
    assert recorder.received == []


def test_signature_verifies_against_the_exact_body(monkeypatch, alarm, webhook):
    """The HMAC must cover the bytes sent, not a re-serialisation of them."""
    secret = "border-post-shared-secret"
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_SECRET", secret)
    _, recorder = webhook

    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)

    request = recorder.received[0]
    expected = hmac.new(secret.encode(), request["body"], hashlib.sha256).hexdigest()
    assert request["headers"]["x-ibvap-signature"] == expected


def test_unsigned_when_no_secret_is_configured(alarm, webhook):
    _, recorder = webhook
    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)
    assert "x-ibvap-signature" not in recorder.received[0]["headers"]


def test_cooldown_collapses_repeats_from_one_camera(alarm, webhook):
    _, recorder = webhook
    for _ in range(5):
        alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)
    time.sleep(0.2)

    assert len(recorder.received) == 1
    assert alarm.get_status()["suppressed_by_cooldown"] == 4


def test_cooldown_does_not_hide_a_second_camera(alarm, webhook):
    """Post B's intrusion must not be swallowed by Post A's."""
    _, recorder = webhook
    alarm.trigger({"type": "alert", "data": make_alert(camera_id=1)})
    alarm.trigger({"type": "alert", "data": make_alert(camera_id=2)})
    deadline = time.time() + 5.0
    while time.time() < deadline and len(recorder.received) < 2:
        time.sleep(0.02)

    assert len(recorder.received) == 2
    assert alarm.get_status()["suppressed_by_cooldown"] == 0


def test_controller_error_is_recorded_not_raised(alarm, webhook):
    _, recorder = webhook
    recorder.reply_status = 503

    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)

    status = alarm.get_status()
    assert status["sent"] == 0
    assert status["failed"] == 1
    assert "503" in status["last_error"]


def test_unreachable_controller_does_not_raise(monkeypatch, alarm):
    """A dead endpoint degrades the alarm, never the surveillance pipeline."""
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_URL", "http://127.0.0.1:1/none")
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_TIMEOUT", 1.0)

    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)

    assert alarm.get_status()["failed"] == 1


def test_disabled_channel_is_inert(monkeypatch, alarm, webhook):
    _, recorder = webhook
    monkeypatch.setattr(settings, "ALARM_ENABLED", False)
    alarm.trigger({"type": "alert", "data": make_alert()})
    time.sleep(0.3)
    assert recorder.received == []


def test_test_alarm_bypasses_the_cooldown(alarm, webhook):
    """Commissioning must not silently no-op because a real alert just fired."""
    _, recorder = webhook
    alarm.trigger({"type": "alert", "data": make_alert()})
    drain(alarm)

    assert alarm.test_alarm() is True
    assert len(recorder.received) == 2
    assert json.loads(recorder.received[1]["body"])["test"] is True


def test_trigger_returns_without_waiting_for_the_network(monkeypatch, alarm):
    """
    The subscriber runs on a camera analytics thread, so it must not block.

    A 3 s webhook timeout against a black hole would stall that camera for 3 s
    if dispatch were synchronous.
    """
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_URL", "http://10.255.255.1:8081/x")
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_TIMEOUT", 3.0)

    started = time.perf_counter()
    alarm.trigger({"type": "alert", "data": make_alert()})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25, f"trigger() blocked the calling thread for {elapsed:.2f}s"


def test_enabled_but_unconfigured_reports_itself(monkeypatch):
    monkeypatch.setattr(settings, "ALARM_ENABLED", True)
    monkeypatch.setattr(settings, "ALARM_WEBHOOK_URL", "")
    monkeypatch.setattr(settings, "ALARM_GPIO_PIN", -1)

    status = AlarmManager().get_status()
    assert status["enabled"] is True
    assert status["configured"] is False
    assert status["sinks"] == "no sink configured"


# --------------------------------------------------------------------------- #
# SMS
# --------------------------------------------------------------------------- #


def test_message_fits_one_segment():
    message = SMSNotifier.format_message(make_alert())
    assert len(message) <= SMS_SINGLE_SEGMENT


def test_message_is_gsm_safe():
    """One non-GSM character would cut the segment budget from 160 to 70."""
    message = SMSNotifier.format_message(make_alert())
    assert message.isascii(), message
    assert "—" not in message


def test_message_keeps_what_an_officer_needs_to_act():
    message = SMSNotifier.format_message(make_alert())
    assert "CRITICAL" in message
    assert "BOP-NORTH-01" in message
    assert "FENCE CROSSED" in message
    assert "IBVAP" in message


def test_short_alert_keeps_its_description():
    message = SMSNotifier.format_message(make_alert(
        title="INTRUSION", description="Crossed inbound.",
        timestamp_ist="12 Sep 10:00 IST"))
    assert "Crossed inbound." in message


def test_overlong_title_is_truncated_not_dropped():
    message = SMSNotifier.format_message(make_alert(title="X" * 400))
    assert len(message) <= SMS_SINGLE_SEGMENT
    assert message.endswith("...")


@pytest.mark.parametrize("raw,expected", [
    ("9876543210", "+919876543210"),
    ("+91 98765 43210", "+919876543210"),
    ("919876543210", "+919876543210"),
    ("+1 415 555 2671", "+14155552671"),
])
def test_numbers_are_normalised_to_e164(raw, expected):
    assert SMSNotifier.to_e164(raw) == expected


def test_msg91_wants_no_plus():
    assert SMSNotifier.to_msg91_mobile("9876543210") == "919876543210"


def test_multiple_recipients_from_one_setting(monkeypatch):
    monkeypatch.setattr(settings, "SMS_TO_NUMBER", "9876543210, +919000000000")
    assert SMSNotifier.recipients() == ["9876543210", "+919000000000"]


def test_no_recipients_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "SMS_TO_NUMBER", "")
    assert SMSNotifier.recipients() == []


def test_sms_disabled_by_default_and_reports_honestly(monkeypatch):
    monkeypatch.setattr(settings, "SMS_ENABLED", False)
    monkeypatch.setattr(settings, "SMS_TO_NUMBER", "")
    status = SMSNotifier().get_status()
    assert status["enabled"] is False
    assert status["configured"] is False
    assert status["failed"] == 0


def test_sms_enabled_without_credentials_does_not_raise(monkeypatch):
    """A misconfigured gateway must cost the SMS, never the pipeline."""
    monkeypatch.setattr(settings, "SMS_ENABLED", True)
    monkeypatch.setattr(settings, "SMS_TO_NUMBER", "9876543210")
    monkeypatch.setattr(settings, "TWILIO_ACCOUNT_SID", "")
    monkeypatch.setattr(settings, "MSG91_AUTH_KEY", "")

    channel = SMSNotifier()
    try:
        channel.handle({"type": "alert", "data": make_alert()})
        time.sleep(0.3)
        assert channel.get_status()["configured"] is False
    finally:
        channel.shutdown(timeout=2.0)


def test_send_ignores_a_non_alert_payload():
    assert SMSNotifier().send({"type": "stats", "data": {}}) is None


# --------------------------------------------------------------------------- #
# Text folding
# --------------------------------------------------------------------------- #


def test_gsm_safe_folds_dashes_and_strips_emoji():
    assert gsm_safe("INTRUSION — FENCE 🚨") == "INTRUSION - FENCE "


def test_gsm_safe_handles_empty():
    assert gsm_safe("") == ""


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #


def test_channel_survives_a_malformed_event(alarm):
    """The EventManager isolates subscribers; the channel must too."""
    for junk in (None, "alert", 42, {"type": "alert"}, {"type": "alert", "data": []}):
        alarm.trigger(junk)
    assert alarm.get_status()["failed"] == 0


def test_event_manager_publishes_to_the_channel(monkeypatch, alarm, webhook):
    """Wire the real EventManager to the channel the way lifespan() does."""
    from core.events import EventManager

    _, recorder = webhook
    events = EventManager.get()
    events.subscribe(alarm.trigger)
    try:
        events.broadcast({"type": "alert", "data": make_alert(camera_id=99)})
        drain(alarm)
    finally:
        events.unsubscribe(alarm.trigger)

    assert len(recorder.received) == 1
    assert json.loads(recorder.received[0]["body"])["camera_id"] == 99
