"""
Alarm / siren escalation.

A sealed CRITICAL or HIGH event is the moment the platform stops being a
recorder and becomes an alarm system.  This module is the bridge to whatever
actually makes noise at the Border Out Post.

Two sinks, either or both
-------------------------

**Webhook** (``ALARM_WEBHOOK_URL``) — an HTTP POST to an external alarm
controller: an IP siren/relay board, a PA controller, a SIP gateway that dials
the guard room, or an HQ dispatch API.  This is the deployable path: it needs
no hardware on the analytics machine and works over the post's existing LAN.
The body is signed with HMAC-SHA256 (``ALARM_WEBHOOK_SECRET``) so the
controller can verify that a trigger really came from IBVAP — an unauthenticated
"sound the siren" endpoint on a shared network is an obvious abuse target.

**GPIO** (``ALARM_GPIO_PIN``) — a relay wired directly to a single-board host
(Raspberry Pi and similar).  The pin is driven to its active level for
``ALARM_GPIO_DURATION_SECONDS`` and then released.  Many relay boards are
active-LOW, so ``ALARM_GPIO_ACTIVE_HIGH=false`` exists to stop a deployment
wiring the siren permanently *on* and only noticing at 2 a.m.

Both sinks are attempted when both are configured, and the trigger counts as
delivered if *any* of them fired — a missing ``RPi.GPIO`` on a plain x86 server
degrades to the webhook rather than failing the alarm.

Threading
---------

:meth:`AlarmManager.trigger` is the ``EventManager`` subscriber.  It only gates
and enqueues; the HTTP request and the multi-second GPIO pulse happen on a
worker thread owned by :class:`core.notify.NotificationChannel`.  That matters:
``_publish`` runs on a camera's analytics thread, and holding it for five
seconds of webhook timeout would stall that camera's surveillance during the
exact incident that raised the alarm.

Extending
---------

Add a sink by writing a ``_fire_<sink>`` method that returns a short reference
string or raises, then call it from :meth:`_deliver`.  Add a whole new channel
(e-mail, radio, MQTT) by subclassing ``NotificationChannel`` in its own module.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Optional

from core.config import settings
from core.notify import NotificationChannel, NotificationError, http_post, json_body
from core.timeutil import fmt_ist, utc_iso

log = logging.getLogger("ibvap.alarm")

__all__ = ["AlarmManager", "get_alarm_manager"]

#: Header carrying the HMAC-SHA256 of the exact request body.
SIGNATURE_HEADER = "X-IBVAP-Signature"
#: Convenience header so a controller can route without parsing the body.
EVENT_HEADER = "X-IBVAP-Event"


class AlarmManager(NotificationChannel):
    """
    Process-wide siren trigger.

    Thread-safe singleton, matching the pattern used by ``EventManager`` and
    ``FaceRecognizer``: one queue, one worker, one set of counters, however
    many camera threads are feeding it.
    """

    name = "alarm"

    _instance: Optional["AlarmManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()
        #: Set once the GPIO library has been probed, so a host without
        #: ``RPi.GPIO`` logs one warning instead of one per alert.
        self._gpio_module = None
        self._gpio_ready = False
        self._gpio_checked = False
        self._gpio_error = ""

    @classmethod
    def get(cls) -> "AlarmManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def is_enabled(self) -> bool:
        return bool(settings.ALARM_ENABLED)

    def is_configured(self) -> bool:
        """True when at least one sink could actually make noise."""
        return bool(self._webhook_url()) or self._gpio_available()

    @property
    def cooldown_seconds(self) -> float:
        return float(settings.ALARM_COOLDOWN_SECONDS)

    @staticmethod
    def _webhook_url() -> str:
        return str(settings.ALARM_WEBHOOK_URL or "").strip()

    @staticmethod
    def _gpio_configured() -> bool:
        return int(settings.ALARM_GPIO_PIN) >= 0

    def describe_sinks(self) -> str:
        """One-line summary for the startup log."""
        sinks = []
        if self._webhook_url():
            sinks.append("webhook")
        if self._gpio_configured():
            sinks.append(
                f"gpio:{int(settings.ALARM_GPIO_PIN)}"
                if self._gpio_available() else "gpio:unavailable"
            )
        return ", ".join(sinks) or "no sink configured"

    # ------------------------------------------------------------------ #
    # Subscriber entry point
    # ------------------------------------------------------------------ #

    def trigger(self, message: dict) -> None:
        """
        ``EventManager`` subscriber — **this is the method to subscribe**.

        Gates on severity and cooldown, then hands the alert to the worker
        thread.  Returns immediately and never raises, so the camera thread
        that sealed the event is not held for the length of a network timeout.
        """
        self.handle(message)

    # ------------------------------------------------------------------ #
    # Delivery
    # ------------------------------------------------------------------ #

    def _deliver(self, alert: dict, *, test: bool = False) -> str:
        """
        Fire every configured sink. Runs on the worker thread and may block.

        Succeeds if *any* sink fired: a webhook controller that is up is a
        working alarm even when the GPIO library is missing, and vice versa.
        Only when nothing fired at all is this a failure worth reporting.
        """
        fired: list[str] = []
        errors: list[str] = []

        if self._gpio_configured():
            try:
                fired.append(self._fire_gpio())
            except Exception as exc:
                errors.append(f"gpio: {exc}")
                log.error("Alarm GPIO pulse failed: %s", exc)

        url = self._webhook_url()
        if url:
            try:
                fired.append(self._fire_webhook(url, alert, test=test))
            except Exception as exc:
                errors.append(f"webhook: {exc}")

        if not fired:
            raise NotificationError(
                "; ".join(errors) or "no alarm sink configured "
                "(set ALARM_WEBHOOK_URL or ALARM_GPIO_PIN)"
            )
        if errors:
            # Partial success is still an alarm, but the operator needs to know
            # one of their sinks is down before the night they rely on it.
            log.warning("Alarm fired via %s but %s", ", ".join(fired), "; ".join(errors))
        return ", ".join(fired)

    # -- webhook -------------------------------------------------------- #

    def build_payload(self, alert: dict, *, test: bool = False) -> dict:
        """
        The JSON an alarm controller receives.

        Deliberately flat and self-describing: a relay board's firmware should
        not have to understand IBVAP's schema to decide whether to sound a
        siren.  ``action`` and ``severity`` alone are enough to act on.
        """
        return {
            "action": "trigger_alarm",
            "source": "IBVAP",
            "version": settings.VERSION,
            "test": bool(test),
            "alert_id": alert.get("id"),
            "alert_type": alert.get("alert_type", ""),
            "severity": alert.get("severity", ""),
            "title": alert.get("title", ""),
            "camera_id": alert.get("camera_id"),
            "camera_name": alert.get("camera_name", ""),
            "timestamp_ist": alert.get("timestamp_ist") or fmt_ist(),
            "timestamp_utc": alert.get("timestamp") or utc_iso(),
            "track_id": alert.get("track_id", 0),
            "description": alert.get("description", ""),
            "details": alert.get("details", {}) or {},
        }

    def _fire_webhook(self, url: str, alert: dict, *, test: bool = False) -> str:
        body = json_body(self.build_payload(alert, test=test))
        headers = {EVENT_HEADER: str(alert.get("alert_type") or "alert")}

        secret = str(settings.ALARM_WEBHOOK_SECRET or "")
        if secret:
            # Sign the exact bytes on the wire, not a re-serialisation of the
            # payload — key order or separators differing by one character
            # would make every signature fail verification.
            headers[SIGNATURE_HEADER] = hmac.new(
                secret.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()

        status, text = http_post(
            url, body, headers=headers, timeout=float(settings.ALARM_WEBHOOK_TIMEOUT)
        )
        if not (200 <= status < 300):
            raise NotificationError(f"HTTP {status} from alarm controller: {text[:160]}")
        log.info(
            "Alarm webhook accepted %s (%s) — HTTP %d",
            alert.get("alert_type", "alert"), alert.get("camera_name", ""), status,
        )
        return f"webhook:{status}"

    # -- GPIO ----------------------------------------------------------- #

    def _gpio_available(self) -> bool:
        """
        True when a pin is configured *and* a GPIO library is importable.

        Probed once.  On an x86 analytics server ``RPi.GPIO`` will never
        appear, and re-importing it per alert would cost an ImportError on the
        alarm path at the worst possible moment.
        """
        if not self._gpio_configured():
            return False
        with self._lock:
            if self._gpio_checked:
                return self._gpio_ready
            self._gpio_checked = True
            try:
                import RPi.GPIO as GPIO

                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                pin = int(settings.ALARM_GPIO_PIN)
                # Initialise to the *inactive* level. Setting up an output pin
                # without an explicit initial state leaves it undefined, which
                # on an active-LOW relay board means the siren sounds the
                # moment the service starts.
                GPIO.setup(pin, GPIO.OUT, initial=self._gpio_level(active=False))
                self._gpio_module = GPIO
                self._gpio_ready = True
                log.info(
                    "Alarm GPIO ready — pin %d (active %s)",
                    pin, "HIGH" if settings.ALARM_GPIO_ACTIVE_HIGH else "LOW",
                )
            except Exception as exc:
                self._gpio_ready = False
                self._gpio_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "Alarm GPIO pin %s configured but unusable (%s) — "
                    "falling back to the webhook sink",
                    settings.ALARM_GPIO_PIN, exc,
                )
            return self._gpio_ready

    @staticmethod
    def _gpio_level(*, active: bool) -> int:
        """Map "siren on/off" to a pin level, honouring active-LOW wiring."""
        high = bool(settings.ALARM_GPIO_ACTIVE_HIGH)
        return int(active == high)

    def _fire_gpio(self) -> str:
        """
        Drive the relay for the configured duration, then release it.

        The ``finally`` is the important part: if the process is killed or the
        sleep is interrupted mid-pulse, the pin must still be returned to its
        inactive level rather than leaving the siren latched on.
        """
        if not self._gpio_available():
            raise NotificationError(self._gpio_error or "GPIO library unavailable")

        gpio = self._gpio_module
        pin = int(settings.ALARM_GPIO_PIN)
        duration = max(0.1, float(settings.ALARM_GPIO_DURATION_SECONDS))

        # One pulse at a time: two overlapping alerts must not have the second
        # one's release cut the first one short.
        with self._lock:
            try:
                gpio.output(pin, self._gpio_level(active=True))
                time.sleep(duration)
            finally:
                try:
                    gpio.output(pin, self._gpio_level(active=False))
                except Exception as exc:  # pragma: no cover - hardware only
                    log.error("Could not release alarm GPIO pin %d: %s", pin, exc)
        log.info("Alarm siren pulsed on GPIO %d for %.1fs", pin, duration)
        return f"gpio:{pin}"

    # ------------------------------------------------------------------ #
    # Operator hooks
    # ------------------------------------------------------------------ #

    def test_alarm(self) -> bool:
        """
        Fire a synthetic alarm so an operator can prove the wiring works.

        Blocks — it is meant to be called from the REST test hook, which runs
        off the event loop.  Bypasses the severity and cooldown gates on
        purpose: a commissioning test that silently did nothing because a real
        alert fired a minute ago would be worse than useless.  Failure detail
        is available from :meth:`get_status` as ``last_error``.
        """
        alert = {
            "id": None,
            "alert_type": "test_alarm",
            "severity": "CRITICAL",
            "title": "IBVAP TEST ALARM",
            "camera_id": None,
            "camera_name": "SYSTEM",
            "timestamp_ist": fmt_ist(),
            "timestamp": utc_iso(),
            "track_id": 0,
            "description": (
                "Manual alarm test triggered from the IBVAP API. "
                "No intrusion has been detected."
            ),
            "details": {"test": True},
        }
        ok, detail = self.dispatch_now(alert, test=True)
        if ok:
            log.info("Alarm test succeeded (%s)", detail)
        else:
            log.warning("Alarm test failed: %s", detail)
        return ok

    def status_extra(self) -> dict:
        return {
            "webhook_configured": bool(self._webhook_url()),
            "webhook_signed": bool(settings.ALARM_WEBHOOK_SECRET),
            "webhook_timeout_seconds": float(settings.ALARM_WEBHOOK_TIMEOUT),
            "gpio_pin": int(settings.ALARM_GPIO_PIN),
            "gpio_configured": self._gpio_configured(),
            "gpio_available": self._gpio_available(),
            "gpio_error": self._gpio_error,
            "gpio_duration_seconds": int(settings.ALARM_GPIO_DURATION_SECONDS),
            "gpio_active_high": bool(settings.ALARM_GPIO_ACTIVE_HIGH),
            "sinks": self.describe_sinks(),
        }


def get_alarm_manager() -> AlarmManager:
    """Convenience accessor, mirroring ``get_face_recognizer`` / ``get_anpr_processor``."""
    return AlarmManager.get()
