"""
SMS escalation.

A siren wakes the post; an SMS reaches the section commander who is not at the
post.  This module turns a sealed CRITICAL or HIGH event into a single-segment
text message and hands it to a gateway.

Providers
---------

**Twilio** (``SMS_PROVIDER=twilio``) — the international default.  Needs
``pip install twilio``; the import is optional, and a host without it reports
the provider as unavailable instead of failing to start.

**MSG91** (``SMS_PROVIDER=msg91``) — an Indian gateway, reached over plain HTTP
with no extra dependency.  Uses the v5 *flow* API, because Indian TRAI/DLT
rules require transactional SMS to be sent against a registered template
rather than as free text.  The alert is passed into the template as named
variables (``MESSAGE``, ``SEVERITY``, ``CAMERA``, ``TIME``), so a registered
template such as ``IBVAP ALERT: ##SEVERITY## at ##CAMERA## ##TIME##`` works
without code changes.

Whichever provider is primary, the other is tried as a fallback when it is also
configured — a border post's connectivity is exactly the thing that fails
during an incident, and two routes out are worth the twenty lines.

Message shape
-------------

One SMS, one segment.  See :func:`core.notify.gsm_safe` for why a single em
dash in an alert title would otherwise triple the cost and risk truncation.

Threading
---------

:meth:`SMSNotifier.handle` is the ``EventManager`` subscriber; it gates and
enqueues only.  The provider call runs on a worker thread, because
``_publish`` executes on a camera's analytics thread and an SMS gateway on a
congested link can take seconds to answer.

:meth:`SMSNotifier.send` is the *synchronous* sender and returns the provider
message id.  Do not subscribe it to the EventManager — it blocks.

Extending
---------

Add ``_send_via_<name>`` returning a list of provider message ids (or raising),
and list ``<name>`` in :meth:`_provider_order` and :meth:`_provider_ready`.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from core.config import settings
from core.notify import NotificationChannel, NotificationError, gsm_safe, http_post, json_body
from core.timeutil import fmt_ist, utc_iso

log = logging.getLogger("ibvap.sms")

__all__ = ["SMSNotifier", "get_sms_notifier", "TWILIO_AVAILABLE"]

try:  # optional dependency — absence disables one provider, not the platform
    from twilio.rest import Client as _TwilioClient

    TWILIO_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the install
    _TwilioClient = None  # type: ignore[assignment]
    TWILIO_AVAILABLE = False

#: A GSM-7 message is billed and delivered in 160-character segments.
SMS_SINGLE_SEGMENT = 160

#: MSG91 v5 transactional flow endpoint.
MSG91_FLOW_URL = "https://control.msg91.com/api/v5/flow/"

_DIGITS_RE = re.compile(r"\D+")


class SMSNotifier(NotificationChannel):
    """
    Process-wide SMS escalation.

    Thread-safe singleton, matching ``FaceRecognizer`` / ``ANPRProcessor``:
    one queue, one worker, one cached provider client, however many camera
    threads are feeding it.
    """

    name = "sms"

    _instance: Optional["SMSNotifier"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()
        #: Twilio's client opens a connection pool; build it once, and rebuild
        #: only if the credentials in settings actually change.
        self._twilio_client = None
        self._twilio_client_sid = ""

    @classmethod
    def get(cls) -> "SMSNotifier":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def is_enabled(self) -> bool:
        return bool(settings.SMS_ENABLED)

    def is_configured(self) -> bool:
        """True when at least one provider has everything it needs."""
        return bool(self.recipients()) and bool(self._provider_order())

    @property
    def cooldown_seconds(self) -> float:
        return float(settings.SMS_COOLDOWN_SECONDS)

    @staticmethod
    def recipients() -> list[str]:
        """
        Destination numbers.

        ``SMS_TO_NUMBER`` accepts a comma-separated list as well as a single
        number: a post escalates to the duty officer *and* the section
        commander, and making that need a second setting would have meant
        deployments quietly notifying only one of them.
        """
        raw = str(settings.SMS_TO_NUMBER or "")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _provider_ready(self, provider: str) -> bool:
        """Does this provider have credentials *and* a usable client library?"""
        if provider == "twilio":
            return bool(
                TWILIO_AVAILABLE
                and settings.TWILIO_ACCOUNT_SID
                and settings.TWILIO_AUTH_TOKEN
                and settings.TWILIO_FROM_NUMBER
            )
        if provider == "msg91":
            return bool(settings.MSG91_AUTH_KEY and settings.MSG91_TEMPLATE_ID)
        return False

    def _provider_order(self) -> list[str]:
        """
        Configured providers, primary first, fallback second.

        An unconfigured provider is simply absent from the list, so "try the
        fallback" and "there is no fallback" are the same code path.
        """
        primary = str(settings.SMS_PROVIDER or "twilio").strip().lower()
        candidates = [primary] + [p for p in ("twilio", "msg91") if p != primary]
        return [p for p in candidates if self._provider_ready(p)]

    # ------------------------------------------------------------------ #
    # Message formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_message(alert: dict) -> str:
        """
        Render one alert as a single-segment SMS.

        The long form carries the free-text description.  When that pushes the
        message past one segment — which the longer titles always do — the
        description is the first thing dropped: the title already names the
        event, and what an officer needs to act on is *how bad, what, where,
        when*.  Only if even that does not fit is the text hard-truncated.
        """
        severity = str(alert.get("severity") or "ALERT").upper()
        title = str(alert.get("title") or alert.get("alert_type") or "EVENT")
        camera = str(
            alert.get("camera_name")
            or (f"CAM-{alert.get('camera_id')}" if alert.get("camera_id") is not None
                else "SYSTEM")
        )
        stamp = str(alert.get("timestamp_ist") or fmt_ist())
        description = str(alert.get("description") or "")
        track = alert.get("track_id") or 0

        full = gsm_safe(
            f"[IBVAP ALERT] {severity}: {title}\n"
            f"Camera: {camera} | Time: {stamp}\n"
            f"{description}\n"
            f"Track ID: {track}"
        )
        if len(full) <= SMS_SINGLE_SEGMENT:
            return full

        compact = gsm_safe(
            f"[IBVAP] {severity}: {title} | {camera} | {stamp} | Trk {track}"
        )
        if len(compact) <= SMS_SINGLE_SEGMENT:
            return compact
        return compact[:SMS_SINGLE_SEGMENT - 3].rstrip() + "..."

    # ------------------------------------------------------------------ #
    # Number normalisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_cc() -> str:
        return _DIGITS_RE.sub("", str(settings.SMS_DEFAULT_COUNTRY_CODE or "91")) or "91"

    @classmethod
    def to_e164(cls, number: str) -> str:
        """
        Twilio requires ``+<country><number>``.

        A bare 10-digit Indian mobile is what an operator naturally types into
        ``.env``, and sending that unqualified is rejected by the gateway, so
        the configured country code is applied when one is clearly missing.
        """
        raw = str(number or "").strip()
        if raw.startswith("+"):
            return "+" + _DIGITS_RE.sub("", raw)
        digits = _DIGITS_RE.sub("", raw)
        cc = cls._default_cc()
        if len(digits) <= 10:
            return f"+{cc}{digits}"
        return f"+{digits}"

    @classmethod
    def to_msg91_mobile(cls, number: str) -> str:
        """MSG91 wants country code and number with no ``+`` and no spaces."""
        return cls.to_e164(number).lstrip("+")

    # ------------------------------------------------------------------ #
    # Delivery
    # ------------------------------------------------------------------ #

    def _deliver(self, alert: dict, *, test: bool = False) -> str:
        """
        Send via the primary provider, falling back to the secondary.

        Runs on the worker thread and may block.  Raises when every configured
        provider has failed; the base class records the reason.
        """
        recipients = self.recipients()
        if not recipients:
            raise NotificationError("no recipient configured (set SMS_TO_NUMBER)")

        providers = self._provider_order()
        if not providers:
            raise NotificationError(
                "no usable SMS provider — check SMS_PROVIDER and its credentials"
                + ("" if TWILIO_AVAILABLE else " (the twilio package is not installed)")
            )

        message = self.format_message(alert)
        errors: list[str] = []

        for provider in providers:
            try:
                sender = getattr(self, f"_send_via_{provider}")
                references = sender(message, recipients, alert)
                if not references:
                    raise NotificationError("provider accepted nothing")
                log.info(
                    "SMS sent via %s to %d recipient(s): %s",
                    provider, len(references), alert.get("alert_type", "alert"),
                )
                return f"{provider}:{','.join(str(r) for r in references)}"
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                log.warning("SMS via %s failed (%s)", provider, exc)

        raise NotificationError("; ".join(errors))

    # -- Twilio --------------------------------------------------------- #

    def _twilio(self):
        """Cached Twilio client, rebuilt only when the credentials change."""
        sid = str(settings.TWILIO_ACCOUNT_SID or "")
        with self._lock:
            if self._twilio_client is not None and self._twilio_client_sid == sid:
                return self._twilio_client
            if not TWILIO_AVAILABLE:
                raise NotificationError("the twilio package is not installed")
            self._twilio_client = _TwilioClient(sid, str(settings.TWILIO_AUTH_TOKEN or ""))
            self._twilio_client_sid = sid
            return self._twilio_client

    def _send_via_twilio(self, message: str, recipients: list[str], alert: dict) -> list[str]:
        client = self._twilio()
        from_number = str(settings.TWILIO_FROM_NUMBER or "").strip()
        sent: list[str] = []
        errors: list[str] = []

        for number in recipients:
            try:
                result = client.messages.create(
                    body=message, from_=from_number, to=self.to_e164(number)
                )
                sent.append(str(getattr(result, "sid", "") or "sent"))
            except Exception as exc:
                # One bad number in the list must not cost the other officers
                # their alert.
                errors.append(f"{number}: {exc}")

        if not sent:
            raise NotificationError("; ".join(errors) or "no message accepted")
        if errors:
            log.warning("Twilio partially delivered — %s", "; ".join(errors))
        return sent

    # -- MSG91 ---------------------------------------------------------- #

    def _send_via_msg91(self, message: str, recipients: list[str], alert: dict) -> list[str]:
        """
        Send through MSG91's v5 flow API.

        The alert text rides in a configurable template variable
        (``MSG91_MESSAGE_VAR``) alongside structured ``SEVERITY`` / ``CAMERA``
        / ``TIME`` variables, so a DLT-registered template can use whichever
        shape it was approved with.
        """
        variables = {
            str(settings.MSG91_MESSAGE_VAR or "MESSAGE"): message,
            "SEVERITY": str(alert.get("severity") or ""),
            "CAMERA": str(alert.get("camera_name") or ""),
            "TIME": str(alert.get("timestamp_ist") or fmt_ist()),
            "EVENT": str(alert.get("title") or alert.get("alert_type") or ""),
        }
        payload = {
            "template_id": str(settings.MSG91_TEMPLATE_ID or ""),
            "short_url": "0",
            "recipients": [
                {"mobiles": self.to_msg91_mobile(number), **variables}
                for number in recipients
            ],
        }
        sender_id = str(settings.MSG91_SENDER_ID or "").strip()
        if sender_id:
            payload["sender"] = sender_id

        status, text = http_post(
            MSG91_FLOW_URL,
            json_body(payload),
            headers={
                "authkey": str(settings.MSG91_AUTH_KEY or ""),
                "accept": "application/json",
            },
            timeout=float(settings.SMS_TIMEOUT),
        )
        if not (200 <= status < 300):
            raise NotificationError(f"HTTP {status}: {text[:160]}")
        # MSG91 answers 200 with {"type": "error", "message": "..."} for a bad
        # template or auth key, so the status code alone does not mean sent.
        lowered = text.lower()
        if '"type":"error"' in lowered.replace(" ", "") or '"status":"error"' in lowered.replace(" ", ""):
            raise NotificationError(f"MSG91 rejected the request: {text[:160]}")
        return [f"msg91-{status}"]

    # ------------------------------------------------------------------ #
    # Operator hooks
    # ------------------------------------------------------------------ #

    def send(self, alert_payload: dict) -> Optional[str]:
        """
        Send one alert **synchronously** and return the provider reference.

        Blocking by design — this is for the REST test hook, a CLI or a test,
        all of which want to know the outcome.  Do **not** subscribe this to
        the ``EventManager``: subscribe :meth:`handle`, which queues instead of
        holding a camera thread for the length of a gateway round-trip.

        Returns the provider message id on success, ``None`` on failure; the
        reason is available from :meth:`get_status` as ``last_error``.
        """
        from core.notify import unwrap_alert

        alert = unwrap_alert(alert_payload)
        if alert is None:
            log.debug("SMS send(): payload was not an alert, ignoring")
            return None
        ok, detail = self.dispatch_now(alert)
        return detail if ok else None

    def test_sms(self) -> bool:
        """
        Send a synthetic message so an operator can prove the route works.

        Bypasses the severity and cooldown gates on purpose: a commissioning
        test that silently did nothing because a real alert fired two minutes
        ago would be worse than useless.
        """
        alert = {
            "id": None,
            "alert_type": "test_sms",
            "severity": "CRITICAL",
            "title": "IBVAP TEST ALERT",
            "camera_id": None,
            "camera_name": "SYSTEM",
            "timestamp_ist": fmt_ist(),
            "timestamp": utc_iso(),
            "track_id": 0,
            "description": "Manual SMS test from the IBVAP API. No intrusion detected.",
            "details": {"test": True},
        }
        ok, detail = self.dispatch_now(alert, test=True)
        if ok:
            log.info("SMS test succeeded (%s)", detail)
        else:
            log.warning("SMS test failed: %s", detail)
        return ok

    def status_extra(self) -> dict:
        configured = self._provider_order()
        return {
            "provider": str(settings.SMS_PROVIDER or "twilio").lower(),
            "providers_ready": configured,
            "fallback_provider": configured[1] if len(configured) > 1 else None,
            "twilio_installed": TWILIO_AVAILABLE,
            "recipients": len(self.recipients()),
            "timeout_seconds": float(settings.SMS_TIMEOUT),
            "max_segment_chars": SMS_SINGLE_SEGMENT,
        }


def get_sms_notifier() -> SMSNotifier:
    """Convenience accessor, mirroring ``get_face_recognizer`` / ``get_anpr_processor``."""
    return SMSNotifier.get()
