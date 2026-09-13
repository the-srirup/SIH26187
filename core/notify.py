"""
Shared plumbing for outbound notification channels (siren, SMS, …).

Why this module exists
----------------------

``EventManager._publish`` calls every subscriber **synchronously, on the thread
that sealed the event** — and that thread is a camera's analytics thread.  A
webhook to an unreachable siren controller, or an SMS gateway on a congested
link, blocks for whole seconds.  Doing that inline would stall analytics for
that camera for the duration: the escalation channel would degrade the
surveillance it exists to support, and it would do so precisely during the
incident that triggered it.

So every channel here is a **bounded queue plus a worker thread**.  The
subscriber callback does three cheap things — unwrap, gate, enqueue — and
returns in microseconds.  All network and GPIO work happens on the channel's
own daemon thread.  The queue is bounded, so an endpoint that hangs forever
costs a fixed amount of memory and a drop counter rather than unbounded growth.

This mirrors the decision already made for the WebSocket layer in
``api.main.ConnectionManager``: publication is decoupled from persistence, and
a slow consumer can never stall a producer.

Extending
---------

To add a provider (an e-mail relay, a radio gateway, an MQTT topic):

1. Subclass :class:`NotificationChannel`.
2. Set ``name`` and implement ``is_enabled``, ``is_configured``,
   ``cooldown_seconds`` and ``_deliver``.
3. Subscribe the instance's ``handle`` method to the ``EventManager``.

``_deliver`` runs on the worker thread, may block, and signals failure by
raising — the base class records the error, counts it, and keeps the worker
alive.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional

from core.config import settings
from core.timeutil import fmt_ist

log = logging.getLogger("ibvap.notify")

__all__ = [
    "NotificationChannel",
    "NotificationError",
    "SEVERITY_RANK",
    "gsm_safe",
    "http_post",
    "json_body",
    "severity_at_least",
    "unwrap_alert",
]


class NotificationError(RuntimeError):
    """A channel could not deliver an alert. Carries an operator-facing reason."""


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #

#: Ordering for the escalation gate.  Kept here rather than derived from
#: ``cv.rules.SEVERITY_BY_TYPE`` because that maps *type -> severity*; what a
#: channel needs is *severity -> how loud*, which is a different question and
#: must stay answerable for a severity string read back from the database.
SEVERITY_RANK: dict[str, int] = {
    "INFO": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def severity_at_least(severity: Optional[str], minimum: str) -> bool:
    """
    True when ``severity`` is at least as loud as ``minimum``.

    An unrecognised severity is treated as MEDIUM — the same fallback
    ``cv.rules.severity_for`` uses — so a future alert type cannot accidentally
    escalate to a siren just because nobody added it to the table yet.
    """
    have = SEVERITY_RANK.get(str(severity or "").strip().upper(), SEVERITY_RANK["MEDIUM"])
    want = SEVERITY_RANK.get(str(minimum or "").strip().upper(), SEVERITY_RANK["HIGH"])
    return have >= want


def unwrap_alert(message: object) -> Optional[dict]:
    """
    Pull the sealed alert out of whatever the ``EventManager`` published.

    ``record()`` publishes ``{"type": "alert", "data": {...}}`` while
    ``broadcast()`` publishes stats, status and checkpoint envelopes on the
    *same* subscriber list — a channel that did not filter would try to SMS a
    once-per-second stats tick.  A bare alert dict is also accepted so a
    channel can be driven directly from a test, a CLI or the REST test hooks.
    """
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if kind is not None:
        if kind != "alert":
            return None
        data = message.get("data")
        return data if isinstance(data, dict) else None
    # No envelope: accept it only if it actually looks like a sealed alert,
    # so a malformed broadcast is dropped rather than half-interpreted.
    if "alert_type" in message and "severity" in message:
        return message
    return None


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

#: Characters this project's own alert titles contain that are *not* in the
#: GSM 03.38 alphabet.  See :func:`gsm_safe` for why that matters.
_GSM_SUBSTITUTIONS = {
    "—": "-",     # em dash: appears in almost every alert title
    "–": "-",     # en dash
    "‑": "-",     # non-breaking hyphen
    "…": "...",   # ellipsis
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "×": "x",
    " ": " ",     # non-breaking space
    "→": "->",
}


def gsm_safe(text: str) -> str:
    """
    Fold a message to characters an SMS can carry in a single 160-char segment.

    This project's titles are full of em dashes (``INTRUSION — FENCE CROSSED
    (INBOUND)``) and its icons are emoji.  A single character outside the GSM
    03.38 alphabet re-encodes the *whole* message as UCS-2, which cuts the
    per-segment budget from 160 characters to 70 — so a 120-character alert
    silently becomes two or three billed segments, or is truncated by the
    gateway mid-sentence.  Folding to ASCII first keeps one alert in one
    segment, which is both cheaper and more reliable on a weak link.
    """
    if not text:
        return ""
    for bad, good in _GSM_SUBSTITUTIONS.items():
        text = text.replace(bad, good)
    # Anything still outside ASCII (emoji, Devanagari) would force UCS-2.
    return text.encode("ascii", "ignore").decode("ascii")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def json_body(payload: dict) -> bytes:
    """Serialise a payload deterministically so an HMAC over it is reproducible."""
    return json.dumps(
        payload, default=str, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def http_post(
    url: str,
    body: bytes,
    *,
    headers: Optional[dict] = None,
    timeout: float = 5.0,
) -> tuple[int, str]:
    """
    POST raw bytes and return ``(status_code, response_text)``.

    Takes bytes rather than an object because the alarm webhook signs exactly
    what it sends: re-serialising a payload to compute an HMAC and then
    serialising it again to transmit is how signature mismatches are born.

    Uses ``requests`` when installed and falls back to ``urllib.request``
    otherwise, so the webhook still works on a minimal install that skipped the
    optional subsystems.  A transport failure raises; a non-2xx response is
    *returned*, so the caller can put the endpoint's own error text into the
    status page instead of a generic "request failed".
    """
    hdrs = {
        "Content-Type": "application/json",
        "User-Agent": f"IBVAP/{settings.VERSION}",
    }
    hdrs.update(headers or {})

    try:
        import requests
    except ImportError:
        requests = None  # type: ignore[assignment]

    if requests is not None:
        response = requests.post(url, data=body, headers=hdrs, timeout=timeout)
        return int(response.status_code), (response.text or "")[:500]

    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # a reply, not a transport failure
        return int(exc.code), exc.read(4096).decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Channel base
# --------------------------------------------------------------------------- #


class NotificationChannel:
    """
    One outbound escalation route: gate, queue, deliver, account.

    Subclasses provide the transport.  Everything that must not be got wrong
    twice — thread safety, the severity gate, the cooldown, the bounded queue,
    the worker that must never die, the status accounting — lives here once.
    """

    #: Short identifier used in logs, thread names and the status payload.
    name = "notification"

    def __init__(self, queue_size: Optional[int] = None) -> None:
        self._lock = threading.RLock()
        size = int(queue_size if queue_size is not None else settings.NOTIFY_QUEUE_SIZE)
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=max(1, size))
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

        #: ``(camera_id, alert_type) -> monotonic time of the last escalation``.
        self._cooldowns: dict[tuple, float] = {}

        self._sent = 0
        self._failed = 0
        self._dropped = 0
        self._suppressed = 0
        self._last_reference = ""
        self._last_sent_ist = ""
        self._last_error = ""
        self._last_error_ist = ""

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    def is_enabled(self) -> bool:
        """Has the operator switched this channel on?"""
        raise NotImplementedError

    def is_configured(self) -> bool:
        """Does the channel have somewhere to actually deliver to?"""
        raise NotImplementedError

    @property
    def cooldown_seconds(self) -> float:
        """Minimum quiet period per (camera, alert type) for this channel."""
        raise NotImplementedError

    def _deliver(self, alert: dict, *, test: bool = False) -> str:
        """
        Transmit one alert. Runs on the worker thread and may block.

        Returns a provider reference (message SID, HTTP status, GPIO pin) for
        the audit trail.  Raises :class:`NotificationError` — or anything else
        — to report failure; the base class records it and stays alive.
        """
        raise NotImplementedError

    def status_extra(self) -> dict:
        """Provider-specific fields merged into :meth:`get_status`."""
        return {}

    # ------------------------------------------------------------------ #
    # Gating
    # ------------------------------------------------------------------ #

    def _cooldown_for(self, alert_type: str) -> float:
        """
        How long this channel stays quiet about one alert type.

        ``ALERT_COOLDOWNS`` is tuned for the *event log*, where 4 s between
        fence crossings is the right granularity for an operator scrolling a
        timeline.  A siren or a billed SMS at that rate is alarm fatigue —
        exactly the failure this cooldown exists to prevent — so the channel's
        own floor wins whenever it is the larger of the two.  A per-type value
        can therefore only ever make a channel *quieter* (``camera_offline`` at
        120 s stays 120 s), never chattier than the operator asked for.
        """
        try:
            per_type = float(settings.ALERT_COOLDOWNS.get(alert_type, 0.0))
        except Exception:  # pragma: no cover - defensive against bad config
            per_type = 0.0
        return max(per_type, float(self.cooldown_seconds))

    def _claim_cooldown(self, alert: dict) -> bool:
        """
        Atomically take the escalation slot for this alert, or refuse it.

        Keyed by **camera as well as type**: suppressing an intrusion at Post B
        because Post A raised one thirty seconds ago would discard exactly the
        event an operator most needs.  Repeat alerts from a single camera are
        still collapsed, which is the flooding this guards against.
        """
        alert_type = str(alert.get("alert_type") or "unknown")
        key = (alert.get("camera_id"), alert_type)
        window = self._cooldown_for(alert_type)
        now = time.monotonic()

        with self._lock:
            last = self._cooldowns.get(key)
            if last is not None and (now - last) < window:
                return False
            self._cooldowns[key] = now
            # Bound the table: a long-running post cycles through many cameras
            # and types, and nothing else ever removes an entry.
            if len(self._cooldowns) > 512:
                self._cooldowns = {
                    k: v for k, v in self._cooldowns.items() if (now - v) < 3600.0
                }
            return True

    # ------------------------------------------------------------------ #
    # Subscriber entry point
    # ------------------------------------------------------------------ #

    def handle(self, message: dict) -> None:
        """
        ``EventManager`` subscriber. Cheap, non-blocking, never raises.

        Subscribe *this* method — never a delivery method.  It returns in
        microseconds so the camera thread that sealed the event goes straight
        back to analytics.
        """
        try:
            if not self.is_enabled():
                return
            alert = unwrap_alert(message)
            if alert is None:
                return
            if not severity_at_least(alert.get("severity"), settings.NOTIFY_MIN_SEVERITY):
                return
            if not self._claim_cooldown(alert):
                with self._lock:
                    self._suppressed += 1
                return
            self._enqueue(alert)
        except Exception as exc:  # pragma: no cover - the pipeline comes first
            # EventManager already isolates subscribers; this is the second
            # belt, because a channel must never be the reason an event is lost.
            log.warning("%s: could not queue event (%s)", self.name, exc)

    def _enqueue(self, alert: dict) -> None:
        self._ensure_worker()
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            # One line per burst, not one per drop: a hung endpoint would
            # otherwise fill the disk with log faster than it drops events.
            if dropped == 1 or dropped % 25 == 0:
                log.warning(
                    "%s queue full — %d event(s) dropped; endpoint is not keeping up",
                    self.name, dropped,
                )

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        Bring the dispatch thread up now, before the first event arrives.

        Called when a channel is armed at startup.  Without it the worker is
        created lazily on the first escalation, so an armed, correctly
        configured channel reports ``worker_alive: false`` on the status page
        until something bad happens — which reads as "broken" exactly when an
        operator is checking that it is not.
        """
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        """Start the dispatch thread on first use, and restart it if it died."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run, name=f"ibvap-{self.name}", daemon=True
            )
            self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.dispatch_now(alert)
            except Exception as exc:  # pragma: no cover - the worker must survive
                log.exception("%s worker error: %s", self.name, exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # Delivery
    # ------------------------------------------------------------------ #

    def dispatch_now(self, alert: dict, *, test: bool = False) -> tuple[bool, str]:
        """
        Deliver one alert **on the calling thread**, recording the outcome.

        Used by the worker, and directly by the REST test hooks — FastAPI runs
        those off the event loop, so blocking there is safe.
        """
        if not self.is_configured():
            detail = f"{self.name} is enabled but not configured"
            self._record_failure(detail)
            return False, detail
        try:
            reference = self._deliver(alert, test=test)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._record_failure(detail)
            log.error("%s delivery failed: %s", self.name, detail)
            return False, detail
        self._record_success(reference)
        return True, reference

    def _record_success(self, reference: str) -> None:
        with self._lock:
            self._sent += 1
            self._last_reference = str(reference or "")
            self._last_sent_ist = fmt_ist()

    def _record_failure(self, detail: str) -> None:
        with self._lock:
            self._failed += 1
            self._last_error = str(detail or "")[:300]
            self._last_error_ist = fmt_ist()

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Operator-facing state: is it on, is it wired up, is it working?"""
        with self._lock:
            status = {
                "channel": self.name,
                "enabled": bool(self.is_enabled()),
                "configured": bool(self.is_configured()),
                "min_severity": str(settings.NOTIFY_MIN_SEVERITY).upper(),
                "cooldown_seconds": float(self.cooldown_seconds),
                "queued": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "worker_alive": bool(self._worker is not None and self._worker.is_alive()),
                "sent": self._sent,
                "failed": self._failed,
                "dropped": self._dropped,
                "suppressed_by_cooldown": self._suppressed,
                "last_reference": self._last_reference,
                "last_sent_ist": self._last_sent_ist,
                "last_error": self._last_error,
                "last_error_ist": self._last_error_ist,
            }
        status.update(self.status_extra())
        return status

    def shutdown(self, timeout: float = 3.0) -> None:
        """Stop the worker. Queued events are abandoned, not retried."""
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
