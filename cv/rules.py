"""
Rules engine — the *analytics* half of IBVAP.

Everything in this module is **deterministic, rule-based geometry and
timing** applied to the tracker's output.  It is not a learned behaviour
model, and the dashboard labels it accordingly: YOLO produces
``AI DETECTION`` results, this module produces ``RULE-BASED EVENT ANALYSIS``.

Rules implemented
-----------------
``FenceRule``         tripwire — line crossing, with ENTRY / EXIT direction
``ZoneRule``          restricted-area intrusion (enter / exit / sustained)
``LoiterRule``        dwell-time inside a polygon
``DirectionRule``     movement against an allowed direction of travel
``NightMovementRule`` sustained movement while the *scene* is dark

All spatial questions are delegated to :mod:`cv.geometry`, so zone membership
and line crossing have exactly one implementation between them.

The two mechanisms that keep an operator from being flooded
----------------------------------------------------------
**Hysteresis, not per-frame opinions.**  Every rule that has a notion of
"inside" or "which side" uses a dead band around its boundary.  Within the
band a rule holds its previous conclusion instead of forming a new one.  A
foot point resting on a boundary therefore produces one event, not an endless
alternating stream.

**Debouncing.**  :class:`RuleEngine` enforces a cooldown per
``(rule, track, event)``, with per-event-type durations from
``settings.ALERT_COOLDOWNS``.

Geometry uses the **foot point** (bottom-centre of the box — where the subject
contacts the ground) rather than the box centre, which is what makes a fence
agree with what an operator sees on the ground plane.  A rule may select
``"center"`` instead via its ``reference_point`` parameter, for an overhead or
wall-mounted view.

Root causes fixed in this rewrite
---------------------------------
* **Crossing rules tested the infinite line, not the drawn segment.**  A
  tripwire across a gate fired for anyone walking across the far end of the
  frame that happened to lie on the same extended line.  Crossings are now a
  trajectory-versus-segment intersection.
* **Crossing rules demanded N further frames on the new side.**  A crossing is
  an instantaneous geometric event; requiring three more frames of persistence
  meant a fast vehicle — which crosses and leaves the frame — was never
  reported.  Confirmation is now provided by a displacement band, which
  rejects jitter *without* rejecting speed, and works across multi-frame gaps.
* **``ZoneRule`` declared an exit with zero confirmation** while requiring
  three frames to declare an entry.  That asymmetry produced the ``enter`` /
  ``zone_exit`` pairs a fraction of a second apart that filled the event log.
* **``LoiterRule`` discarded its state the instant a foot point fell outside**,
  so one jittery frame at the boundary reset the dwell clock and the threshold
  was effectively unreachable.  Dwell now survives a configurable excursion.
* **``NightMovementRule`` accumulated travel forever and alerted once per
  track for all time.**  Travel is now measured in a rolling window and the
  rule can re-arm.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.config import settings
from cv.geometry import LineGeometry, ZoneGeometry, distance, reference_point

log = logging.getLogger("ibvap.cv.rules")

AlertType = Literal[
    "entry", "exit", "enter", "zone_exit", "loiter",
    "wrong_direction", "zone_presence", "night_movement",
]

#: Severity assigned to each rule outcome. Kept here so the UI, the log and
#: the API all agree on how loud an event is.
SEVERITY_BY_TYPE: dict[str, str] = {
    "entry": "CRITICAL",
    "exit": "HIGH",
    "enter": "HIGH",
    "zone_exit": "LOW",
    "loiter": "HIGH",
    "zone_presence": "MEDIUM",
    "wrong_direction": "HIGH",
    "night_movement": "HIGH",
    "watchlist_match": "CRITICAL",
    "face_detected": "LOW",
    "anpr_detection": "MEDIUM",
    "human_detected": "MEDIUM",
    "vehicle_detected": "MEDIUM",
    "camera_offline": "HIGH",
    "system_error": "HIGH",
}


def severity_for(alert_type: str) -> str:
    return SEVERITY_BY_TYPE.get(alert_type, "MEDIUM")


def cooldown_for(alert_type: str, default: Optional[float] = None) -> float:
    """
    Cooldown for one event type.

    Per-type values matter: a tripwire crossing and a loitering alert describe
    very different things, and forcing both through one global interval either
    floods the operator with crossings or hides a second genuine intrusion.
    """
    fallback = settings.DEBOUNCE_SECONDS if default is None else default
    try:
        return float(settings.ALERT_COOLDOWNS.get(alert_type, fallback))
    except Exception:  # pragma: no cover - defensive against bad config
        return float(fallback)


@dataclass
class Alert:
    """A triggered rule outcome, ready for persistence / WebSocket broadcast."""

    rule_name: str
    rule_type: str
    track_id: int
    alert_type: AlertType
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)
    severity: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = severity_for(self.alert_type)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "track_id": self.track_id,
            "alert_type": self.alert_type,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "description": self.description,
            "details": self.details,
        }


def _resolve_time(value: Optional[float]) -> float:
    """
    Resolve a caller-supplied frame timestamp.

    Uses an explicit ``None`` check rather than a truthiness test: media time
    for an uploaded video legitimately starts at ``0.0``, and treating that as
    "not supplied" silently substituted wall-clock time, which made every
    dwell/elapsed computation nonsense for offline analysis.
    """
    return time.time() if value is None else float(value)


def _class_of(detection: Any) -> str:
    return (getattr(detection, "class_name", "") or "").lower()


def _age_of(detection: Any, fallback: int) -> int:
    age = getattr(detection, "age", None)
    return int(age) if isinstance(age, int) else fallback


class BaseRule(ABC):
    """Abstract base class — every rule implements ``update`` and ``forget``."""

    rule_type: str = "base"

    def __init__(self, name: str, reference: str = "foot") -> None:
        self.name = name
        self.reference = reference or "foot"

    @abstractmethod
    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        """Process one track's reference point for this frame."""

    def forget(self, track_id: int) -> None:
        """Drop per-track state for a track that has gone away."""

    def reset(self) -> None:
        """Drop *all* per-track state (source reconnect, file loop)."""

    def geometry(self) -> list:
        """Coordinates for rendering the rule on the video overlay."""
        return []

    def describe(self) -> dict:
        """Operator-facing summary of how this rule is configured."""
        return {"name": self.name, "type": self.rule_type,
                "reference_point": self.reference}


# --------------------------------------------------------------------------- #
# Crossing rules — tripwire and direction
# --------------------------------------------------------------------------- #


@dataclass
class _Trail:
    """
    Per-track memory for a crossing rule.

    ``committed_point``/``committed_side`` are the heart of the anti-jitter
    design: only an observation at least ``band`` pixels clear of the line is
    allowed to *commit* a side.  Observations inside the band are recorded but
    change nothing, so a subject standing astride the fence with ±3 px of box
    jitter never flips state, while a vehicle that crosses decisively commits
    both sides within two frames.
    """

    last_point: tuple[float, float]
    last_seen: float
    committed_point: Optional[tuple[float, float]] = None
    committed_side: int = 0
    frames: int = 1
    last_fire: dict[str, float] = field(default_factory=dict)


class _CrossingRule(BaseRule):
    """
    Shared machinery for :class:`FenceRule` and :class:`DirectionRule`.

    Both answer the same geometric question — "did this track cross this
    segment, and in which direction?" — and previously each carried its own
    copy of the answer, including the same infinite-line bug.
    """

    def __init__(
        self,
        name: str,
        x1: float, y1: float, x2: float, y2: float,
        *,
        reference: str = "foot",
        band: Optional[float] = None,
        min_displacement: Optional[float] = None,
        rearm_seconds: Optional[float] = None,
        min_track_age: Optional[int] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__(name, reference)
        self.line = LineGeometry(x1, y1, x2, y2)
        self.band = float(
            settings.CROSSING_MIN_DISPLACEMENT * 2.0 if band is None else band
        )
        self.min_displacement = float(
            settings.CROSSING_MIN_DISPLACEMENT
            if min_displacement is None else min_displacement
        )
        self.rearm_seconds = float(
            settings.CROSSING_REARM_SECONDS if rearm_seconds is None else rearm_seconds
        )
        self.min_track_age = int(
            settings.CROSSING_MIN_TRACK_AGE if min_track_age is None else min_track_age
        )
        self.classes = {c.lower() for c in classes} if classes else set()
        self._state: dict[int, _Trail] = {}

    # -- description ---------------------------------------------------- #
    def geometry(self) -> list:
        return self.line.as_list()

    def describe(self) -> dict:
        return {**super().describe(),
                "band_px": self.band,
                "rearm_seconds": self.rearm_seconds,
                "classes": sorted(self.classes) or "all"}

    # -- evaluation ----------------------------------------------------- #
    def _evaluate(self, track_id: int, point: tuple[int, int], now: float,
                  detection: Any = None):
        """
        Advance this track's trail and report a crossing, if one happened.

        Returns ``(CrossingResult | None, trail)``.
        """
        trail = self._state.get(track_id)
        signed = self.line.distance_to(point)
        side = self.line.side(point)
        committed_now = signed >= self.band

        if trail is None:
            trail = _Trail(last_point=point, last_seen=now)
            if committed_now:
                trail.committed_point = point
                trail.committed_side = side
            self._state[track_id] = trail
            return None, trail

        gap = now - trail.last_seen
        trail.frames += 1
        trail.last_point = point
        trail.last_seen = now

        # A track reacquired after a long occlusion may have crossed and come
        # back, or crossed twice. Claiming a single crossing over that gap would
        # be a guess, so the trail restarts instead of inventing an event.
        if gap > settings.CROSSING_MAX_GAP_SECONDS:
            trail.committed_point = point if committed_now else None
            trail.committed_side = side if committed_now else 0
            return None, trail

        if not committed_now:
            # Inside the dead band — record nothing, decide nothing.
            return None, trail

        previous_point = trail.committed_point
        previous_side = trail.committed_side
        trail.committed_point = point
        trail.committed_side = side

        if previous_point is None or previous_side == 0 or previous_side == side:
            return None, trail

        if _age_of(detection, trail.frames) < self.min_track_age:
            return None, trail

        result = self.line.crossing(previous_point, point, self.min_displacement)
        return (result if result.crossed else None), trail

    def _rearmed(self, trail: _Trail, direction: str, now: float) -> bool:
        last = trail.last_fire.get(direction)
        if last is not None and now - last < self.rearm_seconds:
            return False
        trail.last_fire[direction] = now
        return True

    def _class_allowed(self, detection: Any) -> bool:
        if not self.classes:
            return True
        name = _class_of(detection)
        return (not name) or name in self.classes

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()


class FenceRule(_CrossingRule):
    """
    Virtual fence / tripwire — a finite line segment with direction.

    The directed segment ``(x1, y1) -> (x2, y2)`` splits the scene.  A
    reference point travelling from the **right** of that direction to its
    **left** is an ``entry``; the reverse is an ``exit`` (right-hand rule).

    An alert means the object's *trajectory* actually intersected the drawn
    segment.  Being merely near the line, or on the same infinite line but
    past its end, is not a crossing and produces nothing.
    """

    rule_type = "fence"

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        detection = kwargs.get("detection")
        if not self._class_allowed(detection):
            return None

        result, trail = self._evaluate(track_id, foot_point, now, detection)
        if result is None:
            return None
        if not self._rearmed(trail, result.direction, now):
            return None

        inbound = result.direction == "entry"
        return Alert(
            rule_name=self.name,
            rule_type=self.rule_type,
            track_id=track_id,
            alert_type=result.direction,  # type: ignore[arg-type]
            timestamp=now,
            description=(
                f"Crossed virtual fence '{self.name}' "
                f"({'inbound' if inbound else 'outbound'})"
            ),
            details={
                "line": self.geometry(),
                "direction": result.direction,
                "crossing_point": ([round(result.point[0], 1), round(result.point[1], 1)]
                                   if result.point else None),
                "foot_point": [int(foot_point[0]), int(foot_point[1])],
                "displacement_px": round(result.displacement, 1),
                "reference_point": self.reference,
            },
        )


class DirectionRule(_CrossingRule):
    """
    Wrong-direction movement across a directed line.

    ``allowed_direction`` is ``"entry"`` or ``"exit"``; a crossing the other
    way raises ``wrong_direction``.  The direction comes from comparing which
    side the track was committed to *before* the crossing with which side it is
    committed to after — never from a single bounding box.
    """

    rule_type = "direction"

    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float,
                 allowed_direction: str = "entry", **kwargs) -> None:
        super().__init__(name, x1, y1, x2, y2, **kwargs)
        allowed = (allowed_direction or "entry").lower()
        if allowed not in ("entry", "exit"):
            raise ValueError("allowed_direction must be 'entry' or 'exit'")
        self.allowed = allowed

    def describe(self) -> dict:
        return {**super().describe(), "allowed_direction": self.allowed}

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        detection = kwargs.get("detection")
        if not self._class_allowed(detection):
            return None

        result, trail = self._evaluate(track_id, foot_point, now, detection)
        if result is None:
            return None
        if result.direction == self.allowed:
            return None
        if not self._rearmed(trail, f"wrong:{result.direction}", now):
            return None

        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="wrong_direction", timestamp=now,
            description=(
                f"Movement against permitted flow at '{self.name}' "
                f"(allowed: {self.allowed}, observed: {result.direction})"
            ),
            details={
                "line": self.geometry(),
                "allowed": self.allowed,
                "actual": result.direction,
                "crossing_point": ([round(result.point[0], 1), round(result.point[1], 1)]
                                   if result.point else None),
                "displacement_px": round(result.displacement, 1),
                "reference_point": self.reference,
            },
        )


# --------------------------------------------------------------------------- #
# Zone (polygon intrusion + presence dwell)
# --------------------------------------------------------------------------- #


@dataclass
class _ZoneState:
    """
    Per-track zone membership with hysteresis and an exit grace period.

    ``outside_since`` is what makes a missed detection harmless: leaving is a
    *sustained* condition, so one frame in which the foot point lands outside
    (or the detector drops the box entirely) does not end the visit.
    """

    inside: bool = False
    enter_confirm: int = 0
    entered_at: float = 0.0
    outside_since: Optional[float] = None
    presence_alerted: bool = False
    last_seen: float = 0.0


class ZoneRule(BaseRule):
    """
    Restricted-area intrusion.

    Emits ``enter`` once the reference point is convincingly inside for
    ``ANCHOR_CONFIRMATION_FRAMES`` frames, ``zone_exit`` once it has been
    convincingly outside for ``ZONE_EXIT_GRACE_SECONDS``, and
    ``zone_presence`` when a subject stays inside beyond the dwell threshold.

    "Convincingly" is the fix for the flapping that dominated the old event
    log: the polygon boundary is given ``ZONE_BOUNDARY_MARGIN`` pixels of
    thickness, and a point inside that band leaves the current state alone.
    """

    rule_type = "zone"

    def __init__(
        self,
        name: str,
        points: list,
        presence_seconds: Optional[float] = None,
        *,
        reference: str = "foot",
        margin: Optional[float] = None,
        exit_grace: Optional[float] = None,
        enter_frames: Optional[int] = None,
        classes: Optional[list[str]] = None,
        exit_alerts: Optional[bool] = None,
    ) -> None:
        super().__init__(name, reference)
        self.zone = ZoneGeometry(points)
        self.presence_seconds = float(
            presence_seconds if presence_seconds is not None
            else settings.ZONE_PRESENCE_SECONDS
        )
        self.margin = float(
            settings.ZONE_BOUNDARY_MARGIN if margin is None else margin
        )
        self.exit_grace = float(
            settings.ZONE_EXIT_GRACE_SECONDS if exit_grace is None else exit_grace
        )
        self.enter_frames = max(1, int(
            settings.ANCHOR_CONFIRMATION_FRAMES if enter_frames is None else enter_frames
        ))
        self.exit_alerts = bool(
            settings.ZONE_EXIT_ALERTS_ENABLED if exit_alerts is None else exit_alerts
        )
        self.classes = {c.lower() for c in classes} if classes else set()
        self._state: dict[int, _ZoneState] = {}

    # -- description ---------------------------------------------------- #
    @property
    def polygon(self):
        """The shapely polygon — retained for callers that inspect geometry."""
        return self.zone.polygon

    def geometry(self) -> list:
        return self.zone.as_list()

    def describe(self) -> dict:
        return {**super().describe(),
                "presence_seconds": self.presence_seconds,
                "boundary_margin_px": self.margin,
                "exit_grace_seconds": self.exit_grace,
                "classes": sorted(self.classes) or "all"}

    # -- evaluation ----------------------------------------------------- #
    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        detection = kwargs.get("detection")
        if self.classes:
            name = _class_of(detection)
            if name and name not in self.classes:
                return None

        membership = self.zone.membership(foot_point, self.margin)
        state = self._state.setdefault(track_id, _ZoneState())
        state.last_seen = now

        # ``None`` means the point is inside the boundary's dead band: hold the
        # current conclusion rather than forming a new one.
        if membership is None:
            membership = state.inside

        if membership:
            state.outside_since = None
            if not state.inside:
                state.enter_confirm += 1
                if state.enter_confirm < self.enter_frames:
                    return None
                state.inside = True
                state.entered_at = now
                state.presence_alerted = False
                return Alert(
                    rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
                    alert_type="enter", timestamp=now,
                    description=f"Entered restricted zone '{self.name}'",
                    details={
                        "polygon": self.geometry(),
                        "foot_point": [int(foot_point[0]), int(foot_point[1])],
                        "reference_point": self.reference,
                        "confirmation_frames": self.enter_frames,
                    },
                )

            # Already inside — check sustained presence.
            if not state.presence_alerted and state.entered_at:
                dwell = now - state.entered_at
                if dwell >= self.presence_seconds:
                    state.presence_alerted = True
                    return Alert(
                        rule_name=self.name, rule_type=self.rule_type,
                        track_id=track_id, alert_type="zone_presence", timestamp=now,
                        description=(
                            f"Sustained presence in restricted zone '{self.name}' "
                            f"for {dwell:.0f}s"
                        ),
                        details={
                            "polygon": self.geometry(),
                            "dwell_seconds": round(dwell, 1),
                            "threshold_seconds": self.presence_seconds,
                        },
                    )
            return None

        # Outside. Leaving must be sustained for the grace period.
        state.enter_confirm = 0
        if not state.inside:
            state.outside_since = None
            return None

        if state.outside_since is None:
            state.outside_since = now
            return None
        if now - state.outside_since < self.exit_grace:
            return None

        dwell = (state.outside_since - state.entered_at) if state.entered_at else 0.0
        state.inside = False
        state.outside_since = None
        state.presence_alerted = False
        if not self.exit_alerts:
            return None
        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="zone_exit", timestamp=now,
            description=f"Left restricted zone '{self.name}'",
            details={
                "polygon": self.geometry(),
                "dwell_seconds": round(max(0.0, dwell), 1),
                "exit_grace_seconds": self.exit_grace,
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()


# --------------------------------------------------------------------------- #
# Loitering
# --------------------------------------------------------------------------- #


@dataclass
class _LoiterState:
    """
    Per-track loitering state.

    ``entered_at`` is the start of the *visit*, not of the current continuous
    detection.  A subject may drop out of the zone — or out of the detector —
    for up to the grace period without restarting the clock, which is the
    whole reason this rule now works on real footage.
    """

    entered_at: float
    last_inside: float
    frames_inside: int = 1
    outside_since: Optional[float] = None
    alerted_at: Optional[float] = None
    last_seen: float = 0.0


class LoiterRule(BaseRule):
    """
    Loitering — a tracked subject dwelling inside a polygon.

    Dwell is measured from timestamps supplied by the pipeline, so the rule
    behaves identically at 5 FPS and 30 FPS, and identically when an uploaded
    video is analysed faster than real time (media time is passed in that
    case).

    Robustness requirements this rule is built to meet:

    * **multiple people** — state is per track id, so concurrent subjects have
      independent clocks;
    * **temporary detection loss** — a gap shorter than the grace period does
      not reset the visit;
    * **boundary jitter** — the polygon boundary has a margin, so a foot point
      wobbling across the edge does not reset the visit either;
    * **leaving and returning** — an absence longer than the grace period ends
      the visit, and a later return starts a genuinely new one;
    * **one alert per visit** — with optional re-alerting for a subject who
      simply stays, controlled by ``LOITER_REALERT_SECONDS``.
    """

    rule_type = "loiter"

    def __init__(
        self,
        name: str,
        points: list,
        dwell_seconds: Optional[float] = None,
        *,
        reference: str = "foot",
        margin: Optional[float] = None,
        exit_grace: Optional[float] = None,
        realert_seconds: Optional[float] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__(name, reference)
        self.zone = ZoneGeometry(points)
        self.dwell_seconds = float(
            dwell_seconds if dwell_seconds is not None else settings.LOITER_SECONDS
        )
        self.margin = float(
            settings.ZONE_BOUNDARY_MARGIN if margin is None else margin
        )
        self.exit_grace = float(
            settings.LOITER_EXIT_GRACE_SECONDS if exit_grace is None else exit_grace
        )
        self.realert_seconds = float(
            settings.LOITER_REALERT_SECONDS if realert_seconds is None
            else realert_seconds
        )
        if classes is None:
            classes = list(settings.LOITER_CLASSES or [])
        self.classes = {c.lower() for c in classes}
        self._state: dict[int, _LoiterState] = {}
        #: Longest dwell this rule has actually observed, and how many visits it
        #: has seen. Without this the rule is silent in exactly the case an
        #: operator most needs to understand: nobody has yet stayed long enough.
        #: "No alert" and "the threshold is set above anything that happens
        #: here" look identical from the outside, and the second is a
        #: configuration mistake the operator can fix — but only if they can see
        #: it. Reported through the camera's stats, never as a fabricated alert.
        self.longest_dwell_seen = 0.0
        self.visits_seen = 0
        self.alerts_raised = 0

    # -- description ---------------------------------------------------- #
    @property
    def polygon(self):
        return self.zone.polygon

    def geometry(self) -> list:
        return self.zone.as_list()

    def describe(self) -> dict:
        return {**super().describe(),
                "dwell_seconds": self.dwell_seconds,
                "exit_grace_seconds": self.exit_grace,
                "realert_seconds": self.realert_seconds,
                "classes": sorted(self.classes) or "all",
                "longest_dwell_seen": round(self.longest_dwell_seen, 1),
                "visits_seen": self.visits_seen,
                "alerts_raised": self.alerts_raised,
                # The operator-facing explanation of a quiet zone.
                "threshold_reachable": (
                    self.longest_dwell_seen >= self.dwell_seconds
                    if self.visits_seen else None
                )}

    # -- evaluation ----------------------------------------------------- #
    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        detection = kwargs.get("detection")

        # Loitering is a human behaviour; a parked car is not loitering. When
        # the caller supplies no class information (unit tests, a bare
        # trajectory) the rule cannot filter and does not pretend to.
        if self.classes:
            name = _class_of(detection)
            if name and name not in self.classes:
                self._state.pop(track_id, None)
                return None

        state = self._state.get(track_id)
        membership = self.zone.membership(foot_point, self.margin)
        if membership is None:
            # In the boundary dead band: treat as still inside if we were.
            membership = state is not None and state.outside_since is None

        if not membership:
            if state is None:
                return None
            if state.outside_since is None:
                state.outside_since = now
            state.last_seen = now
            if now - state.outside_since >= self.exit_grace:
                # A genuine departure — the visit is over.
                self._state.pop(track_id, None)
            return None

        if state is None:
            # First confirmed frame inside: the start of this visit.
            self._state[track_id] = _LoiterState(
                entered_at=now, last_inside=now, last_seen=now
            )
            self.visits_seen += 1
            return None

        state.outside_since = None
        state.last_inside = now
        state.last_seen = now
        state.frames_inside += 1

        if state.frames_inside < max(1, settings.ANCHOR_CONFIRMATION_FRAMES):
            return None

        dwell = now - state.entered_at
        if dwell > self.longest_dwell_seen:
            self.longest_dwell_seen = dwell
        if dwell < self.dwell_seconds:
            return None

        if state.alerted_at is not None:
            if self.realert_seconds <= 0:
                return None
            if now - state.alerted_at < self.realert_seconds:
                return None

        state.alerted_at = now
        self.alerts_raised += 1
        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="loiter", timestamp=now,
            description=(
                f"Loitering in '{self.name}' for {dwell:.0f}s "
                f"(threshold {self.dwell_seconds:.0f}s)"
            ),
            details={
                "polygon": self.geometry(),
                "dwell_seconds": round(dwell, 1),
                "threshold_seconds": self.dwell_seconds,
                "reference_point": self.reference,
                "object_class": _class_of(detection) or "unknown",
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()


# --------------------------------------------------------------------------- #
# Night movement
# --------------------------------------------------------------------------- #


@dataclass
class _NightState:
    """Rolling travel history for one track, used by the night rule."""

    #: (timestamp, point) samples inside the measurement window.
    trail: deque = field(default_factory=lambda: deque(maxlen=256))
    alerted_at: Optional[float] = None
    last_seen: float = 0.0


class NightMovementRule(BaseRule):
    """
    Sustained movement while the **scene** is dark.

    This rule is deliberately not a clock rule.  It fires only when the
    pipeline reports ``is_night`` — which :mod:`cv.scene` derives from measured
    frame illumination, not from the time of day — *and* the track has actually
    moved.  The three conditions are kept separate on purpose:

        scene darkness  (cv.scene)
              AND
        object movement (this rule)
              AND
        sustained for a window / minimum travel
              =
        NIGHT MOVEMENT event

    Travel is measured over a rolling ``NIGHT_MOVEMENT_WINDOW``.  The previous
    implementation accumulated travel for the lifetime of the track, so a
    subject shuffling in place for several minutes eventually crossed the
    threshold, and — because the alert flag was never cleared — each track
    could report at most one event for all time.
    """

    rule_type = "night"

    def __init__(
        self,
        name: str = "night_movement",
        min_travel: Optional[float] = None,
        *,
        window_seconds: Optional[float] = None,
        debounce_seconds: Optional[float] = None,
        reference: str = "foot",
    ) -> None:
        super().__init__(name, reference)
        self.min_travel = float(
            settings.NIGHT_MOVEMENT_MIN_TRAVEL if min_travel is None else min_travel
        )
        self.window_seconds = float(
            settings.NIGHT_MOVEMENT_WINDOW if window_seconds is None else window_seconds
        )
        self.debounce_seconds = float(
            settings.NIGHT_MOVEMENT_DEBOUNCE if debounce_seconds is None
            else debounce_seconds
        )
        #: Net displacement required, as a fraction of ``min_travel``.
        self.net_ratio = float(settings.NIGHT_MOVEMENT_NET_RATIO)
        #: Speed-based qualifier, for objects that cross the frame faster than
        #: they can be observed for the full distance window.
        self.min_speed = float(settings.NIGHT_MOVEMENT_MIN_SPEED)
        self.min_observation = float(settings.NIGHT_MOVEMENT_MIN_OBSERVATION)
        self.min_net_floor = float(settings.NIGHT_MOVEMENT_MIN_NET_FLOOR)
        self._state: dict[int, _NightState] = {}

    def describe(self) -> dict:
        return {**super().describe(),
                "min_travel_px": self.min_travel,
                "window_seconds": self.window_seconds,
                "debounce_seconds": self.debounce_seconds,
                "min_speed_px_s": self.min_speed,
                "min_observation_seconds": self.min_observation,
                "min_net_floor_px": self.min_net_floor}

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))

        # Scene darkness is supplied by the analytics layer, which measures it.
        if not kwargs.get("is_night", False):
            self._state.pop(track_id, None)
            return None

        state = self._state.get(track_id)
        if state is None:
            state = _NightState()
            self._state[track_id] = state
        state.last_seen = now
        state.trail.append((now, (float(foot_point[0]), float(foot_point[1]))))

        # Drop samples that fell out of the measurement window.
        cutoff = now - self.window_seconds
        while len(state.trail) > 2 and state.trail[0][0] < cutoff:
            state.trail.popleft()
        if len(state.trail) < 2:
            return None

        points = [p for _, p in state.trail]
        travelled = 0.0
        for previous, current in zip(points, points[1:]):
            step = distance(previous, current)
            # A single-frame teleport is an identity re-association, not travel.
            if step <= 200.0:
                travelled += step

        # Path length alone is not enough. Box jitter of a couple of pixels on a
        # stationary subject accumulates ~60 px of *path* per second at 15 FPS
        # while the subject has not gone anywhere — which would report a parked
        # vehicle or a sleeping sentry as night movement. Requiring net
        # displacement as well means the object must actually have travelled
        # across the scene. (Pacing on the spot is loitering, and the loiter
        # rule is what reports it.)
        net = distance(points[0], points[-1])
        observed = state.trail[-1][0] - state.trail[0][0]

        # Two ways to qualify, because one absolute distance threshold is biased
        # by how long the object happens to stay in view.
        #
        # This is the measured cause of "night detection works for people but
        # not vehicles". On this project's night clip a person is tracked for
        # 4.2 s and accumulates 338 px — trivially over the 45 px bar — while a
        # car is detected at higher confidence (0.75) but tracked for only 0.8 s
        # and accumulates 37 px, and is rejected. The car was never the problem:
        # a pedestrian dawdles through frame for seconds, a vehicle crosses it
        # in under one, so a pure distance test quietly encodes "slow, long-lived
        # subject" as its definition of movement.
        #
        # So a *rate* qualifies too: 37 px in 0.8 s is 46 px/s, which is
        # sustained movement by any reading. The rate is computed from net
        # displacement rather than path length precisely so box jitter — which
        # inflates path while going nowhere — cannot satisfy it, and it still
        # requires a real observation window and a floor on net displacement so
        # a one-frame flicker cannot trigger an alert.
        by_distance = travelled >= self.min_travel and net >= self.min_travel * self.net_ratio
        by_speed = (
            observed >= self.min_observation
            and len(points) >= 3
            and net >= self.min_net_floor
            and (net / max(observed, 1e-6)) >= self.min_speed
        )
        if not (by_distance or by_speed):
            return None
        qualifier = "distance" if by_distance else "speed"
        if state.alerted_at is not None and now - state.alerted_at < self.debounce_seconds:
            return None

        state.alerted_at = now
        window = max(1e-6, state.trail[-1][0] - state.trail[0][0])
        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="night_movement", timestamp=now,
            description=(
                f"Movement detected in a dark scene "
                f"({travelled:.0f}px travelled, {net:.0f}px net, in {window:.1f}s"
                f", {net / max(window, 1e-6):.0f}px/s — qualified by {qualifier})"
            ),
            details={
                "travelled_px": round(travelled, 1),
                "net_displacement_px": round(net, 1),
                "speed_px_per_s": round(net / max(window, 1e-6), 1),
                "qualified_by": qualifier,
                "min_travel_px": self.min_travel,
                "window_seconds": round(window, 2),
                "night_source": kwargs.get("night_source", "darkness"),
                "scene_darkness": kwargs.get("scene_darkness"),
                "mean_luma": kwargs.get("mean_luma"),
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class RuleEngine:
    """
    Applies every rule for one camera and enforces the alert debouncer.

    ``update`` is called once per analytics frame with the tracker output.
    State for tracks that stop appearing is garbage-collected on a timer so a
    multi-hour run does not accumulate dictionaries.
    """

    def __init__(self, camera_id: str, debounce_seconds: Optional[float] = None) -> None:
        self.camera_id = str(camera_id)
        #: ``None`` means "use the per-type cooldown table"; an explicit value
        #: overrides it, which is what the tests and tuning experiments want.
        self.debounce_override = (
            None if debounce_seconds is None else float(debounce_seconds)
        )
        self.debounce_seconds = (
            settings.DEBOUNCE_SECONDS if debounce_seconds is None
            else float(debounce_seconds)
        )
        self._rules: dict[str, BaseRule] = {}
        self._debounce: dict[tuple[str, int, str], float] = {}
        self._suppressed: dict[str, int] = {}
        self._seen_tracks: dict[int, float] = {}
        #: Set on first update so the GC clock follows the caller's timeline
        #: (wall clock for live cameras, media time for uploaded video).
        self._last_gc: Optional[float] = None

    # -- rule management ------------------------------------------------ #
    def add_rule(self, rule: BaseRule) -> None:
        if rule.name in self._rules:
            raise ValueError(
                f"Rule '{rule.name}' already exists for camera {self.camera_id}"
            )
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        self._rules.pop(name, None)
        self._debounce = {k: v for k, v in self._debounce.items() if k[0] != name}

    def clear(self) -> None:
        self._rules.clear()
        self._debounce.clear()
        self._seen_tracks.clear()

    def get_rules(self) -> list[BaseRule]:
        return list(self._rules.values())

    def diagnostics(self) -> list[dict]:
        """
        What each armed rule has actually observed.

        Exposed so a zone that has raised no alerts can explain itself. A silent
        rule is ambiguous — correctly quiet, or configured past anything that
        happens in this scene — and only the rule knows which.
        """
        return [rule.describe() for rule in self._rules.values()]

    def reset_state(self) -> None:
        """
        Drop every rule's per-track state without unloading the rules.

        Used when a source reconnects or a looping video file restarts: track
        ids are reissued and objects appear to teleport across the frame, which
        would otherwise manufacture phantom fence crossings at the seam.
        """
        for rule in self._rules.values():
            try:
                rule.reset()
            except Exception:  # pragma: no cover - a rule must not block a reset
                log.warning("Rule '%s' failed to reset", rule.name)
        self._debounce.clear()
        self._seen_tracks.clear()

    @property
    def has_rules(self) -> bool:
        return bool(self._rules)

    def describe(self) -> list[dict]:
        return [rule.describe() for rule in self._rules.values()]

    # -- per-frame evaluation ------------------------------------------- #
    def _cooldown(self, alert_type: str) -> float:
        if self.debounce_override is not None:
            return self.debounce_override
        return cooldown_for(alert_type)

    def update(self, detections: list, timestamp: Optional[float] = None,
               context: Optional[dict] = None) -> list[Alert]:
        """
        Evaluate every rule against every detection for the current frame.

        ``context`` carries frame-level facts the rules may need — scene
        darkness and its supporting measurements.
        """
        if not self._rules:
            return []

        now = _resolve_time(timestamp)
        ctx: dict[str, Any] = dict(context or {})
        triggered: list[Alert] = []

        for det in detections:
            tid = det.track_id
            self._seen_tracks[tid] = now
            # Each rule may want a different reference point (ground contact for
            # a ground-plane fence, box centre for an overhead view), so resolve
            # it per rule from the box rather than trusting one precomputed value.
            bbox = getattr(det, "bbox", None)
            default_point = getattr(det, "foot", None)

            for rule in self._rules.values():
                if bbox is not None and getattr(rule, "reference", "foot") != "foot":
                    point = reference_point(bbox, rule.reference)
                else:
                    point = default_point if default_point is not None else \
                        (reference_point(bbox) if bbox is not None else (0, 0))
                try:
                    alert = rule.update(
                        track_id=tid, foot_point=point, timestamp=now,
                        detection=det, **ctx,
                    )
                except Exception as exc:  # one bad rule must not stop the rest
                    log.exception("Rule '%s' raised: %s", rule.name, exc)
                    continue
                if alert is None:
                    continue

                key = (rule.name, tid, alert.alert_type)
                last = self._debounce.get(key)
                if last is not None and now - last < self._cooldown(alert.alert_type):
                    self._suppressed[alert.alert_type] = (
                        self._suppressed.get(alert.alert_type, 0) + 1
                    )
                    continue
                self._debounce[key] = now
                alert.details.setdefault("camera_id", self.camera_id)
                alert.details.setdefault("object_class", getattr(det, "class_name", ""))
                triggered.append(alert)

        self._gc(now)
        return triggered

    @property
    def suppressed_counts(self) -> dict:
        """How many events the debouncer absorbed, per type — for diagnostics."""
        return dict(self._suppressed)

    def _gc(self, now: float) -> None:
        """Evict per-track rule state and debounce keys for departed tracks."""
        if self._last_gc is None:
            self._last_gc = now
            return
        if now - self._last_gc < 30.0:
            return
        self._last_gc = now
        ttl = settings.TRACK_STATE_TTL

        stale = [tid for tid, seen in self._seen_tracks.items() if now - seen > ttl]
        for tid in stale:
            self._seen_tracks.pop(tid, None)
            for rule in self._rules.values():
                rule.forget(tid)

        if stale:
            dead = set(stale)
            self._debounce = {
                k: v for k, v in self._debounce.items()
                if k[1] not in dead and now - v <= ttl
            }
            log.debug("Camera %s: released state for %d departed tracks",
                      self.camera_id, len(stale))


__all__ = [
    "Alert", "AlertType", "BaseRule", "FenceRule", "ZoneRule", "LoiterRule",
    "DirectionRule", "NightMovementRule", "RuleEngine",
    "severity_for", "cooldown_for", "SEVERITY_BY_TYPE",
]
