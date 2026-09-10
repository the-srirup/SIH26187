"""
Rules engine — the *analytics* half of IBVAP.

Everything in this module is **deterministic, rule-based geometry and
timing** applied to the tracker's output.  It is not a learned behaviour
model, and the dashboard labels it accordingly: YOLO produces
``AI DETECTION`` results, this module produces ``RULE-BASED EVENT ANALYSIS``.

Rules implemented
-----------------
``FenceRule``       virtual fence line crossing, with ENTRY / EXIT direction
``ZoneRule``        polygon intrusion (enter / exit)
``LoiterRule``      dwell-time inside a polygon
``DirectionRule``   movement against an allowed direction of travel
``NightMovementRule`` sustained movement during configured night hours

Two mechanisms keep the operator from being flooded:

**Anchor confirmation** — a rule only fires once the foot point has held the
new side/zone for ``ANCHOR_CONFIRMATION_FRAMES`` consecutive analytics frames.
A swaying branch or a shadow oscillating across the line never reaches the
threshold.

**Debouncing** — :class:`RuleEngine` enforces a cooldown per
``(rule, track, event)``, so one person walking along a fence produces one
intrusion event, not forty.

Geometry uses the **foot point** (bottom-centre of the box, i.e. where the
subject contacts the ground) rather than the box centre, which is what makes
the fence agree with what an operator sees on the ground plane.

Bugs fixed relative to the previous implementation
--------------------------------------------------
* ``ZoneRule`` wrote its ``was-inside`` state *before* comparing against it,
  so the transition test was always false and the rule could never fire.
* ``LoiterRule`` never recorded an entry timestamp, so elapsed dwell was
  always ~0 and the rule could never fire.
* ``DirectionRule`` never cleared its crossing state, so it re-fired on every
  subsequent frame and relied entirely on the debouncer to hide it.
* Per-track dictionaries grew without bound; state is now garbage-collected.
"""
from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from core.config import settings

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


class BaseRule(ABC):
    """Abstract base class — every rule implements ``update`` and ``forget``."""

    rule_type: str = "base"

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        """Process one track's foot point for this frame."""

    def forget(self, track_id: int) -> None:
        """Drop per-track state for a track that has gone away."""

    def geometry(self) -> list:
        """Coordinates for rendering the rule on the video overlay."""
        return []


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _resolve_time(value: Optional[float]) -> float:
    """
    Resolve a caller-supplied frame timestamp.

    Uses an explicit ``None`` check rather than a truthiness test: media time
    for an uploaded video legitimately starts at ``0.0``, and treating that as
    "not supplied" silently substituted wall-clock time, which made every
    dwell/elapsed computation nonsense for offline analysis.
    """
    return time.time() if value is None else float(value)


def _side_of_line(p1: np.ndarray, p2: np.ndarray, point: tuple[float, float]) -> int:
    """Sign of the 2-D cross product: +1 left of the directed line, -1 right."""
    cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
    if cross > 1e-9:
        return 1
    if cross < -1e-9:
        return -1
    return 0


def _make_polygon(points: list) -> Polygon:
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# --------------------------------------------------------------------------- #
# Fence (line crossing)
# --------------------------------------------------------------------------- #


@dataclass
class _CrossState:
    side: int = 0
    pending: Optional[str] = None   # "entry" | "exit" awaiting confirmation
    confirm: int = 0
    last_seen: float = 0.0


class FenceRule(BaseRule):
    """
    Virtual fence — line crossing with direction.

    The directed line ``(x1, y1) -> (x2, y2)`` splits the frame in two.  A
    foot point moving from the right side to the left side is an **ENTRY**;
    the reverse is an **EXIT** (right-hand-rule convention).  The crossing is
    only reported once the new side has been held for
    ``ANCHOR_CONFIRMATION_FRAMES`` consecutive frames.
    """

    rule_type = "fence"

    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float) -> None:
        super().__init__(name)
        self.line = LineString([(x1, y1), (x2, y2)])
        self.p1 = np.array([x1, y1], dtype=float)
        self.p2 = np.array([x2, y2], dtype=float)
        self._state: dict[int, _CrossState] = {}

    def geometry(self) -> list:
        return [[float(self.p1[0]), float(self.p1[1])],
                [float(self.p2[0]), float(self.p2[1])]]

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        side = _side_of_line(self.p1, self.p2, foot_point)
        state = self._state.setdefault(track_id, _CrossState())
        state.last_seen = now

        if side == 0:
            # Exactly on the line — hold state, do not count a confirmation.
            return None

        prev_side = state.side
        state.side = side

        if prev_side != 0 and prev_side != side:
            # A crossing just happened; start confirming the new side.
            state.pending = "entry" if (prev_side == -1 and side == 1) else "exit"
            state.confirm = 1
            return None

        if state.pending is None:
            return None

        state.confirm += 1
        if state.confirm < max(1, settings.ANCHOR_CONFIRMATION_FRAMES):
            return None

        direction = state.pending
        state.pending = None
        state.confirm = 0
        return Alert(
            rule_name=self.name,
            rule_type=self.rule_type,
            track_id=track_id,
            alert_type=direction,  # type: ignore[arg-type]
            timestamp=now,
            description=(
                f"Crossed virtual fence '{self.name}' "
                f"({'inbound' if direction == 'entry' else 'outbound'})"
            ),
            details={
                "line": self.geometry(),
                "direction": direction,
                "foot_point": [int(foot_point[0]), int(foot_point[1])],
                "confirmation_frames": settings.ANCHOR_CONFIRMATION_FRAMES,
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Zone (polygon intrusion + presence dwell)
# --------------------------------------------------------------------------- #


@dataclass
class _ZoneState:
    inside: bool = False
    confirm: int = 0
    entered_at: float = 0.0
    presence_alerted: bool = False
    last_seen: float = 0.0


class ZoneRule(BaseRule):
    """
    Restricted-zone intrusion.

    Emits ``enter`` once the foot point has been inside for
    ``ANCHOR_CONFIRMATION_FRAMES`` frames, ``zone_exit`` when it leaves, and
    ``zone_presence`` when an object stays inside longer than
    ``ZONE_PRESENCE_SECONDS``.
    """

    rule_type = "zone"

    def __init__(self, name: str, points: list, presence_seconds: Optional[float] = None) -> None:
        super().__init__(name)
        self.polygon = _make_polygon(points)
        self._prepared = prep(self.polygon)   # ~3x faster repeated contains()
        self.presence_seconds = (
            presence_seconds if presence_seconds is not None else settings.ZONE_PRESENCE_SECONDS
        )
        self._state: dict[int, _ZoneState] = {}

    def geometry(self) -> list:
        return [[float(x), float(y)] for x, y in self.polygon.exterior.coords]

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        inside = self._prepared.contains(Point(foot_point))
        state = self._state.setdefault(track_id, _ZoneState())
        state.last_seen = now
        was_inside = state.inside

        if not inside:
            if was_inside:
                state.inside = False
                state.confirm = 0
                state.presence_alerted = False
                dwell = now - state.entered_at if state.entered_at else 0.0
                return Alert(
                    rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
                    alert_type="zone_exit", timestamp=now,
                    description=f"Left restricted zone '{self.name}'",
                    details={"polygon": self.geometry(), "dwell_seconds": round(dwell, 1)},
                )
            state.confirm = 0
            return None

        state.confirm += 1

        if not was_inside:
            if state.confirm < max(1, settings.ANCHOR_CONFIRMATION_FRAMES):
                return None
            state.inside = True
            state.entered_at = now
            return Alert(
                rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
                alert_type="enter", timestamp=now,
                description=f"Entered restricted zone '{self.name}'",
                details={
                    "polygon": self.geometry(),
                    "foot_point": [int(foot_point[0]), int(foot_point[1])],
                },
            )

        # Already inside — check sustained presence.
        if not state.presence_alerted and state.entered_at:
            dwell = now - state.entered_at
            if dwell >= self.presence_seconds:
                state.presence_alerted = True
                return Alert(
                    rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
                    alert_type="zone_presence", timestamp=now,
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

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Loitering
# --------------------------------------------------------------------------- #


class LoiterRule(BaseRule):
    """
    Loitering — a track dwelling inside a polygon for ``dwell_seconds``.

    Dwell is measured with wall-clock timestamps supplied by the pipeline, so
    the rule behaves identically at 5 FPS and at 30 FPS, and identically when
    an uploaded video is analysed faster than real time (the pipeline passes
    media time in that case).
    """

    rule_type = "loiter"

    def __init__(self, name: str, points: list, dwell_seconds: Optional[float] = None) -> None:
        super().__init__(name)
        self.polygon = _make_polygon(points)
        self._prepared = prep(self.polygon)
        self.dwell_seconds = float(
            dwell_seconds if dwell_seconds is not None else settings.LOITER_SECONDS
        )
        # track_id -> [entered_at, confirm_frames, has_alerted, last_seen]
        self._state: dict[int, list] = {}

    def geometry(self) -> list:
        return [[float(x), float(y)] for x, y in self.polygon.exterior.coords]

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        inside = self._prepared.contains(Point(foot_point))

        if not inside:
            self._state.pop(track_id, None)
            return None

        state = self._state.get(track_id)
        if state is None:
            # First confirmed frame inside — this is the entry instant that the
            # previous implementation never recorded.
            self._state[track_id] = [now, 1, False, now]
            return None

        state[1] += 1
        state[3] = now
        if state[2]:                      # already alerted for this visit
            return None
        if state[1] < max(1, settings.ANCHOR_CONFIRMATION_FRAMES):
            return None

        dwell = now - state[0]
        if dwell < self.dwell_seconds:
            return None

        state[2] = True
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
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Wrong-direction movement
# --------------------------------------------------------------------------- #


class DirectionRule(BaseRule):
    """
    Wrong-direction movement across a directed line.

    ``allowed_direction`` is ``"entry"`` or ``"exit"``; a confirmed crossing
    the other way raises ``wrong_direction``.
    """

    rule_type = "direction"

    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float,
                 allowed_direction: str = "entry") -> None:
        super().__init__(name)
        if allowed_direction not in ("entry", "exit"):
            raise ValueError("allowed_direction must be 'entry' or 'exit'")
        self.allowed = allowed_direction
        self.line = LineString([(x1, y1), (x2, y2)])
        self.p1 = np.array([x1, y1], dtype=float)
        self.p2 = np.array([x2, y2], dtype=float)
        self._state: dict[int, _CrossState] = {}

    def geometry(self) -> list:
        return [[float(self.p1[0]), float(self.p1[1])],
                [float(self.p2[0]), float(self.p2[1])]]

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        side = _side_of_line(self.p1, self.p2, foot_point)
        state = self._state.setdefault(track_id, _CrossState())
        state.last_seen = now

        if side == 0:
            return None

        prev_side = state.side
        state.side = side

        if prev_side != 0 and prev_side != side:
            state.pending = "entry" if (prev_side == -1 and side == 1) else "exit"
            state.confirm = 1
            return None

        if state.pending is None:
            return None

        state.confirm += 1
        if state.confirm < max(1, settings.ANCHOR_CONFIRMATION_FRAMES):
            return None

        actual = state.pending
        state.pending = None          # clear regardless — no repeat firing
        state.confirm = 0
        if actual == self.allowed:
            return None

        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="wrong_direction", timestamp=now,
            description=(
                f"Movement against permitted flow at '{self.name}' "
                f"(allowed: {self.allowed}, observed: {actual})"
            ),
            details={"line": self.geometry(), "allowed": self.allowed, "actual": actual},
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Night movement
# --------------------------------------------------------------------------- #


class NightMovementRule(BaseRule):
    """
    Sustained movement during night hours.

    A stationary object (parked vehicle, a bush) never triggers this: the
    track must accumulate ``NIGHT_MOVEMENT_MIN_TRAVEL`` pixels of foot-point
    travel while the pipeline reports night conditions.  Night is decided by
    the IST clock, optionally reinforced by measured frame luminance, so the
    rule still demonstrates correctly on daytime footage that is genuinely
    dark (tunnel, heavy shade) or with ``FORCE_NIGHT_MODE``.
    """

    rule_type = "night"

    def __init__(self, name: str = "night_movement",
                 min_travel: Optional[float] = None) -> None:
        super().__init__(name)
        self.min_travel = float(
            min_travel if min_travel is not None else settings.NIGHT_MOVEMENT_MIN_TRAVEL
        )
        # track_id -> [last_point, travelled_px, alerted, last_seen]
        self._state: dict[int, list] = {}

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Optional[Alert]:
        now = _resolve_time(kwargs.get("timestamp"))
        if not kwargs.get("is_night", False):
            self._state.pop(track_id, None)
            return None

        state = self._state.get(track_id)
        if state is None:
            self._state[track_id] = [foot_point, 0.0, False, now]
            return None

        last_point, travelled, alerted, _ = state
        step = math.dist(foot_point, last_point)
        # Ignore single-frame teleports caused by an ID re-association.
        if step > 200:
            step = 0.0
        travelled += step
        self._state[track_id] = [foot_point, travelled, alerted, now]

        if alerted or travelled < self.min_travel:
            return None

        self._state[track_id][2] = True
        return Alert(
            rule_name=self.name, rule_type=self.rule_type, track_id=track_id,
            alert_type="night_movement", timestamp=now,
            description=(
                f"Movement detected during night hours "
                f"({travelled:.0f}px of tracked travel)"
            ),
            details={
                "travelled_px": round(travelled, 1),
                "min_travel_px": self.min_travel,
                "night_source": kwargs.get("night_source", "clock"),
            },
        )

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)


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
        self.debounce_seconds = float(
            debounce_seconds if debounce_seconds is not None else settings.DEBOUNCE_SECONDS
        )
        self._rules: dict[str, BaseRule] = {}
        self._debounce: dict[tuple[str, int, str], float] = {}
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

    @property
    def has_rules(self) -> bool:
        return bool(self._rules)

    # -- per-frame evaluation ------------------------------------------- #
    def update(self, detections: list, timestamp: Optional[float] = None,
               context: Optional[dict] = None) -> list[Alert]:
        """
        Evaluate every rule against every detection for the current frame.

        ``context`` carries frame-level facts the rules may need — currently
        ``is_night`` and ``night_source``.
        """
        if not self._rules:
            return []

        now = _resolve_time(timestamp)
        ctx: dict[str, Any] = context or {}
        triggered: list[Alert] = []

        for det in detections:
            tid = det.track_id
            self._seen_tracks[tid] = now
            foot = det.foot
            for rule in self._rules.values():
                try:
                    alert = rule.update(track_id=tid, foot_point=foot, timestamp=now, **ctx)
                except Exception as exc:  # one bad rule must not stop the rest
                    log.exception("Rule '%s' raised: %s", rule.name, exc)
                    continue
                if alert is None:
                    continue

                key = (rule.name, tid, alert.alert_type)
                last = self._debounce.get(key, 0.0)
                if now - last < self.debounce_seconds:
                    continue
                self._debounce[key] = now
                alert.details.setdefault("camera_id", self.camera_id)
                alert.details.setdefault("object_class", getattr(det, "class_name", ""))
                triggered.append(alert)

        self._gc(now)
        return triggered

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
