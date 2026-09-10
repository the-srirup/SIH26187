"""
Rules engine for IBVAP — line crossing, zone intrusion, loitering, and
wrong-direction detection with per-track debouncing.

All rules operate on the foot point (ground contact) and stable track_id
produced by ``Detector.track()``.

Foot-Point Anchor Logic:
    Each rule requires the foot point (bottom-centre of bounding box) to
    be a stable anchor within the detection zone for at least
    ANCHOR_CONFIRMATION_FRAMES consecutive frames before triggering an
    alert. This prevents false alarms from environmental factors like
    swaying tree branches, passing shadows, or transient occlusions.

Alert Debouncer:
    The RuleEngine maintains a _debounce dictionary keyed by
    (rule_name, track_id, alert_type) that enforces a minimum time
    between consecutive alerts for the same object+rule combination.
    Even after anchor confirmation, a new alert only fires if the
    debounce window has elapsed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from core.config import settings

# Type aliases for alert payloads
AlertType = Literal["entry", "exit", "enter", "loiter", "wrong_direction"]


@dataclass
class Alert:
    """Triggered rule alert with metadata for logging / WebSocket broadcast."""
    rule_name: str
    rule_type: str
    track_id: int
    alert_type: AlertType
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "track_id": self.track_id,
            "alert_type": self.alert_type,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class BaseRule(ABC):
    """Abstract base class — every rule must implement ``update``."""

    def __init__(self, name: str) -> None:
        self.name = name
        # track_id -> last alert timestamp (for internal debouncing if needed)
        self._last_alert: dict[int, float] = {}

    @abstractmethod
    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Alert | None:
        """Process a single track's foot point. Return Alert if triggered, else None."""
        ...


class FenceRule(BaseRule):
    """
    Virtual fence — detects line crossing (entry / exit) using 2-D cross product.

    Foot-Point Anchor Logic:
        A crossing is only confirmed when the foot point (bottom-centre of
        the bounding box) has been on the new side of the line for at least
        ``ANCHOR_CONFIRMATION_FRAMES`` consecutive frames. This prevents false
        alarms from environmental shifts — a swaying tree branch or passing
        shadow will oscillate across the line briefly but never reach the
        anchor threshold.

    The directed line from ``(x1, y1)`` → ``(x2, y2)`` defines "entry" as
    crossing from left to right (cross product sign change − → +) and "exit"
    as right to left (+ → −).  This matches the right-hand rule convention.
    """

    def __init__(
        self,
        name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        super().__init__(name)
        self.line = LineString([(x1, y1), (x2, y2)])
        self.p1 = np.array([x1, y1], dtype=float)
        self.p2 = np.array([x2, y2], dtype=float)
        # track_id -> which side of the line the track was on last frame (−1 / +1 / 0)
        self._side: dict[int, int] = {}
        # track_id -> consecutive frame count on current side (anchor confirmation)
        self._anchor_count: dict[int, int] = {}

    @staticmethod
    def _cross(p1: np.ndarray, p2: np.ndarray, pt: np.ndarray) -> float:
        """2-D cross product (p2 - p1) × (pt - p1). Positive = left of directed line."""
        return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])

    def _side_of(self, foot: tuple[int, int]) -> int:
        val = self._cross(self.p1, self.p2, np.array(foot, dtype=float))
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Alert | None:
        """Update fence rule for a track.

        Foot-Point Anchor Logic:
        - Track which side of the fence the foot point is on
        - When crossing to a new side, start counting consecutive frames on that side
        - Only trigger an alert after ANCHOR_CONFIRMATION_FRAMES frames have passed
        - This prevents false alarms from environmental noise (branches, shadows)
        """
        from core.config import settings

        side = self._side_of(foot_point)
        prev_side = self._side.get(track_id, 0)

        # Always update the side tracking for this track
        self._side[track_id] = side

        # Track if we're in a crossing state (just crossed, waiting for anchor)
        if not hasattr(self, '_crossed'):
            self._crossed: dict[int, tuple[int, str]] = {}  # track_id -> (anchor_count, crossing_direction)

        crossed_state = self._crossed.get(track_id, (0, None))
        anchor_count, crossing_type = crossed_state

        # Check for side change (crossing event) - only when actually on a side, not on the line
        side_changed = prev_side != 0 and prev_side != side

        if side_changed:
            # We just crossed - determine direction and reset counter
            if prev_side == -1 and side == 1:
                crossing_type = "entry"
            elif prev_side == 1 and side == -1:
                crossing_type = "exit"
            else:
                crossing_type = None
            anchor_count = 0  # Will start counting from 0

        # Count frames on current side (only if not on the line)
        if side != 0:
            anchor_count += 1
            self._crossed[track_id] = (anchor_count, crossing_type)
        else:
            # On the line - keep previous state but don't increment
            # When object moves off the line to a side, it will be treated as a continuation
            pass

        # Anchor confirmation check
        anchor_frames = getattr(settings, 'ANCHOR_CONFIRMATION_FRAMES', 5)
        alert: Alert | None = None

        if anchor_count >= anchor_frames and crossing_type:
            if crossing_type == "entry":
                alert = Alert(
                    rule_name=self.name,
                    rule_type="fence",
                    track_id=track_id,
                    alert_type="entry",
                    details={"line": [float(c) for coord in self.line.coords for c in coord]},
                )
                crossing_type = None  # Reset for next crossing
            elif crossing_type == "exit":
                alert = Alert(
                    rule_name=self.name,
                    rule_type="fence",
                    track_id=track_id,
                    alert_type="exit",
                    details={"line": [float(c) for coord in self.line.coords for c in coord]},
                )
                crossing_type = None  # Reset for next crossing

            self._crossed[track_id] = (anchor_count, crossing_type)

        return alert


class ZoneRule(BaseRule):
    """
    Zone (polygon) intrusion — detects enter / exit using Shapely point-in-polygon.

    Foot-Point Anchor Logic:
        The foot point must be inside the polygon for at least
        ``ANCHOR_CONFIRMATION_FRAMES`` consecutive frames before a zone
        intrusion alert is triggered. A swaying branch or passing shadow
        that briefly enters the zone will not exceed the threshold.
    """

    def __init__(self, name: str, points: list[tuple[float, float]]) -> None:
        super().__init__(name)
        self.polygon = Polygon(points)
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)  # attempt self-repair
        # track_id -> was_inside (bool)
        self._inside: dict[int, bool] = {}
        # track_id -> consecutive frames inside the zone (anchor confirmation)
        self._inside_count: dict[int, int] = {}

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Alert | None:
        """Update zone rule for a track.

        Foot-Point Anchor Logic:
        - Count consecutive frames where the foot point is inside the zone
        - Only trigger 'enter' alert after ANCHOR_CONFIRMATION_FRAMES
        - This prevents false alarms from shadows, swaying branches, etc.
        """
        from core.config import settings

        pt = Point(foot_point)
        inside = self.polygon.contains(pt)

        # Count consecutive frames inside the zone
        if inside:
            self._inside_count[track_id] = self._inside_count.get(track_id, 0) + 1
        else:
            self._inside_count[track_id] = 0

        self._inside[track_id] = inside

        anchor_frames = getattr(settings, 'ANCHOR_CONFIRMATION_FRAMES', 5)
        alert: Alert | None = None

        # Exit doesn't need anchor - we know for sure they left
        if self._inside.get(track_id, False) != inside:
            if inside:
                # Entered zone - need anchor confirmation
                if self._inside_count[track_id] >= anchor_frames:
                    alert = Alert(
                        rule_name=self.name,
                        rule_type="zone",
                        track_id=track_id,
                        alert_type="enter",
                        details={"polygon": [list(c) for c in self.polygon.exterior.coords]},
                    )
            else:
                # Exited zone - no anchor needed for exit
                alert = Alert(
                    rule_name=self.name,
                    rule_type="zone",
                    track_id=track_id,
                    alert_type="exit",
                    details={"polygon": [list(c) for c in self.polygon.exterior.coords]},
                )

        return alert


class LoiterRule(BaseRule):
    """
    Loitering detection — triggers when a track dwells inside a polygon for
    ``dwell_seconds`` continuous seconds.

    Uses wall-clock timestamps passed via ``timestamp`` kwarg (or ``time.time()``
    if omitted) so it works correctly with variable frame rates.
    """

    def __init__(
        self,
        name: str,
        points: list[tuple[float, float]],
        dwell_seconds: float | None = None,
    ) -> None:
        super().__init__(name)
        self.polygon = Polygon(points)
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)
        self.dwell_seconds = dwell_seconds if dwell_seconds is not None else settings.LOITER_SECONDS
        # track_id -> (entry_timestamp, has_alerted)
        self._state: dict[int, tuple[float, bool]] = {}
        # track_id -> consecutive frames inside the zone (anchor confirmation)
        self._inside_count: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        foot_point: tuple[int, int],
        timestamp: float | None = None,
        **kwargs,
    ) -> Alert | None:
        from core.config import settings

        ts = timestamp if timestamp is not None else time.time()
        pt = Point(foot_point)
        inside = self.polygon.contains(pt)

        entry_ts, has_alerted = self._state.get(track_id, (ts, False))
        inside_count = self._inside_count.get(track_id, 0)

        # Anchor confirmation: count consecutive frames inside the zone
        if inside:
            self._inside_count[track_id] = inside_count + 1
        else:
            self._inside_count[track_id] = 0
            # Left the zone — reset
            self._state.pop(track_id, None)
            return None

        # Loiter requires both dwell time AND anchor confirmation
        alert: Alert | None = None
        anchor_frames = getattr(settings, 'ANCHOR_CONFIRMATION_FRAMES', 5)
        dwell_met = (ts - entry_ts) >= self.dwell_seconds
        anchor_met = self._inside_count[track_id] >= anchor_frames

        if not has_alerted and dwell_met and anchor_met:
            alert = Alert(
                rule_name=self.name,
                rule_type="loiter",
                track_id=track_id,
                alert_type="loiter",
                details={
                    "polygon": [list(c) for c in self.polygon.exterior.coords],
                    "dwell_seconds": self.dwell_seconds,
                    "actual_dwell": ts - entry_ts,
                },
            )
            self._state[track_id] = (entry_ts, True)

        return alert


class DirectionRule(BaseRule):
    """
    Wrong-direction detection — fires when a track crosses a directed line
    in the *disallowed* direction.

    Foot-Point Anchor Logic:
        The foot point must be on the wrong side of the line for at least
        ``ANCHOR_CONFIRMATION_FRAMES`` consecutive frames before a
        ``wrong_direction`` alert is triggered. This prevents false alarms
        from environmental factors like swaying branches that momentarily
        cross the line.

    ``allowed_direction`` must be "entry" (left→right cross = + cross product)
    or "exit" (right→left cross = − cross product).  Any crossing opposite to
    the allowed direction triggers ``wrong_direction``.
    """

    def __init__(
        self,
        name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        allowed_direction: Literal["entry", "exit"],
    ) -> None:
        super().__init__(name)
        if allowed_direction not in ("entry", "exit"):
            raise ValueError("allowed_direction must be 'entry' or 'exit'")
        self.allowed = allowed_direction
        self.line = LineString([(x1, y1), (x2, y2)])
        self.p1 = np.array([x1, y1], dtype=float)
        self.p2 = np.array([x2, y2], dtype=float)
        # track_id -> previous side (−1 / +1 / 0)
        self._side: dict[int, int] = {}
        # track_id -> consecutive frames on current side (anchor confirmation)
        self._anchor_count: dict[int, int] = {}

    @staticmethod
    def _cross(p1: np.ndarray, p2: np.ndarray, pt: np.ndarray) -> float:
        return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])

    def _side_of(self, foot: tuple[int, int]) -> int:
        val = self._cross(self.p1, self.p2, np.array(foot, dtype=float))
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def update(self, track_id: int, foot_point: tuple[int, int], **kwargs) -> Alert | None:
        """Update direction rule for a track.

        Foot-Point Anchor Logic:
        - Track which side of the line the foot point is on
        - When crossing to a new side, start counting consecutive frames on that side
        - Only trigger after ANCHOR_CONFIRMATION_FRAMES to avoid false alarms
        """
        from core.config import settings

        side = self._side_of(foot_point)
        prev_side = self._side.get(track_id, 0)

        # Always update the side tracking for this track
        self._side[track_id] = side

        # Track crossing state
        if not hasattr(self, '_crossed'):
            self._crossed: dict[int, tuple[int, str]] = {}

        crossed_state = self._crossed.get(track_id, (0, None))
        anchor_count, crossing_type = crossed_state

        # Check for side change (crossing event) - only when actually on a side, not on the line
        side_changed = prev_side != 0 and prev_side != side

        if side_changed:
            # We just crossed - determine direction and reset counter
            if prev_side == -1 and side == 1:
                crossing_type = "entry"
            elif prev_side == 1 and side == -1:
                crossing_type = "exit"
            else:
                crossing_type = None
            anchor_count = 0

        # Count frames on current side
        if side != 0:
            anchor_count += 1
            self._crossed[track_id] = (anchor_count, crossing_type)

        anchor_frames = getattr(settings, 'ANCHOR_CONFIRMATION_FRAMES', 5)
        alert: Alert | None = None

        if anchor_count >= anchor_frames and crossing_type:
            # Check if direction is wrong (opposite of allowed)
            # crossing_type contains the direction we crossed in
            if crossing_type != self.allowed:
                alert = Alert(
                    rule_name=self.name,
                    rule_type="direction",
                    track_id=track_id,
                    alert_type="wrong_direction",
                    details={
                        "line": [float(c) for coord in self.line.coords for c in coord],
                        "allowed": self.allowed,
                        "actual": crossing_type,
                    },
                )

        return alert


class RuleEngine:
    """
    Manages multiple rules per camera, applies global debouncing, and returns
    triggered alerts ready for persistence / WebSocket broadcast.
    """

    def __init__(self, camera_id: str, debounce_seconds: float | None = None) -> None:
        self.camera_id = camera_id
        self.debounce_seconds = debounce_seconds if debounce_seconds is not None else settings.DEBOUNCE_SECONDS
        self._rules: dict[str, BaseRule] = {}
        # (rule_name, track_id, alert_type) -> last_alert_timestamp
        self._debounce: dict[tuple[str, int, str], float] = {}

    def add_rule(self, rule: BaseRule) -> None:
        if rule.name in self._rules:
            raise ValueError(f"Rule '{rule.name}' already exists for camera {self.camera_id}")
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        self._rules.pop(name, None)
        # Clean debounce keys for this rule
        self._debounce = {k: v for k, v in self._debounce.items() if k[0] != name}

    def clear(self) -> None:
        self._rules.clear()
        self._debounce.clear()

    def update(self, detections: list) -> list[Alert]:
        """
        Process all detections for the current frame.

        Parameters
        ----------
        detections : list[Detection]
            Output from ``Detector.track()`` — each item has ``track_id`` and ``foot``.

        Returns
        -------
        list[Alert]
            Alerts that fired this frame (after debouncing).
        """
        triggered: list[Alert] = []
        now = time.time()

        for det in detections:
            tid = det.track_id
            foot = det.foot
            for rule in self._rules.values():
                alert = rule.update(track_id=tid, foot_point=foot, timestamp=now)
                if alert is None:
                    continue

                # Debounce key
                key = (rule.name, tid, alert.alert_type)
                last = self._debounce.get(key, 0.0)
                if now - last >= self.debounce_seconds:
                    self._debounce[key] = now
                    triggered.append(alert)

        return triggered

    def get_rules(self) -> list[BaseRule]:
        return list(self._rules.values())