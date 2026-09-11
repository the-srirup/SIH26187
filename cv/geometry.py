"""
Shared geometry primitives for the rule engine.

Every rule that asks a spatial question asks it *here*.  Before this module
existed, ``FenceRule``, ``DirectionRule``, ``ZoneRule`` and ``LoiterRule``
each carried their own copy of the side-of-line and point-in-polygon logic,
which is how the two line rules ended up sharing the same defect: both tested
the point against the **infinite** line through the operator's two clicks
instead of against the drawn **segment**, so a tripwire across a gate fired
for anyone walking across the far end of the frame on the same extended line.

Two ideas carry most of the correctness here.

**A crossing is a property of a trajectory, not of a point.**
Asking "which side is this object on?" once per frame cannot distinguish a
genuine crossing from an object that was simply never seen on the other side.
:class:`LineGeometry.crossing` instead takes the *movement segment*
``previous_point -> current_point`` and intersects it with the fence segment.
That is exact, and — crucially — it is correct across a multi-frame gap: a
motorcycle that jumps 180 px between analysed frames still has a movement
segment that cuts the fence, so it is detected at 15 FPS just as reliably as
at 60.  The old point-sampling approach needed the object to be *observed* on
both sides, then to survive three further frames of confirmation, which is
exactly what a fast vehicle never does.

**Zone membership needs hysteresis, not a fresh opinion every frame.**
A foot point sitting on a polygon edge flips in and out with sub-pixel box
jitter.  :class:`ZoneGeometry` therefore exposes a signed distance to the
boundary so rules can require a margin before believing a transition, rather
than trusting a bare ``contains()``.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

Point2 = tuple[float, float]

#: Below this movement (in pixels) a track is treated as stationary for the
#: purpose of crossing tests.  Box jitter on a stationary subject is routinely
#: 1-3 px; demanding real displacement is what stops a fence from firing on a
#: person standing still beside it.
DEFAULT_MIN_DISPLACEMENT = 3.0


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #


def side_of_line(p1: Point2, p2: Point2, point: Point2, epsilon: float = 1e-9) -> int:
    """
    Which side of the directed line ``p1 -> p2`` a point lies on.

    Returns ``+1`` (left of travel), ``-1`` (right) or ``0`` (on the line).
    This is the 2-D cross product's sign.  Note it describes the *infinite*
    line: it answers "which half-plane", never "did it cross the segment".
    Use :meth:`LineGeometry.crossing` for crossings.
    """
    cross = ((p2[0] - p1[0]) * (point[1] - p1[1])
             - (p2[1] - p1[1]) * (point[0] - p1[0]))
    if cross > epsilon:
        return 1
    if cross < -epsilon:
        return -1
    return 0


def distance(a: Point2, b: Point2) -> float:
    return math.dist(a, b)


def make_polygon(points: Sequence[Sequence[float]]) -> Polygon:
    """
    Build a valid polygon from operator-clicked vertices.

    A hand-drawn outline is frequently self-intersecting (the operator crosses
    their own line while tracing a compound shape).  ``buffer(0)`` repairs
    that into a valid geometry instead of leaving a polygon whose
    ``contains()`` results are undefined.
    """
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) < 3:
        raise ValueError("a polygon needs at least 3 points")
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
        # buffer(0) on a bow-tie can yield a MultiPolygon; keep the largest part.
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.area <= 0:
        raise ValueError("polygon has no area")
    return poly


# --------------------------------------------------------------------------- #
# Line geometry — tripwires and direction rules
# --------------------------------------------------------------------------- #


class CrossingResult:
    """
    The outcome of one trajectory-versus-fence test.

    ``crossed`` is the only field a rule must check; the rest explain *why*,
    and are carried into the event details so an operator reviewing the log
    can see the actual geometry that fired.
    """

    __slots__ = ("crossed", "direction", "from_side", "to_side",
                 "point", "displacement")

    def __init__(
        self,
        crossed: bool = False,
        direction: str = "",
        from_side: int = 0,
        to_side: int = 0,
        point: Optional[Point2] = None,
        displacement: float = 0.0,
    ) -> None:
        self.crossed = crossed
        self.direction = direction          # "entry" | "exit" | ""
        self.from_side = from_side
        self.to_side = to_side
        self.point = point                  # where the trajectory met the line
        self.displacement = displacement

    def __bool__(self) -> bool:             # pragma: no cover - convenience
        return self.crossed

    def __repr__(self) -> str:              # pragma: no cover - debugging aid
        return (f"CrossingResult(crossed={self.crossed}, "
                f"direction={self.direction!r}, "
                f"{self.from_side}->{self.to_side})")


class LineGeometry:
    """
    A finite, directed fence segment.

    Direction convention: travelling from the **right** of ``p1 -> p2`` to its
    **left** is an ``entry``; the reverse is an ``exit``.  That is the
    right-hand rule, and it means an operator who draws a line left-to-right
    across a road gets "entry" for traffic moving *towards* the camera, which
    matches how people describe a checkpoint.
    """

    __slots__ = ("p1", "p2", "_segment", "length")

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.p1: Point2 = (float(x1), float(y1))
        self.p2: Point2 = (float(x2), float(y2))
        self.length = distance(self.p1, self.p2)
        if self.length <= 0:
            raise ValueError("a fence line needs two distinct points")
        self._segment = LineString([self.p1, self.p2])

    # -- description ---------------------------------------------------- #
    def as_list(self) -> list[list[float]]:
        return [[self.p1[0], self.p1[1]], [self.p2[0], self.p2[1]]]

    def side(self, point: Point2) -> int:
        return side_of_line(self.p1, self.p2, point)

    def distance_to(self, point: Point2) -> float:
        """Perpendicular distance to the *segment* (not the infinite line)."""
        return float(self._segment.distance(Point(point)))

    # -- the crossing test ---------------------------------------------- #
    def crossing(
        self,
        previous: Optional[Point2],
        current: Point2,
        min_displacement: float = DEFAULT_MIN_DISPLACEMENT,
    ) -> CrossingResult:
        """
        Did the movement ``previous -> current`` cross this fence segment?

        The test is a segment-segment intersection, so it is exact for any
        line orientation (horizontal, vertical, diagonal — no special cases)
        and holds across arbitrarily large single-frame displacements.

        A crossing requires three things, all of which matter:

        * the trajectory actually **intersects the drawn segment** — not its
          infinite extension, which is what previously let a fence fire at the
          far side of the frame;
        * the two endpoints lie on **opposite sides**, which rejects a
          trajectory that merely grazes an endpoint and returns;
        * the object **moved** at least ``min_displacement`` px, which rejects
          box jitter on a subject standing on the line.
        """
        if previous is None:
            return CrossingResult()

        travelled = distance(previous, current)
        if travelled < max(0.0, min_displacement):
            return CrossingResult(displacement=travelled)

        from_side = self.side(previous)
        to_side = self.side(current)

        # Both endpoints on the same side cannot be a crossing. A zero means
        # an endpoint sat exactly on the line; that is not yet a crossing --
        # the next frame resolves it once the point commits to a side.
        if from_side == 0 or to_side == 0 or from_side == to_side:
            return CrossingResult(from_side=from_side, to_side=to_side,
                                  displacement=travelled)

        movement = LineString([previous, current])
        if not movement.intersects(self._segment):
            # Opposite half-planes, but the path missed the finite fence --
            # the object went around the end of the line.
            return CrossingResult(from_side=from_side, to_side=to_side,
                                  displacement=travelled)

        meeting = movement.intersection(self._segment)
        try:
            where: Point2 = (float(meeting.x), float(meeting.y))
        except AttributeError:
            # Collinear overlap yields a LineString; use its midpoint.
            centroid = meeting.centroid
            where = (float(centroid.x), float(centroid.y))

        return CrossingResult(
            crossed=True,
            direction="entry" if (from_side == -1 and to_side == 1) else "exit",
            from_side=from_side,
            to_side=to_side,
            point=where,
            displacement=travelled,
        )


# --------------------------------------------------------------------------- #
# Zone geometry — restricted areas and loiter areas
# --------------------------------------------------------------------------- #


class ZoneGeometry:
    """
    A polygonal area with a margin-aware membership test.

    ``contains`` is the plain question.  ``membership`` is the one rules should
    ask: it returns a signed distance to the boundary (positive inside), which
    lets a rule demand that a foot point be *convincingly* inside before
    declaring an intrusion and *convincingly* outside before declaring an
    exit.  Without that margin, a foot point resting on the boundary alternates
    every frame and generates an endless enter/exit pair — which is precisely
    the flapping seen in the event log before this rewrite.
    """

    __slots__ = ("polygon", "_prepared", "_boundary")

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        self.polygon = make_polygon(points)
        self._prepared = prep(self.polygon)     # ~3x faster repeated contains()
        self._boundary = self.polygon.exterior

    def as_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.polygon.exterior.coords]

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    def contains(self, point: Point2) -> bool:
        return bool(self._prepared.contains(Point(point)))

    def signed_distance(self, point: Point2) -> float:
        """
        Distance to the boundary; **positive inside**, negative outside.

        Used for hysteresis: require ``>= margin`` to enter and
        ``<= -margin`` to leave, so the boundary has thickness.
        """
        shapely_point = Point(point)
        edge = float(self._boundary.distance(shapely_point))
        return edge if self._prepared.contains(shapely_point) else -edge

    def membership(self, point: Point2, margin: float = 0.0) -> Optional[bool]:
        """
        Ternary membership with a dead band.

        ``True`` = convincingly inside, ``False`` = convincingly outside,
        ``None`` = within ``margin`` px of the boundary, i.e. "do not change
        your mind on this frame".  A rule that treats ``None`` as "hold
        previous state" becomes immune to boundary jitter.
        """
        signed = self.signed_distance(point)
        if margin <= 0:
            return signed >= 0
        if signed >= margin:
            return True
        if signed <= -margin:
            return False
        return None


# --------------------------------------------------------------------------- #
# Reference points
# --------------------------------------------------------------------------- #


def foot_point(bbox: Sequence[float]) -> tuple[int, int]:
    """
    Bottom-centre of a box — where the subject meets the ground.

    This is the right reference for ground-plane rules.  A box *centre* floats
    at chest height, so a person standing just outside a zone registers as
    inside it whenever the camera looks down, which is the normal mounting.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (int(round((x1 + x2) / 2.0)), int(round(y2)))


def centre_point(bbox: Sequence[float]) -> tuple[int, int]:
    """Box centre — the better reference for airborne or wall-mounted views."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))


#: Named reference-point extractors, selectable per rule via params.
REFERENCE_POINTS = {"foot": foot_point, "center": centre_point, "centre": centre_point}


def reference_point(bbox: Sequence[float], kind: str = "foot") -> tuple[int, int]:
    """Resolve a rule's configured reference point, defaulting to the foot."""
    return REFERENCE_POINTS.get((kind or "foot").lower(), foot_point)(bbox)


__all__ = [
    "CrossingResult", "LineGeometry", "ZoneGeometry",
    "side_of_line", "distance", "make_polygon",
    "foot_point", "centre_point", "reference_point", "REFERENCE_POINTS",
    "DEFAULT_MIN_DISPLACEMENT",
]
