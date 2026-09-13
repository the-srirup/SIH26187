"""
Video overlay renderer.

The overlay is what a judge and an operator actually look at, so it is built
to be *read at a glance on a moving image*:

* Labels read ``PERSON #17 | 94%`` — class, stable track id, confidence —
  instead of a 10-pixel ``person 0.51``.
* Every label sits on a filled chip in the box colour with black or white
  text chosen by luminance, so it stays legible over sky, sand or tarmac.
* Colours encode meaning: cyan = person, amber = vehicle, red = an object
  that just triggered an event, yellow = fence, magenta = loiter zone.
* Box thickness is 2px, corner-accented rather than heavy, so it never
  obscures the subject.
* Text placement flips below the box when a label would fall off the top
  edge, and never renders outside the frame.

Rendering is deliberately cheap: no alpha compositing per box (one blended
overlay pass for filled shapes), no per-frame font loading, no PIL.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from core.config import settings
from core.timeutil import fmt_ist_time

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_DUPLEX = cv2.FONT_HERSHEY_DUPLEX

# --- palette (BGR) --------------------------------------------------------- #
C_PERSON = (255, 214, 92)      # cyan-blue
C_VEHICLE = (64, 176, 255)     # amber
C_OTHER = (168, 168, 168)
C_ALERT = (60, 60, 255)        # red
C_FENCE = (0, 226, 255)        # yellow
C_ZONE = (255, 128, 0)         # blue-orange
C_LOITER = (220, 90, 255)      # magenta
C_PLATE = (0, 200, 120)        # green
C_FACE = (200, 120, 255)
C_HUD_BG = (26, 22, 18)
C_HUD_TEXT = (235, 235, 235)


def _text_colour(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on light chips, white on dark ones (Rec. 601 luma)."""
    luma = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (16, 16, 16) if luma > 150 else (255, 255, 255)


def class_colour(class_name: str, alerted: bool = False) -> tuple[int, int, int]:
    if alerted:
        return C_ALERT
    if class_name == "person":
        return C_PERSON
    if class_name in ("car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"):
        return C_VEHICLE
    return C_OTHER


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_label(
    img: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    colour: tuple[int, int, int],
    *,
    scale: float = 0.46,
    thickness: int = 1,
    below: bool = False,
    occupied: Optional[list] = None,
) -> tuple[int, int, int, int]:
    """
    Draw a filled chip with ``text``, clamped to stay inside the frame.

    When ``occupied`` is supplied, the chip is nudged vertically until it no
    longer collides with a label already drawn this frame, and its rectangle is
    appended.  Without this, a fence label and a box label landing in the same
    place rendered on top of each other and both became unreadable.
    """
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, FONT_DUPLEX, scale, thickness)
    pad_x, pad_y = 6, 4
    box_w, box_h = tw + pad_x * 2, th + base + pad_y * 2

    x, y = anchor
    y_top = y + 2 if below else y - box_h - 2
    x = max(0, min(x, w - box_w))

    if occupied is not None:
        step = box_h + 3
        direction = 1 if below else -1
        for _ in range(6):
            candidate = (x, y_top, x + box_w, y_top + box_h)
            if not any(_overlaps(candidate, r) for r in occupied):
                break
            y_top += direction * step
        # If pushing off one edge, come back the other way.
        if y_top < 0 or y_top + box_h > h:
            y_top = y + 2 if not below else y - box_h - 2

    y_top = max(0, min(y_top, h - box_h))
    rect = (x, y_top, x + box_w, y_top + box_h)

    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), colour, -1)
    cv2.putText(
        img, text, (x + pad_x, y_top + box_h - pad_y - base // 2),
        FONT_DUPLEX, scale, _text_colour(colour), thickness, cv2.LINE_AA,
    )
    if occupied is not None:
        occupied.append(rect)
    return rect


def _corner_box(img: np.ndarray, box, colour, thickness: int = 2, corner: int = 14) -> None:
    """A 2px rectangle with thicker corner ticks — reads as a targeting box."""
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    c = min(corner, max(4, (x2 - x1) // 3), max(4, (y2 - y1) // 3))
    t = thickness + 1
    for (px, py, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(img, (px, py), (px + dx * c, py), colour, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * c), colour, t, cv2.LINE_AA)


def draw_detections(
    img: np.ndarray,
    detections: Iterable,
    alerted_tracks: Optional[set] = None,
    occupied: Optional[list] = None,
) -> None:
    """Draw every tracked object with a readable ``CLASS #ID | NN%`` label."""
    alerted = alerted_tracks or set()
    if occupied is None:
        occupied = []
    # Draw nearest (largest) objects first so their labels win the best spot.
    ordered = sorted(
        detections,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        reverse=True,
    )
    for det in ordered:
        flagged = det.track_id in alerted
        colour = class_colour(det.class_name, flagged)
        box = det.draw_bbox if any(det.draw_bbox) else det.bbox
        x1, y1, x2, y2 = (int(v) for v in box)

        _corner_box(img, (x1, y1, x2, y2), colour, 2)
        draw_label(img, f"{det.label} #{det.track_id:02d} | {det.confidence * 100:.0f}%",
                   (x1, y1), colour, occupied=occupied)

        # Ground-contact point — the anchor every rule actually evaluates.
        fx, fy = det.foot
        cv2.drawMarker(img, (int(fx), int(fy)), colour, cv2.MARKER_TILTED_CROSS, 9, 2)

        if flagged:
            draw_label(img, "EVENT", (x1, y2), C_ALERT, scale=0.42,
                       below=True, occupied=occupied)


def draw_rules(img: np.ndarray, rule_shapes: Sequence[dict],
               occupied: Optional[list] = None) -> None:
    """
    Draw fences and zones.

    ``rule_shapes`` is a pre-computed list of
    ``{"type", "name", "geometry", "active"}`` dicts — the pipeline caches
    these in memory so the renderer never touches the database.

    A shape may also carry ``occupied`` and ``breached``. An occupied polygon
    is tinted and labelled for as long as something is inside it, and a
    breached one — occupied past its dwell threshold — is drawn in alert
    colour. That is the difference between a fence that reports a crossing and
    a zone that shows an intrusion in progress: the operator who looks up
    thirty seconds late still sees it.
    """
    if not rule_shapes:
        return
    if occupied is None:
        occupied = []

    filled = None      # armed but empty polygons — a faint hint
    heavy = None       # occupied polygons — a strong, continuous signal
    for shape in rule_shapes:
        geom = shape.get("geometry") or []
        rtype = shape.get("type")
        name = (shape.get("name") or rtype or "").upper()

        if rtype in ("line", "direction") and len(geom) >= 2:
            p1 = (int(geom[0][0]), int(geom[0][1]))
            p2 = (int(geom[1][0]), int(geom[1][1]))
            colour = C_FENCE
            cv2.line(img, p1, p2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(img, p1, p2, colour, 2, cv2.LINE_AA)
            # Direction ticks showing which way is "entry".
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            length = max(1.0, float(np.hypot(vx, vy)))
            nx, ny = -vy / length, vx / length
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.arrowedLine(
                img, mid, (int(mid[0] + nx * 26), int(mid[1] + ny * 26)),
                colour, 2, cv2.LINE_AA, tipLength=0.35,
            )
            # Anchor the label at whichever end sits further from the frame
            # centre, so it lands in empty scene rather than over the action.
            cx = img.shape[1] / 2
            anchor = p1 if abs(p1[0] - cx) >= abs(p2[0] - cx) else p2
            draw_label(img, f"FENCE: {name}", anchor, colour, scale=0.42,
                       occupied=occupied)

        elif rtype in ("zone", "loiter") and len(geom) >= 3:
            is_occupied = bool(shape.get("occupied"))
            is_breached = bool(shape.get("breached"))
            base = C_LOITER if rtype == "loiter" else C_ZONE
            # Three states, three appearances, held for as long as the state
            # holds: armed and empty (thin outline, faint tint), occupied
            # (thicker outline, stronger tint), breached (alert colour). An
            # operator can read the current situation off a still frame.
            colour = C_ALERT if is_breached else base
            weight = 3 if is_occupied else 2
            if heavy is None:
                heavy = img.copy()
            if filled is None:
                filled = img.copy()
            pts = np.array(geom, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(heavy if is_occupied else filled, [pts], colour)
            cv2.polylines(img, [pts], True, colour, weight, cv2.LINE_AA)

            label = "LOITER ZONE" if rtype == "loiter" else "RESTRICTED ZONE"
            if is_occupied:
                count = int(shape.get("count") or 0)
                suffix = f" — OCCUPIED{f' x{count}' if count > 1 else ''}"
                if is_breached:
                    suffix = f" — INTRUSION{f' x{count}' if count > 1 else ''}"
                label = f"{label}: {name}{suffix}"
            else:
                label = f"{label}: {name}"
            draw_label(img, label,
                       (int(geom[0][0]), int(geom[0][1])), colour, scale=0.42,
                       occupied=occupied)

    # Two blend passes, not one per shape: empty polygons stay a faint hint,
    # occupied ones are unmistakable without hiding the subject inside them.
    if filled is not None:
        cv2.addWeighted(filled, 0.16, img, 0.84, 0, dst=img)
    if heavy is not None:
        cv2.addWeighted(heavy, 0.34, img, 0.66, 0, dst=img)


def draw_plates(img: np.ndarray, plates: Iterable,
                occupied: Optional[list] = None) -> None:
    """Draw ANPR results; low-confidence reads are shown as PLATE UNCERTAIN."""
    for plate in plates:
        x1, y1, x2, y2 = (int(v) for v in plate.bbox)
        certain = plate.text_confidence >= settings.ANPR_CONFIDENCE_THRESHOLD
        colour = C_PLATE if certain else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        text = (
            f"PLATE {plate.plate_text} | {plate.text_confidence * 100:.0f}%"
            if certain else "PLATE UNCERTAIN"
        )
        draw_label(img, text, (x1, y2), colour, scale=0.44, below=True,
                   occupied=occupied)


def draw_faces(img: np.ndarray, matches: Iterable,
               occupied: Optional[list] = None) -> None:
    """Draw face boxes. Unmatched faces say FACE DETECTED — never an identity."""
    for match in matches:
        x1, y1, x2, y2 = (int(v) for v in match.bbox)
        if match.matched and match.watchlist_name:
            colour = C_ALERT
            text = f"WATCHLIST: {match.watchlist_name} | {match.similarity * 100:.0f}%"
        else:
            colour = C_FACE
            text = "FACE DETECTED"
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        draw_label(img, text, (x1, y1), colour, scale=0.42, occupied=occupied)


def draw_hud(
    img: np.ndarray,
    *,
    camera_name: str,
    fps: float,
    detections: int,
    online: bool = True,
    night: bool = False,
    timestamp: Optional[float] = None,
    extra: str = "",
) -> None:
    """Bottom status strip: source, IST clock, FPS, live object count."""
    h, w = img.shape[:2]
    strip_h = 26
    y0 = h - strip_h

    band = img[y0:h, 0:w]
    cv2.addWeighted(band, 0.25, np.full_like(band, C_HUD_BG), 0.75, 0, dst=band)

    dot = (80, 220, 120) if online else (60, 60, 255)
    cv2.circle(img, (14, y0 + strip_h // 2), 5, dot, -1, cv2.LINE_AA)

    left = f"{camera_name}"
    cv2.putText(img, left, (28, y0 + 18), FONT_DUPLEX, 0.46, C_HUD_TEXT, 1, cv2.LINE_AA)

    right_bits = [f"{fps:4.1f} FPS", f"OBJ {detections:02d}"]
    if night:
        right_bits.append("NIGHT")
    if extra:
        right_bits.append(extra)
    right_bits.append(fmt_ist_time(timestamp))
    right = "   ".join(right_bits)
    (tw, _), _ = cv2.getTextSize(right, FONT_DUPLEX, 0.46, 1)
    cv2.putText(img, right, (max(0, w - tw - 10), y0 + 18),
                FONT_DUPLEX, 0.46, C_HUD_TEXT, 1, cv2.LINE_AA)


def draw_offline(img: np.ndarray, camera_name: str, reason: str = "") -> np.ndarray:
    """Render an explicit OFFLINE card so a dead camera never shows a stale frame."""
    img[:] = (24, 20, 18)
    h, w = img.shape[:2]
    cv2.putText(img, "SIGNAL LOST", (int(w * 0.5) - 105, int(h * 0.45)),
                FONT_DUPLEX, 0.9, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(img, camera_name, (int(w * 0.5) - 90, int(h * 0.45) + 30),
                FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    if reason:
        (tw, _), _ = cv2.getTextSize(reason, FONT, 0.45, 1)
        cv2.putText(img, reason, (max(8, (w - tw) // 2), int(h * 0.45) + 56),
                    FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def draw_event_banner(img: np.ndarray, text: str, severity: str = "HIGH") -> None:
    """Top banner shown for a moment when an event fires."""
    colour = {
        "CRITICAL": (60, 60, 255), "HIGH": (40, 120, 255),
        "MEDIUM": (0, 190, 255), "LOW": (120, 190, 120),
    }.get(severity, (40, 120, 255))
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 26), colour, -1)
    cv2.putText(img, text[:80], (10, 18), FONT_DUPLEX, 0.5,
                _text_colour(colour), 1, cv2.LINE_AA)
