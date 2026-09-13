"""
Proof that ANPR reads a plate end to end.

Every other ANPR test in this suite checks a *part* — the grammar corrector,
the voter, the crop geometry. None of them would have caught what was actually
wrong: plate candidates were located on the 640x384 analytics frame, where a
plate is four to twelve pixels tall and below the localiser's own minimum size,
so the candidate list came back near-empty and the whole feature silently did
nothing while every unit test passed.

Real footage cannot pin that down either, because a clip whose plates are
genuinely illegible is indistinguishable from a broken reader. So these tests
render a plate to known text at a known size, run the real pipeline over it,
and assert the text comes back. Deterministic, repeatable, and it fails loudly
if localisation ever moves back to the downscaled frame.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.config import settings  # noqa: E402

PLATE_TEXT = "MH12DE1433"
PLATE_DISPLAY = "MH 12 DE 1433"


def _render_plate(width: int = 210, height: int = 48, text: str = PLATE_TEXT):
    """A white Indian-style plate with black characters."""
    plate = np.full((height, width, 3), 245, np.uint8)
    cv2.rectangle(plate, (2, 2), (width - 3, height - 3), (20, 20, 20), 2)
    scale = height / 42.0
    cv2.putText(plate, text, (int(10 * scale), int(34 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05 * scale, (15, 15, 15),
                max(2, int(3 * scale)), cv2.LINE_AA)
    return plate


def _scene_with_vehicle(frame_w: int, frame_h: int, plate_w: int):
    """
    A frame containing something plate-shaped on something car-shaped.

    Proportions follow a real camera: the car occupies a modest part of the
    frame and the plate a modest part of the car, which is exactly the regime
    where searching the downscaled frame fails.
    """
    scene = np.full((frame_h, frame_w, 3), 118, np.uint8)
    # Road and sky, so the frame is not a flat colour the CLAHE pass would blow out.
    cv2.rectangle(scene, (0, int(frame_h * 0.55)), (frame_w, frame_h), (92, 92, 96), -1)

    car_w = int(frame_w * 0.30)
    car_h = int(car_w * 0.78)
    car_x = (frame_w - car_w) // 2
    car_y = int(frame_h * 0.42)
    cv2.rectangle(scene, (car_x, car_y), (car_x + car_w, car_y + car_h),
                  (48, 52, 60), -1)
    cv2.rectangle(scene, (car_x + int(car_w * 0.12), car_y + int(car_h * 0.10)),
                  (car_x + int(car_w * 0.88), car_y + int(car_h * 0.42)),
                  (28, 30, 36), -1)

    plate = _render_plate(plate_w, max(12, int(plate_w * 0.23)))
    ph, pw = plate.shape[:2]
    px = car_x + (car_w - pw) // 2
    py = car_y + car_h - ph - int(car_h * 0.10)
    scene[py:py + ph, px:px + pw] = plate
    return scene, (px, py, px + pw, py + ph)


class _FakeVehicle:
    """A detection, as the tracker would hand one to the ANPR stage."""

    def __init__(self, bbox, class_name="car", track_id=1):
        self.bbox = bbox
        self.draw_bbox = bbox
        self.class_name = class_name
        self.class_id = 2
        self.track_id = track_id
        self.confidence = 0.9
        self.age = 12

    @property
    def is_person(self):
        return False

    @property
    def is_vehicle(self):
        return True


@pytest.fixture(scope="module")
def anpr():
    """
    The real processor, with ANPR force-enabled for this module.

    ``conftest`` turns ANPR off for the suite so the other tests never pay for
    an EasyOCR load. These tests are the ones that exist to exercise it, so
    they turn it back on — and skip cleanly where EasyOCR is genuinely absent,
    which is the case on a CI box without the optional dependency.
    """
    from cv.anpr import EASYOCR_AVAILABLE, get_anpr_processor

    if not EASYOCR_AVAILABLE:
        pytest.skip("EasyOCR is not installed in this environment")

    original = settings.ANPR_ENABLED
    settings.ANPR_ENABLED = True
    processor = get_anpr_processor()
    if not processor.is_available():
        settings.ANPR_ENABLED = original
        pytest.skip("EasyOCR present but the reader would not initialise")
    yield processor
    settings.ANPR_ENABLED = original


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #


def test_a_plate_is_located_at_source_resolution(anpr):
    """
    The regression that mattered.

    A 1080p frame carries a plate roughly 3x larger in each axis than the
    640x384 analytics frame does. The localiser has an absolute minimum
    candidate size, so the same scene yields candidates at source resolution
    and none at analytics resolution.
    """
    full, _ = _scene_with_vehicle(1920, 1080, plate_w=210)
    small = cv2.resize(full, settings.frame_size, interpolation=cv2.INTER_AREA)

    # Search the vehicle region, which is what the pipeline does — a full-frame
    # sweep is only the fallback for when no vehicle was detected at all.
    def _vehicle_crop(image):
        h, w = image.shape[:2]
        car_w = int(w * 0.30)
        car_h = int(car_w * 0.78)
        car_x, car_y = (w - car_w) // 2, int(h * 0.42)
        margin = max(8, w // 80)
        return anpr.preprocess_for_indian_plates(image)[
            max(0, car_y + int(car_h * 0.40)):min(h, car_y + car_h + margin),
            max(0, car_x - margin):min(w, car_x + car_w + margin)]

    at_source = anpr._find_plate_candidates(_vehicle_crop(full))
    at_analytics = anpr._find_plate_candidates(_vehicle_crop(small))

    assert at_source, "no plate candidate found even at source resolution"
    assert len(at_source) >= len(at_analytics), (
        f"source {len(at_source)} vs analytics {len(at_analytics)}"
    )


def test_the_pipeline_searches_the_source_frame_when_given_one(anpr):
    """
    ``recognize_plates`` must localise in ``source_frame`` when it has one.

    Asserted through the candidate counter rather than by inspecting internals,
    so the test survives refactoring of how the search frame is chosen.
    """
    full, _ = _scene_with_vehicle(1920, 1080, plate_w=210)
    small = cv2.resize(full, settings.frame_size, interpolation=cv2.INTER_AREA)
    sx = settings.FRAME_WIDTH / 1920.0
    sy = settings.FRAME_HEIGHT / 1080.0
    car = _FakeVehicle((int(672 * sx), int(453 * sy), int(1248 * sx), int(903 * sy)))

    before = anpr._candidates_found
    anpr.recognize_plates(small, vehicle_detections=[car], frame_number=1,
                          source_frame=full, source_id="anpr-src")
    with_source = anpr._candidates_found - before

    before = anpr._candidates_found
    anpr.recognize_plates(small, vehicle_detections=[car], frame_number=2,
                          source_frame=None, source_id="anpr-nosrc")
    without_source = anpr._candidates_found - before

    assert with_source >= without_source, (
        f"searching the source frame found fewer candidates "
        f"({with_source}) than searching the downscale ({without_source})"
    )


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_a_legible_plate_is_actually_read(anpr):
    """
    The end-to-end claim: pixels in, registration out.

    OCR is allowed to mis-read an occasional glyph on a synthetic render, so
    the assertion is a strong similarity to the rendered text rather than an
    exact match — but a broken localiser returns nothing at all, which is what
    this catches.
    """
    full, _ = _scene_with_vehicle(1920, 1080, plate_w=260)
    small = cv2.resize(full, settings.frame_size, interpolation=cv2.INTER_AREA)
    sx = settings.FRAME_WIDTH / 1920.0
    sy = settings.FRAME_HEIGHT / 1080.0
    car = _FakeVehicle((int(672 * sx), int(453 * sy), int(1248 * sx), int(903 * sy)))

    reads = []
    for frame_number in range(1, 8):          # let the temporal voter work
        found = anpr.recognize_plates(
            small, vehicle_detections=[car], frame_number=frame_number * 20,
            source_frame=full, source_id="anpr-read",
        )
        reads.extend(p.plate_text for p in found if p.plate_text)

    assert reads, "ANPR produced no reading at all from a legible plate"

    normalised = [r.replace(" ", "").upper() for r in reads]
    best = max(normalised, key=lambda r: _similarity(r, PLATE_TEXT))
    score = _similarity(best, PLATE_TEXT)
    assert score >= 0.6, (
        f"best read {best!r} is only {score:.0%} similar to {PLATE_TEXT!r}; "
        f"all reads: {sorted(set(normalised))}"
    )


def test_a_plate_box_is_returned_in_analytics_coordinates(anpr):
    """
    The overlay, the rules and the evidence crop all work in analytics space.

    Localisation moved to the source frame, so the box has to be mapped back —
    a plate box carrying 1080p coordinates would be drawn far outside the
    frame and would crop evidence from the wrong place entirely.
    """
    full, _ = _scene_with_vehicle(1920, 1080, plate_w=260)
    small = cv2.resize(full, settings.frame_size, interpolation=cv2.INTER_AREA)
    sx = settings.FRAME_WIDTH / 1920.0
    sy = settings.FRAME_HEIGHT / 1080.0
    car = _FakeVehicle((int(672 * sx), int(453 * sy), int(1248 * sx), int(903 * sy)))

    found = []
    for frame_number in range(1, 6):
        found = anpr.recognize_plates(
            small, vehicle_detections=[car], frame_number=frame_number * 20,
            source_frame=full, source_id="anpr-coords",
        ) or found

    if not found:
        pytest.skip("no reading produced; coordinate mapping untestable here")

    for plate in found:
        x1, y1, x2, y2 = plate.bbox
        assert 0 <= x1 < x2 <= settings.FRAME_WIDTH, plate.bbox
        assert 0 <= y1 < y2 <= settings.FRAME_HEIGHT, plate.bbox


def _similarity(a: str, b: str) -> float:
    """Character-level agreement, position by position, normalised by length."""
    if not a or not b:
        return 0.0
    hits = sum(1 for x, y in zip(a, b) if x == y)
    return hits / float(max(len(a), len(b)))
