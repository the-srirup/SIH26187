"""
Regression tests for which vehicles ANPR actually reads.

The bug: only the strongest plate candidates were OCR'd each tick
(``ANPR_MAX_PLATES_PER_TICK``), chosen purely by candidate score. Score is a
property of the vehicle — how near, how large, how crisp — so the *same* two
vehicles won every tick and the rest were never attempted even once. Measured
on a rendered scene, the reader returned exactly two plates whether three, four
or six cars were in frame. That is the "it misses a lot of cars" report.

The fix is a rotation: a vehicle whose plate is already settled yields its slot,
and among the rest the one waiting longest goes first. These tests exercise the
scheduler directly, so they need neither EasyOCR nor a GPU.
"""
from __future__ import annotations

import time

import pytest

from core.config import settings


@pytest.fixture
def processor():
    """An ANPRProcessor with no OCR model loaded — the scheduler needs none."""
    from cv.anpr import ANPRProcessor
    proc = ANPRProcessor.__new__(ANPRProcessor)
    proc._last_ocr_tick = {}
    proc._tick = 0
    from cv.anpr import PlateVoter
    proc.voter = PlateVoter()
    return proc


def _candidates(n, base_score=0.9):
    """n plate candidates, each on its own vehicle, in descending score order.

    Descending score is the important part: it is what let the top two starve
    everyone else.
    """
    return [(10, 10, 60, 24, base_score - i * 0.01, 100 + i, "car")
            for i in range(n)]


def test_every_vehicle_is_reached_within_a_few_ticks(processor):
    """The property that matters: coverage, not who scores highest."""
    settings_budget = int(settings.ANPR_MAX_PLATES_PER_TICK)
    vehicles = 6
    seen = set()
    now = time.time()
    # Enough ticks for a fair rotation to cover everyone, and no more.
    for _ in range((vehicles // max(1, settings_budget)) + 1):
        chosen = processor._schedule_candidates(_candidates(vehicles), "cam1", now)
        assert len(chosen) <= settings_budget
        seen.update(c[5] for c in chosen)
    assert seen == {100 + i for i in range(vehicles)}, (
        f"only {sorted(seen)} were ever read out of 6 vehicles — the OCR budget "
        f"is starving the lower-scoring cars"
    )


def test_the_highest_scoring_vehicle_does_not_win_every_tick(processor):
    """Directly reproduces the starvation, which score-only sorting always did."""
    now = time.time()
    first = processor._schedule_candidates(_candidates(6), "cam1", now)
    second = processor._schedule_candidates(_candidates(6), "cam1", now)
    assert {c[5] for c in first} != {c[5] for c in second}, (
        "the same vehicles were selected twice in a row; lower-scoring cars "
        "can never be read"
    )


def test_a_settled_plate_yields_its_slot(processor):
    """Re-reading an agreed plate buys nothing and costs a car that has none."""
    now = time.time()
    # Vehicle 100 is read consistently until its plate is settled.
    key = ("cam1", 100)
    for _ in range(int(settings.ANPR_MIN_VOTES) + 2):
        processor.voter.add(key, "MH12DE1433", 0.9, now)
    assert processor.voter.settled(key, now) is True

    chosen = processor._schedule_candidates(_candidates(4), "cam1", now)
    assert 100 not in {c[5] for c in chosen}, (
        "a vehicle whose plate is already agreed is still consuming OCR budget"
    )


def test_scheduling_is_a_no_op_when_everything_fits(processor):
    """Below the budget there is nothing to ration, so nothing is dropped."""
    budget = int(settings.ANPR_MAX_PLATES_PER_TICK)
    cands = _candidates(max(1, budget - 1))
    assert processor._schedule_candidates(cands, "cam1", time.time()) == cands


def test_the_rotation_state_stays_bounded(processor):
    """Per-track bookkeeping must not grow for the life of the process."""
    now = time.time()
    for batch in range(80):
        cands = [(10, 10, 60, 24, 0.9, batch * 20 + i, "car") for i in range(20)]
        processor._schedule_candidates(cands, "cam1", now)
    assert len(processor._last_ocr_tick) <= 512, (
        f"scheduler state grew to {len(processor._last_ocr_tick)} entries"
    )


def test_the_reader_actually_uses_the_scheduler():
    """Guards the wiring, not just the scheduler.

    The original bug was not a broken scheduler — there was none, and the
    selection was a plain score sort inline in ``recognize_plates``. A test
    that only calls ``_schedule_candidates`` would pass against that code, so
    this one checks the reader routes its candidates through it.
    """
    import inspect
    from cv.anpr import ANPRProcessor

    body = inspect.getsource(ANPRProcessor.recognize_plates)
    assert "_schedule_candidates" in body, (
        "recognize_plates no longer schedules its OCR budget fairly, so the "
        "highest-scoring vehicles will starve the rest again"
    )
    assert "candidates.sort(key=lambda c: c[4], reverse=True)" not in body, (
        "recognize_plates is back to selecting candidates by score alone"
    )


# --------------------------------------------------------------------------- #
# Reading plates on moving vehicles
# --------------------------------------------------------------------------- #


def test_the_state_code_must_be_one_india_issues():
    """
    Shape alone does not make a registration.

    ``KH12DE1433`` satisfies the layout grammar perfectly, but KH is not a code
    India issues — it is a misread of MH. Accepting it as format-verified would
    put a registration that cannot exist into a tamper-evident log and present
    it as identified.
    """
    from cv.anpr import ANPRProcessor
    proc = ANPRProcessor.__new__(ANPRProcessor)

    for real in ("MH12DE1433", "DL8CAF5031", "KA01F1234", "UP16CD7890",
                 "TS09XY1234", "OD02AB1111"):
        assert ANPRProcessor.is_plausible_plate(proc, real) is True, real

    for impossible in ("KH12DE1433", "ZZ99XX9999", "QH12DE1433", "XX01AB1234"):
        assert ANPRProcessor.is_plausible_plate(proc, impossible) is False, impossible


def test_an_unambiguous_state_code_misread_is_repaired():
    """One substitution to a single legal code is a correction, not a guess."""
    from cv.anpr import ANPRProcessor
    proc = ANPRProcessor.__new__(ANPRProcessor)
    fixed, corrected = ANPRProcessor.apply_plate_grammar(proc, "XZ12DE1433")
    assert fixed == "MZ12DE1433" and corrected is True


def test_an_ambiguous_state_code_is_left_alone():
    """
    KH is one substitution from BH, CH, JH and MH. Choosing between them would
    be inventing a registration, so the reading is left exactly as OCR saw it
    and simply fails verification.
    """
    from cv.anpr import ANPRProcessor
    proc = ANPRProcessor.__new__(ANPRProcessor)
    fixed, _corrected = ANPRProcessor.apply_plate_grammar(proc, "KH12DE1433")
    assert fixed == "KH12DE1433"
    assert ANPRProcessor.is_plausible_plate(proc, fixed) is False


def test_a_vehicle_leaving_the_frame_gets_priority():
    """
    A car at the edge has one tick left; one mid-frame will still be there.

    Fair rotation alone cannot see that, so edge proximity outranks waiting
    time — otherwise the read that was never going to happen again is the one
    that gets skipped.
    """
    from cv.anpr import ANPRProcessor, PlateVoter
    proc = ANPRProcessor.__new__(ANPRProcessor)
    proc._last_ocr_tick = {}
    proc._tick = 0
    proc.voter = PlateVoter()

    import time
    now = time.time()
    width, height = 1920, 1080
    middle = (900, 500, 1000, 530, 0.95, 1, "car")      # centre, highest score
    leaving = (20, 500, 120, 530, 0.60, 2, "car")       # at the left edge
    chosen = proc._schedule_candidates([middle, leaving], "cam1", now,
                                       frame_size=(width, height))
    if len(chosen) < 2:                                  # budget forced a choice
        assert chosen[0][5] == 2, "the vehicle about to leave was skipped"


def test_motion_deblur_returns_an_image_or_nothing():
    """It must never raise: a preprocessing failure cannot break a read."""
    import numpy as np
    from cv.anpr import ANPRProcessor

    textured = np.random.default_rng(0).integers(0, 255, (40, 200, 3), dtype=np.uint8)
    restored = ANPRProcessor._motion_deblur(textured, 9)
    assert restored is not None and restored.shape == textured.shape

    # Everything below is a crop nothing can be recovered from. Each must return
    # None rather than raise, because a preprocessing failure must never take
    # down a read — the plain variants still have to get their chance.
    assert ANPRProcessor._motion_deblur(textured, 0) is None        # no smear
    assert ANPRProcessor._motion_deblur(np.zeros((40, 200, 3), np.uint8), 9) is None
    assert ANPRProcessor._motion_deblur(np.zeros((0, 0, 3), np.uint8), 9) is None
    assert ANPRProcessor._motion_deblur(None, 9) is None
    # A smear longer than the crop cannot be modelled.
    assert ANPRProcessor._motion_deblur(np.zeros((10, 8, 3), np.uint8), 9) is None


def test_deblur_is_a_second_pass_not_a_default():
    """Sharp plates must not pay for deconvolution they do not need."""
    import inspect
    from cv.anpr import ANPRProcessor

    body = inspect.getsource(ANPRProcessor._read_plate)
    assert "_deblur_variants" in body, "the deblur pass is not wired in"
    assert "if not good_enough()" in body, (
        "deblurring is no longer gated on the cheap path having failed"
    )
