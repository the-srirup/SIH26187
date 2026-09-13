"""
Regression tests for the face recogniser's per-track identity cache.

The bug: ``_track_match_cache`` was only ever pruned by
``_cached_matches_for_frame``, which pops an entry it finds expired *while
looking it up for a detection that is currently on screen*. A subject who
walks out of frame is never looked up again, so their entry was never
reached and never removed. Track ids only ever increase, so on a post left
running the dict grew without a ceiling — the one structure in this module
without one, next to ``_event_debounce`` which caps at 512.

These tests drive the real eviction path with fabricated cache entries rather
than real faces, so they need neither InsightFace nor a GPU.
"""
from __future__ import annotations

import time

import pytest

from core.config import settings


@pytest.fixture
def recognizer():
    """The process-wide recogniser with an empty cache.

    FaceRecognizer is a singleton, so the cache is reset around each test
    rather than constructed fresh — constructing it would load SCRFD+ArcFace.
    """
    from cv.face import FaceRecognizer
    rec = FaceRecognizer.__new__(FaceRecognizer)     # no model load
    rec._track_match_cache = {}
    rec._last_cache_sweep = 0.0
    return rec


def _departed(rec, count, age_seconds, now):
    """Seed `count` tracks last seen `age_seconds` ago."""
    for tid in range(count):
        rec._track_match_cache[("cam1", tid)] = (None, None, 0.0, False,
                                                 now - age_seconds)


def test_departed_tracks_are_released(recognizer):
    now = time.monotonic()
    ttl = settings.FACE_MATCH_CACHE_SECONDS
    _departed(recognizer, 500, ttl * 20, now)        # long gone
    assert len(recognizer._track_match_cache) == 500

    recognizer._last_cache_sweep = 0.0               # force the sweep to run
    recognizer._sweep_track_cache(now)

    assert recognizer._track_match_cache == {}, (
        "tracks that left the scene were never released — this is the leak"
    )


def test_a_briefly_occluded_track_keeps_its_identity(recognizer):
    """The sweep must not evict someone who merely stepped behind a pillar.

    Re-deriving an identity costs an ArcFace embedding, and evicting an
    occluded subject would also break the continuity of their match.
    """
    now = time.monotonic()
    recognizer._track_match_cache[("cam1", 7)] = (3, "WATCHED", 0.9, True,
                                                  now - settings.FACE_MATCH_CACHE_SECONDS * 1.5)
    recognizer._last_cache_sweep = 0.0
    recognizer._sweep_track_cache(now)

    assert ("cam1", 7) in recognizer._track_match_cache


def test_the_sweep_is_rate_limited(recognizer):
    """It must not run on every recognition tick."""
    now = time.monotonic()
    _departed(recognizer, 10, settings.FACE_MATCH_CACHE_SECONDS * 20, now)

    recognizer._last_cache_sweep = now               # just swept
    recognizer._sweep_track_cache(now + 1.0)         # 1 s later
    assert len(recognizer._track_match_cache) == 10, "swept too eagerly"

    from cv.face import _TRACK_CACHE_SWEEP_SECONDS
    recognizer._sweep_track_cache(now + _TRACK_CACHE_SWEEP_SECONDS + 1)
    assert recognizer._track_match_cache == {}


def test_cache_stays_bounded_across_many_departed_tracks(recognizer):
    """The property that actually matters: it does not grow without bound.

    This drives the **real write path** (``_remember_match``, what
    ``_match_faces`` calls for every recognised face) rather than poking the
    dict, so it fails if the eviction is ever unwired from the growth again —
    which is exactly how the original leak existed.
    """
    from cv.face import FaceMatch, _TRACK_CACHE_SWEEP_SECONDS
    import numpy as np

    now = time.monotonic()
    match = FaceMatch(bbox=(0, 0, 10, 10), embedding=np.zeros(512, dtype=np.float32))

    for batch in range(50):
        t = now + batch * (_TRACK_CACHE_SWEEP_SECONDS + 1)
        for i in range(100):                          # 100 new tracks per batch
            recognizer._remember_match("cam1", batch * 100 + i, match, t)

    # 5,000 tracks have come and gone. Only the most recent can still be live.
    assert len(recognizer._track_match_cache) <= 500, (
        f"cache grew to {len(recognizer._track_match_cache)} entries across "
        f"5,000 departed tracks — it is not bounded"
    )


def test_the_write_path_is_what_evicts(recognizer):
    """Guards the wiring, not just the sweep function.

    The original bug was not a broken sweep — there was no sweep on the growth
    path at all. A test that only calls ``_sweep_track_cache`` directly passes
    against the buggy code, so this one goes through ``_remember_match``.
    """
    from cv.face import FaceMatch, _TRACK_CACHE_SWEEP_SECONDS
    import numpy as np

    now = time.monotonic()
    match = FaceMatch(bbox=(0, 0, 10, 10), embedding=np.zeros(512, dtype=np.float32))

    _departed(recognizer, 400, settings.FACE_MATCH_CACHE_SECONDS * 20, now)
    assert len(recognizer._track_match_cache) == 400

    # One ordinary write, far enough after the last sweep to trigger one.
    recognizer._remember_match("cam1", 99_999, match,
                               now + _TRACK_CACHE_SWEEP_SECONDS + 1)

    assert len(recognizer._track_match_cache) == 1, (
        "writing to the cache did not evict departed tracks — the eviction is "
        "not wired to the growth path"
    )
