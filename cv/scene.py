"""
Scene condition estimation — is *this camera's view* dark?

Why this module exists
----------------------
Night analytics used to be decided by ``is_night(datetime.now())``: at 23:00
the platform declared night on every camera in the deployment, including one
pointed at a floodlit checkpost, and at noon it declared day on a camera
inside an unlit culvert.  The clock is not an observation of the scene.

This module answers the question the surveillance platform actually needs —
*is the image dark?* — from the pixels, and does so robustly enough to survive
the conditions a border deployment really presents:

================================  ==========================================
Condition                         How it is handled
================================  ==========================================
Night, unlit                      Low mean luma, high dark-pixel fraction.
Night with a floodlight/streetlamp Mean luma is dragged up by a small bright
                                  region, so mean alone says "day".  The
                                  **dark-pixel fraction** still says night,
                                  and it is weighted accordingly.
IR / night-vision camera          The sensor outputs a *bright* but truly
                                  colourless image, so mean luma says day.
                                  Near-zero **saturation and channel spread**
                                  at moderate brightness is the giveaway.
                                  Claimed only when the brightness signals do
                                  not already say night, so a pitch-black frame
                                  is reported as "dark", not as "infrared".
Daytime, camera inside a tunnel   Dark scene -> night analytics, correctly.
Lens temporarily covered          Dark -> night, and the latch keeps it
                                  stable rather than oscillating.
Headlights sweeping the frame     A single bright frame cannot unlatch night;
                                  ``bright_frames_to_day`` consecutive frames
                                  are required.
Dusk / dawn                       Hysteresis: the threshold to *become* night
                                  is darker than the threshold to *stop being*
                                  night, so the transition happens once
                                  instead of flickering for twenty minutes.
================================  ==========================================

Design
------
Three cheap, complementary signals are measured on a downsampled frame
(~0.1 ms total, versus ~1.5 ms for a full-frame LAB conversion):

``mean_luma``      overall brightness, 0-255.
``dark_fraction``  share of pixels below ``dark_pixel_value`` — the signal
                   that survives bright artificial lights.
``saturation`` +
``channel_spread`` mean HSV saturation and mean per-pixel
                   ``max(BGR) - min(BGR)``.  Together they identify a true
                   monochrome sensor.  Saturation alone cannot: this project's
                   own daytime footage already reads saturation ~7, while a
                   genuine IR frame reads 0 for both.

Those are fused into a single 0-1 ``darkness score``, smoothed with an
exponential moving average, and then **latched**: the state only changes after
a configurable number of consecutive frames agree.  The result is a night
decision that is stable, explainable (every component is reported for the UI),
and completely independent of the wall clock.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.cv.scene")


@dataclass
class SceneCondition:
    """One frame's measured illumination, plus the latched night decision."""

    #: Latched decision — what the rule engine acts on.
    is_night: bool = False
    #: Which signal justified the decision: "", "darkness", "infrared",
    #: "forced", or "clock-hint".
    source: str = ""
    #: Fused 0-1 darkness score after temporal smoothing.
    darkness: float = 0.0
    #: Raw per-frame measurements.
    mean_luma: float = 255.0
    dark_fraction: float = 0.0
    saturation: float = 255.0
    #: Mean per-pixel channel spread; ~0 only for a true monochrome sensor.
    channel_spread: float = 255.0
    #: True when the frame looks like an IR / monochrome night-vision image.
    infrared: bool = False
    #: How many consecutive frames currently agree with the pending change.
    streak: int = 0
    #: Frames measured since this estimator was created.
    samples: int = 0

    def to_dict(self) -> dict:
        return {
            "is_night": self.is_night,
            "source": self.source,
            "darkness": round(self.darkness, 3),
            "mean_luma": round(self.mean_luma, 1),
            "dark_fraction": round(self.dark_fraction, 3),
            "saturation": round(self.saturation, 1),
            "channel_spread": round(self.channel_spread, 2),
            "infrared": self.infrared,
            "streak": self.streak,
            "samples": self.samples,
        }


class SceneIlluminationEstimator:
    """
    Per-camera visual night detector with hysteresis and a confirmation latch.

    One instance belongs to one video source: two cameras at the same BOP
    reach their own conclusions, which is the entire point — the clock could
    never do that.

    Thread-safety: ``measure`` is called only from the owning analytics
    thread, but ``condition`` is read by the API thread, so the latched state
    is published under a lock.
    """

    def __init__(
        self,
        *,
        enter_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
        dark_frames_to_night: Optional[int] = None,
        bright_frames_to_day: Optional[int] = None,
        dark_pixel_value: Optional[int] = None,
        smoothing: Optional[float] = None,
    ) -> None:
        self.enter_threshold = float(
            settings.NIGHT_DARKNESS_ENTER if enter_threshold is None else enter_threshold
        )
        self.exit_threshold = float(
            settings.NIGHT_DARKNESS_EXIT if exit_threshold is None else exit_threshold
        )
        # Hysteresis only exists if the exit threshold is the looser one.
        if self.exit_threshold >= self.enter_threshold:
            self.exit_threshold = max(0.0, self.enter_threshold - 0.10)

        self.dark_frames_to_night = max(1, int(
            settings.NIGHT_CONFIRM_FRAMES if dark_frames_to_night is None
            else dark_frames_to_night
        ))
        self.bright_frames_to_day = max(1, int(
            settings.DAY_CONFIRM_FRAMES if bright_frames_to_day is None
            else bright_frames_to_day
        ))
        self.dark_pixel_value = int(
            settings.NIGHT_DARK_PIXEL_VALUE if dark_pixel_value is None
            else dark_pixel_value
        )
        self.smoothing = float(
            settings.NIGHT_SMOOTHING if smoothing is None else smoothing
        )

        self._lock = threading.Lock()
        self._condition = SceneCondition()
        self._ema: Optional[float] = None
        self._streak = 0
        self._samples = 0
        self._last_logged: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Measurement
    # ------------------------------------------------------------------ #
    @staticmethod
    def _probe(frame: np.ndarray) -> tuple[float, float, float, float]:
        """
        Measure ``(mean_luma, dark_fraction, mean_saturation, channel_spread)``.

        Downsampled to at most ~160 px wide first.  Illumination is a global
        property, so a thumbnail carries it exactly, and measuring on the
        thumbnail is what keeps this affordable on every frame of every camera.

        ``channel_spread`` is the mean per-pixel ``max(B,G,R) - min(B,G,R)``.
        It is the signal that separates a true monochrome IR image from merely
        desaturated colour footage, and mean saturation alone cannot: measured
        on this project's own daytime clips, saturation is already as low as 7
        while the channel spread is ~3.8, whereas a genuine monochrome sensor
        gives exactly 0 for both.
        """
        height, width = frame.shape[:2]
        if width > 160:
            scale = 160.0 / float(width)
            small = cv2.resize(
                frame, (160, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame

        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean_luma = float(grey.mean())

        dark_value = int(settings.NIGHT_DARK_PIXEL_VALUE)
        dark_fraction = float(np.count_nonzero(grey < dark_value)) / float(grey.size)

        # Saturation and channel spread together identify a monochrome sensor.
        hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
        saturation = float(hsv_s.mean())

        as_int = small.astype(np.int16)
        channel_spread = float(
            (as_int.max(axis=2) - as_int.min(axis=2)).mean()
        )

        return mean_luma, dark_fraction, saturation, channel_spread

    def _score(self, mean_luma: float, dark_fraction: float,
               saturation: float, channel_spread: float) -> tuple[float, bool]:
        """
        Fuse the three signals into a 0-1 darkness score.

        ``dark_fraction`` is weighted most heavily because it is the signal
        that stays correct when a lamp, a headlight or a bright sky patch
        inflates the mean.  Luma contributes a normalised term so a uniformly
        dim scene (no truly black pixels, e.g. heavy fog at night) still
        scores as dark.
        """
        # Normalised luma darkness: 1.0 at pitch black, 0.0 at the configured
        # daylight reference brightness.
        reference = max(1.0, float(settings.NIGHT_DAYLIGHT_LUMA))
        luma_darkness = float(np.clip(1.0 - (mean_luma / reference), 0.0, 1.0))

        score = (settings.NIGHT_WEIGHT_DARK_FRACTION * dark_fraction
                 + settings.NIGHT_WEIGHT_LUMA * luma_darkness)
        total_weight = (settings.NIGHT_WEIGHT_DARK_FRACTION
                        + settings.NIGHT_WEIGHT_LUMA)
        score = score / max(1e-6, total_weight)

        # IR / night-vision: a near-colourless image that is *not* already dark
        # by the brightness signals. Such a camera switched to monochrome
        # because it is dark, so the mode itself is evidence of night however
        # bright its IR illuminator makes the frame look.
        #
        # This is deliberately the explanation of last resort. A pitch-black
        # frame also has zero saturation and zero channel spread, but calling it
        # "infrared" would report the wrong reason for a correct decision — it
        # is simply dark, and the brightness signals already say so.
        infrared = (
            settings.NIGHT_DETECT_INFRARED
            and score < self.enter_threshold
            and saturation < settings.NIGHT_IR_SATURATION_MAX
            and channel_spread < settings.NIGHT_IR_CHANNEL_SPREAD_MAX
            and mean_luma < settings.NIGHT_IR_LUMA_MAX
        )
        if infrared:
            score = max(score, settings.NIGHT_IR_SCORE)

        return float(np.clip(score, 0.0, 1.0)), infrared

    def measure(self, frame: np.ndarray) -> SceneCondition:
        """
        Measure one frame and update the latched night state.

        Returns the current :class:`SceneCondition`.  The latch means the
        returned ``is_night`` can legitimately disagree with this single
        frame's ``darkness`` — that is the anti-flicker behaviour working, and
        it is why a camera flash or a passing headlight cannot toggle night
        analytics.
        """
        mean_luma, dark_fraction, saturation, channel_spread = self._probe(frame)
        raw_score, infrared = self._score(
            mean_luma, dark_fraction, saturation, channel_spread
        )

        # Exponential smoothing over the fused score.
        alpha = float(np.clip(self.smoothing, 0.0, 1.0))
        if self._ema is None:
            self._ema = raw_score
        else:
            self._ema = self._ema + alpha * (raw_score - self._ema)
        score = float(self._ema)

        self._samples += 1

        with self._lock:
            currently_night = self._condition.is_night

            # Hysteresis: the bar to become night is higher than the bar to
            # remain night, so dusk crosses once instead of oscillating.
            if currently_night:
                wants_change = score < self.exit_threshold
                needed = self.bright_frames_to_day
            else:
                wants_change = score >= self.enter_threshold
                needed = self.dark_frames_to_night

            if wants_change:
                self._streak += 1
            else:
                self._streak = 0

            flipped = False
            if wants_change and self._streak >= needed:
                currently_night = not currently_night
                self._streak = 0
                flipped = True

            source = ""
            if currently_night:
                source = "infrared" if infrared else "darkness"

            self._condition = SceneCondition(
                is_night=currently_night,
                source=source,
                darkness=score,
                mean_luma=mean_luma,
                dark_fraction=dark_fraction,
                saturation=saturation,
                channel_spread=channel_spread,
                infrared=infrared,
                streak=self._streak,
                samples=self._samples,
            )
            condition = self._condition

        if flipped and self._last_logged != currently_night:
            self._last_logged = currently_night
            log.info(
                "Scene is now %s — darkness=%.2f luma=%.0f dark_px=%.0f%% "
                "sat=%.0f%s",
                "NIGHT" if currently_night else "DAY",
                score, mean_luma, dark_fraction * 100, saturation,
                " (infrared)" if infrared else "",
            )
        return condition

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def condition(self) -> SceneCondition:
        """The latest latched condition — safe to read from any thread."""
        with self._lock:
            return self._condition

    @property
    def is_night(self) -> bool:
        with self._lock:
            return self._condition.is_night

    @property
    def is_dark_enough_to_enhance(self) -> bool:
        """
        Whether low-light enhancement (CLAHE) is worth its cost.

        Keyed off measured luma rather than the night latch, because an IR
        camera is latched to night but its image is already bright — running
        CLAHE on it would amplify sensor noise for nothing.
        """
        with self._lock:
            return self._condition.mean_luma < settings.LOW_LIGHT_THRESHOLD

    def reset(self) -> None:
        """Forget history — used when a source reconnects or a file loops."""
        with self._lock:
            self._condition = SceneCondition()
            self._ema = None
            self._streak = 0
            self._samples = 0


__all__ = ["SceneCondition", "SceneIlluminationEstimator"]
