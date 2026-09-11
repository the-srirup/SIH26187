"""
The shared analytics core.

``FrameAnalyzer`` is the one place where a frame becomes intelligence:

    frame -> preprocess -> detect -> track -> face -> ANPR -> rules -> overlay

A live camera thread and an uploaded-MP4 analysis run both drive *this same
object*, which is what stops the two paths from drifting apart.  The only
difference between them is who supplies the frames and what "now" means:
live analytics use wall-clock time, offline analysis uses media time.

Cost control is explicit rather than accidental:

* YOLO runs on every analysed frame (it is the cheapest useful signal).
* Face recognition runs on a cadence **and** only when a person is present,
  with a hard cap on crops per tick.
* ANPR runs on a cadence **and** only when a vehicle is present — the previous
  build ran EasyOCR on every single frame, including frames with no vehicle
  in them at all.
* CLAHE runs only when the frame is actually dark, and the darkness test is
  done on a downsampled grayscale rather than a full BGR→LAB conversion.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from core.config import settings
from core.timeutil import is_night as clock_is_night
from cv import overlay as ov
from cv.detector import Detection, Detector, ObjectTracker
from cv.rules import Alert as RuleAlert, NightMovementRule, RuleEngine
from cv.scene import SceneCondition, SceneIlluminationEstimator

log = logging.getLogger("ibvap.analytics")


# --------------------------------------------------------------------------- #
# Low-light enhancement
# --------------------------------------------------------------------------- #


class LowLightEnhancer:
    """
    CLAHE on the L channel, applied only when the frame is genuinely dark.

    The darkness probe runs on a 1/8-scale grayscale image (~0.05 ms) instead
    of converting the whole frame to LAB just to read its mean (~1.5 ms).  On
    a bright daytime feed that is the difference between paying 1.5 ms every
    frame forever and paying almost nothing.
    """

    def __init__(
        self,
        clip_limit: Optional[float] = None,
        tile_grid_size: tuple[int, int] = (8, 8),
        luminance_threshold: Optional[float] = None,
    ) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit if clip_limit is not None else settings.CLAHE_CLIP_LIMIT,
            tileGridSize=tile_grid_size,
        )
        self.luminance_threshold = (
            luminance_threshold if luminance_threshold is not None
            else settings.LOW_LIGHT_THRESHOLD
        )
        self.last_luma = 255.0

    def measure_luma(self, frame: np.ndarray) -> float:
        small = cv2.resize(frame, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST)
        self.last_luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        return self.last_luma

    def is_dark(self, frame: Optional[np.ndarray] = None) -> bool:
        luma = self.measure_luma(frame) if frame is not None else self.last_luma
        return luma < self.luminance_threshold

    def maybe_enhance(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns ``(frame, was_enhanced)``."""
        if not self.is_dark(frame):
            return frame, False
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisResult:
    """Everything one analysed frame produced."""

    frame: np.ndarray                       # annotated, ready to stream/encode
    raw_frame: np.ndarray                   # unannotated, for evidence snapshots
    detections: list[Detection] = field(default_factory=list)
    alerts: list[RuleAlert] = field(default_factory=list)
    faces: list = field(default_factory=list)
    plates: list = field(default_factory=list)
    timestamp: float = 0.0
    is_night: bool = False
    night_source: str = ""
    #: Full illumination measurement behind the night decision, so the UI can
    #: show *why* a camera is in night mode instead of asserting that it is.
    scene: Optional[SceneCondition] = None
    enhanced: bool = False
    inference_ms: float = 0.0
    total_ms: float = 0.0
    person_count: int = 0
    vehicle_count: int = 0


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #


class FrameAnalyzer:
    """
    One analytics context — one camera, or one uploaded-video session.

    Owns the per-stream tracker and rule state.  The YOLO model itself is
    shared process-wide via :meth:`Detector.get`.
    """

    def __init__(
        self,
        source_id: str,
        display_name: str = "",
        detector: Optional[Detector] = None,
        enable_face: Optional[bool] = None,
        enable_anpr: Optional[bool] = None,
        frame_rate: Optional[int] = None,
    ) -> None:
        self.source_id = str(source_id)
        self.display_name = display_name or f"SOURCE {source_id}"
        self.detector = detector or Detector.get()
        self.tracker = ObjectTracker(frame_rate=frame_rate)
        self.rules = RuleEngine(camera_id=self.source_id)
        self.enhancer = LowLightEnhancer()
        #: Night is decided from this camera's own pixels, never from the clock.
        self.scene = SceneIlluminationEstimator()

        self.enable_face = settings.FACE_ENABLED if enable_face is None else enable_face
        self.enable_anpr = settings.ANPR_ENABLED if enable_anpr is None else enable_anpr

        self._frame_index = 0
        self._last_face_frame = -10_000
        self._last_anpr_frame = -10_000
        self._faces: list = []
        self._plates: list = []
        self._rule_shapes: list[dict] = []
        self._night_rule: Optional[NightMovementRule] = None

        # Per-track first-sighting bookkeeping for presence events.
        self._announced_tracks: dict[int, float] = {}
        #: Per-class floor. Track ids churn whenever objects occlude one another,
        #: so a purely per-track debounce cannot stop the flood: the old build
        #: issued a fresh "HUMAN DETECTED" for every new id, and the event log
        #: shows 20 ids inside 50 seconds of sample footage.
        self._announced_classes: dict[str, float] = {}

        self._face_recognizer = None
        self._anpr = None

    # ------------------------------------------------------------------ #
    # Rule configuration
    # ------------------------------------------------------------------ #
    def set_rules(self, rule_rows: list[dict]) -> None:
        """
        Rebuild the rule engine from plain dicts.

        Callers pass already-deserialised rows so the analytics loop never
        touches the database — the previous build ran a SQL query *per frame*
        just to redraw the fence.
        """
        from cv.rules import DirectionRule, FenceRule, LoiterRule, ZoneRule

        self.rules.clear()
        self._rule_shapes = []

        for row in rule_rows:
            rtype = row.get("rule_type")
            geom = row.get("geometry") or []
            params = row.get("params") or {}
            name = row.get("name") or f"{rtype}_{row.get('id', '?')}"
            # Operator-tunable knobs travel in the rule's params JSON, so a
            # dwell threshold or a reference point can be changed per rule from
            # the UI without touching global configuration. Anything absent
            # falls back to the configured default.
            reference = params.get("reference_point") or "foot"
            classes = params.get("classes") or None
            try:
                if rtype == "line" and len(geom) >= 2:
                    self.rules.add_rule(FenceRule(
                        name, *geom[0], *geom[1],
                        reference=reference, classes=classes,
                        rearm_seconds=params.get("rearm_seconds"),
                    ))
                elif rtype == "zone" and len(geom) >= 3:
                    self.rules.add_rule(ZoneRule(
                        name, geom, params.get("presence_seconds"),
                        reference=reference, classes=classes,
                        margin=params.get("boundary_margin"),
                        exit_grace=params.get("exit_grace_seconds"),
                        exit_alerts=params.get("exit_alerts"),
                    ))
                elif rtype == "loiter" and len(geom) >= 3:
                    self.rules.add_rule(LoiterRule(
                        name, geom, params.get("dwell_seconds"),
                        reference=reference, classes=classes,
                        margin=params.get("boundary_margin"),
                        exit_grace=params.get("exit_grace_seconds"),
                        realert_seconds=params.get("realert_seconds"),
                    ))
                elif rtype == "direction" and len(geom) >= 2:
                    self.rules.add_rule(DirectionRule(
                        name, *geom[0], *geom[1],
                        allowed_direction=params.get("allowed_direction", "entry"),
                        reference=reference, classes=classes,
                        rearm_seconds=params.get("rearm_seconds"),
                    ))
                else:
                    continue
            except Exception as exc:
                log.warning("[%s] skipping malformed rule %s: %s", self.source_id, name, exc)
                continue

            self._rule_shapes.append(
                {"type": rtype, "name": name, "geometry": geom,
                 "params": params, "active": True}
            )

        # Night movement is a standing rule, not an operator-drawn shape.
        self._night_rule = NightMovementRule()
        try:
            self.rules.add_rule(self._night_rule)
        except ValueError:
            pass

        log.info("[%s] %d rule(s) armed", self.source_id, len(self._rule_shapes))

    @property
    def rule_shapes(self) -> list[dict]:
        return list(self._rule_shapes)

    # ------------------------------------------------------------------ #
    # Lazily-loaded optional subsystems
    # ------------------------------------------------------------------ #
    def _face(self):
        if self._face_recognizer is None and self.enable_face:
            from cv.face import get_face_recognizer

            self._face_recognizer = get_face_recognizer()
        return self._face_recognizer

    def _anpr_processor(self):
        if self._anpr is None and self.enable_anpr:
            from cv.anpr import get_anpr_processor

            self._anpr = get_anpr_processor()
        return self._anpr

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def analyse(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        *,
        annotate: bool = True,
        fps: float = 0.0,
        online: bool = True,
    ) -> AnalysisResult:
        """Run the full pipeline over one frame."""
        t_start = time.perf_counter()
        now = time.time() if timestamp is None else float(timestamp)
        self._frame_index += 1

        # -- preprocess: resize to the analytics resolution --------------- #
        # The pre-resize frame is retained (by reference — no copy) purely so
        # ANPR can crop plate pixels from it. Detection and tracking are happy
        # at 640x384, but OCR accuracy on small text is governed by glyph
        # height, and a 1080p feed carries three times the linear detail that
        # the resize is about to discard.
        source_frame = frame
        if frame.shape[1] != settings.FRAME_WIDTH or frame.shape[0] != settings.FRAME_HEIGHT:
            frame = cv2.resize(
                frame, settings.frame_size, interpolation=cv2.INTER_LINEAR
            )

        scene = self._night_state(frame)
        night, night_source = scene.is_night, scene.source

        work = frame
        enhanced = False
        # Enhance based on measured brightness, not on the night latch: an IR
        # camera is latched to night but its image is already bright, and running
        # CLAHE on it would only amplify sensor noise.
        if scene.mean_luma < settings.LOW_LIGHT_THRESHOLD:
            work, enhanced = self.enhancer.maybe_enhance(frame)

        # -- detect + track ----------------------------------------------- #
        xyxy, conf, cls, inference_ms = self.detector.raw_detect(work)
        detections = self.tracker.update(xyxy, conf, cls, self.detector.names)

        persons = [d for d in detections if d.is_person]
        vehicles = [d for d in detections if d.is_vehicle]

        # -- face (cadenced, person-gated) -------------------------------- #
        # The pre-resize frame goes to the face stage for the same reason it
        # goes to ANPR: a face is ~1/7 of a person, so at 640x384 it is barely a
        # dozen pixels tall and SCRFD never had anything to work with.
        self._faces = self._run_face(work, detections, persons, source_frame)

        # -- ANPR (cadenced, vehicle-gated) ------------------------------- #
        self._plates = self._run_anpr(work, vehicles, source_frame)

        # -- rules --------------------------------------------------------- #
        context = {
            "is_night": night,
            "night_source": night_source,
            "scene_darkness": round(scene.darkness, 3),
            "mean_luma": round(scene.mean_luma, 1),
        }
        alerts = self.rules.update(detections, timestamp=now, context=context)
        alerts.extend(self._presence_alerts(detections, now))

        # -- overlay -------------------------------------------------------- #
        annotated = work
        if annotate:
            annotated = work.copy()
            # One shared collision map for the frame, so a fence label, a box
            # label and a plate label can never render on top of each other.
            occupied: list = []
            ov.draw_rules(annotated, self._rule_shapes, occupied)
            ov.draw_detections(
                annotated, detections, {a.track_id for a in alerts}, occupied
            )
            if self._faces:
                ov.draw_faces(annotated, self._faces, occupied)
            if self._plates:
                ov.draw_plates(annotated, self._plates, occupied)
            ov.draw_hud(
                annotated,
                camera_name=self.display_name,
                fps=fps,
                detections=len(detections),
                online=online,
                night=night,
                timestamp=None,
            )
            if alerts:
                top = max(alerts, key=lambda a: a.severity == "CRITICAL")
                ov.draw_event_banner(
                    annotated,
                    f"{top.alert_type.replace('_', ' ').upper()} — TRACK #{top.track_id}",
                    top.severity,
                )

        return AnalysisResult(
            frame=annotated,
            raw_frame=work,
            detections=detections,
            alerts=alerts,
            faces=list(self._faces),
            plates=list(self._plates),
            timestamp=now,
            is_night=night,
            night_source=night_source,
            scene=scene,
            enhanced=enhanced,
            inference_ms=inference_ms,
            total_ms=(time.perf_counter() - t_start) * 1000.0,
            person_count=len(persons),
            vehicle_count=len(vehicles),
        )

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    def _night_state(self, frame: np.ndarray) -> SceneCondition:
        """
        Decide whether night analytics apply, from **what the camera sees**.

        This is the inversion of the original logic.  It used to read::

            if clock_is_night(now):        # 23:00 -> night, unconditionally
                return True, "clock"
            if dark(frame):
                return True, "luminance"

        so at 23:00 every camera in the deployment was declared night — a
        floodlit checkpost included — and at noon a camera inside an unlit
        culvert was declared day.  The clock was consulted first and the pixels
        only as a fallback, which is backwards: the scene is the observation and
        the clock is not.

        Now the measurement decides (see :mod:`cv.scene`, which fuses dark-pixel
        fraction, mean luma and colour saturation, then latches the result over
        several frames).  The clock survives only as an optional *hint* that can
        slightly lower the threshold for a borderline scene, and it is disabled
        by default.  A bright scene is never called night, whatever the hour.
        """
        condition = self.scene.measure(frame)

        if settings.FORCE_NIGHT_MODE:
            # Explicit operator override for demonstrating the night rule with
            # daytime footage. Reported honestly as "forced" so the dashboard
            # never presents an override as a measurement.
            return SceneCondition(
                is_night=True, source="forced", darkness=condition.darkness,
                mean_luma=condition.mean_luma, dark_fraction=condition.dark_fraction,
                saturation=condition.saturation, infrared=condition.infrared,
                streak=condition.streak, samples=condition.samples,
            )

        if (not condition.is_night
                and settings.NIGHT_USE_CLOCK_HINT
                and clock_is_night(None, settings.NIGHT_START_HOUR,
                                   settings.NIGHT_END_HOUR)):
            # A hint, not a trigger: it can only promote a scene that is already
            # close to the darkness threshold, never a bright one.
            threshold = settings.NIGHT_DARKNESS_ENTER - settings.NIGHT_CLOCK_HINT_BONUS
            if condition.darkness >= threshold:
                return SceneCondition(
                    is_night=True, source="darkness+clock-hint",
                    darkness=condition.darkness, mean_luma=condition.mean_luma,
                    dark_fraction=condition.dark_fraction,
                    saturation=condition.saturation, infrared=condition.infrared,
                    streak=condition.streak, samples=condition.samples,
                )
        return condition

    def _run_face(self, frame: np.ndarray, detections: list, persons: list,
                  source_frame: Optional[np.ndarray] = None) -> list:
        recognizer = self._face()
        if recognizer is None or not getattr(recognizer, "_enabled", False):
            return []
        if not persons:
            return []
        if self._frame_index - self._last_face_frame < settings.FACE_RECOGNITION_EVERY_N_FRAMES:
            return recognizer.cached_matches(detections, self.source_id)
        self._last_face_frame = self._frame_index
        try:
            return recognizer.recognize(frame, detections=detections,
                                        frame_number=self._frame_index,
                                        source_frame=source_frame,
                                        source_id=self.source_id)
        except Exception as exc:
            log.warning("[%s] face stage failed: %s", self.source_id, exc)
            return []

    def _run_anpr(self, frame: np.ndarray, vehicles: list,
                  source_frame: Optional[np.ndarray] = None) -> list:
        processor = self._anpr_processor()
        if processor is None or not processor.is_available():
            return []
        if not vehicles:
            # No vehicle in frame means no plate. The previous build still ran
            # a full-frame candidate search plus OCR here, every single frame.
            return []
        if self._frame_index - self._last_anpr_frame < settings.ANPR_EVERY_N_FRAMES:
            return processor.cached_detections(self.source_id)
        self._last_anpr_frame = self._frame_index
        try:
            return processor.recognize_plates(
                frame, vehicle_detections=vehicles,
                frame_number=self._frame_index,
                source_frame=source_frame,
                source_id=self.source_id,
            )
        except Exception as exc:
            log.warning("[%s] ANPR stage failed: %s", self.source_id, exc)
            return []

    def _presence_alerts(self, detections: list, now: float) -> list[RuleAlert]:
        """
        First-sighting events for humans and vehicles.

        These are plain detection notifications (``AI DETECTION``), separate
        from rule outcomes, and are debounced per track so a person standing
        in frame produces one event, not one per frame.
        """
        if not settings.PRESENCE_ALERTS_ENABLED:
            return []

        out: list[RuleAlert] = []
        for det in detections:
            if det.confidence < settings.PRESENCE_MIN_CONFIDENCE:
                continue
            if not (det.is_person or det.is_vehicle):
                continue
            # Require the track to be established — suppresses one-frame ghosts.
            if det.age < max(2, settings.ANCHOR_CONFIRMATION_FRAMES):
                continue
            last = self._announced_tracks.get(det.track_id)
            if last is not None and now - last < settings.PRESENCE_DEBOUNCE_SECONDS:
                continue

            kind = "human_detected" if det.is_person else "vehicle_detected"

            # A per-class floor on top of the per-track debounce. Track ids are
            # reissued whenever objects occlude one another, so "one event per
            # track" alone does not bound the event rate — the old log shows a
            # fresh HUMAN DETECTED for each of 20 ids inside 50 seconds. This is
            # a rate limit, not a filter: a genuinely new subject arriving after
            # the cooldown still produces its own event.
            class_last = self._announced_classes.get(kind)
            if (class_last is not None
                    and now - class_last < settings.PRESENCE_CLASS_COOLDOWN_SECONDS):
                continue

            self._announced_tracks[det.track_id] = now
            self._announced_classes[kind] = now
            out.append(
                RuleAlert(
                    rule_name="detection",
                    rule_type="detection",
                    track_id=det.track_id,
                    alert_type=kind,  # type: ignore[arg-type]
                    timestamp=now,
                    description=f"{det.label} #{det.track_id} detected in view",
                    details={
                        "object_class": det.class_name,
                        "confidence": round(det.confidence, 3),
                        "bbox": list(det.bbox),
                    },
                )
            )

        if len(self._announced_tracks) > 512:
            cutoff = now - settings.TRACK_STATE_TTL
            self._announced_tracks = {
                k: v for k, v in self._announced_tracks.items() if v > cutoff
            }
        return out

    def reset_tracking(self, *, reset_scene: bool = False) -> None:
        """
        Restart tracking — used when a source reconnects or a file loops.

        Rule state must be dropped alongside the tracker. A looping video file
        reissues track ids and every object appears to teleport across the frame
        at the seam; a fence rule holding the previous lap's trajectory would
        manufacture a phantom crossing on the first frame of the new lap.

        The illumination estimator is kept by default — a reconnecting camera is
        still pointed at the same scene, so discarding a converged measurement
        would only re-introduce a latch delay. A genuinely new source passes
        ``reset_scene=True``.
        """
        self.tracker.reset()
        self.rules.reset_state()
        self._announced_tracks.clear()
        self._announced_classes.clear()
        if reset_scene:
            self.scene.reset()
