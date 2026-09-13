

from __future__ import annotations

import logging
import threading
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

class LowLightEnhancer:
 
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
  
        if not self.is_dark(frame):
            return frame, False
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True


# --------------------------------------------------------------------------- #
# Optional-stage model loading
# --------------------------------------------------------------------------- #

#: Loaders already running, so a dozen camera threads asking for the same model
#: start exactly one load between them.
_loading: set[str] = set()
_loading_lock = threading.Lock()


def _load_in_background(name: str, build) -> None:
    """
    Build an optional model on a thread of its own, once.

    Constructing the face and ANPR stacks costs seconds — measured at ~6.3 s for
    SCRFD+ArcFace and ~5.6 s for EasyOCR, more when weights move onto CUDA. The
    first analysed frame used to pay that *on the camera's analytics thread*,
    inside ``analyse()``, before the cheap "is there even a person in view?"
    test. The effects were all of a piece and all bad: the camera reported
    PROCESSING and ONLINE while publishing nothing for ~12-20 s, so the operator
    saw a dead tile; removing it took five seconds instead of a third of one,
    because the thread could not reach its own ``while self._running`` check
    until the load returned; and the model construction holds the GIL in long
    stretches, which stalls the event loop serving the dashboard, the MJPEG
    streams and every API call. Every one of those reads as "the UI is lagging
    and the camera will not go away".

    So the pipeline never waits for a model. It asks whether one is ready,
    skips its stage if not, and this brings the model up behind it.
    """
    with _loading_lock:
        if name in _loading:
            return
        _loading.add(name)

    def _run() -> None:
        started = time.time()
        try:
            build()
            log.info("%s model ready after %.1fs", name.upper(), time.time() - started)
        except Exception as exc:
            log.error("%s model could not be loaded: %s", name.upper(), exc)
        finally:
            with _loading_lock:
                _loading.discard(name)

    threading.Thread(target=_run, name=f"load-{name}", daemon=True).start()


class _AsyncStage:
    """
    Runs one enrichment tick at a time, off the analytics thread.

    Face recognition and ANPR are *enrichment*: the pipeline already has a
    cache that answers for them between cadence ticks, and every consumer
    re-anchors to the current frame's detections. What it did not have was any
    protection against the tick itself being slow — and on a CPU-only face
    stack a tick is very slow. Measured here: 5-6 s per tick with three people
    in view, on a 15 fps pipeline whose whole frame budget is 66 ms.

    Synchronously, that cost was not paid by the face feature. It was paid by
    the camera (no frames published for seconds at a time), by removal (the
    thread could not reach its stop check), and by the entire server, because
    a stage that holds the GIL in long stretches starves the event loop behind
    the dashboard, the MJPEG streams and every API call. One camera looking at
    people was enough to make the whole platform feel broken.

    So a tick is submitted and the pipeline moves on. One tick may be in flight
    at a time, which is the backpressure: a stage that cannot keep up simply
    runs less often instead of queueing work that is already stale.
    """

    #: One budget per stage *name*, shared by every camera in the process.
    #:
    #: The limiter has to be global, because the resource it protects is. Each
    #: analyzer holding its own 25% budget means eight cameras can consume 200%
    #: of a core between them, and that is not a thought experiment: running
    #: eight streams at once, average inference rose from 33 ms to 229 ms and
    #: aggregate throughput collapsed from 96 fps to 10.6 while every camera's
    #: face stage ticked independently. One budget, and one tick in flight at a
    #: time, keeps the cost of an optional stage flat as cameras are added —
    #: which is the only way "add another camera" stays a safe thing to do.
    _budgets: dict = {}
    _budget_lock = threading.Lock()

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._pending: Any = None
        #: How long the last tick took, measured for diagnostics.
        self._last_seconds = 0.0
        with _AsyncStage._budget_lock:
            _AsyncStage._budgets.setdefault(
                name, {"busy": False, "next_allowed": 0.0, "last_seconds": 0.0})

    @property
    def busy(self) -> bool:
        """True while *any* camera's tick of this stage is running."""
        with _AsyncStage._budget_lock:
            return bool(_AsyncStage._budgets[self._name]["busy"])

    @property
    def last_seconds(self) -> float:
        with _AsyncStage._budget_lock:
            return float(_AsyncStage._budgets[self._name]["last_seconds"])

    def submit(self, work) -> bool:
        """
        Start ``work()`` on a worker, unless a tick of this stage is already
        running anywhere in the process, or the stage has already had its share
        of the machine. Returns whether one started.

        Both gates are process-wide. A camera that loses the race simply serves
        its cached result for another cadence and tries again — which is the
        backpressure: adding cameras spreads a fixed budget more thinly instead
        of multiplying the load.
        """
        now = time.monotonic()
        with _AsyncStage._budget_lock:
            budget = _AsyncStage._budgets[self._name]
            if budget["busy"] or now < budget["next_allowed"]:
                return False
            budget["busy"] = True

        started = time.monotonic()

        def _run() -> None:
            outcome = None
            try:
                outcome = work()
            except Exception as exc:
                log.warning("%s stage failed: %s", self._name, exc)
            finally:
                elapsed = max(0.0, time.monotonic() - started)
                duty = min(1.0, max(0.01, float(settings.STAGE_MAX_DUTY)))
                with self._lock:
                    if outcome is not None:
                        self._pending = outcome
                    self._last_seconds = elapsed
                with _AsyncStage._budget_lock:
                    slot = _AsyncStage._budgets[self._name]
                    slot["last_seconds"] = elapsed
                    # Idle long enough that the stage averages `duty` of wall
                    # time across the whole process. A cheap tick barely delays
                    # the next one; an expensive one backs itself off.
                    slot["next_allowed"] = (
                        time.monotonic() + elapsed * (1.0 / duty - 1.0))
                    slot["busy"] = False

        threading.Thread(target=_run, name=f"stage-{self._name}",
                         daemon=True).start()
        return True

    @classmethod
    def reset_budgets(cls) -> None:
        """Clear every shared gate — used by the hard reset and by tests."""
        with cls._budget_lock:
            for slot in cls._budgets.values():
                slot.update({"busy": False, "next_allowed": 0.0,
                             "last_seconds": 0.0})

    def take(self):
        """The most recent completed tick, once. ``None`` if nothing is new."""
        with self._lock:
            out, self._pending = self._pending, None
            return out


@dataclass
class AnalysisResult:
    """Everything one analysed frame produced."""

    frame: np.ndarray                       # annotated, ready to stream/encode
    #: The unannotated frame. Evidence snapshots, ANPR crops and face crops are
    #: all cut from this one: the overlay burns the camera name, HUD text and
    #: bounding boxes into ``frame``, and an evidence image with a label drawn
    #: across the subject is not what the camera saw. (An OCR probe pointed at
    #: the annotated frame once read the camera's own name back as a plate.)
    raw_frame: np.ndarray                   # unannotated, for evidence snapshots
    detections: list[Detection] = field(default_factory=list)
    alerts: list[RuleAlert] = field(default_factory=list)
    faces: list = field(default_factory=list)
    plates: list = field(default_factory=list)
    timestamp: float = 0.0
    is_night: bool = False
    night_source: str = ""

    scene: Optional[SceneCondition] = None
    enhanced: bool = False
    inference_ms: float = 0.0
    total_ms: float = 0.0
    person_count: int = 0
    vehicle_count: int = 0
    #: The unannotated frame the ANPR tick read, when its results came from an
    #: earlier frame than this one. Plate boxes are in that frame's coordinates,
    #: so the evidence crop has to be cut from it and not from the current one.
    plates_frame: Optional[np.ndarray] = None
    #: Live occupancy per zone / loiter area — the continuous counterpart to the
    #: discrete enter and exit events. One entry per rule that has an inside.
    zones: list = field(default_factory=list)



class FrameAnalyzer:
 

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


        self._announced_tracks: dict[int, float] = {}

        self._announced_classes: dict[str, float] = {}

        self._face_recognizer = None
        self._anpr = None
        #: Enrichment stages, each running at most one tick at a time.
        self._face_stage = _AsyncStage("face")
        self._anpr_stage = _AsyncStage("anpr")
        #: The frame the last completed ANPR tick read.
        self._plates_frame: Optional[np.ndarray] = None

    
    def set_rules(self, rule_rows: list[dict]) -> None:
       
        from cv.rules import DirectionRule, FenceRule, LoiterRule, ZoneRule

        self.rules.clear()
        self._rule_shapes = []

        for row in rule_rows:
            rtype = row.get("rule_type")
            geom = row.get("geometry") or []
            params = row.get("params") or {}
            name = row.get("name") or f"{rtype}_{row.get('id', '?')}"
            
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


    def _face(self):
        """
        The face recogniser, or ``None`` while it is still loading.

        Returning ``None`` costs a few skipped face ticks on a cold start and
        keeps the pipeline running; blocking here cost the whole camera.
        """
        if self._face_recognizer is not None or not self.enable_face:
            return self._face_recognizer

        from cv.face import face_ready, get_face_recognizer

        if not face_ready():
            _load_in_background("face", get_face_recognizer)
            return None
        self._face_recognizer = get_face_recognizer()
        return self._face_recognizer

    def _anpr_processor(self):
        """The ANPR processor, or ``None`` while EasyOCR is still loading."""
        if self._anpr is not None or not self.enable_anpr:
            return self._anpr

        from cv.anpr import anpr_ready, preload_anpr

        if not anpr_ready():
            # preload_anpr, not get_anpr_processor: the expensive part is the
            # EasyOCR reader that is_available() builds, and constructing the
            # processor without it would report "ready" while leaving the real
            # stall in place for the first frame that contains a vehicle.
            _load_in_background("anpr", preload_anpr)
            return None
        from cv.anpr import get_anpr_processor

        self._anpr = get_anpr_processor()
        return self._anpr

 
    def analyse(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        *,
        annotate: bool = True,
        fps: float = 0.0,
        online: bool = True,
    ) -> AnalysisResult:
        
        t_start = time.perf_counter()
        now = time.time() if timestamp is None else float(timestamp)
        self._frame_index += 1

    
        source_frame = frame
        if frame.shape[1] != settings.FRAME_WIDTH or frame.shape[0] != settings.FRAME_HEIGHT:
            frame = cv2.resize(
                frame, settings.frame_size, interpolation=cv2.INTER_LINEAR
            )

        scene = self._night_state(frame)
        night, night_source = scene.is_night, scene.source

        work = frame
        enhanced = False
       
        if scene.mean_luma < settings.LOW_LIGHT_THRESHOLD:
            work, enhanced = self.enhancer.maybe_enhance(frame)

        
        xyxy, conf, cls, inference_ms = self.detector.raw_detect(work)
        detections = self.tracker.update(xyxy, conf, cls, self.detector.names)

        persons = [d for d in detections if d.is_person]
        vehicles = [d for d in detections if d.is_vehicle]

       
        self._faces = self._run_face(work, detections, persons, source_frame)

       
        self._plates = self._run_anpr(work, vehicles, source_frame)

      
        context = {
            "is_night": night,
            "night_source": night_source,
            "scene_darkness": round(scene.darkness, 3),
            "mean_luma": round(scene.mean_luma, 1),
        }
        alerts = self.rules.update(detections, timestamp=now, context=context)
        alerts.extend(self._presence_alerts(detections, now))

        # Occupancy is read straight after the rules ran, from the same state
        # those rules just updated, so the continuous signal and the discrete
        # events can never disagree about who is inside.
        occupancy = self.rules.occupancy(now)
        occupied_rules = {
            entry["rule"] for entry in occupancy if entry.get("occupied")
        }
        breached_rules = {
            entry["rule"] for entry in occupancy if entry.get("breached")
        }

      
        annotated = work
        if annotate:
            annotated = work.copy()

            occupied: list = []
            # Tell the renderer which polygons currently contain something, so
            # an occupied zone is visibly different from an empty one for as
            # long as it stays occupied — not only in the frame where somebody
            # crossed the boundary.
            shapes = [
                {**shape,
                 "occupied": shape.get("name") in occupied_rules,
                 "breached": shape.get("name") in breached_rules}
                for shape in self._rule_shapes
            ]
            ov.draw_rules(annotated, shapes, occupied)
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
            plates_frame=self._plates_frame,
            zones=occupancy,
            person_count=len(persons),
            vehicle_count=len(vehicles),
        )

    def _night_state(self, frame: np.ndarray) -> SceneCondition:
        
        condition = self.scene.measure(frame)

        if settings.FORCE_NIGHT_MODE:
        
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
        # "Is there a person in view?" is a length check; "is the recogniser
        # available?" can start a multi-second model load. Asking them in that
        # order means a feed that never sees a person never pays for face
        # recognition at all.
        if not persons or not self.enable_face:
            return []
        recognizer = self._face()
        if recognizer is None or not getattr(recognizer, "_enabled", False):
            return []

        # Due a tick, and nothing already running? Start one and carry on.
        # The frames are copied because the capture thread reuses its buffers.
        due = (self._frame_index - self._last_face_frame
               >= settings.FACE_RECOGNITION_EVERY_N_FRAMES)
        if due and not self._face_stage.busy:
            work_frame = frame.copy()
            work_source = None if source_frame is None else source_frame.copy()
            work_dets = list(detections or ())
            index, stream = self._frame_index, self.source_id

            def _tick():
                # force=True: the cadence is decided here, so letting the
                # recogniser apply its own would make every other tick a no-op.
                recognizer.recognize(
                    work_frame, detections=work_dets, frame_number=index,
                    force=True, source_frame=work_source, source_id=stream,
                )
                return True

            if self._face_stage.submit(_tick):
                self._last_face_frame = self._frame_index

        # Answer from the identity cache either way. This is not a degraded
        # path: it rebuilds each match against the *current* person boxes, so
        # the overlay, the events and the evidence crop are all in step with
        # the frame being analysed — only the identity is carried forward.
        self._face_stage.take()
        return recognizer.cached_matches(detections, self.source_id)

    def _run_anpr(self, frame: np.ndarray, vehicles: list,
                  source_frame: Optional[np.ndarray] = None) -> list:
        # Same order as the face stage: no vehicle in view means EasyOCR is
        # never even asked for, let alone loaded.
        if not vehicles or not self.enable_anpr:
            return []
        processor = self._anpr_processor()
        if processor is None or not processor.is_available():
            return []

        due = (self._frame_index - self._last_anpr_frame
               >= settings.ANPR_EVERY_N_FRAMES)
        if due and not self._anpr_stage.busy:
            work_frame = frame.copy()
            work_source = None if source_frame is None else source_frame.copy()
            work_vehicles = list(vehicles)
            index, stream = self._frame_index, self.source_id

            def _tick():
                processor.recognize_plates(
                    work_frame, vehicle_detections=work_vehicles,
                    frame_number=index, source_frame=work_source,
                    source_id=stream,
                )
                # Hand back the pixels the plate boxes belong to; unlike face
                # matches, a plate box is not re-anchored to a current
                # detection, so its evidence crop must come from this frame.
                return work_source if work_source is not None else work_frame

            if self._anpr_stage.submit(_tick):
                self._last_anpr_frame = self._frame_index

        done = self._anpr_stage.take()
        if done is not None:
            self._plates_frame = done
        return processor.cached_detections(self.source_id)

    def _presence_alerts(self, detections: list, now: float) -> list[RuleAlert]:
       
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
       
        self.tracker.reset()
        self.rules.reset_state()
        self._announced_tracks.clear()
        self._announced_classes.clear()
        if reset_scene:
            self.scene.reset()
