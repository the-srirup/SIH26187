"""
Application configuration.

Every tunable in IBVAP lives here — detection thresholds, inference
resolution, processing cadence, rule timings, evidence retention, night
hours, upload limits.  Values are read from environment variables / ``.env``
so an operator can retune a Border Out Post deployment without touching
code.  No magic numbers are scattered through the pipeline.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    PROJECT_NAME: str = "IBVAP"
    VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite:///./alerts.db"

    # ------------------------------------------------------------------ #
    # Media / storage
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    DASHBOARD_DIR: Path = BASE_DIR / "dashboard"
    ALERTS_DIR: Path = BASE_DIR / "alerts"
    CLIPS_DIR: Path = BASE_DIR / "clips"
    SNAPSHOTS_DIR: Path = ALERTS_DIR / "snapshots"
    VIDEOS_DIR: Path = BASE_DIR / "videos"                    # uploaded sources
    PROCESSED_DIR: Path = BASE_DIR / "videos" / "processed"    # annotated renders
    #: MP4 files registered as live camera sources. Kept separate from the
    #: offline-analysis uploads so removing such a camera can safely delete its
    #: own file without ever touching an evidence artefact or another session.
    SOURCES_DIR: Path = BASE_DIR / "videos" / "sources"
    LOG_DIR: Path = BASE_DIR / "logs"

    #: Default camera feed used when seeding a fresh install.
    DEFAULT_CAMERA_URL: str = ""

    # ------------------------------------------------------------------ #
    # Detection model
    # ------------------------------------------------------------------ #
    MODEL_PATH: str = "yolo11n.pt"
    #: "auto" picks CUDA when available and falls back to CPU cleanly.
    DEVICE: str = "auto"
    #: FP16 on CUDA only; ignored on CPU (torch CPU has no fast fp16 path).
    USE_HALF: bool = True
    #: Inference letterbox size. Benchmarked on this pipeline (RTX 4060, yolo11n,
    #: 640x384 input): 640 gave 0.38 detections/frame vs 0.32 at 480 — double the
    #: vehicle recall — at a *lower* median latency (17.4 ms vs 19.3 ms), because
    #: the model is small enough that fixed overhead dominates and 640 needs no
    #: awkward letterbox padding. On a CPU-only host 640 costs ~1.8x the pixels;
    #: set INFERENCE_IMGSZ=480 in .env there.
    INFERENCE_IMGSZ: int = 640
    #: Below 0.30 the extra detections were marginal and added flicker; above it
    #: distant vehicles were dropped.
    DEFAULT_CONFIDENCE: float = 0.30
    NMS_IOU: float = 0.45
    MAX_DETECTIONS: int = 50
    #: Batch frames from concurrent cameras into one forward pass. Measured on
    #: an RTX 4060 with yolo11n@640: batch 4 costs 21.0 ms total (5.25 ms per
    #: frame) versus 4 x 17.5 ms served one at a time — a 3.3x throughput gain,
    #: because the per-call launch overhead dominates a model this small.
    #: The batcher never waits to fill a batch, so single-camera latency is
    #: unchanged. Set false to force strictly sequential inference.
    INFERENCE_BATCHING: bool = True
    INFERENCE_BATCH_MAX: int = 8
    #: COCO ids kept for border surveillance:
    #: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 6 train, 7 truck, 8 boat
    DETECT_CLASSES: list[int] = [0, 1, 2, 3, 5, 6, 7, 8]
    #: Minimum box area (px^2) below which a detection is ignored.
    MIN_OBJECT_AREA: int = 400

    # ------------------------------------------------------------------ #
    # Tracking (ByteTrack — one independent instance per video source)
    # ------------------------------------------------------------------ #
    TRACK_HIGH_THRESH: float = 0.45
    TRACK_LOW_THRESH: float = 0.10
    #: Confidence needed to start a *new* track. Was 0.50, which is above the
    #: score a motion-blurred car or scooter typically gets — such an object was
    #: detected but never tracked, so it had no identity, triggered no rule and
    #: appeared to the operator as a missed detection. 0.32 sits just above the
    #: 0.30 detector floor so blurred fast movers get an identity immediately.
    NEW_TRACK_THRESH: float = 0.32
    #: How long a lost track survives before it is discarded, in SECONDS.
    #: Configured as a duration because that is the actual intent — a subject
    #: walking behind a truck is occluded for a couple of seconds regardless of
    #: the analytics cadence. It is converted to a frame count per stream.
    TRACK_LOST_SECONDS: float = 3.0
    TRACK_BUFFER_MIN_FRAMES: int = 30
    #: Retained for backward compatibility with existing .env files; the
    #: effective value is derived from TRACK_LOST_SECONDS.
    TRACK_BUFFER: int = 45
    #: ByteTrack association gate. The cost is ``1 - IoU*score``, so 0.85 needed
    #: ``IoU*score > 0.15`` — for a vehicle that moves more than its own length
    #: between analysed frames IoU is 0 and association was impossible, which
    #: is what produced 20 new track IDs in 50 seconds of sample footage.
    #: Raising the gate lets the Kalman prediction carry fast objects; the
    #: prediction (not raw overlap) then does the matching.
    MATCH_THRESH: float = 0.92
    #: Exponential smoothing factor for drawn boxes (0 = off, 1 = no smoothing).
    #: Applied to rendering only — rule geometry uses the raw foot point.
    BOX_SMOOTHING: float = 0.55

    # ------------------------------------------------------------------ #
    # Video pipeline
    # ------------------------------------------------------------------ #
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 384
    #: Analytics cadence. The capture thread always runs at source speed and
    #: keeps only the newest frame, so raising this never builds a backlog.
    TARGET_FPS: int = 15
    #: Seconds without a decoded frame before a source is flagged OFFLINE.
    CAMERA_TIMEOUT: float = 6.0
    RECONNECT_INTERVAL: float = 4.0
    #: Loop finite video files (demo behaviour for the sample clip).
    LOOP_FILE_SOURCES: bool = True
    #: JPEG quality for the MJPEG stream (encoded once per frame, shared).
    JPEG_QUALITY: int = 72
    #: How far above its nominal frame rate a LIVE source may be decoded.
    #:
    #: A buffered network stream (YouTube Live HLS especially) hands FFmpeg
    #: whole segments at once, and an unpaced capture loop will decode them as
    #: fast as the CPU allows — measured at 370-750 fps on a 30 fps stream,
    #: which starved the analytics thread feeding off it down to 1.5 fps and
    #: made the whole dashboard sluggish. Headroom above 1.0 still lets a source
    #: that fell behind sprint back to the live edge.
    LIVE_CAPTURE_HEADROOM: float = 1.5

    # ------------------------------------------------------------------ #
    # Night detection — VISUAL, not clock-based
    # ------------------------------------------------------------------ #
    # Night is decided from what the camera actually sees (see cv/scene.py).
    # The old build asked the clock: at 23:00 every camera was declared "night"
    # including one aimed at a floodlit checkpost, and a camera inside an unlit
    # culvert was declared "day" at noon. The clock is not an observation.
    #
    # Three measured signals are fused into a 0-1 darkness score:
    # dark-pixel fraction (weighted highest — it survives a streetlamp or
    # headlights inflating the mean), normalised mean luma, and colour
    # saturation / channel spread to recognise IR night-vision.
    #
    # Calibrated against this project's own footage: real daytime clips here
    # score 0.00-0.05, a dark scene with a bright streetlamp scores ~0.90, and
    # pitch black scores 1.00 — so the 0.55 threshold has a wide margin on
    # both sides rather than sitting between two adjacent measurements.

    #: Darkness score at or above which the scene becomes night.
    NIGHT_DARKNESS_ENTER: float = 0.55
    #: Score below which night is released. Lower than ENTER on purpose: the
    #: gap is hysteresis, and it is what stops dusk from flickering for twenty
    #: minutes between the two states.
    NIGHT_DARKNESS_EXIT: float = 0.40
    #: Consecutive agreeing frames before the night latch flips. One dark frame
    #: (a truck's shadow, an auto-exposure hunt) must never arm night analytics.
    NIGHT_CONFIRM_FRAMES: int = 12
    #: Consecutive bright frames before night is released — deliberately slower
    #: than arming it, so sweeping headlights cannot disarm night analytics.
    DAY_CONFIRM_FRAMES: int = 20
    #: A pixel below this grey value counts toward the dark-pixel fraction.
    NIGHT_DARK_PIXEL_VALUE: int = 50
    #: Mean luma treated as full daylight when normalising the luma term.
    NIGHT_DAYLIGHT_LUMA: float = 110.0
    #: Relative weights of the two brightness signals.
    NIGHT_WEIGHT_DARK_FRACTION: float = 0.65
    NIGHT_WEIGHT_LUMA: float = 0.35
    #: EMA factor applied to the fused score (0 = frozen, 1 = no smoothing).
    NIGHT_SMOOTHING: float = 0.25

    #: Recognise IR / night-vision cameras, which output a *bright* but
    #: colourless image that mean luma alone reads as daylight.
    NIGHT_DETECT_INFRARED: bool = True
    #: A true monochrome sensor gives saturation ~0 and per-pixel channel
    #: spread ~0. Measured on this project's footage, genuinely desaturated
    #: *colour* daylight still reads saturation ~7 and spread ~3.8, so these
    #: ceilings separate the two cases with margin instead of guessing.
    NIGHT_IR_SATURATION_MAX: float = 3.0
    NIGHT_IR_CHANNEL_SPREAD_MAX: float = 1.5
    NIGHT_IR_LUMA_MAX: float = 160.0
    #: Darkness score attributed to a confirmed IR frame.
    NIGHT_IR_SCORE: float = 0.75

    #: Optional *hint* only — never a trigger. When true, the clock window can
    #: nudge a borderline scene, but a bright scene is never called night.
    NIGHT_USE_CLOCK_HINT: bool = False
    #: How much the clock hint may lower the enter threshold, at most.
    NIGHT_CLOCK_HINT_BONUS: float = 0.10
    NIGHT_START_HOUR: int = 19       # IST — only used for the optional hint
    NIGHT_END_HOUR: int = 6          # IST

    #: Mean luma below which CLAHE enhancement kicks in.
    LOW_LIGHT_THRESHOLD: float = 80.0
    CLAHE_CLIP_LIMIT: float = 2.5
    #: Force night analytics regardless of the scene — for demonstrating the
    #: night rule with daytime footage. Reported as night_source="forced" so
    #: the dashboard never passes a forced state off as a measurement.
    FORCE_NIGHT_MODE: bool = False

    #: Night-movement rule: sustained travel required, measured within a
    #: rolling window so a subject shuffling for ten minutes does not slowly
    #: accumulate its way to an alert.
    NIGHT_MOVEMENT_MIN_TRAVEL: float = 45.0    # px of travel before alerting
    NIGHT_MOVEMENT_WINDOW: float = 6.0         # seconds the travel must occur in
    #: Net displacement required, as a fraction of NIGHT_MOVEMENT_MIN_TRAVEL.
    #: Path length alone is satisfied by a couple of pixels of box jitter
    #: accumulating over a second, which would report a parked vehicle as
    #: movement; requiring net displacement means the object actually crossed
    #: some of the scene.
    NIGHT_MOVEMENT_NET_RATIO: float = 0.6
    NIGHT_MOVEMENT_DEBOUNCE: float = 45.0      # seconds before a track re-alerts
    #: Alternative, rate-based qualifier for the night-movement rule.
    #:
    #: A pure distance threshold measured over a 6 s window silently encodes
    #: "slow, long-lived subject". Measured on this project's night footage a
    #: person is tracked 4.2 s and travels 338 px, while a car — detected at
    #: *higher* confidence — is tracked 0.8 s and travels 37 px, and was
    #: rejected. That is the whole of "night detection works for people but not
    #: vehicles": a vehicle crosses the frame faster than the rule watches it.
    #:
    #: 40 px/s of NET displacement is motion no jitter can fake (jitter
    #: inflates path length, not net), and the two floors below keep a one-frame
    #: flicker from qualifying.
    NIGHT_MOVEMENT_MIN_SPEED: float = 40.0         # px/s of net displacement
    NIGHT_MOVEMENT_MIN_OBSERVATION: float = 0.4    # s the track must be watched
    NIGHT_MOVEMENT_MIN_NET_FLOOR: float = 20.0     # px net, absolute floor

    # ------------------------------------------------------------------ #
    # Rules / analytics
    # ------------------------------------------------------------------ #
    #: Global per-(rule, track, event) cooldown — the alert debouncer.
    DEBOUNCE_SECONDS: float = 12.0
    #: Consecutive confirmations before a *zone* rule believes an entry, in
    #: frames of the analytics cadence. Crossing rules do NOT use this: a
    #: crossing is an instantaneous geometric event, and requiring N further
    #: frames of persistence is exactly why fast vehicles were never reported.
    ANCHOR_CONFIRMATION_FRAMES: int = 3

    # -- crossing rules (tripwire / direction) ------------------------- #
    #: Minimum travel (px) between two observations for a crossing to count.
    #: Rejects sub-pixel box jitter on a subject standing on the line.
    CROSSING_MIN_DISPLACEMENT: float = 4.0
    #: A track must be this many frames old before it can trigger a crossing,
    #: so a one-frame false positive appearing across the line is ignored.
    CROSSING_MIN_TRACK_AGE: int = 2
    #: Seconds before the same track may report the same crossing direction
    #: again — stops a subject loitering astride the line from machine-gunning
    #: events, while a genuine cross-back in the other direction still fires.
    CROSSING_REARM_SECONDS: float = 3.0
    #: Ignore a crossing inferred across a gap longer than this (seconds). A
    #: track reacquired after a long occlusion may have crossed and returned,
    #: so claiming a single crossing would be a guess, not an observation.
    CROSSING_MAX_GAP_SECONDS: float = 2.0

    # -- zone rules (restricted area) ---------------------------------- #
    #: Dead band (px) around a zone boundary. Inside it, a rule holds its
    #: previous opinion instead of forming a new one — this is what stops the
    #: enter/exit flapping that filled the old event log with pairs of
    #: contradictory alerts a fraction of a second apart.
    ZONE_BOUNDARY_MARGIN: float = 6.0
    #: How long a track must be continuously outside before an exit is
    #: declared. Absorbs a missed detection or a one-frame box wobble.
    ZONE_EXIT_GRACE_SECONDS: float = 1.5
    #: Restricted-zone presence threshold (seconds) before a dwell alert.
    ZONE_PRESENCE_SECONDS: float = 5.0
    #: Emit the low-severity "left the zone" event. Kept ON: with the exit
    #: grace period in place this is now one informative event per genuine
    #: departure (carrying the dwell time), not the flapping noise it used to
    #: be — suppressing it would hide a legitimate event rather than fix one.
    ZONE_EXIT_ALERTS_ENABLED: bool = True

    # -- loitering ------------------------------------------------------ #
    LOITER_SECONDS: float = 15.0
    #: A tracked subject may leave the loiter zone for this long without the
    #: dwell timer resetting. The old rule discarded its state the instant a
    #: foot point fell outside, so a single jittery frame at the boundary reset
    #: the clock to zero and the threshold was effectively never reached.
    LOITER_EXIT_GRACE_SECONDS: float = 3.0
    #: Re-alert interval for a subject that keeps loitering past the first
    #: alert. 0 disables re-alerting (one event per visit).
    LOITER_REALERT_SECONDS: float = 120.0
    #: Object classes the loiter rule considers. Loitering is a human
    #: behaviour; a parked car should not be reported as loitering.
    LOITER_CLASSES: list[str] = ["person"]

    #: Presence/first-sighting events for plain human & vehicle detections.
    PRESENCE_ALERTS_ENABLED: bool = True
    PRESENCE_MIN_CONFIDENCE: float = 0.55
    PRESENCE_DEBOUNCE_SECONDS: float = 60.0
    #: Suppress a presence event unless the scene has been quiet for this long
    #: for that class. Track IDs churn when objects occlude each other, and a
    #: per-track debounce alone therefore cannot stop the flood; this is a
    #: per-(camera, class) floor that does not depend on ID stability.
    PRESENCE_CLASS_COOLDOWN_SECONDS: float = 20.0
    #: Drop rule state for tracks unseen for this long (prevents dict growth).
    TRACK_STATE_TTL: float = 120.0

    # -- per-event-type cooldowns -------------------------------------- #
    #: Overrides DEBOUNCE_SECONDS per alert type. Tuned so a single subject
    #: cannot produce a wall of events while genuinely distinct incidents are
    #: still all reported. Anything absent falls back to DEBOUNCE_SECONDS.
    ALERT_COOLDOWNS: dict[str, float] = {
        "entry": 4.0,
        "exit": 4.0,
        "enter": 8.0,
        "zone_exit": 8.0,
        "zone_presence": 30.0,
        "loiter": 30.0,
        "wrong_direction": 6.0,
        "night_movement": 45.0,
        "human_detected": 20.0,
        "vehicle_detected": 20.0,
        "anpr_detection": 30.0,
        "face_detected": 60.0,
        "watchlist_match": 45.0,
        "camera_offline": 120.0,
    }

    # ------------------------------------------------------------------ #
    # Evidence
    # ------------------------------------------------------------------ #
    EVIDENCE_ENABLED: bool = True
    #: Rolling pre-alert buffer, in seconds of analytics-rate frames.
    CLIP_PRE_SECONDS: float = 4.0
    CLIP_POST_SECONDS: float = 4.0
    #: Retention guard — oldest evidence is pruned past these caps.
    MAX_EVIDENCE_MB: int = 2048
    MAX_EVIDENCE_AGE_DAYS: int = 30
    EVIDENCE_SWEEP_INTERVAL: float = 900.0     # seconds between retention sweeps

    # ------------------------------------------------------------------ #
    # Face detection / watchlist recognition
    # ------------------------------------------------------------------ #
    FACE_ENABLED: bool = True
    FACE_RECOGNITION_EVERY_N_FRAMES: int = 12
    FACE_DET_SIZE: int = 320          # SCRFD input; 640 is 4x the pixels
    FACE_MIN_HEIGHT: int = 32
    FACE_MATCH_CACHE_SECONDS: float = 30.0
    FACE_SIMILARITY_THRESHOLD: float = 0.55
    FACE_ALERT_DEBOUNCE_SECONDS: float = 45.0
    #: Max person crops embedded per cadence tick — bounds worst-case latency.
    FACE_MAX_CROPS_PER_TICK: int = 3
    #: Emit a "face detected" event even when the face is not on the watchlist.
    FACE_DETECTION_ALERTS: bool = True
    FACE_DETECT_DEBOUNCE_SECONDS: float = 60.0

    # ------------------------------------------------------------------ #
    # ANPR
    # ------------------------------------------------------------------ #
    ANPR_ENABLED: bool = True
    #: OCR is expensive — only run it on this cadence, and only when a vehicle
    #: is actually present in the frame.
    ANPR_EVERY_N_FRAMES: int = 10
    ANPR_MAX_PLATES_PER_TICK: int = 2
    #: OCR confidence below which a plate is reported as PLATE UNCERTAIN
    #: instead of a fabricated-looking number.
    ANPR_CONFIDENCE_THRESHOLD: float = 0.45
    #: Confidence required before an ANPR event is written to the log.
    ANPR_ALERT_CONFIDENCE: float = 0.55
    #: Candidate plate geometry. These are now *relative* to the region being
    #: searched rather than absolute pixel counts: the old code required a
    #: candidate at least 50x15 px, which at the 640x384 analytics resolution
    #: only matches a vehicle filling most of the frame — so on any normal feed
    #: the candidate list came back empty and ANPR silently did nothing.
    ANPR_MIN_PLATE_AREA: int = 120
    ANPR_MIN_PLATE_WIDTH_FRAC: float = 0.06   # of the searched region's width
    ANPR_MIN_PLATE_WIDTH_PX: int = 18
    ANPR_MIN_PLATE_HEIGHT_PX: int = 7
    #: Upper bounds, also relative to the searched region. A plate is a small
    #: feature of a vehicle; without these the full-frame fallback proposed a
    #: whole-frame "plate" and OCR read the burnt-in overlay text out of it.
    ANPR_MAX_PLATE_WIDTH_FRAC: float = 0.75
    ANPR_MAX_PLATE_HEIGHT_FRAC: float = 0.45
    ANPR_MIN_ASPECT_RATIO: float = 1.8
    ANPR_MAX_ASPECT_RATIO: float = 6.5
    #: Plate crops are upscaled to this height before OCR. A plate is often
    #: 40x12 px at analytics resolution; EasyOCR cannot read glyphs that small,
    #: and the previous build passed the crop through at native size.
    ANPR_PLATE_TARGET_HEIGHT: int = 64
    ANPR_PLATE_MAX_UPSCALE: float = 6.0
    ANPR_ALERT_DEBOUNCE_SECONDS: float = 30.0
    ANPR_LANGUAGES: str = "en"

    #: Temporal consensus. A single frame's OCR is a noisy observation, so reads
    #: of the same tracked vehicle are accumulated and decided by per-character
    #: majority vote (WB12AB1234 / WB12AB1284 / WB12AB1234 -> WB12AB1234).
    ANPR_CONSENSUS_ENABLED: bool = True
    #: Reads of one vehicle required before a plate is published.
    ANPR_MIN_VOTES: int = 2
    #: Reads kept per tracked vehicle when voting.
    ANPR_VOTE_HISTORY: int = 12
    #: Votes older than this (seconds) are discarded — a new vehicle may reuse
    #: a recycled track id.
    ANPR_VOTE_WINDOW_SECONDS: float = 20.0

    # ------------------------------------------------------------------ #
    # Video upload / offline analysis
    # ------------------------------------------------------------------ #
    UPLOAD_MAX_MB: int = 512
    UPLOAD_ALLOWED_EXTENSIONS: list[str] = [".mp4"]
    #: Analyse at most every Nth decoded frame of an uploaded file. 1 = all.
    UPLOAD_FRAME_STRIDE: int = 2
    #: Write an annotated MP4 render of the analysed upload.
    UPLOAD_RENDER_OUTPUT: bool = True
    UPLOAD_MAX_CONCURRENT: int = 1
    #: Uploaded-video sessions retained in memory.
    UPLOAD_KEEP_SESSIONS: int = 25

    # ------------------------------------------------------------------ #
    # Integrity (tamper-evident hash chain)
    # ------------------------------------------------------------------ #
    #: Seconds between Merkle checkpoints over the alert chain.
    CHECKPOINT_INTERVAL: float = 300.0

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    #: CORS origins. "*" is convenient on a closed LAN; set explicit origins
    #: for anything reachable beyond the BOP network.
    CORS_ORIGINS: list[str] = ["*"]
    #: Cap on concurrent MJPEG viewers per camera.
    MAX_STREAM_CLIENTS: int = 12

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.FRAME_WIDTH, self.FRAME_HEIGHT)

    @property
    def clip_pre_frames(self) -> int:
        return max(1, int(self.CLIP_PRE_SECONDS * self.TARGET_FPS))

    @property
    def clip_post_frames(self) -> int:
        return max(1, int(self.CLIP_POST_SECONDS * self.TARGET_FPS))

    #: Structured, dated evidence tree for ANPR crops and face crops:
    #: ``evidence/<kind>/camera_<id>/<YYYY>/<MM>/``. Kept separate from the flat
    #: snapshot directory because these artefacts are produced per detection
    #: rather than per alert, and a single BOP generates tens of thousands a
    #: month — a flat folder makes both retention sweeps and an investigator's
    #: "that afternoon, that camera" search impractical.
    EVIDENCE_DIR: Path = ALERTS_DIR / "evidence"

    @property
    def evidence_roots(self) -> tuple[Path, ...]:
        """Directories the API is permitted to serve files from."""
        return (
            self.SNAPSHOTS_DIR.resolve(),
            self.CLIPS_DIR.resolve(),
            self.EVIDENCE_DIR.resolve(),
            self.VIDEOS_DIR.resolve(),
            self.PROCESSED_DIR.resolve(),
            self.SOURCES_DIR.resolve(),
        )

    def ensure_dirs(self) -> None:
        """Create runtime directories that don't yet exist."""
        for d in (
            self.ALERTS_DIR, self.CLIPS_DIR, self.SNAPSHOTS_DIR,
            self.EVIDENCE_DIR, self.VIDEOS_DIR, self.PROCESSED_DIR,
            self.SOURCES_DIR, self.STATIC_DIR, self.LOG_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
