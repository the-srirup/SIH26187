"""
FastAPI application — the HTTP / WebSocket layer for IBVAP.

Concurrency notes (this is where the original build's dashboard lag lived):

* **Streaming is async.**  MJPEG generators are ``async def`` and await a new
  frame.  The previous ``def`` generator ran in Starlette's threadpool and
  held one of its 40 slots *for the life of the connection*; four cameras
  across a few browser tabs starved every other endpoint, because every
  ``def`` route shares that same pool.
* **Frames are encoded once.**  The camera thread encodes one JPEG per frame
  and every viewer receives those bytes.  The old generator re-encoded on a
  33 ms timer per client, mostly re-encoding frames it had already sent.
* **Alerts are pushed, not polled.**  The event manager hands new events to an
  asyncio queue the moment they are sealed, replacing a 2 Hz database poll.
* **Nothing expensive runs on the event loop.**  Inference lives in camera
  threads, video analysis in worker threads, chain verification in the
  threadpool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tempfile
from logging.handlers import RotatingFileHandler
import threading
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cv2
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Query, Request,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core import models
from core.analysis import (
    AnalysisManager, UploadValidationError, ensure_upload_camera,
    rules_for_camera, sanitize_filename,
)
from core.camera import CameraManager, CameraProcessor, FrameBuffer
from core.video_source import PLAYBACK_MAX_SPEED, PLAYBACK_MIN_SPEED
from core.config import settings
from core.database import SessionLocal, init_db
from core.evidence import (
    USAGE_MAX_AGE_SECONDS, evidence_usage, is_safe_evidence_path,
    measure_evidence_usage, sweep_evidence,
)
from core.events import EventManager, serialize_alert_row
from cv.rules import RED_ALERT_SEVERITIES, SEVERITY_BY_TYPE
from core.sources import (
    KIND_FILE, KIND_LIVE, SourceError, fresh_start, get_live_camera, hard_reset,
    reconcile_on_start, register_camera, retire_camera, startup_cameras,
    store_source_video, visible_cameras,
)
from core.hashchain import (
    chain_status, create_checkpoint, export_integrity_certificate_to_json,
    generate_integrity_certificate, latest_chain_hash, list_checkpoints,
    verify_chain, verify_checkpoint,
)
from core.timeutil import (
    fmt_ist, fmt_ist_time, ist_iso, now_ist, now_utc, start_of_ist_day, utc_iso,
)

log = logging.getLogger("ibvap.api")

#: Filled at startup; the loop that WebSocket fan-out is scheduled onto.
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def configure_logging() -> None:
    """
    Operational logging with IST timestamps.

    Per-frame chatter is silenced; what remains is what an operator would want
    in an incident review — connections, model load, events, evidence.
    """
    from core.timeutil import IST

    class ISTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):  # noqa: N802 - stdlib API
            from datetime import datetime

            stamp = datetime.fromtimestamp(record.created, IST)
            return stamp.strftime(datefmt or "%H:%M:%S IST")

    handler = logging.StreamHandler()
    handler.setFormatter(ISTFormatter("[%(asctime)s] %(levelname)-7s %(name)-18s %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Rotating, not plain. A ``FileHandler`` never truncates: this project's
        # own log reached 2.6 MB and had no upper bound at all, which on an
        # unattended Border Out Post is a disk that fills months after anyone
        # last looked at it — the slowest and least obvious way for a
        # surveillance system to stop working. Bounded here the same way
        # evidence is bounded by the retention sweep.
        file_handler = RotatingFileHandler(
            settings.LOG_DIR / "ibvap.log",
            maxBytes=settings.LOG_MAX_MB * 1024 * 1024,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            ISTFormatter("[%(asctime)s] %(levelname)-7s %(name)-18s %(message)s",
                         datefmt="%d %b %Y %H:%M:%S IST")
        )
        root.addHandler(file_handler)
    except OSError as exc:
        log.warning("File logging unavailable: %s", exc)

    for noisy in ("ultralytics", "urllib3", "PIL", "matplotlib", "easyocr", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# WebSocket connection manager
# --------------------------------------------------------------------------- #


class ConnectionManager:
    """Fan-out to dashboard clients with per-client backpressure."""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> asyncio.Queue:
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._connections[websocket] = queue
        return queue

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Forget a client and release what it was holding. Idempotent.

        Three things go, and all three matter. The dictionary entry, or the
        fan-out keeps trying to feed a socket nobody is reading — and because
        the dict is keyed by the ``WebSocket`` object, that entry is also the
        last reference keeping the connection alive, so a closed tab would
        never be collected. The queue, which is bounded at 100 messages but
        those messages are alert payloads, so a few abandoned tabs are real
        memory. And the socket itself: a browser that vanished without a close
        frame leaves the server side half-open until something closes it.
        """
        async with self._lock:
            queue = self._connections.pop(websocket, None)
        if queue is not None:
            # Drop any payloads this client never read.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:      # pragma: no cover - race only
                    break
        try:
            await websocket.close()
        except Exception:
            # Already closed, or closing — either way there is nothing to free.
            pass

    def broadcast_threadsafe(self, message: dict) -> None:
        """
        Publish from *any* thread (camera / analysis workers live off-loop).

        A client whose queue is full is skipped rather than awaited, so one
        stalled browser tab can never slow a camera thread.
        """
        loop = _event_loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout, message)
        except RuntimeError:
            pass

    def _fanout(self, message: dict) -> None:
        """
        Hand the message to every client's queue, never blocking on any of them.

        A full queue means that client is not keeping up. Its message is
        dropped rather than awaited: the alternative is one stalled browser tab
        applying backpressure all the way to a camera thread.
        """
        for queue in list(self._connections.values()):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    async def close_all(self) -> None:
        """Disconnect every client — used on shutdown so no socket is left open."""
        async with self._lock:
            sockets = list(self._connections)
        for websocket in sockets:
            await self.disconnect(websocket)

    @property
    def client_count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop

    configure_logging()
    log.info("=" * 66)
    log.info("IBVAP %s — Intelligent Border Video Analytics Platform", settings.VERSION)
    log.info("Startup at %s", fmt_ist())
    log.info("=" * 66)

    _event_loop = asyncio.get_running_loop()
    settings.ensure_dirs()
    init_db()

    EventManager.get().subscribe(ws_manager.broadcast_threadsafe)
    channels = _subscribe_notification_channels()

    # Load every model off the event loop so startup never blocks the server —
    # and so no camera thread is ever the one that pays for a model.
    await asyncio.to_thread(_warm_models)

    db = SessionLocal()
    try:
        # 1. Make the database agree with reality before anything reads it.
        #    Nothing is online until a processor in *this* process says so;
        #    an ungraceful exit leaves the previous run's flags behind.
        reconcile_on_start(db)

        # 2. Optionally begin as though freshly installed.
        if settings.FRESH_START:
            fresh_start(db)

        # 3. Bring the surviving cameras back up.
        if not settings.AUTOSTART_CAMERAS:
            log.info("Camera auto-start disabled (AUTOSTART_CAMERAS=false) — "
                     "the dashboard starts empty")
        else:
            cameras = startup_cameras(db)
            manager = CameraManager.get()
            started = 0
            for cam in cameras:
                # One unreachable source must not prevent the others starting.
                if manager.add_camera(cam) is not None:
                    started += 1
                else:
                    log.warning("Camera %d (%s) could not be started at boot",
                                cam.id, cam.name)
            log.info("Auto-started %d of %d camera(s)", started, len(cameras))
    except Exception as exc:
        log.exception("Camera startup failed: %s", exc)
    finally:
        db.close()

    tasks = [
        asyncio.create_task(_stats_broadcaster()),
        asyncio.create_task(_checkpoint_worker()),
        asyncio.create_task(_evidence_sweeper()),
    ]
    log.info("API ready on http://%s:%d", settings.HOST, settings.PORT)

    try:
        yield
    finally:
        log.info("Shutting down…")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Close dashboard sockets before the cameras go, so no client is left
        # holding a connection to a server that has stopped producing.
        await ws_manager.close_all()
        CameraManager.get().stop_all()
        # Leave the database honest for the next boot. This is best-effort by
        # nature — a killed process never gets here — which is why
        # reconcile_on_start does the same job unconditionally at startup.
        try:
            db = SessionLocal()
            try:
                reconcile_on_start(db)
            finally:
                db.close()
        except Exception as exc:
            log.debug("Could not clear online flags at shutdown: %s", exc)
        for channel, callback in channels:
            # Unsubscribe first: a camera still draining could otherwise seal
            # an event and queue it onto a worker that is already stopping.
            EventManager.get().unsubscribe(callback)
            channel.shutdown()
        log.info("Shutdown complete at %s", fmt_ist())


def _subscribe_notification_channels() -> list:
    """
    Arm the outbound escalation channels the operator has switched on.

    An unconfigured channel is *not* subscribed: there is no point paying the
    severity and cooldown checks on every event to reach a sink that does not
    exist, and a channel left out here still reports its state honestly on
    ``/api/system/notifications``.

    Note which method is subscribed. ``trigger`` / ``handle`` only gate and
    enqueue; the blocking provider call happens on the channel's own worker
    thread. Subscribing a delivery method instead (``SMSNotifier.send``) would
    put an SMS gateway round-trip on the camera analytics thread that sealed
    the event.
    """
    #: ``(channel, subscribed callback)`` — the callback is kept because it is
    #: what ``unsubscribe`` has to be handed back at shutdown, and it is not
    #: always ``channel.handle``.
    channels: list[tuple] = []
    events = EventManager.get()

    if settings.ALARM_ENABLED:
        try:
            from core.alarm import AlarmManager

            alarm = AlarmManager.get()
            alarm.start()
            events.subscribe(alarm.trigger)
            channels.append((alarm, alarm.trigger))
            log.info("Alarm channel armed — sinks: %s", alarm.describe_sinks())
            if not alarm.is_configured():
                log.warning(
                    "ALARM_ENABLED is set but no sink is configured — "
                    "set ALARM_WEBHOOK_URL or ALARM_GPIO_PIN"
                )
        except Exception as exc:
            # A broken escalation channel must cost the feature, never the boot.
            log.error("Alarm channel unavailable: %s", exc)

    if settings.SMS_ENABLED:
        try:
            from core.sms import SMSNotifier

            sms = SMSNotifier.get()
            sms.start()
            events.subscribe(sms.handle)
            channels.append((sms, sms.handle))
            log.info(
                "SMS channel armed — providers: %s, recipients: %d",
                ", ".join(sms.get_status()["providers_ready"]) or "none",
                len(sms.recipients()),
            )
            if not sms.is_configured():
                log.warning(
                    "SMS_ENABLED is set but the channel is not configured — "
                    "check SMS_TO_NUMBER and the provider credentials"
                )
        except Exception as exc:
            log.error("SMS channel unavailable: %s", exc)

    if not channels:
        log.info("No outbound escalation channel enabled (alarm/SMS off)")
    return channels


def _warm_models() -> None:
    """
    Load every model the pipeline can use, before any camera starts.

    YOLO was already loaded here; face and ANPR were not, and that asymmetry
    was the bug. ``FrameAnalyzer`` built those two lazily **inside
    ``analyse()``**, so the cost — measured on this machine at 6.3 s for
    SCRFD+ArcFace and 5.6 s for EasyOCR, more onto CUDA — landed on the
    analytics thread of whichever camera happened to be first. For those ~12-20
    seconds that camera read frames and published none, while reporting
    PROCESSING and ONLINE: a dead tile on a dashboard that insisted the camera
    was fine. Removing it then took seconds rather than milliseconds, because
    the thread could not reach its own stop check, and the loads hold the GIL
    in long stretches, so the event loop behind the dashboard, the MJPEG
    streams and every API call stalled with it. That is the "everything lags
    and the camera will not go away" report, and it recurred on every single
    start of the program.

    The three loads are independent, so they run concurrently: on this machine
    that turns ~18 s of startup into about the cost of the slowest one. A model
    that cannot be loaded costs its feature, never the boot.
    """
    loaders = [("detector", _load_detector)]
    if settings.FACE_ENABLED:
        from cv.face import preload_face

        loaders.append(("face", preload_face))
    if settings.ANPR_ENABLED:
        from cv.anpr import preload_anpr

        loaders.append(("anpr", preload_anpr))

    started = time.time()
    threads = [
        threading.Thread(target=fn, name=f"warm-{name}", daemon=True)
        for name, fn in loaders
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.info("Models ready in %.1fs (%s)", time.time() - started,
             ", ".join(name for name, _ in loaders))


def _load_detector() -> None:
    try:
        from cv.detector import Detector

        Detector.get()
    except Exception as exc:
        log.error("Detector unavailable: %s", exc)


# --------------------------------------------------------------------------- #
# Background tasks
# --------------------------------------------------------------------------- #


async def _stats_broadcaster() -> None:
    """Push live runtime stats to dashboards once a second."""
    while True:
        try:
            await asyncio.sleep(1.0)
            if ws_manager.client_count == 0:
                continue
            manager = CameraManager.get()
            ws_manager.broadcast_threadsafe({
                "type": "stats",
                "data": {
                    **manager.aggregate(),
                    "cameras": manager.stats(),
                    "server_time_ist": fmt_ist_time(),
                    "server_time_iso": ist_iso(),
                },
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("Stats broadcast skipped: %s", exc)


async def _checkpoint_worker() -> None:
    """Seal a Merkle checkpoint over new events at a fixed interval."""
    while True:
        try:
            await asyncio.sleep(settings.CHECKPOINT_INTERVAL)
            checkpoint = await asyncio.to_thread(create_checkpoint)
            if checkpoint:
                ws_manager.broadcast_threadsafe(
                    {"type": "checkpoint", "data": checkpoint.to_dict()}
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Checkpoint worker error: %s", exc)


async def _evidence_sweeper() -> None:
    """
    Enforce evidence retention, and keep the storage figure warm.

    Two jobs on two clocks, which is why this is one loop rather than a sleep
    for the sweep interval.

    Measuring the footprint is one ``stat`` per evidence file — 190-1685 ms on
    a real store of 2,449 files — so it must never happen on a request.
    ``/api/stats`` and ``/api/system/info`` both report it, so both read a
    cached figure instead. But a cache is only useful while it is warm: with
    the refresh tied to the 900-second sweep and the cache valid for 60, one
    request in every fifteen still paid the full walk. Refreshing on the
    cache's own cadence closes that, and the sweep keeps its own slower one.

    Everything runs through ``asyncio.to_thread``, so neither job touches the
    event loop that serves the dashboard.
    """
    # Measure once up front so the very first dashboard load is a cache hit.
    await asyncio.to_thread(measure_evidence_usage)
    refresh_every = max(5.0, USAGE_MAX_AGE_SECONDS * 0.5)
    next_sweep = time.time() + settings.EVIDENCE_SWEEP_INTERVAL
    while True:
        try:
            await asyncio.sleep(refresh_every)
            if time.time() >= next_sweep:
                next_sweep = time.time() + settings.EVIDENCE_SWEEP_INTERVAL
                await asyncio.to_thread(sweep_evidence)
            # Re-measure after a sweep, and on the cache cadence otherwise.
            await asyncio.to_thread(measure_evidence_usage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Evidence sweep error: %s", exc)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="IBVAP — Intelligent Border Video Analytics Platform",
    description=(
        "Software-defined surveillance for existing CCTV: detection, tracking, "
        "virtual fencing, rule-based behaviour analysis, and a tamper-evident "
        "hash-chained event log. All timestamps are Indian Standard Time."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

settings.ensure_dirs()
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


def get_db_session():
    """
    Request-scoped database session, always closed.

    Every synchronous route takes this rather than opening its own session, so
    a connection cannot be leaked by an early ``return`` or a raised
    ``HTTPException`` — the ``finally`` runs either way. Routes that hand work
    to a worker thread open their own session *inside that thread* instead,
    because a Session is not safe to share across threads.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def active_processors() -> dict[int, "CameraProcessor"]:
    """
    The live ``{camera_id: CameraProcessor}`` registry.

    There is exactly one, owned by :class:`~core.camera.CameraManager`, and
    this is a read-only view of it. That single-owner rule is deliberate: a
    second dictionary tracking the same threads is not a safety net, it is a
    way for the two to disagree — and a processor present in one and absent
    from the other is precisely the zombie this whole path exists to prevent.
    The manager registers and starts a processor under one lock, and unregisters
    and signals it under the same lock, so the registry never contains a
    processor that has been stopped.
    """
    return {proc.camera_id: proc for proc in CameraManager.get().list_cameras()}


def _camera_names(db: Session) -> dict[int, str]:
    return {c.id: c.name for c in db.query(models.Camera.id, models.Camera.name).all()}


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """
    Serve the operator console.

    Explicitly un-cached: the browser was happily serving a stale
    ``index.html`` across restarts, so a redeployed dashboard silently kept
    loading the previous build's stylesheet. Static assets are separately
    version-stamped in the HTML.
    """
    index = settings.DASHBOARD_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard not installed")
    return FileResponse(
        index,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"},
    )


@app.get("/health")
def health(db: Session = Depends(get_db_session)):
    """
    Liveness + a one-glance view of every *current* source.

    ``visible_cameras`` — not a raw query — is what keeps this endpoint
    consistent with the rest of the system. This route used to select every
    non-upload row, archived ones included, so a deployment that had removed
    42 cameras reported all 42 back with ``is_active=false`` and ``fps=0``:
    an operator (and a monitoring probe) reading that saw a system apparently
    carrying dozens of dead cameras, and the removals looked like they had
    silently failed. The list is now exactly the cameras the dashboard shows,
    and the runtime lookup is over live processors only, so the cost is
    proportional to what is actually running.
    """
    live = {proc.camera_id: proc for proc in CameraManager.get().list_cameras()}
    cameras = visible_cameras(db)
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": utc_iso(),
        "timestamp_ist": fmt_ist(),
        "timezone": "Asia/Kolkata (IST, UTC+05:30)",
        "cameras_registered": len(cameras),
        "cameras_running": len(live),
        "cameras": [
            {
                "id": c.id,
                "name": c.name,
                "is_active": c.is_active,
                "is_online": c.id in live and live[c.id].is_online,
                "fps": round(live[c.id].stats()["fps"], 1) if c.id in live else 0.0,
            }
            for c in cameras
        ],
    }


@app.get("/api/system/info")
def system_info(db: Session = Depends(get_db_session)):
    """Real device/runtime facts — nothing here is assumed or hardcoded."""
    import sys

    from cv.anpr import get_anpr_processor
    from cv.face import get_face_recognizer

    try:
        from cv.detector import Detector

        detector_metrics = Detector.get().metrics()
    except Exception as exc:
        detector_metrics = {"error": str(exc)}

    manager = CameraManager.get()
    return {
        "version": settings.VERSION,
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "detector": detector_metrics,
        "face": get_face_recognizer().get_metrics(),
        "anpr": get_anpr_processor().get_metrics(),
        #: Canonical analytics frame size. Rule coordinates live in this space,
        #: so the dashboard sizes its drawing canvas from here rather than
        #: hard-coding it — otherwise retuning FRAME_WIDTH/FRAME_HEIGHT would
        #: silently misplace every zone and tripwire drawn afterwards.
        "frame": {"width": settings.FRAME_WIDTH, "height": settings.FRAME_HEIGHT},
        "limits": {
            "upload_max_mb": settings.UPLOAD_MAX_MB,
            "allowed_extensions": settings.UPLOAD_ALLOWED_EXTENSIONS,
            "max_stream_clients": settings.MAX_STREAM_CLIENTS,
        },
        "pipeline": {
            "frame_size": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
            "target_fps": settings.TARGET_FPS,
            "inference_imgsz": settings.INFERENCE_IMGSZ,
            "confidence": settings.DEFAULT_CONFIDENCE,
            "debounce_seconds": settings.DEBOUNCE_SECONDS,
        },
        #: Night detection is visual. Reporting the thresholds (and the fact
        #: that the clock is only an optional hint) keeps the dashboard honest
        #: about how the decision is actually made.
        "night_detection": {
            "mode": "forced" if settings.FORCE_NIGHT_MODE else "visual",
            "enter_threshold": settings.NIGHT_DARKNESS_ENTER,
            "exit_threshold": settings.NIGHT_DARKNESS_EXIT,
            "confirm_frames_to_night": settings.NIGHT_CONFIRM_FRAMES,
            "confirm_frames_to_day": settings.DAY_CONFIRM_FRAMES,
            "detects_infrared": settings.NIGHT_DETECT_INFRARED,
            "clock_used_as_hint_only": settings.NIGHT_USE_CLOCK_HINT,
            "clock_window_ist": (
                f"{settings.NIGHT_START_HOUR:02d}:00–{settings.NIGHT_END_HOUR:02d}:00"
                if settings.NIGHT_USE_CLOCK_HINT else "not used"
            ),
        },
        #: What the dashboard should escalate on. Served rather than hard-coded
        #: so an operator retuning NOTIFY_MIN_SEVERITY does not also have to
        #: edit the front end.
        "alerting": {
            "notify_min_severity": settings.NOTIFY_MIN_SEVERITY,
            "red_alert_severities": list(RED_ALERT_SEVERITIES),
            "severity_by_type": dict(SEVERITY_BY_TYPE),
            "severity_order": dict(models.SEVERITY_ORDER),
        },
        "rules": {
            "loiter_seconds": settings.LOITER_SECONDS,
            "loiter_exit_grace_seconds": settings.LOITER_EXIT_GRACE_SECONDS,
            "zone_presence_seconds": settings.ZONE_PRESENCE_SECONDS,
            "zone_boundary_margin_px": settings.ZONE_BOUNDARY_MARGIN,
            "zone_exit_grace_seconds": settings.ZONE_EXIT_GRACE_SECONDS,
            "crossing_rearm_seconds": settings.CROSSING_REARM_SECONDS,
            "alert_cooldowns": settings.ALERT_COOLDOWNS,
        },
        "cameras": manager.stats(),
        "aggregate": manager.aggregate(),
        "integrity": chain_status(db),
        "evidence": evidence_usage(),
        "websocket_clients": ws_manager.client_count,
        "server_time_ist": fmt_ist(),
        "timezone": "Asia/Kolkata (IST, UTC+05:30)",
    }


@app.get("/api/system/time")
def system_time():
    """Authoritative IST clock — the dashboard header syncs against this."""
    return {
        "utc": utc_iso(),
        "ist_iso": ist_iso(),
        "ist_display": fmt_ist(),
        "ist_time": fmt_ist_time(),
        "timezone": "Asia/Kolkata",
        "utc_offset": "+05:30",
    }


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def serialize_camera(cam: models.Camera, proc=None) -> dict:
    kind = cam.source_kind or KIND_LIVE
    data = {
        "id": cam.id,
        "name": cam.name,
        "url": cam.url,
        "location": cam.location or "",
        "is_active": bool(cam.is_active),
        "is_online": bool(cam.is_online),
        "source_kind": kind,
        #: The dashboard labels a recorded source as such, so nobody mistakes a
        #: looping video file for a live feed from the border.
        "is_file_source": kind == KIND_FILE,
        "source_label": {
            KIND_FILE: "VIDEO FILE",
            "upload": "OFFLINE ANALYSIS",
        }.get(kind, "LIVE FEED"),
        "source_name": Path(cam.url).name if kind == KIND_FILE else cam.url,
        "stream_url": f"/stream/{cam.id}",
        "snapshot_url": f"/api/cameras/{cam.id}/snapshot",
    }
    if proc is not None:
        data["runtime"] = proc.stats()
        data["is_online"] = proc.is_online
    return data


@app.get("/api/cameras")
def list_cameras(
    include_uploads: bool = Query(False),
    db: Session = Depends(get_db_session),
):
    manager = CameraManager.get()
    return [
        serialize_camera(cam, manager.get_camera(cam.id))
        for cam in visible_cameras(db, include_uploads=include_uploads)
    ]


@app.post("/api/cameras", status_code=201)
async def create_camera(
    name: str = Form(...),
    url: str = Form(...),
    location: str = Form(""),
    is_active: bool = Form(True),
):
    """
    Register a network camera: RTSP / HTTP URL, or a webcam index.

    ``async`` + ``to_thread`` for the same reason as the delete endpoint.
    Starting a camera takes the manager lock, constructs the analyzer and
    spawns two threads, and if a removal happens to be signalling at that
    moment it waits behind it. On a plain ``def`` route that wait occupies one
    of Starlette's shared threadpool workers — the pool every other synchronous
    endpoint draws from — so adding a camera could make the rest of the
    dashboard hitch at exactly the moment the operator is watching it.
    """
    def _create() -> dict:
        db = SessionLocal()
        try:
            cam = register_camera(
                db, name=name, url=url, location=location,
                is_active=is_active, source_kind=KIND_LIVE,
            )
            proc = CameraManager.get().add_camera(cam) if is_active else None
            return serialize_camera(cam, proc)
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_create)
    except SourceError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/cameras/upload", status_code=201)
async def create_camera_from_video(
    file: UploadFile = File(...),
    name: str = Form(...),
    location: str = Form(""),
    is_active: bool = Form(True),
    loop: bool = Form(True),
):
    """
    Register an **MP4 file as a camera source**.

    The file becomes a first-class source: it is stored under
    ``videos/sources/``, registered with ``source_kind="file"`` and started on
    the same pipeline as an RTSP stream, so it appears in the Live Camera grid
    and gets the same detection, tracking, zones, tripwires, ANPR, events and
    evidence.  ``LiveSource`` paces it to its own frame rate and loops it at
    EOF, so it behaves like the camera that recorded it rather than being
    decoded as fast as the CPU allows.

    This is distinct from ``/api/analysis/upload``, which analyses a recording
    once, offline, and produces a report.
    """
    if not (name or "").strip():
        raise HTTPException(400, "Camera name is required")

    limit = settings.UPLOAD_MAX_MB * 1024 * 1024
    settings.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(settings.SOURCES_DIR))
    tmp_path = Path(tmp_name)
    written = 0
    try:
        # Streamed to disk in bounded chunks — a large upload is never held in
        # memory, and the size cap is enforced while receiving rather than after.
        with os.fdopen(tmp_fd, "wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise SourceError(
                        f"File exceeds the {settings.UPLOAD_MAX_MB} MB limit."
                    )
                handle.write(chunk)
    except SourceError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(413, str(exc))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.exception("Video source upload failed: %s", exc)
        raise HTTPException(500, "Upload failed while writing to disk")
    finally:
        await file.close()

    try:
        stored, probe = await asyncio.to_thread(
            store_source_video, tmp_path, file.filename or "source.mp4"
        )
    except SourceError as exc:
        raise HTTPException(400, str(exc))

    db = SessionLocal()
    try:
        try:
            cam = register_camera(
                db, name=name, url=str(stored), location=location,
                is_active=is_active, source_kind=KIND_FILE,
            )
        except SourceError as exc:
            stored.unlink(missing_ok=True)
            raise HTTPException(400, str(exc))

        proc = CameraManager.get().add_camera(cam) if is_active else None
        payload = serialize_camera(cam, proc)
    finally:
        db.close()

    payload["video"] = {
        "filename": Path(stored).name,
        "width": probe.get("width"),
        "height": probe.get("height"),
        "fps": round(float(probe.get("fps") or 0.0), 2),
        "frame_count": probe.get("frame_count"),
        "duration_seconds": probe.get("duration_seconds"),
        "size_mb": round(float(probe.get("size_bytes", 0)) / (1024 * 1024), 2),
        "loops": bool(loop),
    }
    return payload


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: int, db: Session = Depends(get_db_session)):
    cam = get_live_camera(db, camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return serialize_camera(cam, CameraManager.get().get_camera(camera_id))


@app.put("/api/cameras/{camera_id}")
def update_camera(
    camera_id: int,
    name: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    is_active: Optional[bool] = Form(None),
    db: Session = Depends(get_db_session),
):
    cam = get_live_camera(db, camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    was_active, old_url = cam.is_active, cam.url
    if name is not None:
        cam.name = name.strip()[:120]
    if url is not None:
        cam.url = url.strip()[:500]
    if location is not None:
        cam.location = location.strip()[:200]
    if is_active is not None:
        cam.is_active = is_active
    db.commit()
    db.refresh(cam)

    manager = CameraManager.get()
    url_changed = url is not None and url.strip() != old_url
    if was_active and not cam.is_active:
        manager.remove_camera(camera_id)
    elif cam.is_active and (not was_active or url_changed):
        manager.remove_camera(camera_id)
        manager.add_camera(cam)

    return serialize_camera(cam, manager.get_camera(camera_id))


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: int):
    """
    Remove a camera: stop its threads, release its handles, then delete the row.

    The order is the whole point, and it is **runtime first, database second**:

    1. look the running :class:`CameraProcessor` up in the live registry;
    2. signal it to stop — its ``threading.Event`` is set, so the analytics
       loop ends at its next turn and any back-off wait aborts immediately;
    3. unregister it, so nothing can hand out a reference to a dead processor;
    4. join both of its threads under a bounded timeout, so a thread parked in
       a native read can never hang the request;
    5. release the OpenCV capture and free the frame buffers;
    6. **only then** touch SQLite.

    Doing the database first is what produced the reported failure: the row
    vanished from the dashboard while the pipeline kept running, invisible,
    holding the camera device and a share of the GPU — and with nothing left
    in any listing to point at it, nothing would ever stop it. Worse, a
    still-running processor seals events against a ``camera_id`` that is about
    to disappear, which fails on the foreign key from inside a camera thread.

    :func:`core.sources.retire_camera` performs steps 1-6 in that order and
    then decides whether the row can be deleted outright or must be archived
    to keep the evidence hash chain intact. It is idempotent, so a
    double-clicked button or a retried request returns success, not a 500.

    ``background_join=True`` moves only step 4 — the *waiting* — off the
    request. Everything that makes the camera dead has already happened
    synchronously by then, so the operator's click returns in milliseconds
    instead of blocking on a socket timeout they gain nothing from.

    The route is ``async`` and hands its work to ``asyncio.to_thread`` because
    a plain ``def`` route would occupy one of Starlette's 40 shared threadpool
    workers — the same pool every other synchronous endpoint needs — for the
    duration. Removing several dead cameras in a row stalled the dashboard
    that way.
    """
    def _retire() -> dict:
        # A Session belongs to one thread. This runs on a worker, so it opens
        # its own and closes it in a finally — never the request's session.
        db = SessionLocal()
        try:
            running = active_processors().get(camera_id)
            if running is not None:
                log.info("CAMERA_REMOVE cam=%d (%s) — stopping pipeline before "
                         "touching the database", camera_id, running.name)
            return retire_camera(db, camera_id, background_join=True)
        finally:
            db.close()

    try:
        result = await asyncio.to_thread(_retire)
    except Exception as exc:
        log.exception("Camera removal failed for %d: %s", camera_id, exc)
        raise HTTPException(500, f"Could not remove camera: {exc}")

    # Assert the contract on the way out rather than assuming it. If a
    # processor for this id is somehow still registered, that is the zombie
    # this endpoint exists to prevent, and it must be visible in the log.
    if camera_id in active_processors():
        log.error("CAMERA_REMOVE_INCOMPLETE cam=%d: a processor is still "
                  "registered after retirement", camera_id)
    return result


@app.post("/api/cameras/{camera_id}/restart")
def restart_camera(camera_id: int, db: Session = Depends(get_db_session)):
    cam = get_live_camera(db, camera_id)
    manager = CameraManager.get()
    # Both checks matter. The row read can be stale by the time we act on it,
    # and the tombstone is set before the row is archived — so a restart that
    # raced a removal would otherwise start a brand-new pipeline for a camera
    # the operator had just deleted, with nothing left to stop it.
    if not cam or manager.is_retired(camera_id):
        raise HTTPException(404, "Camera not found")
    manager.remove_camera(camera_id)
    proc = manager.add_camera(cam)
    return {"ok": proc is not None, "camera": serialize_camera(cam, proc)}


@app.post("/api/cameras/{camera_id}/playback")
def set_camera_playback(
    camera_id: int,
    paused: Optional[bool] = Form(None),
    speed: Optional[float] = Form(None),
    db: Session = Depends(get_db_session),
):
    """
    Pause / resume a camera's tile, and set its review speed (0.5x - 2x).

    Playback is a *view* control, not a pipeline switch. On a recording it
    pauses the decoder, so the footage waits where the operator stopped it. On
    a live camera it holds the published picture while capture, analytics and
    event sealing carry on underneath — pausing a real camera's analysis to
    look at a frame would blind the post at exactly the wrong moment.

    Speed applies only where the source is paced exactly (a recording); the
    response says so in ``speed_supported`` rather than silently ignoring it.
    """
    if get_live_camera(db, camera_id) is None:
        raise HTTPException(404, "Camera not found")
    proc = CameraManager.get().get_camera(camera_id)
    if proc is None:
        raise HTTPException(409, "Camera is not running")

    if speed is not None and not (PLAYBACK_MIN_SPEED <= speed <= PLAYBACK_MAX_SPEED):
        raise HTTPException(
            400,
            f"speed must be between {PLAYBACK_MIN_SPEED} and {PLAYBACK_MAX_SPEED}",
        )
    if paused is None and speed is None:
        raise HTTPException(400, "Provide 'paused', 'speed', or both")

    state = proc.set_playback(paused=paused, speed=speed)
    log.info("Camera %d playback: paused=%s speed=%.2fx",
             camera_id, state["paused"], state["speed"])
    return {"camera_id": camera_id, **state}


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: int):
    """A single current JPEG — used to seed the fence-drawing canvas."""
    jpeg = FrameBuffer.get().get_jpeg(camera_id)
    if jpeg is None:
        raise HTTPException(404, "No frame available yet for this camera")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# Rules (virtual fences and zones)
# --------------------------------------------------------------------------- #


def serialize_rule(rule: models.Rule) -> dict:
    try:
        geometry = json.loads(rule.geometry) if rule.geometry else []
    except (json.JSONDecodeError, TypeError):
        geometry = []
    try:
        params = json.loads(rule.params) if rule.params else {}
    except (json.JSONDecodeError, TypeError):
        params = {}
    return {
        "id": rule.id,
        "camera_id": rule.camera_id,
        "rule_type": rule.rule_type,
        "name": rule.name or f"{rule.rule_type}-{rule.id}",
        "geometry": geometry,
        "params": params,
        "is_active": bool(rule.is_active),
        "created_at_ist": fmt_ist(rule.created_at) if rule.created_at else "",
    }


_RULE_TYPES = {"line", "zone", "loiter", "direction"}
_MIN_POINTS = {"line": 2, "direction": 2, "zone": 3, "loiter": 3}


def _validate_geometry(rule_type: str, geometry) -> list:
    if rule_type not in _RULE_TYPES:
        raise HTTPException(400, f"Unknown rule type '{rule_type}'. "
                                 f"Expected one of {sorted(_RULE_TYPES)}")
    if not isinstance(geometry, list):
        raise HTTPException(400, "geometry must be a list of [x, y] points")
    needed = _MIN_POINTS[rule_type]
    if len(geometry) < needed:
        raise HTTPException(400, f"A '{rule_type}' rule needs at least {needed} points")

    cleaned = []
    for point in geometry:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise HTTPException(400, "Each geometry point must be [x, y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            raise HTTPException(400, "Geometry coordinates must be numeric")
        # Coordinates are in analytics frame space; clamp rather than reject so
        # a drag that ends just outside the canvas still produces a usable rule.
        cleaned.append([
            max(0.0, min(x, settings.FRAME_WIDTH)),
            max(0.0, min(y, settings.FRAME_HEIGHT)),
        ])
    return cleaned


@app.get("/api/cameras/{camera_id}/rules")
def list_rules(camera_id: int, db: Session = Depends(get_db_session)):
    if get_live_camera(db, camera_id) is None:
        raise HTTPException(404, "Camera not found")
    rules = db.query(models.Rule).filter(models.Rule.camera_id == camera_id).all()
    return [serialize_rule(r) for r in rules]


@app.post("/api/cameras/{camera_id}/rules", status_code=201)
def create_rule(
    camera_id: int,
    rule_type: str = Form(...),
    geometry: str = Form("[]"),
    params: str = Form("{}"),
    name: str = Form(""),
    is_active: bool = Form(True),
    db: Session = Depends(get_db_session),
):
    if get_live_camera(db, camera_id) is None:
        raise HTTPException(404, "Camera not found")

    try:
        geometry_data = json.loads(geometry)
    except json.JSONDecodeError:
        raise HTTPException(400, "geometry must be valid JSON")
    try:
        params_data = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "params must be valid JSON")
    if not isinstance(params_data, dict):
        raise HTTPException(400, "params must be a JSON object")

    cleaned = _validate_geometry(rule_type, geometry_data)
    label = (name or "").strip()[:200] or f"{rule_type.upper()}-{int(time.time()) % 10000}"

    rule = models.Rule(
        camera_id=camera_id, rule_type=rule_type, name=label,
        geometry=json.dumps(cleaned), params=json.dumps(params_data),
        is_active=is_active, created_at=utc_iso(),
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)

    CameraManager.get().reload_camera_rules(camera_id)
    log.info("Rule armed on camera %d: %s (%s, %d points)",
             camera_id, label, rule_type, len(cleaned))
    return serialize_rule(rule)


@app.put("/api/rules/{rule_id}")
def update_rule(
    rule_id: int,
    rule_type: Optional[str] = Form(None),
    geometry: Optional[str] = Form(None),
    params: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    is_active: Optional[bool] = Form(None),
    db: Session = Depends(get_db_session),
):
    rule = db.query(models.Rule).filter(models.Rule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Rule not found")

    effective_type = rule_type or rule.rule_type
    if rule_type is not None:
        if rule_type not in _RULE_TYPES:
            raise HTTPException(400, f"Unknown rule type '{rule_type}'")
        rule.rule_type = rule_type
    if geometry is not None:
        try:
            rule.geometry = json.dumps(_validate_geometry(effective_type, json.loads(geometry)))
        except json.JSONDecodeError:
            raise HTTPException(400, "geometry must be valid JSON")
    if params is not None:
        try:
            parsed = json.loads(params)
        except json.JSONDecodeError:
            raise HTTPException(400, "params must be valid JSON")
        if not isinstance(parsed, dict):
            raise HTTPException(400, "params must be a JSON object")
        rule.params = json.dumps(parsed)
    if name is not None:
        rule.name = name.strip()[:200]
    if is_active is not None:
        rule.is_active = is_active

    db.commit()
    db.refresh(rule)
    CameraManager.get().reload_camera_rules(rule.camera_id)
    return serialize_rule(rule)


@app.delete("/api/rules/{rule_id}")
def delete_rule(rule_id: int, db: Session = Depends(get_db_session)):
    rule = db.query(models.Rule).filter(models.Rule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Rule not found")
    camera_id, label = rule.camera_id, rule.name
    db.delete(rule)
    db.commit()
    CameraManager.get().reload_camera_rules(camera_id)
    log.info("Rule removed from camera %d: %s", camera_id, label)
    return {"ok": True, "removed": rule_id}


@app.delete("/api/cameras/{camera_id}/rules")
def clear_rules(camera_id: int, db: Session = Depends(get_db_session)):
    removed = db.query(models.Rule).filter(models.Rule.camera_id == camera_id).delete()
    db.commit()
    CameraManager.get().reload_camera_rules(camera_id)
    return {"ok": True, "removed": removed}


# --------------------------------------------------------------------------- #
# Alerts / event log
# --------------------------------------------------------------------------- #


@app.get("/api/alerts")
def list_alerts(
    camera_id: Optional[int] = Query(None),
    alert_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    source_type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    from_ts: Optional[str] = Query(None, description="ISO-8601; UTC or IST offset"),
    to_ts: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """Filterable event log. Timestamps in and out are timezone-aware."""
    from core.timeutil import to_utc

    query = db.query(models.Alert)
    if camera_id is not None:
        query = query.filter(models.Alert.camera_id == camera_id)
    if alert_type:
        query = query.filter(models.Alert.alert_type == alert_type)
    if severity:
        query = query.filter(models.Alert.severity == severity.upper())
    if track_id is not None:
        query = query.filter(models.Alert.track_id == track_id)
    if source_type:
        query = query.filter(models.Alert.source_type == source_type)
    if session_id:
        query = query.filter(models.Alert.session_id == session_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(models.Alert.description.ilike(pattern) |
                             models.Alert.details_json.ilike(pattern))
    if from_ts:
        parsed = to_utc(from_ts)
        if parsed:
            query = query.filter(models.Alert.timestamp >= parsed.isoformat())
    if to_ts:
        parsed = to_utc(to_ts)
        if parsed:
            query = query.filter(models.Alert.timestamp <= parsed.isoformat())

    total = query.count()
    rows = query.order_by(desc(models.Alert.id)).offset(offset).limit(limit).all()
    names = _camera_names(db)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": [serialize_alert_row(r, names.get(r.camera_id, "")) for r in rows],
    }


@app.get("/api/alerts/export.pdf")
def export_alerts_pdf(
    camera_id: Optional[int] = Query(None),
    alert_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    source_type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    from_ts: Optional[str] = Query(None),
    to_ts: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db_session),
):
    """
    The filtered event log as a printable PDF.

    Takes the same filters as ``GET /api/alerts`` so what is printed is exactly
    what the operator is looking at. The report states its own scope — the
    filters, the number of events shown against the number that matched, and
    whether the hash chain verified at the moment of printing — because a table
    of rows with no provenance proves nothing about the log it came from.
    """
    from core.report import build_event_log_pdf, pdf_available

    if not pdf_available():
        raise HTTPException(
            503,
            "PDF export needs the 'reportlab' package: pip install reportlab",
        )

    from core.timeutil import to_utc

    query = db.query(models.Alert)
    if camera_id is not None:
        query = query.filter(models.Alert.camera_id == camera_id)
    if alert_type:
        query = query.filter(models.Alert.alert_type == alert_type)
    if severity:
        query = query.filter(models.Alert.severity == severity.upper())
    if track_id is not None:
        query = query.filter(models.Alert.track_id == track_id)
    if source_type:
        query = query.filter(models.Alert.source_type == source_type)
    if session_id:
        query = query.filter(models.Alert.session_id == session_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(models.Alert.description.ilike(pattern) |
                             models.Alert.details_json.ilike(pattern))
    if from_ts:
        parsed = to_utc(from_ts)
        if parsed:
            query = query.filter(models.Alert.timestamp >= parsed.isoformat())
    if to_ts:
        parsed = to_utc(to_ts)
        if parsed:
            query = query.filter(models.Alert.timestamp <= parsed.isoformat())

    total = query.count()
    rows = query.order_by(desc(models.Alert.id)).limit(limit).all()
    names = _camera_names(db)
    alerts = [serialize_alert_row(r, names.get(r.camera_id, "")) for r in rows]

    # Measured now, not assumed: a report that asserts its own integrity has to
    # have checked. Never let a verification failure deny the operator a print.
    integrity = None
    try:
        result = verify_chain(db)
        integrity = {
            "valid": bool(result.valid),
            "verified": int(result.total_alerts),
            "breaks": list(result.breaks or []),
            "message": result.message,
        }
    except Exception as exc:
        log.warning("Integrity check failed while building the PDF: %s", exc)

    filters = {
        "camera": names.get(camera_id, camera_id) if camera_id is not None else None,
        "event type": alert_type, "severity": severity, "track": track_id,
        "source": source_type, "session": session_id, "search": search,
        "from": from_ts, "to": to_ts,
    }

    try:
        pdf = build_event_log_pdf(
            alerts, total_matching=total, filters=filters, integrity=integrity,
        )
    except Exception as exc:
        log.exception("PDF generation failed: %s", exc)
        raise HTTPException(500, f"Could not build the PDF: {exc}")

    # A filename that sorts chronologically and survives every filesystem:
    # no spaces, no colons, IST because every timestamp in the report is IST.
    filename = f"ibvap-event-log-{now_ist().strftime('%Y%m%d-%H%M')}-IST.pdf"
    log.info("Event log PDF: %d of %d event(s), %d KB",
             len(alerts), total, len(pdf) // 1024)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: int, db: Session = Depends(get_db_session)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    cam = db.query(models.Camera).filter(models.Camera.id == alert.camera_id).first()
    payload = serialize_alert_row(alert, cam.name if cam else "")
    payload["camera_location"] = cam.location if cam else ""
    return payload


def _serve_evidence(path_value: str, media_type: str, download_name: str = ""):
    """Serve an evidence file only if it resolves inside an approved directory."""
    if not path_value:
        raise HTTPException(404, "No evidence recorded for this event")
    if not is_safe_evidence_path(path_value):
        log.warning("Blocked evidence path outside approved roots: %s", path_value)
        raise HTTPException(403, "Evidence path is outside the permitted store")
    path = Path(path_value)
    if not path.is_file():
        raise HTTPException(404, "Evidence file is missing from disk")
    return FileResponse(path, media_type=media_type, filename=download_name or path.name)


@app.get("/api/alerts/{alert_id}/snapshot")
def get_alert_snapshot(
    alert_id: int,
    clean: bool = Query(False, description="Return the unannotated frame"),
    db: Session = Depends(get_db_session),
):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    path = alert.snapshot_path
    if clean:
        try:
            details = json.loads(alert.details_json or "{}")
            path = details.get("clean_snapshot") or path
        except (json.JSONDecodeError, TypeError):
            pass
    return _serve_evidence(path, "image/jpeg")


@app.get("/api/alerts/{alert_id}/clip")
def get_alert_clip(alert_id: int, db: Session = Depends(get_db_session)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    return _serve_evidence(alert.clip_path, "video/mp4")


@app.get("/api/alerts/{alert_id}/plate")
def get_alert_plate(alert_id: int, db: Session = Depends(get_db_session)):
    """
    The cropped number plate recorded for this event.

    ANPR crops are stored against the reading that produced them, so an event
    in the log can be opened directly onto the pixels its registration was read
    from — which is what makes the plate evidence rather than an assertion.
    """
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    reading = (
        db.query(models.ANPRDetection)
        .filter(models.ANPRDetection.alert_id == alert_id)
        .order_by(desc(models.ANPRDetection.id))
        .first()
    )
    if reading is None or not reading.evidence_path:
        raise HTTPException(404, "No plate crop stored for this event")
    return _serve_evidence(reading.evidence_path, "image/jpeg")


@app.get("/api/stats")
def get_stats(
    camera_id: Optional[int] = Query(None),
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db_session),
):
    """
    Event statistics. "Today" means today **in IST**, not UTC — otherwise the
    operator's daily count would roll over at 05:30 local.
    """
    cutoff = (now_utc() - timedelta(hours=hours)).isoformat()

    base = db.query(models.Alert).filter(models.Alert.timestamp >= cutoff)
    if camera_id:
        base = base.filter(models.Alert.camera_id == camera_id)

    by_type = dict(
        base.with_entities(models.Alert.alert_type, func.count(models.Alert.id))
        .group_by(models.Alert.alert_type).all()
    )
    by_severity = dict(
        base.with_entities(models.Alert.severity, func.count(models.Alert.id))
        .group_by(models.Alert.severity).all()
    )
    by_camera = dict(
        base.with_entities(models.Alert.camera_id, func.count(models.Alert.id))
        .group_by(models.Alert.camera_id).all()
    )

    day_start = start_of_ist_day().isoformat()
    today = db.query(func.count(models.Alert.id)).filter(
        models.Alert.timestamp >= day_start
    ).scalar() or 0

    manager = CameraManager.get()
    return {
        "window_hours": hours,
        "total_alerts": base.count(),
        "today": today,
        "today_since_ist": fmt_ist(day_start),
        "by_type": by_type,
        "by_severity": by_severity,
        "by_camera": {str(k): v for k, v in by_camera.items()},
        "live": manager.aggregate(),
        "evidence": evidence_usage(),
        "generated_at_ist": fmt_ist(),
    }


# --------------------------------------------------------------------------- #
# MJPEG streaming
# --------------------------------------------------------------------------- #


class _StreamSlots:
    """
    Per-camera MJPEG viewer budget.

    Counting is done here rather than inside the generator because a generator
    that is never iterated (client vanished between accept and first read) would
    otherwise leak its slot forever.
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def acquire(self, camera_id: int) -> bool:
        limit = max(1, int(settings.MAX_STREAM_CLIENTS))
        with self._lock:
            current = self._counts.get(camera_id, 0)
            if current >= limit:
                return False
            self._counts[camera_id] = current + 1
            return True

    def release(self, camera_id: int) -> None:
        with self._lock:
            current = self._counts.get(camera_id, 0) - 1
            if current > 0:
                self._counts[camera_id] = current
            else:
                self._counts.pop(camera_id, None)

    def snapshot(self) -> dict[int, int]:
        with self._lock:
            return dict(self._counts)


_stream_slots = _StreamSlots()


async def _mjpeg_stream(camera_id: int, request: Request):
    """
    Async MJPEG generator.

    Sends the JPEG the camera thread already encoded — a frame is never
    re-encoded per viewer — and exits as soon as the client disconnects.

    New frames are detected by polling a monotonic sequence counter (a dict
    lookup under a lock) rather than blocking a worker thread. Handing each
    frame's wait to ``asyncio.to_thread`` cost one threadpool dispatch per
    frame per viewer; at 15 fps with a handful of open tabs that starves the
    very pool every synchronous endpoint depends on.

    The poll interval is what that costs in latency, and it is the last delay
    in the chain before the browser, so it is kept short — 5 ms, which is a
    dict lookup under a lock 200 times a second per viewer and immeasurable
    next to a single JPEG encode. It used to be one half of a frame interval
    (33 ms at 15 fps), which on top of the capture and analytics stages was a
    third of the total end-to-end delay for nothing.

    The disconnect check is the expensive part of this loop — it opens a cancel
    scope and reads from the ASGI channel — so it runs on its own slower timer
    rather than on every poll. Nothing is lost by that: when a client really
    goes away the server cancels this task, and the check is a belt-and-braces
    second signal, not the primary one.
    """
    buffer = FrameBuffer.get()
    last_seq = -1
    idle_since = time.monotonic()
    poll = min(0.005, 1.0 / max(10, settings.TARGET_FPS * 4))
    disconnect_every = 0.5
    next_disconnect_check = 0.0
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "

    while True:
        now = time.monotonic()
        if now >= next_disconnect_check:
            next_disconnect_check = now + disconnect_every
            if await request.is_disconnected():
                break

        seq = buffer.sequence(camera_id)
        if seq <= last_seq:
            # Nothing new yet. Hold the connection open while a camera
            # reconnects rather than tearing the viewer's stream down, but
            # do not keep a dead stream alive forever.
            if now - idle_since > 120:
                break
            await asyncio.sleep(poll)
            continue

        jpeg = buffer.get_jpeg(camera_id)
        last_seq = seq
        idle_since = now
        if jpeg is None:
            await asyncio.sleep(poll)
            continue
        yield boundary + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"


@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: int, request: Request):
    """MJPEG stream — drops straight into any ``<img src>``."""
    db = SessionLocal()
    try:
        # An archived camera has no stream. Checking here is what stops a
        # browser tab left open on a removed camera from holding a connection
        # that quietly resurrects it in the UI.
        exists = get_live_camera(db, camera_id) is not None
    finally:
        db.close()
    if not exists:
        raise HTTPException(404, "Camera not found")

    # Enforce the viewer cap. ``MAX_STREAM_CLIENTS`` was configured but never
    # actually applied, so an unbounded number of MJPEG connections could be
    # opened against one camera — trivially, by a dashboard whose tiles were
    # reconnecting in a loop. Each one holds a generator, a socket and a slot in
    # the server's connection budget, so the cap has to be real for the
    # reconnect backoff on the client to be worth anything.
    if not _stream_slots.acquire(camera_id):
        raise HTTPException(
            503,
            f"Too many viewers on camera {camera_id} "
            f"(limit {settings.MAX_STREAM_CLIENTS}). Close another view and retry.",
        )

    async def _guarded():
        try:
            async for chunk in _mjpeg_stream(camera_id, request):
                yield chunk
        finally:
            _stream_slots.release(camera_id)

    return StreamingResponse(
        _guarded(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache",
                 "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    Real-time channel: alerts, live stats, analysis progress, checkpoints.

    Two independent coroutines — one draining the outbound queue, one reading
    client pings — so a silent client never blocks delivery and a chatty one
    never delays the next alert.
    """
    queue = await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connected",
            "data": {
                "version": settings.VERSION,
                "server_time_ist": fmt_ist(),
                "timezone": "Asia/Kolkata",
                "recent": EventManager.get().recent[-25:],
            },
        })

        async def _pump():
            while True:
                message = await queue.get()
                await websocket.send_json(message)

        async def _drain():
            while True:
                await websocket.receive_text()

        pump = asyncio.create_task(_pump())
        drain = asyncio.create_task(_drain())
        try:
            done, pending = await asyncio.wait(
                {pump, drain}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            # Cancel *and await*. A cancelled task that is never awaited stays
            # pending on the loop and, if it later raises, surfaces as a bare
            # "Task exception was never retrieved" with no context. Gathering
            # them here means a closed tab leaves nothing behind on the loop.
            for task in (pump, drain):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pump, drain, return_exceptions=True)

        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                raise exc
    except WebSocketDisconnect:
        # The normal ending: the tab was closed or refreshed.
        pass
    except (asyncio.CancelledError, RuntimeError) as exc:
        # Server shutting down, or the socket went away mid-send.
        log.debug("WebSocket ended: %s", exc)
    except Exception as exc:
        log.debug("WebSocket closed: %s", exc)
    finally:
        # Runs on every path, including a raised exception and a cancelled
        # task, so a client can never be left in the fan-out list.
        await ws_manager.disconnect(websocket)


# --------------------------------------------------------------------------- #
# Video upload and offline analysis
# --------------------------------------------------------------------------- #


@app.post("/api/analysis/upload", status_code=202)
async def upload_video(
    file: UploadFile = File(...),
    camera_id: Optional[int] = Form(None),
    apply_rules: bool = Form(True),
):
    """
    Accept an MP4 and start analysing it through the live pipeline.

    The upload is streamed to a temporary file in bounded chunks (never read
    whole into memory), validated by extension, size, container signature and
    decodability, then moved into managed storage under a server-generated
    name.  Analysis runs on a worker thread; progress arrives over WebSocket.
    """
    manager = AnalysisManager.get()
    original = sanitize_filename(file.filename or "upload.mp4")

    limit = settings.UPLOAD_MAX_MB * 1024 * 1024
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(settings.VIDEOS_DIR))
    tmp_path = Path(tmp_name)
    written = 0
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise UploadValidationError(
                        f"File exceeds the {settings.UPLOAD_MAX_MB} MB limit."
                    )
                handle.write(chunk)
    except UploadValidationError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(413, str(exc))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.exception("Upload failed: %s", exc)
        raise HTTPException(500, "Upload failed while writing to disk")
    finally:
        await file.close()

    try:
        session_uid, stored, probe = await asyncio.to_thread(
            manager.store_upload, tmp_path, original
        )
    except UploadValidationError as exc:
        raise HTTPException(400, str(exc))

    rules = rules_for_camera(camera_id) if (apply_rules and camera_id) else []
    owner_camera = camera_id or ensure_upload_camera()

    job = manager.submit(session_uid, stored, original, probe, owner_camera, rules)
    log.info("Analysis queued: %s (%s, %.1fs, %d rule(s))",
             original, session_uid[:8], probe.get("duration_seconds", 0), len(rules))
    return {
        "session_id": session_uid,
        "filename": original,
        "video": {
            "width": probe.get("width"), "height": probe.get("height"),
            "fps": round(float(probe.get("fps") or 0), 2),
            "frame_count": probe.get("frame_count"),
            "duration_seconds": probe.get("duration_seconds"),
            "size_mb": round(probe.get("size_bytes", 0) / (1024 * 1024), 2),
        },
        "rules_applied": len(rules),
        "uploaded_at_ist": fmt_ist(),
        "status": job.status,
    }


@app.get("/api/analysis")
def list_analyses(db: Session = Depends(get_db_session)):
    """Active jobs first, then completed sessions recovered from the database."""
    manager = AnalysisManager.get()
    live = manager.list_jobs()
    live_ids = {job["session_id"] for job in live}

    rows = (
        db.query(models.AnalysisSession)
        .order_by(models.AnalysisSession.id.desc())
        .limit(50).all()
    )
    historical = [
        {
            "session_id": r.session_uid,
            "filename": r.filename,
            "status": r.status,
            "error": r.error or "",
            "progress": 100.0 if r.status == "completed" else 0.0,
            "processed_frames": r.processed_frames,
            "analysed_frames": r.analysed_frames,
            "total_frames": r.total_frames,
            "duration_seconds": r.duration_seconds,
            "source_fps": r.source_fps,
            "processing_fps": r.processing_fps,
            "detections_total": r.detections_total,
            "persons": r.persons_seen,
            "vehicles": r.vehicles_seen,
            "alerts": r.alerts_generated,
            "resolution": f"{r.width}x{r.height}",
            "size_mb": round((r.size_bytes or 0) / (1024 * 1024), 2),
            "created_at_ist": r.created_at_ist,
            "completed_at_ist": r.completed_at_ist,
            "has_output": bool(r.output_path),
            "output_url": f"/api/analysis/{r.session_uid}/video" if r.output_path else None,
            "camera_id": r.camera_id,
        }
        for r in rows if r.session_uid not in live_ids
    ]
    return {"active": live, "history": historical}


@app.get("/api/analysis/{session_id}")
def get_analysis(session_id: str, db: Session = Depends(get_db_session)):
    job = AnalysisManager.get().get_job(session_id)
    if job is not None:
        return job.snapshot()

    row = (
        db.query(models.AnalysisSession)
        .filter(models.AnalysisSession.session_uid == session_id).first()
    )
    if row is None:
        raise HTTPException(404, "Analysis session not found")
    return {
        "session_id": row.session_uid, "filename": row.filename,
        "status": row.status, "error": row.error or "",
        "progress": 100.0 if row.status == "completed" else 0.0,
        "analysed_frames": row.analysed_frames, "total_frames": row.total_frames,
        "duration_seconds": row.duration_seconds, "processing_fps": row.processing_fps,
        "detections_total": row.detections_total, "persons": row.persons_seen,
        "vehicles": row.vehicles_seen, "alerts": row.alerts_generated,
        "created_at_ist": row.created_at_ist, "completed_at_ist": row.completed_at_ist,
        "has_output": bool(row.output_path),
        "output_url": f"/api/analysis/{row.session_uid}/video" if row.output_path else None,
        "camera_id": row.camera_id,
    }


@app.get("/api/analysis/{session_id}/preview")
def analysis_preview(session_id: str):
    """Latest annotated frame from a running analysis."""
    job = AnalysisManager.get().get_job(session_id)
    if job is None or job.preview is None:
        raise HTTPException(404, "No preview frame available")
    return StreamingResponse(iter([job.preview]), media_type="image/jpeg",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/analysis/{session_id}/video")
def analysis_video(session_id: str, db: Session = Depends(get_db_session)):
    """Download / play the annotated render of an analysed upload."""
    job = AnalysisManager.get().get_job(session_id)
    path = job.output_path if job else ""
    if not path:
        row = (
            db.query(models.AnalysisSession)
            .filter(models.AnalysisSession.session_uid == session_id).first()
        )
        path = row.output_path if row else ""
    return _serve_evidence(path, "video/mp4", f"ibvap_analysed_{session_id[:8]}.mp4")


@app.get("/api/analysis/{session_id}/alerts")
def analysis_alerts(session_id: str, db: Session = Depends(get_db_session)):
    """Every event produced by one uploaded-video analysis."""
    rows = (
        db.query(models.Alert)
        .filter(models.Alert.session_id == session_id)
        .order_by(models.Alert.id.asc()).all()
    )
    names = _camera_names(db)
    return {
        "session_id": session_id,
        "count": len(rows),
        "alerts": [serialize_alert_row(r, names.get(r.camera_id, "")) for r in rows],
    }


@app.post("/api/analysis/{session_id}/cancel")
def cancel_analysis(session_id: str):
    if not AnalysisManager.get().cancel(session_id):
        raise HTTPException(404, "No running analysis with that id")
    return {"ok": True, "session_id": session_id}


# --------------------------------------------------------------------------- #
# ANPR and face detection records
# --------------------------------------------------------------------------- #


def _detection_evidence_url(kind: str, row_id: int, path: str) -> Optional[str]:
    return f"/api/{kind}/{row_id}/evidence" if path else None


@app.get("/api/anpr/detections")
def list_anpr_detections(
    camera_id: Optional[int] = Query(None),
    plate: Optional[str] = Query(None, description="Exact or partial plate text"),
    status: Optional[str] = Query(None, description="published | uncertain"),
    hours: int = Query(168, ge=1, le=8760),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """
    Plate readings, newest first.

    This queries the typed ``anpr_detections`` table rather than pattern-matching
    the alert log's JSON, so "every sighting of this registration" is an indexed
    lookup instead of a scan that can match the wrong field.
    """
    since = (now_utc() - timedelta(hours=hours)).isoformat()
    query = db.query(models.ANPRDetection).filter(
        models.ANPRDetection.timestamp >= since
    )
    if camera_id is not None:
        query = query.filter(models.ANPRDetection.camera_id == camera_id)
    if plate:
        needle = plate.replace(" ", "").upper()
        query = query.filter(models.ANPRDetection.plate_text.like(f"%{needle}%"))
    if status:
        query = query.filter(models.ANPRDetection.processing_status == status)

    total = query.count()
    rows = (query.order_by(desc(models.ANPRDetection.id))
            .offset(offset).limit(limit).all())
    names = _camera_names(db)
    return {
        "total": total,
        "count": len(rows),
        "offset": offset,
        "detections": [{
            "id": r.id,
            "camera_id": r.camera_id,
            "camera_name": names.get(r.camera_id, f"CAM-{r.camera_id:02d}"),
            "alert_id": r.alert_id,
            "timestamp": r.timestamp,
            "timestamp_ist": r.timestamp_ist,
            "plate_text": r.plate_text,
            "plate_display": r.plate_display or r.plate_text,
            "plate_raw": r.plate_raw,
            "confidence": round(float(r.confidence or 0.0), 3),
            "format_verified": bool(r.format_verified),
            "votes": r.votes,
            "consensus": bool(r.consensus),
            "vehicle_class": r.vehicle_class,
            "vehicle_track_id": r.vehicle_track_id,
            "status": r.processing_status,
            "source_type": r.source_type,
            "has_evidence": bool(r.evidence_path),
            "evidence_url": _detection_evidence_url("anpr", r.id, r.evidence_path),
            "evidence_sha256": r.evidence_sha256,
        } for r in rows],
    }


@app.get("/api/anpr/detections/{detection_id}/evidence")
def get_anpr_evidence(detection_id: int, db: Session = Depends(get_db_session)):
    row = db.query(models.ANPRDetection).filter(
        models.ANPRDetection.id == detection_id).first()
    if not row or not row.evidence_path:
        raise HTTPException(404, "No evidence image for this detection")
    return _serve_evidence(row.evidence_path, "image/jpeg")


@app.get("/api/faces/detections")
def list_face_detections(
    camera_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None, description="matched | unknown"),
    hours: int = Query(168, ge=1, le=8760),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """
    Face detections, newest first.

    ``recognition_status`` is reported exactly as recorded: ``unknown`` means no
    watchlist entry matched above threshold, never that a face was unclear.
    """
    since = (now_utc() - timedelta(hours=hours)).isoformat()
    query = db.query(models.FaceDetection).filter(
        models.FaceDetection.timestamp >= since
    )
    if camera_id is not None:
        query = query.filter(models.FaceDetection.camera_id == camera_id)
    if status:
        query = query.filter(models.FaceDetection.recognition_status == status)

    total = query.count()
    rows = (query.order_by(desc(models.FaceDetection.id))
            .offset(offset).limit(limit).all())
    names = _camera_names(db)
    return {
        "total": total,
        "count": len(rows),
        "offset": offset,
        "detections": [{
            "id": r.id,
            "camera_id": r.camera_id,
            "camera_name": names.get(r.camera_id, f"CAM-{r.camera_id:02d}"),
            "alert_id": r.alert_id,
            "timestamp": r.timestamp,
            "timestamp_ist": r.timestamp_ist,
            "confidence": round(float(r.confidence or 0.0), 3),
            "track_id": r.track_id,
            "recognition_status": r.recognition_status,
            "identity_id": r.identity_id,
            "identity_name": r.identity_name,
            "similarity": round(float(r.similarity or 0.0), 3),
            "similarity_threshold": round(float(r.similarity_threshold or 0.0), 3),
            "source_type": r.source_type,
            "has_evidence": bool(r.evidence_path),
            "evidence_url": _detection_evidence_url("faces", r.id, r.evidence_path),
            "evidence_sha256": r.evidence_sha256,
        } for r in rows],
    }


@app.get("/api/faces/detections/{detection_id}/evidence")
def get_face_evidence(detection_id: int, db: Session = Depends(get_db_session)):
    row = db.query(models.FaceDetection).filter(
        models.FaceDetection.id == detection_id).first()
    if not row or not row.evidence_path:
        raise HTTPException(404, "No evidence image for this detection")
    return _serve_evidence(row.evidence_path, "image/jpeg")


# --------------------------------------------------------------------------- #
# Integrity — tamper-evident hash-chained event log
# --------------------------------------------------------------------------- #


@app.get("/api/integrity/verify")
async def verify_integrity():
    """
    Walk the entire chain and report the measured result.

    Runs in the threadpool: verification is O(events) SHA-256 work and must
    not block the event loop while a demo audience is watching the stream.
    """
    result = await asyncio.to_thread(verify_chain)
    payload = result.to_dict()
    payload["scheme"] = "SHA-256 hash chain (tamper-evident, local)"
    payload["headline"] = (
        "INTEGRITY VERIFIED" if result.valid else "INTEGRITY COMPROMISED"
    )
    return payload


@app.get("/api/integrity/tip")
def get_chain_tip(db: Session = Depends(get_db_session)):
    return {**chain_status(db), "checked_at_ist": fmt_ist()}


@app.post("/api/integrity/checkpoint")
async def make_checkpoint():
    """Seal every event added since the last checkpoint into a Merkle root."""
    checkpoint = await asyncio.to_thread(create_checkpoint)
    if checkpoint is None:
        return {"created": False, "message": "No new events to checkpoint."}
    return {"created": True, "checkpoint": checkpoint.to_dict()}


@app.get("/api/integrity/checkpoints")
def get_checkpoints(limit: int = Query(50, ge=1, le=500),
                    db: Session = Depends(get_db_session)):
    return {"checkpoints": [c.to_dict() for c in list_checkpoints(db, limit)]}


@app.get("/api/integrity/checkpoints/{checkpoint_uid}/verify")
async def verify_one_checkpoint(checkpoint_uid: str):
    return await asyncio.to_thread(verify_checkpoint, checkpoint_uid)


@app.post("/api/integrity/certificate")
async def make_certificate(
    start_time: Optional[str] = Form(None),
    end_time: Optional[str] = Form(None),
    issued_to: str = Form("Evidentiary Review"),
):
    """Export a verifiable integrity certificate covering a time range."""
    from core.timeutil import to_utc

    start = to_utc(start_time) if start_time else (now_utc() - timedelta(days=1))
    end = to_utc(end_time) if end_time else now_utc()
    if start is None or end is None:
        raise HTTPException(400, "start_time / end_time must be ISO-8601")

    def _build():
        db = SessionLocal()
        try:
            return generate_integrity_certificate(
                db, start.isoformat(), end.isoformat(), issued_to
            )
        finally:
            db.close()

    certificate = await asyncio.to_thread(_build)
    return JSONResponse(
        content=json.loads(export_integrity_certificate_to_json(certificate)),
        headers={
            "Content-Disposition":
                f'attachment; filename="ibvap_integrity_{certificate.certificate_id[:8]}.json"'
        },
    )


# --------------------------------------------------------------------------- #
# Watchlist (face recognition)
# --------------------------------------------------------------------------- #


@app.get("/api/watchlist")
def list_watchlist():
    from cv.face import get_face_recognizer

    recognizer = get_face_recognizer()
    return {"entries": recognizer.get_watchlist(), "status": recognizer.get_metrics()}


@app.post("/api/watchlist", status_code=201)
async def add_to_watchlist(
    name: str = Form(...),
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    from cv.face import get_face_recognizer

    recognizer = get_face_recognizer()
    if not recognizer._enabled:
        raise HTTPException(503, "Face recognition unavailable (InsightFace not installed)")

    content = await image.read()
    await image.close()
    if not content:
        raise HTTPException(400, "Empty image file")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Reference image must be under 20 MB")

    suffix = Path(sanitize_filename(image.filename or "face.jpg")).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(400, "Reference image must be JPG, PNG, BMP or WebP")

    try:
        meta = json.loads(metadata)
        if not isinstance(meta, dict):
            meta = {}
    except json.JSONDecodeError:
        meta = {}

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(content)
        watchlist_id = await asyncio.to_thread(
            recognizer.add_watchlist_entry, name.strip()[:120], tmp_name, meta
        )
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

    if watchlist_id is None:
        raise HTTPException(400, "No face could be detected in the uploaded image")
    return {"id": watchlist_id, "name": name.strip()[:120]}


@app.delete("/api/watchlist/{watchlist_id}")
def remove_from_watchlist(watchlist_id: int):
    from cv.face import get_face_recognizer

    if not get_face_recognizer().remove_watchlist_entry(watchlist_id):
        raise HTTPException(404, "Watchlist entry not found")
    return {"ok": True}


@app.put("/api/watchlist/threshold")
def set_watchlist_threshold(threshold: float = Form(...)):
    from cv.face import get_face_recognizer

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "threshold must be between 0.0 and 1.0")
    recognizer = get_face_recognizer()
    recognizer.set_threshold(threshold)
    return {"threshold": recognizer._similarity_threshold}


@app.get("/api/face/test/{camera_id}")
async def test_face_on_camera(camera_id: int):
    """Run face detection on a camera's current frame — a diagnostic probe."""
    from cv.face import get_face_recognizer

    frame = FrameBuffer.get().get_clean_frame(camera_id)
    if frame is None:
        raise HTTPException(404, "No frame available for this camera")

    recognizer = get_face_recognizer()
    if not recognizer._enabled:
        raise HTTPException(503, "Face recognition unavailable")

    matches = await asyncio.to_thread(
        recognizer.recognize, frame, None, 0, True
    )
    return {
        "camera_id": camera_id,
        "count": len(matches),
        "threshold": recognizer._similarity_threshold,
        "matches": [
            {
                "bbox": list(m.bbox),
                "track_id": m.track_id,
                "identified": bool(m.matched),
                "watchlist_name": m.watchlist_name if m.matched else None,
                "similarity": round(float(m.similarity), 4),
                "det_score": round(float(m.det_score), 4),
                "label": (f"WATCHLIST: {m.watchlist_name}" if m.matched
                          else "FACE DETECTED (identity not established)"),
            }
            for m in matches
        ],
        "checked_at_ist": fmt_ist(),
    }


# --------------------------------------------------------------------------- #
# ANPR
# --------------------------------------------------------------------------- #


@app.get("/api/anpr/status")
def get_anpr_status():
    from cv.anpr import get_anpr_processor

    return get_anpr_processor().get_metrics()


@app.get("/api/anpr/test/{camera_id}")
async def test_anpr_on_camera(camera_id: int):
    """Run plate localisation + OCR on a camera's current frame."""
    from cv.anpr import get_anpr_processor

    frame = FrameBuffer.get().get_clean_frame(camera_id)
    if frame is None:
        raise HTTPException(404, "No frame available for this camera")

    processor = get_anpr_processor()
    if not processor.is_available():
        raise HTTPException(503, "ANPR unavailable (EasyOCR not installed or disabled)")

    detections = await asyncio.to_thread(processor.recognize_plates, frame, None, 0)
    return {
        "camera_id": camera_id,
        "count": len(detections),
        "detections": [
            {
                "bbox": list(d.bbox),
                "plate_text": (d.plate_text
                               if d.text_confidence >= settings.ANPR_CONFIDENCE_THRESHOLD
                               else "PLATE UNCERTAIN"),
                "raw_text": d.plate_text,
                "ocr_confidence": round(d.text_confidence, 4),
                "localization_confidence": round(d.confidence, 4),
                "certain": d.text_confidence >= settings.ANPR_CONFIDENCE_THRESHOLD,
                "vehicle_class": d.vehicle_class,
            }
            for d in detections
        ],
        "checked_at_ist": fmt_ist(),
    }


# --------------------------------------------------------------------------- #
# Notifications — alarm siren & SMS
# --------------------------------------------------------------------------- #


def _alarm():
    from core.alarm import AlarmManager

    return AlarmManager.get()


def _sms():
    from core.sms import SMSNotifier

    return SMSNotifier.get()


@app.get("/api/system/notifications")
def notification_status():
    """
    State of every outbound escalation channel.

    Answers the three questions an operator actually has: is it switched on,
    does it have somewhere to deliver to, and has anything failed since boot.
    Reports honestly when a channel is off — a channel that is disabled is not
    an error, and must not be rendered as one.
    """
    alarm = _alarm().get_status()
    sms = _sms().get_status()
    return {
        "alarm": alarm,
        "sms": sms,
        "min_severity": str(settings.NOTIFY_MIN_SEVERITY).upper(),
        "escalating": [
            channel["channel"]
            for channel in (alarm, sms)
            if channel["enabled"] and channel["configured"]
        ],
        "checked_at_ist": fmt_ist(),
    }


@app.get("/api/system/alarm/status")
def alarm_status():
    """Alarm channel only — config, sinks, delivery counters, last error."""
    return _alarm().get_status()


@app.get("/api/system/sms/status")
def sms_status():
    """SMS channel only — provider, recipients, delivery counters, last error."""
    return _sms().get_status()


@app.post("/api/system/alarm/test")
async def test_alarm_endpoint():
    """
    Fire a test alarm so an operator can prove the siren wiring works.

    Runs off the event loop: the webhook can take its full timeout and a GPIO
    pulse deliberately holds its pin for several seconds.
    """
    alarm = _alarm()
    if not alarm.is_enabled():
        raise HTTPException(503, "Alarm channel is disabled (set ALARM_ENABLED=true)")
    if not alarm.is_configured():
        raise HTTPException(
            503, "Alarm channel has no sink configured "
                 "(set ALARM_WEBHOOK_URL or ALARM_GPIO_PIN)"
        )

    ok = await asyncio.to_thread(alarm.test_alarm)
    status = alarm.get_status()
    return {
        "ok": ok,
        "sinks": status["sinks"],
        "reference": status["last_reference"] if ok else "",
        "error": "" if ok else status["last_error"],
        "tested_at_ist": fmt_ist(),
    }


@app.post("/api/system/sms/test")
async def test_sms_endpoint():
    """Send a test SMS so an operator can prove the gateway route works."""
    sms = _sms()
    if not sms.is_enabled():
        raise HTTPException(503, "SMS channel is disabled (set SMS_ENABLED=true)")
    if not sms.is_configured():
        raise HTTPException(
            503, "SMS channel is not configured "
                 "(set SMS_TO_NUMBER and the provider credentials)"
        )

    ok = await asyncio.to_thread(sms.test_sms)
    status = sms.get_status()
    return {
        "ok": ok,
        "provider": status["provider"],
        "recipients": status["recipients"],
        "reference": status["last_reference"] if ok else "",
        "error": "" if ok else status["last_error"],
        "tested_at_ist": fmt_ist(),
    }


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #


@app.post("/api/system/evidence/sweep")
async def trigger_evidence_sweep():
    return await asyncio.to_thread(sweep_evidence)


@app.post("/api/system/hard-reset")
async def system_hard_reset(
    request: Request,
    confirm: bool = Query(
        False, description="Must be true. Guards against an accidental POST."
    ),
    wipe_evidence: Optional[bool] = Query(
        None,
        description="Also delete snapshots, clips, ANPR/face crops, source "
                    "videos and processed renders. Defaults to "
                    "HARD_RESET_WIPE_EVIDENCE.",
    ),
    token: str = Query("", description="HARD_RESET_TOKEN, if one is configured."),
):
    """
    Wipe the platform back to a clean, immediately usable state.

    **This destroys the audit chain.** Every camera, rule, event, checkpoint,
    analysis session, plate reading, face record and watchlist entry is
    deleted, every pipeline is stopped, and integrity verification restarts
    from genesis. It exists because an operator preparing a demonstration
    needs one honest way to start over, and because the alternative — deleting
    "most" of the log — would leave verification permanently broken.

    Three separate things must line up before it runs: the feature must be
    enabled, ``?confirm=true`` must be present, and the configured token (if
    any) must match. Evidence on disk is kept unless explicitly wiped, since a
    truncated database can be restored from a backup and deleted footage
    cannot.
    """
    if not settings.HARD_RESET_ENABLED:
        raise HTTPException(
            403,
            "Hard reset is disabled on this deployment. Set HARD_RESET_ENABLED=true, "
            "or run 'python manage.py hard-reset' on the host.",
        )
    if not confirm:
        raise HTTPException(
            400,
            "Hard reset deletes every camera, rule and sealed event, and resets "
            "the integrity chain. Repeat the request with ?confirm=true if that "
            "is what you intend.",
        )

    expected = (settings.HARD_RESET_TOKEN or "").strip()
    if expected:
        supplied = (request.headers.get("X-Reset-Token") or token or "").strip()
        # Constant-time: this is a shared secret, and the comparison is cheap.
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(
                403, "Invalid or missing reset token (X-Reset-Token)."
            )

    actor = request.client.host if request.client else "unknown"

    def _reset() -> dict:
        db = SessionLocal()
        try:
            return hard_reset(db, wipe_evidence=wipe_evidence, actor=actor)
        finally:
            db.close()

    try:
        result = await asyncio.to_thread(_reset)
    except Exception as exc:
        log.exception("Hard reset failed: %s", exc)
        raise HTTPException(500, f"Hard reset failed: {exc}")

    # Tell every open dashboard to empty itself rather than waiting for the
    # next poll to disagree with what it is showing.
    ws_manager.broadcast_threadsafe({"type": "system_reset", "data": result})
    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a stack trace to a client; always leave one in the log."""
    log.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": request.url.path,
                 "timestamp_ist": fmt_ist()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
