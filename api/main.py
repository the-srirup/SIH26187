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
import tempfile
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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core import models
from core.analysis import (
    AnalysisManager, UploadValidationError, ensure_upload_camera,
    rules_for_camera, sanitize_filename,
)
from core.camera import CameraManager, FrameBuffer
from core.config import settings
from core.database import SessionLocal, init_db
from core.evidence import evidence_usage, is_safe_evidence_path, sweep_evidence
from core.events import EventManager, serialize_alert_row
from core.sources import (
    KIND_FILE, KIND_LIVE, SourceError, get_live_camera, register_camera,
    retire_camera, startup_cameras, store_source_video, visible_cameras,
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
        file_handler = logging.FileHandler(settings.LOG_DIR / "ibvap.log", encoding="utf-8")
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
        async with self._lock:
            self._connections.pop(websocket, None)

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
        for queue in list(self._connections.values()):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

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

    # Load the detector off the event loop so startup never blocks the server.
    await asyncio.to_thread(_load_detector)

    db = SessionLocal()
    try:
        cameras = startup_cameras(db)
        manager = CameraManager.get()
        started = 0
        for cam in cameras:
            # One unreachable source must not prevent the others from starting.
            if manager.add_camera(cam) is not None:
                started += 1
            else:
                log.warning("Camera %d (%s) could not be started at boot",
                            cam.id, cam.name)
        log.info("Auto-started %d of %d camera(s)", started, len(cameras))
    except Exception as exc:
        log.exception("Camera auto-start failed: %s", exc)
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
        CameraManager.get().stop_all()
        log.info("Shutdown complete at %s", fmt_ist())


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
    """Enforce evidence retention so an unattended deployment cannot fill disk."""
    while True:
        try:
            await asyncio.sleep(settings.EVIDENCE_SWEEP_INTERVAL)
            await asyncio.to_thread(sweep_evidence)
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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    """Liveness + a one-glance view of every registered source."""
    manager = CameraManager.get()
    live = {proc.camera_id: proc for proc in manager.list_cameras()}
    cameras = db.query(models.Camera).filter(models.Camera.source_kind != "upload").all()
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": utc_iso(),
        "timestamp_ist": fmt_ist(),
        "timezone": "Asia/Kolkata (IST, UTC+05:30)",
        "cameras": [
            {
                "id": c.id,
                "name": c.name,
                "is_active": c.is_active,
                "is_online": c.id in live and live[c.id].is_online,
                "fps": round(live[c.id]._fps, 1) if c.id in live else 0.0,
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
def create_camera(
    name: str = Form(...),
    url: str = Form(...),
    location: str = Form(""),
    is_active: bool = Form(True),
    db: Session = Depends(get_db_session),
):
    """Register a network camera: RTSP / HTTP URL, or a webcam index."""
    try:
        cam = register_camera(
            db, name=name, url=url, location=location,
            is_active=is_active, source_kind=KIND_LIVE,
        )
    except SourceError as exc:
        raise HTTPException(400, str(exc))

    proc = CameraManager.get().add_camera(cam) if is_active else None
    return serialize_camera(cam, proc)


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
def delete_camera(camera_id: int, db: Session = Depends(get_db_session)):
    """
    Remove a camera: stop its threads, release its handles, drop its rules.

    Delegates to :func:`core.sources.retire_camera`, which tears the runtime
    down before touching the database and preserves sealed evidence.  The call
    is idempotent — removing an already-removed camera returns success rather
    than 404, so a double-click or a retried request cannot produce an error.
    """
    try:
        return retire_camera(db, camera_id)
    except Exception as exc:
        log.exception("Camera removal failed for %d: %s", camera_id, exc)
        raise HTTPException(500, f"Could not remove camera: {exc}")


@app.post("/api/cameras/{camera_id}/restart")
def restart_camera(camera_id: int, db: Session = Depends(get_db_session)):
    cam = get_live_camera(db, camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    manager = CameraManager.get()
    manager.remove_camera(camera_id)
    proc = manager.add_camera(cam)
    return {"ok": proc is not None, "camera": serialize_camera(cam, proc)}


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
    very pool every synchronous endpoint depends on. The poll interval bounds
    the added latency to a few milliseconds.
    """
    buffer = FrameBuffer.get()
    last_seq = -1
    idle_since = time.monotonic()
    poll = 1.0 / max(10, settings.TARGET_FPS * 2)
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "

    while True:
        if await request.is_disconnected():
            break

        seq = buffer.sequence(camera_id)
        if seq <= last_seq:
            # Nothing new yet. Hold the connection open while a camera
            # reconnects rather than tearing the viewer's stream down, but
            # do not keep a dead stream alive forever.
            if time.monotonic() - idle_since > 120:
                break
            await asyncio.sleep(poll)
            continue

        jpeg = buffer.get_jpeg(camera_id)
        last_seq = seq
        idle_since = time.monotonic()
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
        done, pending = await asyncio.wait(
            {pump, drain}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WebSocket closed: %s", exc)
    finally:
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
# Maintenance
# --------------------------------------------------------------------------- #


@app.post("/api/system/evidence/sweep")
async def trigger_evidence_sweep():
    return await asyncio.to_thread(sweep_evidence)


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
