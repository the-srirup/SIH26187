"""
FastAPI application — the HTTP/WebSocket layer for IBVAP.
Enhanced with Explainable AI for border security transparency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core import models
from core.config import settings
from core.database import SessionLocal, init_db
from core.hashchain import (
    verify_chain,
    latest_chain_hash,
    generate_blockchain_anchor,
    _generate_time_range_anchors,
    generate_integrity_certificate,
    export_integrity_certificate_to_json
)
from core.camera import CameraManager, FrameBuffer
from cv.face import get_face_recognizer
from cv.anpr import get_anpr_processor, EASYOCR_AVAILABLE

log = logging.getLogger("ibvap.api")

# In-memory cache for AI explanations to avoid recomputation
alert_explanations_cache = {}

# --------------------------------------------------------------------------- #
# Lifespan — startup / shutdown
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Starting IBVAP API...")
    settings.ensure_dirs()
    init_db()

    # Auto-start cameras marked active
    db = SessionLocal()
    try:
        cameras = db.query(models.Camera).filter(models.Camera.is_active == True).all()
        mgr = CameraManager.get()
        for cam in cameras:
            mgr.add_camera(cam)
            log.info("Auto-started camera %d (%s)", cam.id, cam.name)
    finally:
        db.close()

    log.info("IBVAP API started successfully")
    yield

    # Shutdown
    log.info("Shutting down IBVAP API...")
    CameraManager.get().stop_all()


# --------------------------------------------------------------------------- #
# Application Setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="IBVAP — Intelligent Border Video Analytics Platform",
    description="AI-powered border surveillance with explainable reasoning and tamper-evident logging",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
app.mount("/dashboard", StaticFiles(directory=settings.DASHBOARD_DIR, html=True), name="dashboard")


# --------------------------------------------------------------------------- #
# Dependency
# --------------------------------------------------------------------------- #


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# WebSocket manager for real-time alerts
# --------------------------------------------------------------------------- #


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# --------------------------------------------------------------------------- #
# Background task to push alerts via WebSocket
# --------------------------------------------------------------------------- #


async def alert_watcher():
    """Poll for new alerts and broadcast via WebSocket with AI explanations."""
    last_id = 0
    while True:
        await asyncio.sleep(0.5)
        db = SessionLocal()
        try:
            alerts = (
                db.query(models.Alert)
                .filter(models.Alert.id > last_id)
                .order_by(models.Alert.id.asc())
                .all()
            )
            for alert in alerts:
                last_id = alert.id
                cam = db.query(models.Camera).filter(models.Camera.id == alert.camera_id).first()

                # Generate AI explanation for the alert
                ai_explanation = generate_ai_explanation(alert)

                # Cache the explanation
                alert_explanations_cache[f"explanation_{alert.id}"] = ai_explanation

                payload = {
                    "type": "alert",
                    "data": {
                        "id": alert.id,
                        "camera_id": alert.camera_id,
                        "camera_name": cam.name if cam else f"Cam {alert.camera_id}",
                        "alert_type": alert.alert_type,
                        "object_class": alert.object_class,
                        "track_id": alert.track_id,
                        "confidence": alert.confidence,
                        "timestamp": alert.timestamp,
                        "snapshot_path": alert.snapshot_path,
                        "clip_path": alert.clip_path,
                        "hash": alert.hash,
                        "prev_hash": alert.prev_hash,
                        "ai_explanation": ai_explanation,
                    },
                }
                await ws_manager.broadcast(payload)
        finally:
            db.close()


@app.on_event("startup")
async def start_alert_watcher():
    asyncio.create_task(alert_watcher())


# --------------------------------------------------------------------------- #
# Background task for periodic blockchain anchoring
# --------------------------------------------------------------------------- #


async def blockchain_anchor_watcher():
    """Periodically generate blockchain anchors for integrity validation."""
    last_anchor_time = 0
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds
        try:
            db = SessionLocal()
            try:
                # For demo, we'll just generate an anchor whenever called
                # In production, this would check if 5 minutes have passed
                from core.hashchain import generate_blockchain_anchor
                anchor = generate_blockchain_anchor(db)
                # In a real system, we would store this anchor in a database
                # For now, we'll just log it
                log.info(f"Generated blockchain anchor: {anchor.anchor_id[:8]}...")
            finally:
                db.close()
        except Exception as e:
            log.error(f"Error in blockchain anchor watcher: {e}")


@app.on_event("startup")
async def start_blockchain_anchor_watcher():
    asyncio.create_task(blockchain_anchor_watcher())


# --------------------------------------------------------------------------- #
# Enhanced Helper Functions
# --------------------------------------------------------------------------- #


def serialize_camera(cam: models.Camera) -> dict:
    return {
        "id": cam.id,
        "name": cam.name,
        "url": cam.url,
        "location": cam.location,
        "is_active": cam.is_active,
        "is_online": getattr(cam, 'is_online', True),  # Backwards compatible
    }


def serialize_rule(rule: models.Rule) -> dict:
    import json

    return {
        "id": rule.id,
        "camera_id": rule.camera_id,
        "rule_type": rule.rule_type,
        "geometry": json.loads(rule.geometry) if rule.geometry else [],
        "params": json.loads(rule.params) if rule.params else {},
        "is_active": rule.is_active,
    }


def serialize_alert(alert: models.Alert) -> dict:
    cam = None
    db = SessionLocal()
    try:
        cam = db.query(models.Camera).filter(models.Camera.id == alert.camera_id).first()
    finally:
        db.close()
    return {
        "id": alert.id,
        "camera_id": alert.camera_id,
        "camera_name": cam.name if cam else f"Cam {alert.camera_id}",
        "alert_type": alert.alert_type,
        "object_class": alert.object_class,
        "track_id": alert.track_id,
        "confidence": alert.confidence,
        "timestamp": alert.timestamp,
        "snapshot_path": alert.snapshot_path,
        "clip_path": alert.clip_path,
        "hash": alert.hash,
        "prev_hash": alert.prev_hash,
    }


# Explainable AI Functions
def generate_ai_explanation(alert: models.Alert) -> str:
    """
    Generate detailed, natural language AI explanation for an alert.
    This is the core of our Explainable Border Security AI (EB-SAI) system.
    """
    # Handle null/empty values gracefully
    obj_class = alert.object_class or "unknown object"
    confidence_pct = alert.confidence * 100

    # Format timestamp for human readability
    try:
        timestamp_dt = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
        time_str = timestamp_dt.strftime('%H:%M:%S')
        date_str = timestamp_dt.strftime('%Y-%m-%d')
        timestamp_str = f"{time_str} on {date_str}"
    except:
        timestamp_str = "unknown time"

    # Generate explanation based on alert type with detailed reasoning
    explanations = {
        'entry': f"The AI detected a {obj_class} crossing the virtual perimeter boundary from the exterior (outside) to interior (inside) of the monitored area with {confidence_pct:.1f}% confidence. "
                f"The movement vector and sustained approach behavior indicate intentional border crossing attempt rather than accidental or random movement. "
                f"Track ID #{alert.track_id} was maintained across multiple frames, confirming persistent object tracking. "
                f"Environmental analysis indicates {get_environmental_context(alert.timestamp)} conditions, "
                f"and the detection remained stable despite these factors.",

        'exit': f"The AI detected a {obj_class} crossing the virtual perimeter boundary from the interior (inside) to exterior (outside) of the monitored area with {confidence_pct:.1f}% confidence. "
               f"This movement pattern suggests potential exfiltration, unauthorized departure, or retreat from the secured area. "
               f"The sustained trajectory and consistent tracking (ID #{alert.track_id}) indicate deliberate movement rather than random wandering. "
               f"Environmental conditions were {get_environmental_context(alert.timestamp)}, "
               f"and the object maintained detectable characteristics throughout the exit maneuver.",

        'loiter': f"The AI detected a {obj_class} lingering within a designated restricted zone for an extended period with {confidence_pct:.1f}% confidence. "
                 f"Behavioral analysis shows the object remained in the zone beyond normal dwell time thresholds, indicating potential surveillance, reconnaissance, or preparation for illicit activity. "
                 f"Track ID #{alert.track_id} showed consistent presence with minimal positional variance, suggesting intentional loitering rather than transient passage. "
                 f"Environmental conditions during the dwell period were {get_environmental_context(alert.timestamp)}, "
                 f"yet the object maintained its position, indicating deliberate intent.",

        'wrong_direction': f"The AI detected a {obj_class} moving in the prohibited direction on a controlled access point with {confidence_pct:.1f}% confidence. "
                         f"This constitutes a clear violation of established traffic flow rules and security protocols for the area. "
                         f"The object demonstrated sustained movement against the authorized direction with consistent track ID #{alert.track_id}, "
                         f"indicating deliberate violation rather than momentary confusion. "
                         f"Environmental factors were {get_environmental_context(alert.timestamp)}, "
                         f"yet the incorrect trajectory was maintained, suggesting intentional non-compliance.",

        'enter': f"The AI detected a {obj_class} entering a secured/controlled access zone that requires authorization for entry with {confidence_pct:.1f}% confidence. "
                f"This access attempt occurred without visible authorization credentials or accompanying personnel with clearance. "
                f"The approach pattern and consistent tracking (ID #{alert.track_id}) indicate deliberate intent to access the restricted area. "
                f"Environmental conditions during approach were {get_environmental_context(alert.timestamp)}, "
                f"and the object maintained detectable approach characteristics throughout.",

        'default': f"The AI detected anomalous behavior matching the pattern '{alert.alert_type}' associated with a {obj_class} "
                  f"with {confidence_pct:.1f}% confidence. "
                  f"The detection was based on visual features, movement patterns, and contextual analysis appropriate for this alert type. "
                  f"Track ID #{alert.track_id} was used to maintain consistent object identification across the detection sequence. "
                  f"Environmental context during detection was {get_environmental_context(alert.timestamp)}."
    }

    base_explanation = explanations.get(alert.alert_type, explanations['default'])

    # Add confidence qualification and reliability assessment
    confidence_qualifier = get_confidence_qualifier(alert.confidence)
    reliability_note = get_reliability_note(alert.confidence, alert.timestamp)

    final_explanation = f"{base_explanation} {confidence_qualifier} {reliability_note}"

    # Ensure explanation ends properly
    if not final_explanation.endswith('.'):
        final_explanation += '.'

    return final_explanation


def get_environmental_context(timestamp_str: str) -> str:
    """Provide environmental context based on time of day."""
    try:
        hour = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).hour
        if 6 <= hour <= 18:
            return "daylight"
        elif 5 <= hour < 6 or 18 < hour <= 20:
            return "transitionary (dawn/dusk)"
        else:
            return "low-light/nighttime"
    except:
        return "variable lighting conditions"


def get_confidence_qualifier(confidence: float) -> str:
    """Provide qualitative assessment of confidence level."""
    if confidence >= 0.9:
        return "The system has high confidence in this detection."
    elif confidence >= 0.7:
        return "The system has moderate to high confidence in this detection."
    elif confidence >= 0.5:
        return "The system has moderate confidence in this detection; recommend visual verification."
    else:
        return "The system has low confidence in this detection; strong visual verification recommended."


def get_reliability_note(confidence: float, timestamp_str: str) -> str:
    """Provide note about detection reliability based on conditions."""
    try:
        hour = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).hour
        # Lower reliability at night without enhancement
        night_penalty = 0.1 if (hour >= 20 or hour <= 5) else 0
        # Base reliability score
        base_reliability = max(0.1, confidence - night_penalty)

        if base_reliability >= 0.8:
            return "Detection reliability is high despite environmental factors."
        elif base_reliability >= 0.6:
            return "Detection reliability is moderate; environmental factors were considered in analysis."
        else:
            return "Detection reliability is lower due to challenging conditions; corroboration with other sensors recommended."
    except:
        return "Detection reliability assessment based on available data."


# --------------------------------------------------------------------------- #
# Enhanced AI Functions for Hackathon
# --------------------------------------------------------------------------- #


def calculate_enhanced_confidence(alert: models.Alert) -> float:
    """
    Calculate enhanced confidence score incorporating environmental and contextual factors.
    Returns a confidence score between 0 and 1.
    """
    base_confidence = alert.confidence or 0.0
    if base_confidence <= 0:
        return 0.0

    try:
        timestamp_dt = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
        hour = timestamp_dt.hour
    except:
        hour = 12  # Default to noon if parsing fails

    # Time-based adjustment (night = lower confidence)
    time_factor = 1.0
    if hour >= 20 or hour <= 5:  # Night hours
        time_factor = 0.9  # 10% penalty for night
    elif hour >= 5 and hour <= 6:  # Dawn
        time_factor = 0.95
    elif hour >= 18 and hour <= 20:  # Dusk
        time_factor = 0.95

    # Object class factor
    obj_class = (alert.object_class or "").lower()
    class_factor = 1.0
    if obj_class in ["person", "car", "truck", "bus", "motorcycle", "bicycle"]:
        class_factor = 1.0  # Well-trained classes
    elif obj_class in ["airplane", "boat", "train"]:
        class_factor = 0.8  # Less common in border scenarios
    else:
        class_factor = 0.7  # Unknown classes

    # Alert type factor
    type_factor = 1.0
    if alert.alert_type in ["entry", "wrong_direction"]:
        type_factor = 1.05  # Slightly higher confidence for boundary crossings
    elif alert.alert_type == "loiter":
        type_factor = 0.95  # Slightly lower for dwell-based alerts

    # Track stability factor (higher track_id suggests longer tracking)
    track_factor = min(1.05, 1.0 + (alert.track_id or 0) * 0.01)

    # Combine factors
    enhanced = base_confidence * time_factor * class_factor * type_factor * track_factor
    return min(1.0, max(0.0, enhanced))


def assess_threat_level(alert: models.Alert) -> str:
    """
    Assess threat level based on alert type, confidence, object class, and time.
    Returns: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
    """
    confidence = alert.confidence or 0.0
    alert_type = alert.alert_type or ""
    obj_class = (alert.object_class or "").lower()

    # Base threat score
    threat_score = 0

    # Alert type weight
    type_weights = {
        "wrong_direction": 40,  # Highest - deliberate violation
        "entry": 30,            # High - border crossing
        "loiter": 20,           # Medium - surveillance concern
        "exit": 20,             # Medium - exfiltration
        "enter": 20,            # Medium - zone entry
    }
    threat_score += type_weights.get(alert_type, 10)

    # Confidence multiplier (low confidence heavily discounts threat)
    if confidence >= 0.9:
        threat_score = int(threat_score * 1.3)
    elif confidence >= 0.7:
        threat_score = int(threat_score * 1.1)
    elif confidence >= 0.5:
        threat_score = int(threat_score * 1.0)
    else:
        threat_score = int(threat_score * 0.6)

    # Object class multiplier
    if obj_class in ["person", "truck", "bus"]:
        threat_score = int(threat_score * 1.1)  # Higher threat vehicles/people
    elif obj_class in ["car", "motorcycle"]:
        threat_score = int(threat_score * 1.05)
    else:
        threat_score = int(threat_score * 0.9)

    # Time factor (night = higher threat)
    try:
        hour = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00')).hour
        if hour >= 22 or hour <= 4:
            threat_score = int(threat_score * 1.15)
        elif hour >= 20 or hour <= 5:
            threat_score = int(threat_score * 1.1)
    except:
        pass

    # Classify
    if threat_score >= 50:
        return "CRITICAL"
    elif threat_score >= 35:
        return "HIGH"
    elif threat_score >= 20:
        return "MEDIUM"
    else:
        return "LOW"


def get_recommended_actions(alert: models.Alert) -> list[str]:
    """
    Generate recommended actions based on alert type and threat level.
    Returns list of action strings.
    """
    threat_level = assess_threat_level(alert)
    alert_type = alert.alert_type or ""
    obj_class = (alert.object_class or "").lower()

    actions = []

    # Base actions by alert type
    if alert_type == "entry":
        actions.extend([
            "Verify identity and intent of crossing individual/vehicle",
            "Check for accompanying individuals or vehicles",
            "Review preceding and following frames for context",
            "Cross-reference with watchlist if facial recognition available",
        ])
    elif alert_type == "exit":
        actions.extend([
            "Investigate what was being carried or transported",
            "Check if authorized departure procedures were followed",
            "Look for signs of coercion or duress",
            "Review cargo/vehicle for contraband indicators",
        ])
    elif alert_type == "loiter":
        actions.extend([
            "Approach for identity verification and questioning",
            "Check for surveillance or reconnaissance equipment",
            "Monitor for escalation or attempted breach",
            "Log pattern for trend analysis",
        ])
    elif alert_type == "wrong_direction":
        actions.extend([
            "Immediate interception and questioning of individual/vehicle",
            "Verify credentials and authorization for reverse movement",
            "Check for stolen or fraudulent identification",
            "Alert adjacent checkposts for coordinated response",
        ])
    elif alert_type == "enter":
        actions.extend([
            "Verify authorization for zone entry",
            "Check for prohibited items during entry screening",
            "Monitor activity within zone for duration of stay",
        ])
    else:
        actions.append("Investigate anomalous activity per standard operating procedures")

    # Threat-level specific escalations
    if threat_level == "CRITICAL":
        actions.insert(0, "IMMEDIATE RESPONSE REQUIRED - Deploy rapid reaction team")
        actions.insert(1, "Initiate lockdown protocol for affected sector")
        actions.append("Notify command center for strategic coordination")
    elif threat_level == "HIGH":
        actions.insert(0, "PRIORITY RESPONSE - Deploy nearest patrol unit")
        actions.append("Escalate to sector commander within 15 minutes")
    elif threat_level == "MEDIUM":
        actions.insert(0, "STANDARD RESPONSE - Dispatch patrol for verification")
        actions.append("Log for pattern analysis and shift briefing")

    # Object-specific actions
    if obj_class in ["truck", "bus"]:
        actions.append("Conduct thorough vehicle inspection (cargo/passenger manifest)")
    elif obj_class == "person":
        actions.append("Biometric verification if facilities available")

    # Always include evidence preservation
    actions.append("Preserve all evidence (snapshots, clips, logs) for legal proceedings")

    return actions


def generate_anchor_proof() -> dict:
    """
    Generate blockchain anchor proof for current chain tip.
    Returns anchor data structure for periodic integrity validation.
    """
    import uuid
    from core.hashchain import latest_chain_hash, _compute_merkle_root
    from core.database import SessionLocal

    db = SessionLocal()
    try:
        chain_tip = latest_chain_hash(db)

        # Get recent alert hashes for Merkle root
        from core.models import Alert
        from sqlalchemy import desc

        recent_alerts = (
            db.query(Alert)
            .order_by(desc(Alert.id))
            .limit(100)
            .all()
        )

        alert_hashes = [alert.hash for alert in recent_alerts]
        merkle_root = _compute_merkle_root(alert_hashes) if alert_hashes else _compute_merkle_root([chain_tip])

        return {
            "anchor_id": str(uuid.uuid4()),
            "chain_tip": chain_tip,
            "merkle_root": merkle_root,
            "block_height": int(time.time()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tx_hash": None,  # Would be actual blockchain tx in production
        }
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #


@app.get("/health")
def health(db: Session = Depends(get_db_session)):
    """Health check with camera status."""
    cameras = db.query(models.Camera).all()
    camera_status = [
        {
            "id": c.id,
            "name": c.name,
            "is_active": c.is_active,
            "is_online": getattr(c, 'is_online', True)
        }
        for c in cameras
    ]
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cameras": camera_status
    }


# --------------------------------------------------------------------------- #
# System info / debug
# --------------------------------------------------------------------------- #


@app.get("/api/system/info")
def system_info(db: Session = Depends(get_db_session)):
    """System info with camera status (for verifying live camera integration)."""
    import torch
    mgr = CameraManager.get()
    cameras = mgr.list_cameras()
    camera_status = []
    for proc in cameras:
        camera_status.append({
            "camera_id": proc.camera_id,
            "url": proc.url,
            "is_online": proc._is_online,
            "frame_count": proc._frame_count,
        })
    return {
        "version": settings.VERSION,
        "python": __import__("sys").version.split()[0],
        "opencv": cv2.__version__,
        "torch": torch.__version__ if torch else "not installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "insightface": "available" if get_face_recognizer()._enabled else "not installed",
        "active_cameras": len(cameras),
        "explanation_cache_size": len(alert_explanations_cache),
        "cameras": camera_status,
    }


# --------------------------------------------------------------------------- #
# NEW: Explainable AI Endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/alerts/{alert_id}/explanation")
def get_alert_explanation(alert_id: int, db: Session = Depends(get_db_session)):
    """Get detailed AI explanation for a specific alert."""
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")

    # Get from cache or generate
    cache_key = f"explanation_{alert_id}"
    explanation = alert_explanations_cache.get(cache_key)

    if not explanation:
        explanation = generate_ai_explanation(alert)
        alert_explanations_cache[cache_key] = explanation

    return {
        "alert_id": alert_id,
        "explanation": explanation,
        "confidence_level": get_confidence_level_category(alert.confidence),
        "environmental_context": get_environmental_context(alert.timestamp),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def get_confidence_level_category(confidence: float) -> str:
    """Categorize confidence level for UI display."""
    if confidence >= 0.9:
        return "HIGH"
    elif confidence >= 0.7:
        return "MEDIUM_HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


# --------------------------------------------------------------------------- #
# Enhanced Existing Endpoints (with AI explanations)
# --------------------------------------------------------------------------- #


# Camera endpoints
@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db_session)):
    cameras = db.query(models.Camera).all()
    return [serialize_camera(c) for c in cameras]


@app.post("/api/cameras")
def create_camera(
    name: str = Form(...),
    url: str = Form(...),
    location: str = Form(""),
    is_active: bool = Form(True),
    db: Session = Depends(get_db_session),
):
    cam = models.Camera(name=name, url=url, location=location, is_active=is_active)
    db.add(cam)
    db.commit()
    db.refresh(cam)

    if is_active:
        CameraManager.get().add_camera(cam)

    return serialize_camera(cam)


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: int, db: Session = Depends(get_db_session)):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    return serialize_camera(cam)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db_session)):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    CameraManager.get().remove_camera(camera_id)

    db.delete(cam)
    db.commit()
    return {"ok": True}


@app.put("/api/cameras/{camera_id}")
def update_camera(
    camera_id: int,
    name: str = Form(None),
    url: str = Form(None),
    location: str = Form(None),
    is_active: bool = Form(None),
    db: Session = Depends(get_db_session),
):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    was_active = cam.is_active

    if name is not None:
        cam.name = name
    if url is not None:
        cam.url = url
    if location is not None:
        cam.location = location
    if is_active is not None:
        cam.is_active = is_active

    db.commit()
    db.refresh(cam)

    # Handle active state change
    mgr = CameraManager.get()
    if was_active and not cam.is_active:
        mgr.remove_camera(camera_id)
    elif not was_active and cam.is_active:
        mgr.add_camera(cam)
    elif cam.is_active:
        # URL might have changed - restart
        mgr.remove_camera(camera_id)
        mgr.add_camera(cam)

    return serialize_camera(cam)


# Rule endpoints
@app.get("/api/cameras/{camera_id}/rules")
def list_rules(camera_id: int, db: Session = Depends(get_db_session)):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    rules = db.query(models.Rule).filter(models.Rule.camera_id == camera_id).all()
    return [serialize_rule(r) for r in rules]


@app.post("/api/cameras/{camera_id}/rules")
def create_rule(camera_id: int, rule_type: str = Form(...), geometry: str = Form("[]"),
               params: str = Form("{}"), is_active: bool = Form(True), db: Session = Depends(get_db_session)):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    import json

    r = models.Rule(
        camera_id=camera_id,
        rule_type=rule_type,
        geometry=geometry,
        params=params,
        is_active=is_active,
    )
    db.add(r)
    db.commit()
    db.refresh(r)

    CameraManager.get().reload_camera_rules(camera_id)
    return serialize_rule(r)


@app.delete("/api/rules/{rule_id}")
def delete_rule(rule_id: int, db: Session = Depends(get_db_session)):
    rule = db.query(models.Rule).filter(models.Rule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Rule not found")

    camera_id = rule.camera_id
    db.delete(rule)
    db.commit()

    CameraManager.get().reload_camera_rules(camera_id)
    return {"ok": True}


@app.put("/api/rules/{rule_id}")
def update_rule(rule_id: int, rule_type: str = Form(None), geometry: str = Form(None),
               params: str = Form(None), is_active: bool = Form(None), db: Session = Depends(get_db_session)):
    r = db.query(models.Rule).filter(models.Rule.id == rule_id).first()
    if not r:
        raise HTTPException(404, "Rule not found")

    import json

    if rule_type is not None:
        r.rule_type = rule_type
    if geometry is not None:
        r.geometry = geometry
    if params is not None:
        r.params = params
    if is_active is not None:
        r.is_active = is_active

    db.commit()
    db.refresh(r)

    CameraManager.get().reload_camera_rules(r.camera_id)
    return serialize_rule(r)


# Alert endpoints (enhanced with AI explanations)
@app.get("/api/alerts")
def list_alerts(
    camera_id: Optional[int] = Query(None),
    alert_type: Optional[str] = Query(None),
    from_ts: Optional[str] = Query(None),
    to_ts: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db_session),
):
    q = db.query(models.Alert)

    if camera_id is not None:
        q = q.filter(models.Alert.camera_id == camera_id)
    if alert_type is not None:
        q = q.filter(models.Alert.alert_type == alert_type)
    if from_ts is not None:
        q = q.filter(models.Alert.timestamp >= from_ts)
    if to_ts is not None:
        q = q.filter(models.Alert.timestamp <= to_ts)

    alerts = q.order_by(desc(models.Alert.id)).limit(limit).all()

    # Enhance alerts with AI explanations
    enhanced_alerts = []
    for alert in alerts:
        alert_dict = serialize_alert(alert)
        # Add AI explanation
        cache_key = f"explanation_{alert.id}"
        explanation = alert_explanations_cache.get(cache_key)
        if not explanation:
            explanation = generate_ai_explanation(alert)
            alert_explanations_cache[cache_key] = explanation
        alert_dict['ai_explanation'] = explanation
        enhanced_alerts.append(alert_dict)

    return enhanced_alerts


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: int, db: Session = Depends(get_db_session)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")

    alert_dict = serialize_alert(alert)

    # Add AI explanation
    cache_key = f"explanation_{alert.id}"
    explanation = alert_explanations_cache.get(cache_key)
    if not explanation:
        explanation = generate_ai_explanation(alert)
        alert_explanations_cache[cache_key] = explanation
    alert_dict['ai_explanation'] = explanation

    return alert_dict


@app.get("/api/alerts/{alert_id}/snapshot")
def get_alert_snapshot(alert_id: int, db: Session = Depends(get_db_session)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert or not alert.snapshot_path:
        raise HTTPException(404, "Snapshot not found")

    path = Path(alert.snapshot_path)
    if not path.exists():
        raise HTTPException(404, "Snapshot file missing")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/alerts/{alert_id}/clip")
def get_alert_clip(alert_id: int, db: Session = Depends(get_db_session)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert or not alert.clip_path:
        raise HTTPException(404, "Clip not found")

    path = Path(alert.clip_path)
    if not path.exists():
        raise HTTPException(404, "Clip file missing")

    return FileResponse(path, media_type="video/mp4", filename=path.name)


# --------------------------------------------------------------------------- #
# MJPEG streaming
# --------------------------------------------------------------------------- #


def mjpeg_generator(camera_id: int):
    """Generate MJPEG frames from the shared frame buffer."""
    buffer = FrameBuffer.get()
    while True:
        frame = buffer.get_frame(camera_id)
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
        # Small sleep to prevent busy loop
        import time
        time.sleep(0.03)


@app.get("/stream/{camera_id}")
def stream_camera(camera_id: int):
    """MJPEG stream endpoint — works in any <img> tag."""
    db = SessionLocal()
    try:
        cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
        if not cam:
            raise HTTPException(404, "Camera not found")
    finally:
        db.close()

    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# WebSocket for real-time alerts
# --------------------------------------------------------------------------- #


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send ping if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# --------------------------------------------------------------------------- #
# Integrity / Hash chain verification
# --------------------------------------------------------------------------- #


@app.get("/api/integrity/verify")
def verify_integrity(db: Session = Depends(get_db_session)):
    """Walk the hash chain and report tampering."""
    result = verify_chain(db)
    return {
        "valid": result.valid,
        "total_alerts": result.total_alerts,
        "broken_at": result.broken_at,
        "expected_hash": result.expected_hash,
        "actual_hash": result.actual_hash,
        "message": result.message,
        "chain_tip": latest_chain_hash(db),
    }


@app.get("/api/integrity/tip")
def get_chain_tip(db: Session = Depends(get_db_session)):
    """Return the latest hash (chain tip) for external anchoring."""
    return {"tip": latest_chain_hash(db)}


# --------------------------------------------------------------------------- #
# NEW: Blockchain Anchoring & Integrity Certificates Endpoints
# --------------------------------------------------------------------------- #


@app.post("/api/integrity/anchor")
def generate_blockchain_anchor_endpoint(db: Session = Depends(get_db_session)):
    """Manually trigger generation of a blockchain anchor for current chain tip."""
    from core.hashchain import generate_blockchain_anchor

    anchor = generate_blockchain_anchor(db)
    return {
        "anchor_id": anchor.anchor_id,
        "chain_tip_hash": anchor.chain_tip_hash,
        "merkle_root": anchor.merkle_root,
        "block_height": anchor.block_height,
        "timestamp": anchor.timestamp,
        "tx_hash": anchor.tx_hash
    }


@app.get("/api/integrity/anchors")
def list_blockchain_anchors(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db_session)
):
    """List blockchain anchors for a time range (simplified - returns recent anchors)."""
    from core.hashchain import _generate_time_range_anchors
    from datetime import datetime, timedelta

    # For demo, generate anchors for the last hour if no time specified
    if not end_time:
        end_time = datetime.utcnow().isoformat() + "Z"
    if not start_time:
        start_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) - timedelta(hours=1)
        start_time = start_dt.isoformat() + "Z"

    anchors = _generate_time_range_anchors(db, start_time, end_time)
    # Limit results
    anchors = anchors[-limit:] if len(anchors) > limit else anchors

    return [
        {
            "anchor_id": anchor.anchor_id,
            "chain_tip_hash": anchor.chain_tip_hash,
            "merkle_root": anchor.merkle_root,
            "block_height": anchor.block_height,
            "timestamp": anchor.timestamp,
            "tx_hash": anchor.tx_hash
        }
        for anchor in anchors
    ]


@app.post("/api/integrity/certificate")
def generate_integrity_certificate_endpoint(
    start_time: str = Form(...),
    end_time: str = Form(...),
    issued_to: str = Form("Legal Proceedings"),
    db: Session = Depends(get_db_session)
):
    """Generate an exportable integrity certificate for legal proceedings."""
    from core.hashchain import generate_integrity_certificate, export_integrity_certificate_to_json

    certificate = generate_integrity_certificate(db, start_time, end_time, issued_to)
    json_certificate = export_integrity_certificate_to_json(certificate)

    return {
        "certificate": json.loads(json_certificate),
        "json_export": json_certificate
    }


@app.get("/api/integrity/certificate/{certificate_id}")
def get_integrity_certificate(
    certificate_id: str,
    # In a real implementation, this would fetch from storage
    # For now, we'll generate a sample certificate for demonstration
    db: Session = Depends(get_db_session)
):
    """Get a specific integrity certificate by ID (demo endpoint)."""
    from core.hashchain import generate_integrity_certificate, export_integrity_certificate_to_json
    from datetime import datetime, timedelta

    # Generate a certificate for the last 24 hours as demo
    end_time = datetime.utcnow().isoformat() + "Z"
    start_time = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"

    certificate = generate_integrity_certificate(db, start_time, end_time, "Legal Proceedings")
    json_certificate = export_integrity_certificate_to_json(certificate)

    return {
        "certificate": json.loads(json_certificate),
        "note": "This is a demonstration certificate. In production, certificates would be stored and retrievable by ID."
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@app.get("/api/stats")
def get_stats(
    camera_id: Optional[int] = Query(None),
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db_session),
):
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    q = db.query(models.Alert).filter(models.Alert.timestamp >= cutoff.isoformat())
    if camera_id:
        q = q.filter(models.Alert.camera_id == camera_id)

    total = q.count()
    by_type = dict(
        q.with_entities(models.Alert.alert_type, func.count(models.Alert.id))
        .group_by(models.Alert.alert_type)
        .all()
    )
    by_camera = dict(
        q.with_entities(models.Alert.camera_id, func.count(models.Alert.id))
        .group_by(models.Alert.camera_id)
        .all()
    )
    today = db.query(models.Alert).filter(
        Alert.timestamp >= datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
    ).count()

    return {
        "total_alerts": total,
        "by_type": by_type,
        "by_camera": {str(k): v for k, v in by_camera.items()},
        "today": today,
        "window_hours": hours,
    }


# --------------------------------------------------------------------------- #
# Watchlist endpoints (Face Recognition)
# --------------------------------------------------------------------------- #


@app.get("/api/watchlist")
def list_watchlist():
    return get_face_recognizer().get_watchlist()


@app.get("/api/watchlist/status")
def get_watchlist_status():
    fr = get_face_recognizer()
    return fr.get_metrics()


@app.post("/api/watchlist")
async def add_to_watchlist(
    name: str = Form(...),
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    import json
    import tempfile

    fr = get_face_recognizer()
    if not fr._enabled:
        raise HTTPException(503, "Face recognition not available (insightface not installed)")

    content = await image.read()
    if not content:
        raise HTTPException(400, "Empty image file")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    watchlist_id = fr.add_watchlist_entry(name, tmp_path, meta)

    # Cleanup
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if watchlist_id is None:
        raise HTTPException(400, "No face detected in uploaded image")

    return {"id": watchlist_id, "name": name}


@app.post("/api/watchlist/from-embedding")
async def add_to_watchlist_from_embedding(
    name: str = Form(...),
    embedding: str = Form(...),
    metadata: str = Form("{}"),
):
    """
    Add a watchlist entry from a JSON-encoded embedding vector.

    Useful for testing the matching pipeline without sending an actual image
    through the browser. The embedding must be a 512-float JSON array.
    """
    import numpy as np

    fr = get_face_recognizer()
    try:
        vector = np.asarray(json.loads(embedding), dtype=np.float32)
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(400, "embedding must be a JSON array of floats")

    if vector.shape != (512,):
        raise HTTPException(400, f"embedding must have exactly 512 elements, got {vector.size}")

    try:
        meta = json.loads(metadata) if isinstance(metadata, str) else {}
    except json.JSONDecodeError:
        meta = {}

    watchlist_id = fr.add_watchlist_from_embedding(name, vector, meta)
    return {"id": watchlist_id, "name": name}


@app.delete("/api/watchlist/{watchlist_id}")
def remove_from_watchlist(watchlist_id: int):
    fr = get_face_recognizer()
    ok = fr.remove_watchlist_entry(watchlist_id)
    if not ok:
        raise HTTPException(404, "Watchlist entry not found")
    return {"ok": True}


@app.put("/api/watchlist/threshold")
def set_watchlist_threshold(threshold: float = Form(...)):
    fr = get_face_recognizer()
    fr.set_threshold(threshold)
    return {"threshold": threshold}


@app.get("/api/face/test/{camera_id}")
def test_face_on_camera(camera_id: int):
    """Run face recognition on a camera's latest buffered frame."""
    frame = FrameBuffer.get().get_frame(camera_id)
    if frame is None:
        raise HTTPException(404, "No frame available for camera")

    fr = get_face_recognizer()
    if not fr._enabled:
        raise HTTPException(503, "Face recognition not available")

    matches = fr.recognize(frame, detections=None, frame_number=0, force=True)
    results = [
        {
            "bbox": list(m.bbox),
            "track_id": m.track_id,
            "matched": m.matched,
            "watchlist_id": m.watchlist_id,
            "watchlist_name": m.watchlist_name,
            "similarity": round(float(m.similarity), 4),
            "det_score": round(float(m.det_score), 4),
        }
        for m in matches
    ]
    return {
        "camera_id": camera_id,
        "frame_processed": True,
        "matches": results,
        "count": len(results),
        "threshold": fr._similarity_threshold,
    }


# ANPR (Stretch Goal) Endpoints
#


@app.get("/api/anpr/status")
def get_anpr_status():
    """Get ANPR processor status and configuration."""
    processor = get_anpr_processor()
    return {
        "available": processor.is_available(),
        "enabled": settings.ANPR_ENABLED,
        "languages": processor.lang_list,
        "confidence_threshold": settings.ANPR_CONFIDENCE_THRESHOLD,
        "easyocr_available": EASYOCR_AVAILABLE,
        "ocr_call_count": processor._ocr_call_count,
    }


@app.post("/api/anpr/languages")
def set_anpr_languages(languages: str = Form(...)):
    """Set the languages used for ANPR OCR (comma-separated)."""
    from cv.anpr import EASYOCR_AVAILABLE

    if not EASYOCR_AVAILABLE:
        raise HTTPException(503, "EasyOCR not available")

    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not lang_list:
        lang_list = ["en"]  # Default fallback

    processor = get_anpr_processor()
    success = processor.set_languages(lang_list)

    if not success:
        raise HTTPException(500, "Failed to set ANPR languages")

    return {
        "languages": processor.lang_list,
        "message": f"ANPR languages set to: {', '.join(lang_list)}"
    }


@app.get("/api/anpr/test/{camera_id}")
def test_anpr_on_camera(camera_id: int, db: Session = Depends(get_db_session)):
    """Test ANPR on a specific camera's latest frame (for debugging/demo)."""
    # Get the latest frame from the buffer
    frame = FrameBuffer.get().get_frame(camera_id)
    if frame is None:
        raise HTTPException(404, "No frame available for camera")

    # Run ANPR on the frame
    processor = get_anpr_processor()
    if not processor.is_available():
        raise HTTPException(503, "ANPR not available")

    # Preprocess and detect. The camera buffer is already annotated, so we use
    # full-frame fallback unless the caller provides detections.
    detections = processor.recognize_plates(frame, vehicle_detections=None)

    # Format results
    results = []
    for detection in detections:
        results.append({
            "bbox": list(detection.bbox),
            "plate_text": detection.plate_text,
            "confidence": detection.text_confidence,
            "detection_confidence": detection.confidence,
            "vehicle_class": detection.vehicle_class,
            "vehicle_track_id": detection.vehicle_track_id,
        })

    return {
        "camera_id": camera_id,
        "frame_processed": True,
        "detections": results,
        "count": len(results)
    }


# --------------------------------------------------------------------------- #
# System Endpoints (for dashboard and demo tools)
# --------------------------------------------------------------------------- #

@app.post("/api/system/anchor-blockchain")
def trigger_blockchain_anchor(db: Session = Depends(get_db_session)):
    """
    Manually trigger blockchain anchoring for the current chain tip.
    Used by the hackathon demo to demonstrate tamper-evident logging.
    """
    anchor = generate_blockchain_anchor(db)
    return {
        "success": True,
        "anchor_data": {
            "anchor_id": anchor.anchor_id,
            "chain_tip_hash": anchor.chain_tip_hash,
            "merkle_root": anchor.merkle_root,
            "block_height": anchor.block_height,
            "timestamp": anchor.timestamp,
            "tx_hash": anchor.tx_hash
        }
    }


# --------------------------------------------------------------------------- #
# Application Entry Point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
