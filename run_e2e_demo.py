"""
IBVAP End-to-End Demonstration Runner

Seeds DB with sample video, starts server, waits for processing,
captures all outputs (alerts, snapshots, API responses, integrity reports)
and saves them to DEMO_OUTPUTS/live_test/ for visual verification.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(message)s")
log = logging.getLogger("ibvap.e2e")

# --------------------------------------------------------------------------- #
# 1. Seed database with real sample video
# --------------------------------------------------------------------------- #
from core.config import settings
from core.database import init_db, SessionLocal
from core.models import Camera, Rule

OUT_DIR = Path("DEMO_OUTPUTS/live_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean DB for fresh run
db_path = Path("alerts.db")
if db_path.exists():
    db_path.unlink()

init_db()
db = SessionLocal()

# Point cameras at the REAL downloaded sample video
video_path = str(Path("samples/sample_border_scenario.mp4").resolve())
cam = Camera(
    name="BOP-01 North Perimeter",
    url=video_path,
    location="Northern Border Checkpost (sample video)",
    is_active=True,
)
db.add(cam)
db.commit()
db.refresh(cam)
log.info("Created camera %d → %s", cam.id, video_path)

# Add fence + zone + loiter rules that the video content can trigger
rules_data = [
    {
        "rule_type": "line",
        "geometry": [[200, 240], [568, 240]],
        "params": {"allowed_direction": "entry"},
        "name": "North Perimeter Fence",
    },
    {
        "rule_type": "zone",
        "geometry": [[150, 300], [350, 300], [350, 430], [150, 430]],
        "params": {},
        "name": "Restricted Zone",
    },
    {
        "rule_type": "loiter",
        "geometry": [[400, 100], [620, 100], [620, 400], [400, 400]],
        "params": {"dwell_seconds": 10},
        "name": "Loiter Watch",
    },
]

for rdata in rules_data:
    rule = Rule(
        camera_id=cam.id,
        rule_type=rdata["rule_type"],
        geometry=json.dumps(rdata["geometry"]),
        params=json.dumps(rdata["params"]),
        name=rdata.get("name", ""),
    )
    db.add(rule)
    log.info("Added rule: %s", rdata["name"])
db.commit()
db.close()

# --------------------------------------------------------------------------- #
# 2. Start server in background
# --------------------------------------------------------------------------- #
import subprocess
import signal

# Clear prior output
for p in OUT_DIR.glob("*"):
    p.unlink() if p.is_file() else shutil.rmtree(p)

log.info("Starting IBVAP server with sample video...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    stdout=open(OUT_DIR / "server.log", "w"),
    stderr=subprocess.STDOUT,
)

# Wait for server readiness
for _ in range(60):
    time.sleep(1)
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8000/health")
        if resp.status == 200:
            log.info("Server ready!")
            break
    except Exception:
        continue
else:
    log.error("Server failed to start within 60s")
    server_proc.kill()
    sys.exit(1)

# --------------------------------------------------------------------------- #
# 3. Wait for processing + capture outputs
# --------------------------------------------------------------------------- #
# The sample video is 54 seconds at 12fps with persons/bicycles/cars.
# Let it process most of it, then capture everything.
import urllib.request
import cv2
import numpy as np

log.info("Waiting for video processing...")
time.sleep(20)  # Let ~20 seconds of video process

# --- Capture API responses ---
def fetch_json(url):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=5).read().decode())
    except Exception as e:
        return {"error": str(e)}

results = {}

# Health
results["health"] = fetch_json("http://localhost:8000/health")

# Cameras
results["cameras"] = fetch_json("http://localhost:8000/api/cameras")

# Alerts
results["alerts"] = fetch_json("http://localhost:8000/api/alerts")

# Alert explanations (if any alerts exist)
alerts = results.get("alerts", [])
if isinstance(alerts, list) and len(alerts) > 0:
    alert_id = alerts[0]["id"]
    results["alert_explanation"] = fetch_json(f"http://localhost:8000/api/alerts/{alert_id}/explanation")

# Integrity status
results["integrity_status"] = fetch_json("http://localhost:8000/api/integrity/verify")

# Integrity certificate (POST endpoint)
results["integrity_certificate"] = {"note": "POST endpoint, skipped in GET-only test"}

# ANPR status
results["anpr_status"] = fetch_json("http://localhost:8000/api/anpr/status")

# ANPR languages (POST endpoint)
results["anpr_languages"] = {"note": "POST endpoint, skipped in GET-only test"}

# System info
results["system_info"] = fetch_json("http://localhost:8000/api/system/info")

# --- Capture snapshot frames from MJPEG stream ---
log.info("Capturing stream snapshots...")
try:
    # Open MJPEG stream (camera ID is 1)
    stream_url = "http://localhost:8000/stream/1"
    stream_resp = urllib.request.urlopen(stream_url, timeout=10)
    # Read raw bytes until we find a JPEG boundary
    raw = b""
    jpeg_count = 0
    start_time = time.time()
    while jpeg_count < 3 and (time.time() - start_time) < 10:
        chunk = stream_resp.read(8192)
        if not chunk:
            break
        raw += chunk
        # Find JPEG boundaries
        idx = raw.find(b"\xff\xd8")
        end_idx = raw.find(b"\xff\xd9", idx + 2)
        if idx >= 0 and end_idx >= 0:
            jpeg_data = raw[idx:end_idx+2]
            raw = raw[end_idx+2:]
            jpeg_count += 1
            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                fname = OUT_DIR / f"stream_frame_{jpeg_count}.jpg"
                cv2.imwrite(str(fname), frame)
                log.info("Saved stream frame %d: %s (%dx%d)",
                         jpeg_count, fname, frame.shape[1], frame.shape[0])
except Exception as e:
    log.warning("Stream capture error: %s", e)

# --- Let it process more ---
time.sleep(15)

# --- Second capture after more processing ---
results["alerts_after"] = fetch_json("http://localhost:8000/api/alerts")

# --- Also, generate a test MJPEG frame via direct capture ---
try:
    # Try reading the frame buffer directly using a camera endpoint
    resp = urllib.request.urlopen("http://localhost:8000/stream/1", timeout=5)
    raw2 = b""
    for _ in range(3):
        chunk = resp.read(65536)
        if not chunk:
            break
        raw2 += chunk
    idx = raw2.find(b"\xff\xd8")
    end_idx = raw2.find(b"\xff\xd9", idx + 2)
    if idx >= 0 and end_idx >= 0:
        jpeg_data = raw2[idx:end_idx+2]
        frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(str(OUT_DIR / "live_annotated_frame.jpg"), frame)
            log.info("Saved live annotated frame")
except Exception as e:
    log.warning("Second stream capture error: %s", e)

# --------------------------------------------------------------------------- #
# 4. Save results
# --------------------------------------------------------------------------- #
with open(OUT_DIR / "api_responses.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Check evidence files
snap_dir = Path("alerts/snapshots")
snaps = list(snap_dir.glob("*.jpg")) if snap_dir.exists() else []
clip_dir = Path("clips")
clips = list(clip_dir.glob("*.mp4")) if clip_dir.exists() else []

log.info("=" * 60)
log.info("END-TO-END TEST RESULTS")
log.info("=" * 60)
log.info("Total alerts: %s", len(results.get("alerts", [])))
log.info("Alerts (2nd poll): %s", len(results.get("alerts_after", [])))
log.info("Snapshots found: %d", len(snaps))
log.info("Clips found: %d", len(clips))
log.info("Health: %s", results.get("health", {}))
log.info("Integrity valid: %s", results.get("integrity_status", {}).get("valid"))
log.info("ANPR available: %s", results.get("anpr_status", {}).get("available"))
log.info("")

# Print first few alerts with explanations
if isinstance(alerts, list) and len(alerts) > 0:
    log.info("SAMPLE ALERTS:")
    for i, a in enumerate(alerts[:5]):
        log.info("  Alert %d: type=%s obj=%s conf=%.2f track=%s",
                 a.get("id", i), a.get("alert_type", "?"),
                 a.get("object_class", "?"), a.get("confidence", 0),
                 a.get("track_id", "?"))
        if a.get("explanation"):
            log.info("    Explanation: %s", a["explanation"][:120])
else:
    log.info("⚠️  NO ALERTS GENERATED — check if YOLO detects objects in video")

log.info("")
log.info("Outputs saved to: %s", OUT_DIR)
log.info("  - api_responses.json (all API responses)")
log.info("  - stream_frame_*.jpg (stream snapshots)")
log.info("  - live_annotated_frame.jpg (annotated frame)")
log.info("  - server.log (server output)")

# --------------------------------------------------------------------------- #
# 5. Cleanup
# --------------------------------------------------------------------------- #
server_proc.send_signal(signal.SIGINT)
time.sleep(3)
server_proc.kill()
log.info("Server stopped. E2E test complete.")