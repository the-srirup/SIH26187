"""
IBVAP Demo Runner — generates synthetic video, seeds the database,
and launches the server with a beautiful demo experience.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.config import settings
from core.database import init_db, SessionLocal
from core.models import Camera, Rule

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
log = logging.getLogger("ibvap.demo")


def create_demo_video(path: str, seconds: int = 30):
    """Generate a synthetic test video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        f"{path}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("Created demo video: %s", path)
    except subprocess.CalledProcessError as e:
        log.warning("ffmpeg failed: %s", e.stderr.decode()[:200])
    except FileNotFoundError:
        log.warning("ffmpeg not found — skipping video generation")


def seed_demo():
    """Seed database with demo cameras and rules."""
    init_db()
    db = SessionLocal()

    # Create demo cameras pointing to synthetic video files
    demos = [
        Camera(name="BOP-01 North Perimeter", url="0", location="Northern Border Checkpost"),
        Camera(name="BOP-02 South Perimeter", url="0", location="Southern Border Checkpost"),
        Camera(name="BOP-03 East Checkpoint", url="0", location="Eastern Checkpoint"),
    ]

    for cam in demos:
        existing = db.query(Camera).filter(Camera.name == cam.name).first()
        if not existing:
            db.add(cam)
            log.info("Added camera: %s", cam.name)

    db.commit()

    # Add demo rules for BOP-01
    cam = db.query(Camera).filter(Camera.name == "BOP-01 North Perimeter").first()
    if cam:
        rules_data = [
            {
                "rule_type": "line",
                "geometry": [[100, 240], [540, 240]],
                "params": {"allowed_direction": "entry"},
                "name": "North Perimeter Fence",
            },
            {
                "rule_type": "zone",
                "geometry": [[200, 300], [400, 300], [400, 400], [200, 400]],
                "params": {},
                "name": "Restricted Zone Alpha",
            },
            {
                "rule_type": "loiter",
                "geometry": [[200, 300], [400, 300], [400, 400], [200, 400]],
                "params": {"dwell_seconds": 15},
                "name": "Loiter Detection Zone",
            },
            {
                "rule_type": "direction",
                "geometry": [[320, 0], [320, 480]],
                "params": {"allowed_direction": "entry"},
                "name": "One-Way Checkpoint",
            },
        ]

        for rdata in rules_data:
            existing = db.query(Rule).filter(
                Rule.camera_id == cam.id,
                Rule.rule_type == rdata["rule_type"],
            ).first()
            if not existing:
                rule = Rule(
                    camera_id=cam.id,
                    rule_type=rdata["rule_type"],
                    geometry=json.dumps(rdata["geometry"]),
                    params=json.dumps(rdata["params"]),
                    name=rdata.get("name", ""),
                )
                db.add(rule)
                log.info("Added rule: %s", rdata.get("name", rdata["rule_type"]))

    db.commit()
    db.close()
    log.info("Demo data seeded successfully!")


def run(args):
    """Launch the demo."""
    print("=" * 60)
    print("  IBVAP — Intelligent Border Video Analytics Platform")
    print("  Demo Mode — Generating test data & launching server")
    print("=" * 60)

    # Ensure directories exist
    settings.ensure_dirs()

    # Create demo video files
    demo_dir = Path("videos")
    demo_dir.mkdir(exist_ok=True)
    for i in range(1, 4):
        vid_path = demo_dir / f"demo_cam{i}.mp4"
        create_demo_video(str(vid_path))

    # Seed database
    seed_demo()

    # Update camera URLs to point to demo videos
    db = SessionLocal()
    for cam in db.query(Camera).all():
        cam.url = str(demo_dir / f"demo_cam{cam.id}.mp4")
    db.commit()
    db.close()

    print()
    print("🚀 Starting IBVAP server...")
    print()
    print("  Dashboard:  http://localhost:8000/dashboard")
    print("  API Docs:   http://localhost:8000/docs")
    print("  Health:     http://localhost:8000/health")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    # Launch server
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBVAP Demo Runner")
    parser.add_argument("--skip-server", action="store_true", help="Only seed data, don't start server")
    args = parser.parse_args()

    if args.skip_server:
        seed_demo()
    else:
        run(args)
