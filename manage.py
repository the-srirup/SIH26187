"""
IBVAP CLI — management commands for database, cameras, rules, and deployment.

Usage:
    python manage.py init          Initialize database and create tables
    python manage.py seed          Seed with demo cameras and rules
    python manage.py cameras       List all cameras
    python manage.py camera add    Add a new camera
    python manage.py camera rm     Remove a camera
    python manage.py rules         List rules for a camera
    python manage.py rule add      Add a rule
    python manage.py rule rm       Remove a rule
    python manage.py integrity     Verify hash chain integrity
    python manage.py stats         Show alert statistics
    python manage.py reset         Reset database (dangerous!)
    python manage.py run           Start the server
"""
from __future__ import annotations

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import settings
from core.database import init_db, SessionLocal
from core.models import Camera, Rule
from core.hashchain import verify_chain, latest_chain_hash
from datetime import datetime
from core.camera import CameraManager


def cmd_init(args):
    """Initialize the database."""
    init_db()
    print("✅ Database initialized successfully.")
    print(f"   Database: {settings.DATABASE_URL}")

    # Create default alert directories
    for d in [settings.ALERTS_DIR, settings.CLIPS_DIR, settings.SNAPSHOTS_DIR, settings.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Runtime directories created.")


def cmd_seed(args):
    """Seed the database with demo cameras."""
    init_db()
    db = SessionLocal()

    demos = [
        Camera(name="BOP-01 North", url="0", location="Northern Checkpost"),
        Camera(name="BOP-02 South", url="0", location="Southern Checkpost"),
        Camera(name="BOP-03 East", url="0", location="Eastern Perimeter"),
    ]

    added = 0
    for cam in demos:
        existing = db.query(Camera).filter(Camera.name == cam.name).first()
        if not existing:
            db.add(cam)
            added += 1

    db.commit()

    # Add a demo fence rule
    demo_cam = db.query(Camera).filter(Camera.name == "BOP-01 North").first()
    if demo_cam and not db.query(Rule).filter(Rule.camera_id == demo_cam.id).first():
        rule = Rule(
            camera_id=demo_cam.id,
            rule_type="line",
            geometry=json.dumps([[100, 240], [540, 240]]),
            params=json.dumps({"allowed_direction": "entry"}),
        )
        db.add(rule)
        added += 1

    db.commit()
    db.close()
    print(f"✅ Seeded {added} demo cameras/rules.")
    print("   Note: URL '0' means the default webcam (for testing).")
    print("   Update URLs to your actual camera streams.")


def cmd_cameras(args):
    """List all cameras."""
    db = SessionLocal()
    cameras = db.query(Camera).all()
    if not cameras:
        print("No cameras configured. Run 'manage.py seed' or add cameras via the dashboard.")
        db.close()
        return

    print(f"{'ID':<5} {'Name':<22} {'Location':<22} {'Active':<8} {'URL'}")
    print("-" * 90)
    for c in cameras:
        print(f"{c.id:<5} {c.name:<22} {c.location or 'N/A':<22} {'✓' if c.is_active else '✗':<8} {c.url}")
    db.close()


def cmd_camera_add(args):
    """Add a new camera."""
    init_db()
    db = SessionLocal()
    cam = Camera(name=args.name, url=args.url, location=args.location or "")
    db.add(cam)
    db.commit()
    db.refresh(cam)
    print(f"✅ Camera added: ID={cam.id}, Name='{cam.name}'")
    db.close()


def cmd_camera_rm(args):
    """Remove a camera."""
    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.id == args.camera_id).first()
    if not cam:
        print(f"❌ Camera {args.camera_id} not found.")
        db.close()
        return
    db.delete(cam)
    db.commit()
    print(f"✅ Camera {args.camera_id} removed.")
    db.close()


def cmd_rules(args):
    """List rules for a camera."""
    db = SessionLocal()
    rules = db.query(Rule).filter(Rule.camera_id == args.camera_id).all()
    if not rules:
        print(f"No rules for camera {args.camera_id}.")
        db.close()
        return

    for r in rules:
        geom = json.loads(r.geometry) if r.geometry else []
        params = json.loads(r.params) if r.params else {}
        print(f"ID={r.id}  Type={r.rule_type}  Points={len(geom)}  Params={params}")
    db.close()


def cmd_rule_add(args):
    """Add a new rule."""
    init_db()
    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.id == args.camera_id).first()
    if not cam:
        print(f"❌ Camera {args.camera_id} not found.")
        db.close()
        return

    import json

    geometry = json.loads(args.geometry) if args.geometry else []
    params = json.loads(args.params) if args.params else {}

    rule = Rule(
        camera_id=args.camera_id,
        rule_type=args.rule_type,
        geometry=json.dumps(geometry),
        params=json.dumps(params),
    )
    db.add(rule)
    db.commit()
    print(f"✅ Rule added: ID={rule.id}, Type={rule.rule_type}")
    db.close()
    CameraManager.get().reload_camera_rules(args.camera_id)


def cmd_rule_rm(args):
    """Remove a rule."""
    db = SessionLocal()
    rule = db.query(Rule).filter(Rule.id == args.rule_id).first()
    if not rule:
        print(f"❌ Rule {args.rule_id} not found.")
        db.close()
        return
    db.delete(rule)
    db.commit()
    print(f"✅ Rule {args.rule_id} removed.")
    db.close()
    CameraManager.get().reload_camera_rules(rule.camera_id)


def cmd_integrity(args):
    """Verify hash chain integrity."""
    db = SessionLocal()
    result = verify_chain(db)
    db.close()

    if result.valid:
        print(f"✅ Chain integrity verified — {result.total_alerts} alerts, chain intact.")
        if result.total_alerts > 0:
            print(f"   Chain tip: {latest_chain_hash()[:16]}...")
    else:
        print(f"🚨 INTEGRITY BREACH at alert #{result.broken_at}!")
        print(f"   Expected prev_hash: {result.expected_hash[:16]}...")
        print(f"   Actual prev_hash:   {result.actual_hash[:16]}...")
        print(f"   {result.message}")


def cmd_stats(args):
    """Show alert statistics."""
    from core.database import SessionLocal as _Sess
    from sqlalchemy import desc, func
    from core.models import Alert

    db = _Sess()
    total = db.query(Alert).count()
    by_type = db.query(Alert.alert_type, func.count(Alert.id)).group_by(Alert.alert_type).all()
    today = db.query(Alert).filter(
        Alert.timestamp >= datetime.now().strftime("%Y-%m-%d")
    ).count()

    print(f"Total alerts: {total}")
    print(f"Today: {today}")
    print("By type:")
    for t, c in by_type:
        print(f"  {t}: {c}")
    db.close()


def cmd_reset(args):
    """Reset database (DANGEROUS)."""
    confirm = input("⚠️  This will delete ALL data. Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    db = SessionLocal()
    db.query(Alert).delete()
    db.query(Rule).delete()
    db.query(Camera).delete()
    db.commit()
    db.close()
    print("✅ Database reset.")


def cmd_run(args):
    """Start the server."""
    import uvicorn
    print(f"🚀 Starting IBVAP server on {settings.HOST}:{settings.PORT}")
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)


def main():
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="IBVAP — Intelligent Border Video Analytics Platform (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize database")
    subparsers.add_parser("seed", help="Seed with demo data")
    subparsers.add_parser("cameras", help="List cameras")

    p_add = subparsers.add_parser("camera-add", help="Add a camera")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--url", required=True)
    p_add.add_argument("--location", default="")

    p_rm = subparsers.add_parser("camera-rm", help="Remove a camera")
    p_rm.add_argument("--camera-id", type=int, required=True)

    subparsers.add_parser("rules", help="List rules for a camera")
    p_rules_add = subparsers.add_parser("rule-add", help="Add a rule")
    p_rules_add.add_argument("--camera-id", type=int, required=True)
    p_rules_add.add_argument("--rule-type", required=True, choices=["line", "zone", "loiter", "direction"])
    p_rules_add.add_argument("--geometry", default="[]")
    p_rules_add.add_argument("--params", default="{}")
    p_rules_rm = subparsers.add_parser("rule-rm", help="Remove a rule")
    p_rules_rm.add_argument("--rule-id", type=int, required=True)

    subparsers.add_parser("integrity", help="Verify hash chain")
    subparsers.add_parser("stats", help="Show statistics")
    subparsers.add_parser("reset", help="Reset database (dangerous!)")
    subparsers.add_parser("run", help="Start the server")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "seed": cmd_seed,
        "cameras": cmd_cameras,
        "camera-add": cmd_camera_add,
        "camera-rm": cmd_camera_rm,
        "rules": cmd_rules,
        "rule-add": cmd_rule_add,
        "rule-rm": cmd_rule_rm,
        "integrity": cmd_integrity,
        "stats": cmd_stats,
        "reset": cmd_reset,
        "run": cmd_run,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
