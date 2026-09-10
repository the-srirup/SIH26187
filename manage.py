#!/usr/bin/env python
"""
IBVAP CLI — database, cameras, rules, integrity and server management.

    python manage.py init                 create tables
    python manage.py seed                 seed the demo camera + tripwire
    python manage.py run                  start the server
    python manage.py cameras              list cameras
    python manage.py camera-add           register a camera
    python manage.py camera-rm  --id 2    remove a camera
    python manage.py rules      --camera 1
    python manage.py rule-add   --camera 1 --type line --geometry "[[20,180],[620,180]]"
    python manage.py rule-rm    --id 3
    python manage.py alerts     --limit 20
    python manage.py integrity            verify the tamper-evident chain
    python manage.py checkpoint           seal a Merkle checkpoint
    python manage.py stats
    python manage.py sweep                enforce evidence retention now
    python manage.py reset                wipe the database (destructive)

Examples:
    python manage.py camera-add --name "BOP-02 SOUTH" --url 0 --location "South Gate"
    python manage.py camera-add --name "Gate Cam" --url "rtsp://192.168.1.100:554/stream"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _configure_console() -> bool:
    """
    Make the console UTF-8 capable, and report whether it worked.

    A default Windows console is cp1252 and raises UnicodeEncodeError on the
    tick, cross and ellipsis this CLI prints. Reconfiguring stdout fixes it on
    modern Python; where it cannot, callers fall back to ASCII markers rather
    than crashing a management command over a decorative character.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            return False
    return True


UNICODE_OK = _configure_console()
TICK = "✓" if UNICODE_OK else "[OK]"
CROSS = "✗" if UNICODE_OK else "[!!]"
ELLIPSIS = "…" if UNICODE_OK else "..."

from core.config import settings                                  # noqa: E402
from core.database import SessionLocal, init_db                    # noqa: E402
from core.models import Alert, AnalysisSession, Camera, Checkpoint, Rule  # noqa: E402
from core.timeutil import fmt_ist, utc_iso                         # noqa: E402

DEMO_FENCE = [[20, 180], [620, 180]]


def _session():
    init_db()
    return SessionLocal()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_init(_args) -> int:
    settings.ensure_dirs()
    init_db()
    print(f"Database ready: {settings.DATABASE_URL}")
    print(f"Storage: {settings.ALERTS_DIR}, {settings.CLIPS_DIR}, {settings.VIDEOS_DIR}")
    return 0


def cmd_seed(args) -> int:
    """Seed a demo camera pointed at the bundled sample clip, with a tripwire."""
    settings.ensure_dirs()
    db = _session()
    try:
        source = args.url or "samples/sample_border_scenario.mp4"
        if not str(source).isdigit() and not str(source).startswith(("rtsp", "http")):
            if not Path(source).exists():
                print(f"WARNING: {source} does not exist — the camera will show OFFLINE")

        existing = db.query(Camera).filter(Camera.url == source).first()
        if existing:
            print(f"Camera already registered: #{existing.id} {existing.name}")
            camera = existing
        else:
            camera = Camera(
                name=args.name, url=source, location=args.location,
                is_active=True, is_online=False, source_kind="live",
                created_at=utc_iso(),
            )
            db.add(camera)
            db.commit()
            db.refresh(camera)
            print(f"Camera #{camera.id} '{camera.name}' -> {camera.url}")

        if not db.query(Rule).filter(Rule.camera_id == camera.id).count():
            db.add(Rule(
                camera_id=camera.id, rule_type="line", name="PERIMETER TRIPWIRE",
                geometry=json.dumps(DEMO_FENCE), params="{}",
                is_active=True, created_at=utc_iso(),
            ))
            db.commit()
            print(f"Tripwire armed across {DEMO_FENCE}")
        else:
            print("Camera already has rules — left untouched")

        print("\nStart the platform with:  python manage.py run")
        return 0
    finally:
        db.close()


def cmd_cameras(_args) -> int:
    db = _session()
    try:
        cameras = db.query(Camera).order_by(Camera.id).all()
        if not cameras:
            print("No cameras registered. Run: python manage.py seed")
            return 0
        print(f"{'ID':>3}  {'NAME':<22} {'STATUS':<9} {'KIND':<7} {'RULES':>5}  SOURCE")
        print("-" * 92)
        for cam in cameras:
            rules = db.query(Rule).filter(Rule.camera_id == cam.id).count()
            status = ("ONLINE" if cam.is_online else
                      ("ACTIVE" if cam.is_active else "DISABLED"))
            print(f"{cam.id:>3}  {cam.name[:22]:<22} {status:<9} "
                  f"{(cam.source_kind or 'live'):<7} {rules:>5}  {cam.url[:38]}")
        return 0
    finally:
        db.close()


def cmd_camera_add(args) -> int:
    db = _session()
    try:
        camera = Camera(
            name=args.name, url=args.url, location=args.location or "",
            is_active=not args.disabled, is_online=False,
            source_kind="live", created_at=utc_iso(),
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)
        print(f"Added camera #{camera.id} '{camera.name}' -> {camera.url}")
        print("Restart the server (or use the dashboard) to bring it online.")
        return 0
    finally:
        db.close()


def cmd_camera_rm(args) -> int:
    db = _session()
    try:
        camera = db.query(Camera).filter(Camera.id == args.id).first()
        if not camera:
            print(f"No camera with id {args.id}")
            return 1
        name = camera.name
        db.delete(camera)
        db.commit()
        print(f"Removed camera #{args.id} '{name}' (its rules and alerts too)")
        return 0
    finally:
        db.close()


def cmd_rules(args) -> int:
    db = _session()
    try:
        query = db.query(Rule)
        if args.camera:
            query = query.filter(Rule.camera_id == args.camera)
        rules = query.order_by(Rule.id).all()
        if not rules:
            print("No rules defined.")
            return 0
        print(f"{'ID':>3}  {'CAM':>3}  {'TYPE':<10} {'ACTIVE':<7} {'NAME':<24} GEOMETRY")
        print("-" * 92)
        for rule in rules:
            geometry = (rule.geometry or "")[:34]
            print(f"{rule.id:>3}  {rule.camera_id:>3}  {rule.rule_type:<10} "
                  f"{str(bool(rule.is_active)):<7} {(rule.name or '')[:24]:<24} {geometry}")
        return 0
    finally:
        db.close()


def cmd_rule_add(args) -> int:
    db = _session()
    try:
        if not db.query(Camera).filter(Camera.id == args.camera).first():
            print(f"No camera with id {args.camera}")
            return 1
        try:
            geometry = json.loads(args.geometry)
        except json.JSONDecodeError as exc:
            print(f"Invalid geometry JSON: {exc}")
            return 1

        needed = 2 if args.type in ("line", "direction") else 3
        if not isinstance(geometry, list) or len(geometry) < needed:
            print(f"A '{args.type}' rule needs at least {needed} [x, y] points")
            return 1

        rule = Rule(
            camera_id=args.camera, rule_type=args.type,
            name=args.name or f"{args.type.upper()}-{args.camera}",
            geometry=json.dumps(geometry), params=args.params,
            is_active=True, created_at=utc_iso(),
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)
        print(f"Added rule #{rule.id} ({rule.rule_type}) to camera {args.camera}")
        return 0
    finally:
        db.close()


def cmd_rule_rm(args) -> int:
    db = _session()
    try:
        rule = db.query(Rule).filter(Rule.id == args.id).first()
        if not rule:
            print(f"No rule with id {args.id}")
            return 1
        db.delete(rule)
        db.commit()
        print(f"Removed rule #{args.id}")
        return 0
    finally:
        db.close()


def cmd_alerts(args) -> int:
    db = _session()
    try:
        query = db.query(Alert)
        if args.camera:
            query = query.filter(Alert.camera_id == args.camera)
        if args.type:
            query = query.filter(Alert.alert_type == args.type)
        alerts = query.order_by(Alert.id.desc()).limit(args.limit).all()
        if not alerts:
            print("No events recorded.")
            return 0
        print(f"{'ID':>5}  {'TIME (IST)':<26} {'SEVERITY':<9} {'TYPE':<18} "
              f"{'OBJECT':<12} {'EVIDENCE':<9} HASH")
        print("-" * 108)
        for alert in reversed(alerts):
            obj = f"{(alert.object_class or '-').upper()} #{alert.track_id or 0}"
            evidence = ("SNAP" if alert.snapshot_path else "") + \
                       ("+CLIP" if alert.clip_path else "")
            print(f"{alert.id:>5}  {(alert.timestamp_ist or fmt_ist(alert.timestamp)):<26} "
                  f"{(alert.severity or ''):<9} {alert.alert_type:<18} {obj:<12} "
                  f"{(evidence or '-'):<9} {(alert.hash or '')[:16]}{ELLIPSIS}")
        return 0
    finally:
        db.close()


def cmd_integrity(_args) -> int:
    from core.hashchain import verify_chain

    db = _session()
    try:
        result = verify_chain(db)
        print()
        if result.valid:
            print(f"  {TICK}  INTEGRITY VERIFIED")
        else:
            print("  ✕  INTEGRITY COMPROMISED")
        print()
        print(f"  Scheme        : SHA-256 hash chain (tamper-evident, local)")
        print(f"  Events        : {result.total_alerts}")
        print(f"  Verified at   : {result.verified_at_ist}")
        print(f"  Walk duration : {result.duration_ms} ms")
        print(f"  Chain tip     : {result.chain_tip}")
        print()
        print(f"  {result.message}")
        if not result.valid:
            print(f"  Broken at     : event #{result.broken_at}")
            print(f"  Expected      : {result.expected_hash}")
            print(f"  Stored        : {result.actual_hash}")
        print()
        return 0 if result.valid else 2
    finally:
        db.close()


def cmd_checkpoint(_args) -> int:
    from core.hashchain import create_checkpoint

    db = _session()
    try:
        checkpoint = create_checkpoint(db)
        if checkpoint is None:
            print("No new events to checkpoint.")
            return 0
        print(f"Checkpoint sealed over events "
              f"#{checkpoint.first_alert_id}-#{checkpoint.last_alert_id} "
              f"({checkpoint.alert_count} events)")
        print(f"  Merkle root : {checkpoint.merkle_root}")
        print(f"  Sealed at   : {checkpoint.timestamp_ist}")
        return 0
    finally:
        db.close()


def cmd_stats(_args) -> int:
    from sqlalchemy import func

    from core.evidence import evidence_usage
    from core.timeutil import start_of_ist_day

    db = _session()
    try:
        total = db.query(func.count(Alert.id)).scalar() or 0
        today = db.query(func.count(Alert.id)).filter(
            Alert.timestamp >= start_of_ist_day().isoformat()
        ).scalar() or 0

        print(f"\n  Cameras          : {db.query(func.count(Camera.id)).scalar()}")
        print(f"  Rules            : {db.query(func.count(Rule.id)).scalar()}")
        print(f"  Events (total)   : {total}")
        print(f"  Events (today)   : {today}   [since {fmt_ist(start_of_ist_day())}]")
        print(f"  Checkpoints      : {db.query(func.count(Checkpoint.id)).scalar()}")
        print(f"  Analysis runs    : {db.query(func.count(AnalysisSession.id)).scalar()}")

        by_type = (db.query(Alert.alert_type, func.count(Alert.id))
                   .group_by(Alert.alert_type)
                   .order_by(func.count(Alert.id).desc()).all())
        if by_type:
            print("\n  BY TYPE")
            for alert_type, count in by_type:
                print(f"    {alert_type:<22} {count:>6}")

        by_sev = (db.query(Alert.severity, func.count(Alert.id))
                  .group_by(Alert.severity).all())
        if by_sev:
            print("\n  BY SEVERITY")
            for severity, count in by_sev:
                print(f"    {(severity or '-'):<22} {count:>6}")

        usage = evidence_usage()
        print(f"\n  EVIDENCE")
        print(f"    snapshots            {usage['snapshots']:>6}")
        print(f"    clips                {usage['clips']:>6}")
        print(f"    processed videos     {usage['processed']:>6}")
        print(f"    disk used            {usage['megabytes']:>6} MB "
              f"of {usage['budget_mb']} MB\n")
        return 0
    finally:
        db.close()


def cmd_sweep(_args) -> int:
    from core.evidence import sweep_evidence

    result = sweep_evidence()
    print(f"Removed {result['files_removed']} file(s), "
          f"freed {result['bytes_freed'] / (1024 * 1024):.1f} MB")
    print(f"Remaining: {result['bytes_remaining'] / (1024 * 1024):.1f} MB "
          f"of {result['budget_mb']} MB budget")
    return 0


def cmd_reset(args) -> int:
    if not args.yes:
        answer = input("This deletes ALL cameras, rules and the event log. Type 'yes': ")
        if answer.strip().lower() != "yes":
            print("Aborted.")
            return 1

    from core.database import Base, engine

    Base.metadata.drop_all(bind=engine)
    init_db()
    print("Database reset. Run 'python manage.py seed' to re-seed the demo.")
    return 0


def cmd_run(args) -> int:
    import uvicorn

    settings.ensure_dirs()
    init_db()
    print(f"IBVAP {settings.VERSION} — dashboard at http://{args.host}:{args.port}/dashboard")
    uvicorn.run("api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="IBVAP management CLI",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create database tables")

    p_seed = sub.add_parser("seed", help="seed the demo camera and tripwire")
    p_seed.add_argument("--name", default="BOP-01 NORTH")
    p_seed.add_argument("--url", default="")
    p_seed.add_argument("--location", default="Northern Checkpost — Sector 4")

    sub.add_parser("cameras", help="list cameras")

    p_cadd = sub.add_parser("camera-add", help="register a camera")
    p_cadd.add_argument("--name", required=True)
    p_cadd.add_argument("--url", required=True,
                        help="rtsp://…, http://…, a webcam index like 0, or a file path")
    p_cadd.add_argument("--location", default="")
    p_cadd.add_argument("--disabled", action="store_true")

    p_crm = sub.add_parser("camera-rm", help="remove a camera")
    p_crm.add_argument("--id", type=int, required=True)

    p_rules = sub.add_parser("rules", help="list rules")
    p_rules.add_argument("--camera", type=int)

    p_radd = sub.add_parser("rule-add", help="add a rule")
    p_radd.add_argument("--camera", type=int, required=True)
    p_radd.add_argument("--type", required=True,
                        choices=["line", "zone", "loiter", "direction"])
    p_radd.add_argument("--geometry", required=True, help='e.g. "[[20,180],[620,180]]"')
    p_radd.add_argument("--params", default="{}")
    p_radd.add_argument("--name", default="")

    p_rrm = sub.add_parser("rule-rm", help="remove a rule")
    p_rrm.add_argument("--id", type=int, required=True)

    p_alerts = sub.add_parser("alerts", help="list recent events")
    p_alerts.add_argument("--limit", type=int, default=25)
    p_alerts.add_argument("--camera", type=int)
    p_alerts.add_argument("--type")

    sub.add_parser("integrity", help="verify the tamper-evident hash chain")
    sub.add_parser("checkpoint", help="seal a Merkle checkpoint")
    sub.add_parser("stats", help="show event and storage statistics")
    sub.add_parser("sweep", help="enforce evidence retention now")

    p_reset = sub.add_parser("reset", help="wipe the database (destructive)")
    p_reset.add_argument("--yes", action="store_true", help="skip confirmation")

    p_run = sub.add_parser("run", help="start the server")
    p_run.add_argument("--host", default=settings.HOST)
    p_run.add_argument("--port", type=int, default=settings.PORT)
    p_run.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    handler = globals()[f"cmd_{args.command.replace('-', '_')}"]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
