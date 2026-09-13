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
    python manage.py verify               audit every PS capability on real footage
    python manage.py prune                reclaim archived camera rows
    python manage.py run --fresh          start with an empty dashboard
    python manage.py reset                recreate the schema (destructive)
    python manage.py hard-reset --yes     wipe everything and start clean

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


def cmd_cameras(args) -> int:
    """
    List cameras the operator has.

    Archived cameras are hidden unless ``--all`` is given. They are removed as
    far as anyone using the system is concerned; listing them here would
    contradict every other view and is how a deployment ends up looking as
    though it is carrying dozens of dead sources.
    """
    db = _session()
    try:
        query = db.query(Camera)
        if not args.all:
            query = query.filter(Camera.is_deleted.is_(False))
        cameras = query.order_by(Camera.id).all()
        if not cameras:
            print("No cameras registered. Run: python manage.py seed")
            return 0
        print(f"{'ID':>3}  {'NAME':<22} {'STATUS':<9} {'KIND':<7} {'RULES':>5}  SOURCE")
        print("-" * 92)
        for cam in cameras:
            rules = db.query(Rule).filter(Rule.camera_id == cam.id).count()
            status = ("ARCHIVED" if cam.is_deleted else
                      "ONLINE" if cam.is_online else
                      "ACTIVE" if cam.is_active else "DISABLED")
            print(f"{cam.id:>3}  {cam.name[:22]:<22} {status:<9} "
                  f"{(cam.source_kind or 'live'):<7} {rules:>5}  {cam.url[:38]}")
        if not args.all:
            archived = db.query(Camera).filter(Camera.is_deleted.is_(True)).count()
            if archived:
                print(f"\n({archived} archived camera(s) hidden — "
                      f"'camera-rm' keeps a camera's sealed events. Use --all.)")
        return 0
    finally:
        db.close()


def cmd_camera_add(args) -> int:
    """
    Register a camera through the same path the API uses.

    Building the row here by hand skipped URL validation and the duplicate
    check, so the CLI could create what the API refuses: unreachable sources
    that hang a capture thread, and several rows pointing at one feed, each
    with its own decoder, tracker and rule engine doing identical work. That
    is how this project's database reached four copies of one sample clip.
    """
    from core.sources import SourceError, register_camera

    db = _session()
    try:
        camera = register_camera(
            db, name=args.name, url=args.url, location=args.location or "",
            is_active=not args.disabled, source_kind="live",
            allow_duplicate=bool(args.allow_duplicate),
        )
    except SourceError as exc:
        print(f"{CROSS}  {exc}")
        return 1
    finally:
        db.close()
    print(f"Added camera #{camera.id} '{camera.name}' -> {camera.url}")
    print("Restart the server (or use the dashboard) to bring it online.")
    return 0


def cmd_camera_rm(args) -> int:
    """
    Remove a camera through the same path the API uses.

    The previous implementation issued a bare ``DELETE`` on the row. Against a
    camera referenced by an analysis session or a plate/face record that is a
    ``FOREIGN KEY constraint failed`` and the command simply fails; against a
    camera that owns events it would have cascaded into ``alerts`` — a SHA-256
    hash chain in which every row's hash covers its predecessor's — so the
    audit log would verify as COMPROMISED from then on. ``retire_camera``
    deletes a camera that owns nothing and archives one that owns evidence,
    and it stops the camera's threads in this process first.
    """
    from core.sources import retire_camera

    db = _session()
    try:
        camera = db.query(Camera).filter(Camera.id == args.id).first()
        if camera is None:
            print(f"No camera with id {args.id}")
            return 1
        if camera.is_deleted:
            print(f"Camera #{args.id} '{camera.name}' is already removed.")
            return 0
        result = retire_camera(db, args.id)
    finally:
        db.close()

    print(f"Removed camera #{args.id} '{result.get('name', '')}' "
          f"({result['mode']})")
    print(f"  rules deleted        : {result.get('rules_removed', 0)}")
    if result["mode"] == "archived":
        print(f"  sealed events kept   : {result.get('alerts_retained', 0)}")
        print(f"  analysis runs kept   : {result.get('sessions_retained', 0)}")
        print("  The camera is hidden everywhere; its evidence stays verifiable.")
    if result.get("source_file_removed"):
        print("  source video deleted : yes")
    # The CLI and the API are separate processes. This command retires the row
    # and stops any pipeline *here*, but it cannot reach into a running server
    # to stop that server's camera threads. Saying so plainly beats implying a
    # teardown that did not happen — and beats guessing whether a server is up,
    # which cannot be told apart from this command's own WAL writes.
    print()
    print("If the API server is running, it releases this camera on its "
          "next restart.")
    print(f"To stop it now, use the dashboard or DELETE /api/cameras/{args.id}.")
    return 0


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
            print(f"  Fault type    : {result.break_kind}")
            print(f"  Faults found  : {len(result.breaks)}")
            print(f"  First at      : event #{result.broken_at}")
            for brk in result.breaks[:10]:
                origin = brk.get("forked_from_alert_id")
                print(f"    - #{brk['alert_id']:<6} {brk['kind']:<8}"
                      + (f" (shares predecessor with #{origin})" if origin else ""))
            if len(result.breaks) > 10:
                print(f"    … and {len(result.breaks) - 10} more")
            if result.forks_only:
                print()
                print("  Every event's own digest is valid — no record was altered")
                print("  or deleted. Re-link with: python manage.py chain-repair")
        print()
        return 0 if result.valid else 2
    finally:
        db.close()


def cmd_chain_repair(args) -> int:
    """Re-link a chain forked by concurrent writers. Refuses real tampering."""
    from core.hashchain import repair_chain

    db = _session()
    try:
        result = repair_chain(db, dry_run=args.dry_run, reseal=args.reseal)
        print()
        print(f"  {result['message']}")
        if not result["ok"]:
            print()
            return 2
        if not args.dry_run and result.get("repaired"):
            print(f"  Links rebuilt : {result['repaired']}")
            print("  A 'chain_repaired' event was appended recording this repair.")
        print()
        return 0
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
    """Drop and recreate the schema. Use `hard-reset` for a running system."""
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


def cmd_hard_reset(args) -> int:
    """
    Wipe the platform back to a clean, immediately usable state.

    Unlike ``reset``, this also stops every running camera pipeline, cancels
    in-flight analysis, clears the live frame buffer and the in-memory event
    history, and can wipe the evidence tree — so what is left is a system an
    operator can start adding cameras to, not just an empty schema.
    """
    from core.sources import hard_reset

    wipe = bool(args.evidence)
    if not args.yes:
        print()
        print("  This deletes EVERY camera, rule, sealed event, checkpoint,")
        print("  analysis run, plate reading, face record and watchlist entry.")
        print("  The integrity chain restarts from genesis and cannot be undone.")
        if wipe:
            print("  Snapshots, clips, crops and source videos will ALSO be deleted.")
        print()
        if input("  Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted.")
            return 1

    settings.ensure_dirs()
    db = _session()
    try:
        result = hard_reset(db, wipe_evidence=wipe, actor="manage.py")
    finally:
        db.close()

    rows = result["rows_deleted"]
    print()
    print(f"  {TICK}  HARD RESET COMPLETE  ({result['duration_ms']:.0f} ms)")
    print()
    print(f"  Cameras stopped   : {result['cameras_stopped']}")
    print(f"  Analyses cancelled: {result['analyses_cancelled']}")
    print(f"  Frames cleared    : {result['frames_cleared']}")
    print(f"  Rows deleted      : {result['rows_deleted_total']}")
    for table in sorted(rows):
        if rows[table]:
            print(f"    {table:<20} {rows[table]:>7}")
    if result["evidence"]["wiped"]:
        print(f"  Evidence removed  : {result['evidence']['files_removed']} file(s), "
              f"{result['evidence']['directories_removed']} folder(s)")
    else:
        print("  Evidence          : kept (pass --evidence to delete it)")
    print()
    print("  The system is empty and usable. Add a camera with:")
    print("    python manage.py camera-add --name \"CAM-01\" --url 0")
    print()
    return 0


def cmd_verify(args) -> int:
    """
    Run the capability audit — what the platform actually does, measured.

    Separate from the test suite on purpose. Tests prove the code behaves as
    written; this drives the real pipeline over real footage and reports, per
    problem-statement capability, whether anything came out the other end.
    """
    import verify_capabilities

    argv = ["verify_capabilities", "--seconds", str(args.seconds)]
    for clip in args.clip:
        argv += ["--clip", clip]
    if args.webcam:
        argv.append("--webcam")
    if args.json:
        argv += ["--json", args.json]

    saved, sys.argv = sys.argv, argv
    try:
        return verify_capabilities.main()
    finally:
        sys.argv = saved


def cmd_run(args) -> int:
    """
    Start the server.

    ``--fresh`` and ``--no-autostart`` are applied through the environment
    rather than by mutating ``settings`` here, because ``uvicorn.run`` with
    ``reload=True`` re-imports the application in a *separate process* — a
    setting changed in this one would be silently lost on the reload that
    matters most during development.
    """
    import uvicorn

    if args.fresh:
        os.environ["FRESH_START"] = "true"
    if args.no_autostart:
        os.environ["AUTOSTART_CAMERAS"] = "false"

    settings.ensure_dirs()
    init_db()
    if args.fresh:
        print("FRESH START — every camera is retired at boot; "
              "the dashboard comes up empty.")
    elif args.no_autostart:
        print("Auto-start disabled — registered cameras stay registered but "
              "are not brought up.")
    print(f"IBVAP {settings.VERSION} — dashboard at http://{args.host}:{args.port}/dashboard")
    uvicorn.run("api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


def cmd_prune(args) -> int:
    """
    Reclaim archived camera rows that no longer hold anything.

    Removal archives, rather than deletes, a camera that owns sealed events —
    ``alerts.camera_id`` is a foreign key into a hash-chained log that must not
    lose rows. Correct, and it means archived rows accumulate: this project's
    database reached 49 of them. Once the last dependant of one is gone the row
    is residue, and this removes it. Anything still holding evidence is
    reported and left alone.
    """
    from core.sources import prune_archived

    db = _session()
    try:
        result = prune_archived(db, dry_run=args.dry_run)
    finally:
        db.close()

    removed, kept = result["removed"], result["kept"]
    print()
    print(f"  Archived cameras : {result['archived_total']}")
    print(f"  Reclaimable      : {len(removed)}")
    print(f"  Still holding    : {len(kept)}")
    if removed:
        print()
        for row in removed[:20]:
            print(f"    {'would remove' if args.dry_run else 'removed'} "
                  f"#{row['id']:<4} {row['name'][:32]}")
        if len(removed) > 20:
            print(f"    … and {len(removed) - 20} more")
    if kept:
        print()
        print("  Kept — these still own evidence, and the chain needs their row:")
        for row in kept[:10]:
            owns = ", ".join(f"{k}={v}" for k, v in row.items()
                             if k not in ("id", "name") and v)
            print(f"    #{row['id']:<4} {row['name'][:28]:<28} {owns}")
        if len(kept) > 10:
            print(f"    … and {len(kept) - 10} more")
    print()
    if args.dry_run:
        print("  Nothing was written (--dry-run).")
    elif not removed:
        print("  Nothing to reclaim.")
    print()
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

    p_cams = sub.add_parser("cameras", help="list cameras")
    p_cams.add_argument("--all", action="store_true",
                        help="include archived (removed) cameras")

    p_cadd = sub.add_parser("camera-add", help="register a camera")
    p_cadd.add_argument("--name", required=True)
    p_cadd.add_argument("--url", required=True,
                        help="rtsp://…, http://…, a webcam index like 0, or a file path")
    p_cadd.add_argument("--location", default="")
    p_cadd.add_argument("--disabled", action="store_true")
    p_cadd.add_argument("--allow-duplicate", action="store_true",
                        help="register a source that is already registered "
                             "(doubles the load on one feed; rarely wanted)")

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
    p_repair = sub.add_parser(
        "chain-repair",
        help="re-link a hash chain forked by concurrent writers (refuses tampering)",
    )
    p_repair.add_argument("--dry-run", action="store_true",
                          help="report what would be re-linked without writing")
    p_repair.add_argument("--reseal", action="store_true",
                          help="re-seal every event under the current payload "
                               "schema (needed once after the schema version "
                               "changes; recorded in the log as a reseal)")
    sub.add_parser("checkpoint", help="seal a Merkle checkpoint")
    sub.add_parser("stats", help="show event and storage statistics")
    sub.add_parser("sweep", help="enforce evidence retention now")

    p_verify = sub.add_parser(
        "verify",
        help="audit every problem-statement capability against real footage",
    )
    p_verify.add_argument("--seconds", type=float, default=20.0,
                          help="seconds of each clip to analyse")
    p_verify.add_argument("--webcam", action="store_true",
                          help="also audit face detection against the local camera")
    p_verify.add_argument("--clip", action="append", default=[],
                          help="footage to audit (repeatable)")
    p_verify.add_argument("--json", help="write the full report to this path")

    p_reset = sub.add_parser("reset", help="recreate the schema (destructive)")
    p_reset.add_argument("--yes", action="store_true", help="skip confirmation")

    p_hard = sub.add_parser(
        "hard-reset",
        help="stop everything and wipe cameras, rules, events and evidence",
    )
    p_hard.add_argument("--yes", action="store_true", help="skip confirmation")
    p_hard.add_argument("--evidence", action="store_true",
                        help="also delete snapshots, clips, crops, source "
                             "videos and processed renders")

    p_prune = sub.add_parser(
        "prune", help="reclaim archived camera rows that hold no evidence")
    p_prune.add_argument("--dry-run", action="store_true",
                         help="report what would be removed, write nothing")

    p_run = sub.add_parser("run", help="start the server")
    p_run.add_argument("--host", default=settings.HOST)
    p_run.add_argument("--port", type=int, default=settings.PORT)
    p_run.add_argument("--reload", action="store_true")
    p_run.add_argument("--fresh", action="store_true",
                       help="retire every camera at boot — start as though "
                            "the software were newly installed")
    p_run.add_argument("--no-autostart", action="store_true",
                       help="keep registered cameras but do not bring them up")

    args = parser.parse_args()
    handler = globals()[f"cmd_{args.command.replace('-', '_')}"]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
