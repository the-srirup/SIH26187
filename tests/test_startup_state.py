"""
What survives a termination, and what must not.

The reported symptom was that cameras "were not getting destroyed" when the
project was stopped — the next run came up with the previous run's cameras
already there, one of them showing a green ONLINE dot before any processor
existed. Reproduced by killing a live server with ``taskkill /F``:

    {'id': 1, 'name': 'KILLTEST', 'is_active': 1, 'is_online': 1}

``is_online`` is a *runtime* fact, persisted so the camera list can render
without waiting for the first stats frame. After an ungraceful exit nothing is
receiving anything and the row is simply lying — and a shutdown hook cannot fix
it, because a killed process never reaches one. So reconciliation happens on the
way **in**, where it runs no matter how the previous run ended.

These tests also pin the unbounded stores: rows that accumulate for ever and a
log file that had no size limit at all.
"""
from __future__ import annotations

import pytest

from core.config import settings
from core.models import Alert, Camera, Rule
from core.timeutil import utc_iso


def _camera(db, name="SURVIVOR", *, active=True, online=False, deleted=False):
    cam = Camera(name=name, url=f"rtsp://example/{name}", location="",
                 is_active=active, is_online=online, source_kind="live",
                 is_deleted=deleted, deleted_at="", created_at=utc_iso())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def _seal(db, camera_id):
    from core.events import EventManager
    from cv.rules import Alert as RuleAlert

    return EventManager.get().record(
        camera_id=camera_id,
        rule_alert=RuleAlert(rule_name="s", rule_type="line", track_id=1,
                             alert_type="entry", description="x"),
        frame=None, camera_name="S", capture_evidence=False,
    )


# --------------------------------------------------------------------------- #
# Startup reconciliation
# --------------------------------------------------------------------------- #


def test_a_stale_online_flag_is_cleared_at_startup(db):
    """
    The exact state a hard kill leaves behind.

    Without this, the dashboard's first paint after a crash shows a camera as
    ONLINE with no thread behind it — a system that looks like it is working.
    """
    from core.sources import reconcile_on_start

    killed = _camera(db, "KILLED", active=True, online=True)
    healthy = _camera(db, "OFFLINE-ALREADY", active=True, online=False)

    result = reconcile_on_start(db)

    assert result["stale_online_cleared"] == 1
    db.expire_all()
    assert db.query(Camera).filter(Camera.id == killed.id).one().is_online is False
    assert db.query(Camera).filter(Camera.id == healthy.id).one().is_online is False


def test_reconciliation_leaves_registration_alone(db):
    """It clears a runtime lie, not the operator's configuration."""
    from core.sources import reconcile_on_start

    cam = _camera(db, "CONFIGURED", active=True, online=True)
    db.add(Rule(camera_id=cam.id, rule_type="line", geometry="[[0,0],[1,1]]",
                params="{}", name="wire", is_active=True))
    db.commit()

    reconcile_on_start(db)

    db.expire_all()
    row = db.query(Camera).filter(Camera.id == cam.id).one()
    assert row.is_active is True, "reconciliation must not unregister a camera"
    assert db.query(Rule).filter(Rule.camera_id == cam.id).count() == 1


def test_reconciliation_is_idempotent(db):
    from core.sources import reconcile_on_start

    _camera(db, "A", online=True)
    assert reconcile_on_start(db)["stale_online_cleared"] == 1
    assert reconcile_on_start(db)["stale_online_cleared"] == 0


# --------------------------------------------------------------------------- #
# Fresh start
# --------------------------------------------------------------------------- #


def test_fresh_start_empties_the_dashboard(db):
    """Opening the project should look like opening it for the first time."""
    from core.sources import fresh_start, startup_cameras, visible_cameras

    for n in range(4):
        _camera(db, f"LEFTOVER-{n}", active=True)

    result = fresh_start(db)

    assert result["total"] == 4
    assert visible_cameras(db) == []
    assert startup_cameras(db) == [], "nothing may auto-start after a fresh start"


def test_fresh_start_never_breaks_the_audit_chain(db):
    """
    A camera that owns sealed events is archived, not deleted.

    ``alerts.camera_id`` is a foreign key into a hash chain; removing the row
    would orphan evidence and break verification permanently. Starting clean
    must never be able to do that.
    """
    from core.hashchain import verify_chain
    from core.sources import fresh_start, visible_cameras

    empty = _camera(db, "NO-EVENTS", active=True)
    witness = _camera(db, "HAS-EVENTS", active=True)
    _seal(db, witness.id)
    _seal(db, witness.id)

    result = fresh_start(db)

    assert result["deleted"] == 1, "a camera owning nothing should be deleted"
    assert result["archived"] == 1, "a camera owning evidence must be archived"
    assert visible_cameras(db) == []

    db.expire_all()
    assert db.query(Alert).count() == 2, "sealed events must survive"
    assert verify_chain(db).valid
    assert db.query(Camera).filter(Camera.id == empty.id).first() is None


def test_fresh_start_lets_a_new_camera_take_a_reused_id(db):
    """The tombstones must not refuse the ids the next camera is handed."""
    from core.camera import CameraManager
    from core.sources import fresh_start, register_camera

    _camera(db, "OLD", active=True)
    fresh_start(db)

    fresh = register_camera(db, name="NEW", url="0", is_active=False)
    assert CameraManager.get().is_retired(fresh.id) is False


# --------------------------------------------------------------------------- #
# Reclaiming archived rows
# --------------------------------------------------------------------------- #


def test_prune_reclaims_rows_that_hold_nothing(db):
    from core.sources import prune_archived, retire_camera

    cam = _camera(db, "EPHEMERAL", active=False)
    db.add(Rule(camera_id=cam.id, rule_type="line", geometry="[[0,0],[1,1]]",
                params="{}", name="r", is_active=True))
    db.commit()
    _seal(db, cam.id)
    retire_camera(db, cam.id)                      # archived: it owns an event

    db.expire_all()
    assert db.query(Camera).filter(Camera.id == cam.id).one().is_deleted is True

    # Still holding evidence, so it stays.
    assert prune_archived(db, dry_run=True)["removed"] == []

    # Once the evidence is gone the row is residue.
    db.query(Alert).filter(Alert.camera_id == cam.id).delete()
    db.commit()
    result = prune_archived(db)
    assert [r["id"] for r in result["removed"]] == [cam.id]
    assert db.query(Camera).filter(Camera.id == cam.id).first() is None


def test_prune_never_removes_a_row_the_chain_needs(db):
    """A row with sealed events against it is load-bearing for verification."""
    from core.hashchain import verify_chain
    from core.sources import prune_archived, retire_camera

    cam = _camera(db, "EVIDENCE-HOLDER", active=False)
    _seal(db, cam.id)
    retire_camera(db, cam.id)

    result = prune_archived(db)

    assert result["removed"] == []
    assert [k["id"] for k in result["kept"]] == [cam.id]
    assert result["kept"][0]["alerts"] == 1
    assert verify_chain(db).valid


def test_prune_dry_run_writes_nothing(db):
    from core.sources import prune_archived, retire_camera

    cam = _camera(db, "GONE", active=False)
    retire_camera(db, cam.id)                      # no events: deleted outright
    another = _camera(db, "ARCHIVED", active=False)
    _seal(db, another.id)
    retire_camera(db, another.id)
    db.query(Alert).filter(Alert.camera_id == another.id).delete()
    db.commit()

    before = db.query(Camera).count()
    prune_archived(db, dry_run=True)
    db.expire_all()
    assert db.query(Camera).count() == before


# --------------------------------------------------------------------------- #
# Auto-start
# --------------------------------------------------------------------------- #


def test_autostart_can_be_switched_off(db, monkeypatch):
    """
    Registered but not resurrected.

    The middle ground between a production post that must come back watching
    after a reboot, and a demo that must come up empty.
    """
    monkeypatch.setattr(settings, "AUTOSTART_CAMERAS", False)
    _camera(db, "WOULD-START", active=True)

    from core.sources import startup_cameras

    # The query still finds it — the *decision* not to start is the lifespan's,
    # which is what keeps the camera registered for the next normal boot.
    assert len(startup_cameras(db)) == 1
    assert settings.AUTOSTART_CAMERAS is False


# --------------------------------------------------------------------------- #
# Bounded stores
# --------------------------------------------------------------------------- #


def test_the_operational_log_is_size_bounded():
    """
    A plain ``FileHandler`` never truncates.

    This project's own log reached 2.6 MB with no upper bound at all, which on
    an unattended post is a disk that fills months after anyone last looked —
    the slowest and least obvious way for a surveillance system to stop.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    from api.main import configure_logging

    configure_logging()
    handlers = [h for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)]
    assert handlers, "no file logging configured"
    for handler in handlers:
        assert isinstance(handler, RotatingFileHandler), (
            "the operational log must rotate; a plain FileHandler grows forever"
        )
        assert handler.maxBytes > 0
        assert handler.backupCount > 0

    worst_case_mb = settings.LOG_MAX_MB * (settings.LOG_BACKUP_COUNT + 1)
    assert worst_case_mb <= 512, f"log budget is {worst_case_mb} MB"


def test_an_enrichment_stage_budget_is_shared_across_cameras():
    """
    The limiter must be global, because the resource it protects is.

    Per-camera budgets meant eight cameras could each claim 25% of the machine:
    measured, average inference rose from 33 ms to 229 ms and aggregate
    throughput fell from 96 fps to 10.6. One budget keeps the cost of an
    optional stage flat as cameras are added.
    """
    import time

    from core.analytics import _AsyncStage

    _AsyncStage.reset_budgets()
    stages = [_AsyncStage("face") for _ in range(8)]

    started = [s.submit(lambda: time.sleep(0.05)) for s in stages]
    assert sum(started) == 1, (
        f"{sum(started)} of 8 cameras started a tick at once — the budget is "
        f"not shared"
    )
    assert all(s.busy for s in stages), "busy must be visible to every camera"

    for _ in range(100):
        if not stages[0].busy:
            break
        time.sleep(0.05)
    _AsyncStage.reset_budgets()


def test_resetting_the_budget_reopens_the_gate():
    import time

    from core.analytics import _AsyncStage

    _AsyncStage.reset_budgets()
    stage = _AsyncStage("anpr")
    assert stage.submit(lambda: None) is True
    for _ in range(100):
        if not stage.busy:
            break
        time.sleep(0.02)
    _AsyncStage.reset_budgets()
    assert stage.submit(lambda: None) is True, "reset must clear the back-off"
    for _ in range(100):
        if not stage.busy:
            break
        time.sleep(0.02)
    _AsyncStage.reset_budgets()
