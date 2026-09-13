"""
Camera lifecycle: removal must be complete, and reset must be total.

These tests exist because "Remove Camera" failed in several independent ways at
once, and each way had to be pinned separately:

* the row survived and reappeared on the next refresh (``/health`` selected
  archived cameras);
* the pipeline survived the removal (a ``start()`` that raced a ``stop()``
  spawned threads on a processor nothing referenced afterwards);
* a stale JPEG survived the removal (a capture thread parked in a one-second
  read published *after* the frame buffer had been cleared);
* a concurrent restart resurrected a camera mid-removal;
* and a camera carrying plate or face records could not be deleted at all,
  because those tables carry the same unguarded foreign key that broke the
  original delete.

The contract asserted throughout is the operator's: **removed means gone —
from every listing, and from every thread.**
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.models import (
    Alert, AnalysisSession, ANPRDetection, Camera, Checkpoint, FaceDetection,
    Rule, WatchlistEntry,
)
from core.timeutil import utc_iso


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _StubDetector:
    """Stands in for the YOLO detector so no model is loaded in these tests."""

    def metrics(self) -> dict:
        return {}


def _processor(camera_id: int, url: str = "0", name: str = "STUB"):
    """A real CameraProcessor with a stub detector — no inference, no model."""
    from core.camera import CameraProcessor

    return CameraProcessor(camera_id=camera_id, url=url, name=name,
                           detector=_StubDetector())


def _add_camera(db, name: str = "LC-CAM", url: str = "0", **kwargs) -> Camera:
    cam = Camera(name=name, url=url, location="", is_active=False,
                 is_online=False, source_kind=kwargs.pop("source_kind", "live"),
                 is_deleted=False, deleted_at="", created_at=utc_iso(), **kwargs)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def _seal(db, camera_id: int, alert_type: str = "fence_crossing") -> Alert:
    """One sealed event, written through the real chain so it verifies."""
    from core.events import EventManager
    from cv.rules import Alert as RuleAlert

    payload = EventManager.get().record(
        camera_id=camera_id,
        rule_alert=RuleAlert(rule_name="lc", rule_type="line", track_id=1,
                             alert_type=alert_type, description="lifecycle"),
        frame=None, camera_name="LC", capture_evidence=False,
    )
    assert payload is not None
    return db.query(Alert).filter(Alert.id == payload["id"]).one()


@pytest.fixture(autouse=True)
def _clean_runtime():
    """Every test starts and ends with an empty manager and frame buffer."""
    from core.camera import CameraManager, FrameBuffer

    CameraManager.get().stop_all()
    CameraManager.get().clear_retired()
    FrameBuffer.get().clear()
    yield
    CameraManager.get().stop_all()
    CameraManager.get().clear_retired()
    FrameBuffer.get().clear()


# --------------------------------------------------------------------------- #
# FrameBuffer — a dropped camera can never publish again
# --------------------------------------------------------------------------- #


def test_dropping_a_camera_refuses_frames_still_in_flight():
    """
    The stale-JPEG bug, exactly.

    ``stop()`` cleared the buffer, but the analytics thread was at that moment
    inside a one-second ``read`` and returned afterwards to publish its offline
    card. The entry came straight back for a camera that no longer existed,
    with nothing left running to ever drop it again.
    """
    from core.camera import FrameBuffer

    buffer = FrameBuffer.get()
    frame = np.zeros((8, 8, 3), np.uint8)

    buffer.publish(77, frame, b"jpeg-bytes")
    assert buffer.get_jpeg(77) == b"jpeg-bytes"

    buffer.drop(77)
    buffer.publish(77, frame, b"late-arrival")      # the in-flight frame

    assert buffer.get_jpeg(77) is None
    assert buffer.get_frame(77) is None
    assert buffer.sequence(77) == 0
    assert buffer.is_closed(77)


def test_a_new_camera_may_reuse_a_dropped_id():
    """SQLite reissues ids, so a closed slot has to be reopenable."""
    from core.camera import FrameBuffer

    buffer = FrameBuffer.get()
    buffer.drop(77)
    buffer.open(77)
    buffer.publish(77, np.zeros((8, 8, 3), np.uint8), b"fresh")
    assert buffer.get_jpeg(77) == b"fresh"


def test_clear_empties_the_buffer_completely():
    from core.camera import FrameBuffer

    buffer = FrameBuffer.get()
    for camera_id in (1, 2, 3):
        buffer.publish(camera_id, np.zeros((4, 4, 3), np.uint8), b"x")
    assert buffer.clear() == 3
    assert buffer.tracked_cameras() == []
    assert not buffer.is_closed(1)          # a reset re-arms, it does not seal


# --------------------------------------------------------------------------- #
# CameraProcessor — stop is terminal
# --------------------------------------------------------------------------- #


def test_a_stopped_processor_cannot_be_restarted():
    """
    The zombie-thread race.

    ``add_camera`` registered the processor, released the lock, and only then
    called ``start()``. A ``remove_camera`` landing in that gap popped the
    processor and stopped it — and the ``start()`` that followed brought it
    back up, with no dictionary referencing it and nothing able to stop it
    again. It kept reconnecting and kept trying to seal events for a camera the
    operator had removed.
    """
    proc = _processor(101)
    proc.stop()
    assert proc.is_stopped

    assert proc.start() is False
    assert proc.threads_alive is False
    assert proc._running is False


def test_stopping_twice_is_a_no_op():
    proc = _processor(102)
    proc.stop()
    proc.stop()                              # must not raise
    assert proc.is_stopped


def test_request_stop_is_immediate_and_closes_the_frame_slot():
    """The observable part of teardown must not wait on any thread."""
    from core.camera import FrameBuffer

    proc = _processor(103)
    FrameBuffer.get().publish(103, np.zeros((4, 4, 3), np.uint8), b"live")

    started = time.time()
    assert proc.request_stop() is True
    elapsed = time.time() - started

    assert elapsed < 0.5, "request_stop must not join anything"
    assert proc._running is False
    assert FrameBuffer.get().get_jpeg(103) is None
    assert proc.request_stop() is False      # idempotent


# --------------------------------------------------------------------------- #
# CameraManager — a stopped processor is never in the dict
# --------------------------------------------------------------------------- #


def test_removing_a_camera_forgets_its_processor(db):
    from core.camera import CameraManager

    manager = CameraManager.get()
    cam = _add_camera(db, "MGR-1")
    manager._cameras[cam.id] = _processor(cam.id)

    assert manager.is_running(cam.id) is True
    assert manager.remove_camera(cam.id) is True

    assert manager.is_running(cam.id) is False
    assert manager.get_camera(cam.id) is None
    assert cam.id not in {p.camera_id for p in manager.list_cameras()}
    assert manager.remove_camera(cam.id) is False      # idempotent


def test_a_retired_camera_is_never_started_again(db, monkeypatch):
    """
    Teardown happens before the row is archived, which leaves a window.

    A ``PUT`` or a restart holding a read of the row taken *before* the removal
    would happily start a fresh pipeline inside that window — and nothing would
    ever remove it, because the removal had already run. The tombstone is what
    closes the window.
    """
    from core.camera import CameraManager

    manager = CameraManager.get()
    monkeypatch.setattr(type(manager), "detector",
                        property(lambda self: _StubDetector()))

    cam = _add_camera(db, "MGR-2")
    manager.retire(cam.id)

    assert manager.is_retired(cam.id) is True
    assert manager.add_camera(cam) is None
    assert manager.is_running(cam.id) is False

    manager.release(cam.id)                  # the deliberate undo
    assert manager.is_retired(cam.id) is False


def test_stop_all_signals_every_camera_before_joining_any():
    """
    Teardown cost must be one timeout for the whole system, not one each.

    Stopping serially, the 44 sources this deployment accumulated would have
    taken minutes of joins on threads parked inside FFmpeg — during shutdown or
    a hard reset, with the event loop waiting. That is indistinguishable from
    the hang being reported.
    """
    from core.camera import CameraManager

    manager = CameraManager.get()

    class _SlowStop:
        """A processor whose join overruns, like a thread inside FFmpeg."""

        def __init__(self, camera_id: int) -> None:
            self.camera_id = camera_id
            self.signalled = threading.Event()
            self.is_stopped = False

        def request_stop(self) -> bool:
            self.signalled.set()
            self.is_stopped = True
            return True

        def stop(self, join_timeout: float = 5.0) -> None:
            time.sleep(0.4)

    procs = [_SlowStop(i) for i in range(900, 908)]
    for proc in procs:
        manager._cameras[proc.camera_id] = proc

    started = time.time()
    assert manager.stop_all(join_timeout=2.0) == len(procs)
    elapsed = time.time() - started

    assert all(p.signalled.is_set() for p in procs)
    assert manager.list_cameras() == []
    # Serial teardown would be 8 x 0.4 s = 3.2 s.
    assert elapsed < 2.0, f"stop_all serialised the joins ({elapsed:.2f}s)"


# --------------------------------------------------------------------------- #
# retire_camera — the database half of the contract
# --------------------------------------------------------------------------- #


def test_a_camera_with_only_detections_is_still_deletable(db):
    """
    ``anpr_detections`` and ``face_detections`` carry the same foreign key to
    ``cameras.id`` that ``analysis_sessions`` did, with no cascade and with
    ``PRAGMA foreign_keys=ON``. A camera that had read a plate but never sealed
    an event therefore hit ``FOREIGN KEY constraint failed`` on delete — the
    original Remove Camera bug, reproduced through a newer table.
    """
    from core.sources import retire_camera

    cam = _add_camera(db, "ANPR-CAM")
    db.add(ANPRDetection(camera_id=cam.id, timestamp=utc_iso(),
                         plate_text="MH12AB1234"))
    db.add(FaceDetection(camera_id=cam.id, timestamp=utc_iso(),
                         recognition_status="unknown"))
    db.commit()

    result = retire_camera(db, cam.id)          # must not raise

    assert result["ok"] and result["mode"] == "deleted"
    assert db.query(Camera).filter(Camera.id == cam.id).first() is None
    # The index rows go with the camera: they indexed nothing that survives.
    assert db.query(ANPRDetection).count() == 0
    assert db.query(FaceDetection).count() == 0


def test_a_camera_with_evidence_is_archived_not_deleted(db):
    from core.hashchain import verify_chain
    from core.sources import retire_camera, visible_cameras

    cam = _add_camera(db, "EVID-CAM")
    _seal(db, cam.id)
    _seal(db, cam.id)

    result = retire_camera(db, cam.id)

    assert result["mode"] == "archived"
    assert result["alerts_retained"] == 2
    assert db.query(Alert).count() == 2
    assert verify_chain(db).valid
    assert cam.id not in {c.id for c in visible_cameras(db)}


def test_retirement_tears_the_runtime_down_before_the_row(db):
    """The processor must be dead even though the row is merely archived."""
    from core.camera import CameraManager, FrameBuffer
    from core.sources import retire_camera

    manager = CameraManager.get()
    cam = _add_camera(db, "RUNTIME-CAM")
    proc = _processor(cam.id)
    manager._cameras[cam.id] = proc
    FrameBuffer.get().publish(cam.id, np.zeros((4, 4, 3), np.uint8), b"frame")
    _seal(db, cam.id)

    retire_camera(db, cam.id)

    assert proc.is_stopped
    assert proc.threads_alive is False
    assert manager.is_running(cam.id) is False
    assert manager.is_retired(cam.id) is True
    assert FrameBuffer.get().get_jpeg(cam.id) is None


def test_concurrent_removals_all_succeed_and_remove_once(db):
    """
    A double-clicked button, or a client that retries, must not 500.

    Both requests read ``is_deleted == False``, so both proceed; the state
    transition is a single conditional statement, and the loser reports the
    idempotent answer rather than raising ``StaleDataError`` or a foreign-key
    error from a half-applied delete.
    """
    from core.database import SessionLocal
    from core.sources import retire_camera

    cam = _add_camera(db, "RACE-CAM")
    _seal(db, cam.id)
    camera_id = cam.id

    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def _remove() -> None:
        session = SessionLocal()
        try:
            barrier.wait(timeout=5)
            results.append(retire_camera(session, camera_id))
        except Exception as exc:          # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            session.close()

    threads = [threading.Thread(target=_remove) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors, errors
    assert len(results) == 6
    assert all(r["ok"] for r in results)
    # Exactly one request performed the transition; the rest are idempotent.
    assert sum(1 for r in results if not r.get("already_removed")) == 1

    db.expire_all()
    assert db.query(Camera).filter(Camera.id == camera_id).one().is_deleted


def test_removing_an_unknown_camera_is_not_an_error(db):
    from core.sources import retire_camera

    result = retire_camera(db, 424242)
    assert result["ok"] and result["mode"] == "absent"


# --------------------------------------------------------------------------- #
# Hard reset
# --------------------------------------------------------------------------- #


def _populate(db) -> Camera:
    """A system with something of every kind in it."""
    cam = _add_camera(db, "RESET-CAM")
    db.add(Rule(camera_id=cam.id, rule_type="line", geometry="[[0,0],[1,1]]",
                params="{}", name="r", is_active=True))
    db.add(AnalysisSession(session_uid="uid-reset", filename="c.mp4",
                           stored_path="videos/c.mp4", camera_id=cam.id,
                           status="completed"))
    db.add(ANPRDetection(camera_id=cam.id, timestamp=utc_iso(),
                         plate_text="DL01AB1111"))
    db.add(FaceDetection(camera_id=cam.id, timestamp=utc_iso(),
                         recognition_status="unknown"))
    db.add(WatchlistEntry(name="subject", embedding_json="[]"))
    db.commit()
    _seal(db, cam.id)
    _seal(db, cam.id)
    from core.hashchain import create_checkpoint

    create_checkpoint(db)
    assert db.query(Checkpoint).count() >= 1
    return cam


def test_hard_reset_empties_every_table(db):
    from core.sources import hard_reset

    _populate(db)
    result = hard_reset(db, wipe_evidence=False, actor="test")

    assert result["ok"]
    for model in (Camera, Rule, Alert, Checkpoint, AnalysisSession,
                  ANPRDetection, FaceDetection, WatchlistEntry):
        assert db.query(model).count() == 0, model.__tablename__
    assert result["rows_deleted_total"] > 0


def test_hard_reset_stops_every_pipeline_before_touching_the_database(db):
    from core.camera import CameraManager, FrameBuffer
    from core.sources import hard_reset

    manager = CameraManager.get()
    cam = _populate(db)
    proc = _processor(cam.id)
    manager._cameras[cam.id] = proc
    FrameBuffer.get().publish(cam.id, np.zeros((4, 4, 3), np.uint8), b"frame")

    result = hard_reset(db, wipe_evidence=False, actor="test")

    assert result["cameras_stopped"] == 1
    assert proc.is_stopped and proc.threads_alive is False
    assert manager.list_cameras() == []
    assert FrameBuffer.get().get_jpeg(cam.id) is None


def test_hard_reset_clears_the_in_memory_event_history(db):
    """The dashboard feed lives in memory; a reset it survives is not a reset."""
    from core.events import EventManager
    from core.sources import hard_reset

    cam = _add_camera(db, "FEED-CAM")
    _seal(db, cam.id)
    assert EventManager.get().recent

    hard_reset(db, wipe_evidence=False, actor="test")
    assert EventManager.get().recent == []
    assert EventManager.get().counts() == {}


def test_hard_reset_restarts_the_integrity_chain(db):
    from core.hashchain import GENESIS_HASH, latest_chain_hash, verify_chain
    from core.sources import hard_reset

    _populate(db)
    hard_reset(db, wipe_evidence=False, actor="test")

    assert latest_chain_hash(db) == GENESIS_HASH
    assert verify_chain(db).valid


def test_the_system_is_usable_immediately_after_a_hard_reset(db):
    """The point of the feature: a clean demo, not a broken install."""
    from core.sources import hard_reset, register_camera, visible_cameras

    _populate(db)
    hard_reset(db, wipe_evidence=False, actor="test")

    fresh = register_camera(db, name="AFTER-RESET", url="0", is_active=False)
    assert fresh.id == 1, "ids should start over, not continue from the old set"
    assert [c.id for c in visible_cameras(db)] == [fresh.id]
    # And the new camera is not refused by machinery keeping an old one dead.
    from core.camera import CameraManager

    assert CameraManager.get().is_retired(fresh.id) is False

    _seal(db, fresh.id)
    from core.hashchain import verify_chain

    assert verify_chain(db).valid


def test_hard_reset_keeps_evidence_unless_asked(db, tmp_root):
    from core.config import settings
    from core.sources import hard_reset

    settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    keeper = settings.SNAPSHOTS_DIR / "keep_me.jpg"
    keeper.write_bytes(b"\xff\xd8\xff")

    hard_reset(db, wipe_evidence=False, actor="test")
    assert keeper.exists()

    result = hard_reset(db, wipe_evidence=True, actor="test")
    assert not keeper.exists()
    assert result["evidence"]["wiped"] is True
    assert result["evidence"]["files_removed"] >= 1
    # The directories themselves survive, so the next capture has somewhere to go.
    assert settings.SNAPSHOTS_DIR.is_dir()


def test_hard_reset_never_touches_the_bundled_samples(db):
    """
    ``samples/`` is shipped source material, not evidence.

    A reset that deleted it would leave the documented demo command
    (``camera-add --url samples/sample_border_scenario.mp4``) broken with no
    way back short of a re-clone.
    """
    from core.config import settings
    from core.sources import _evidence_directories, hard_reset

    samples = settings.BASE_DIR / "samples"
    wiped = {d.resolve() for d in _evidence_directories()}
    assert samples.resolve() not in wiped

    before = sorted(p.name for p in samples.iterdir()) if samples.is_dir() else []
    hard_reset(db, wipe_evidence=True, actor="test")
    after = sorted(p.name for p in samples.iterdir()) if samples.is_dir() else []
    assert before == after
