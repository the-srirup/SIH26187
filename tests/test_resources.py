"""
Resource-lifecycle guarantees: nothing this platform starts may outlive its owner.

Every test here pins one of the leaks that makes a long-running deployment
degrade rather than fail — a thread that keeps running after its camera is
removed, a capture device never released, a frame buffer that grows, a
WebSocket the fan-out keeps feeding after the tab closed, a database session
that is never returned to the pool. They are cheap and fast on purpose, because
their value is in running on every commit.
"""
from __future__ import annotations

import ast
import asyncio
import io
import pathlib
import threading
import time

import numpy as np
import pytest

PROJECT = pathlib.Path(__file__).resolve().parent.parent


class _StubDetector:
    """Stands in for YOLO so these tests load no model."""

    names = {0: "person"}

    def raw_detect(self, frame):
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
                np.empty((0,), np.int32), 0.0)


def _processor(camera_id: int, url: str = "0"):
    from core.camera import CameraProcessor

    return CameraProcessor(camera_id=camera_id, url=url, name=f"RES-{camera_id}",
                           detector=_StubDetector())


@pytest.fixture
def client(db):
    """The real app, for the one test that needs a live WebSocket."""
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# 1. The stop event actually stops the loop
# --------------------------------------------------------------------------- #


def test_stop_event_breaks_the_analytics_loop_immediately(db):
    """
    Setting the event must end the loop without waiting out any sleep.

    The loop backs off with waits of 0.2-0.35 s in several places. With a
    boolean flag those were ``time.sleep``, so a camera parked in one served
    the whole sleep before noticing it had been stopped — for no reason, on the
    operator's Remove click. Every one of them is now a wait on the event.
    """
    proc = _processor(9001, url="rtsp://10.255.255.1:554/nowhere")
    assert proc.start() is True
    assert proc._running is True

    time.sleep(0.3)                          # let it get into the loop
    started = time.perf_counter()
    proc.request_stop()
    signalled = (time.perf_counter() - started) * 1000

    assert signalled < 250, f"request_stop blocked for {signalled:.0f} ms"
    assert proc._running is False
    assert proc._stop_event.is_set()

    proc.stop(join_timeout=10.0)
    assert proc.is_stopped


def test_a_stopped_processor_reports_no_live_threads(db):
    proc = _processor(9002, url="0")
    proc.stop()
    assert proc.is_stopped
    assert proc.threads_alive is False


# --------------------------------------------------------------------------- #
# 2. cleanup() releases the capture and the frames
# --------------------------------------------------------------------------- #


def test_cleanup_releases_the_capture_handle(db):
    """``release()`` must actually be called — a held webcam locks the device."""
    from core.camera import FrameBuffer

    proc = _processor(9003)
    released = threading.Event()

    class _FakeCapture:
        def release(self):
            released.set()

    proc.source._cap = _FakeCapture()
    FrameBuffer.get().open(9003)
    FrameBuffer.get().publish(9003, np.zeros((4, 4, 3), np.uint8), b"jpeg")
    assert FrameBuffer.get().get_jpeg(9003) is not None

    proc.cleanup()

    assert released.is_set(), "VideoCapture.release() was never called"
    assert proc.source._cap is None
    assert FrameBuffer.get().get_jpeg(9003) is None
    assert FrameBuffer.get().get_frame(9003) is None

    proc.cleanup()                           # idempotent


def test_cleanup_does_not_release_a_capture_its_thread_is_using(db):
    """
    Releasing a handle out from under a blocking ``read()`` is a native crash.

    The capture thread releases its own handle on the way out instead, so the
    device is always freed — just not from the wrong thread.
    """
    proc = _processor(9004)
    released = threading.Event()

    class _FakeCapture:
        def release(self):
            released.set()

    holding = threading.Event()
    stop = threading.Event()

    def _pretend_blocking_read():
        holding.set()
        stop.wait(5)

    proc.source._cap = _FakeCapture()
    proc.source._thread = threading.Thread(target=_pretend_blocking_read, daemon=True)
    proc.source._thread.start()
    holding.wait(2)

    proc.source.cleanup()
    assert not released.is_set(), "released a handle a live thread was reading"

    stop.set()
    proc.source._thread.join(timeout=3)
    proc.source.cleanup()
    assert released.is_set(), "handle was never released after the thread exited"


# --------------------------------------------------------------------------- #
# 3. Frame buffers are bounded
# --------------------------------------------------------------------------- #


def test_capture_handoff_keeps_only_the_freshest_frame():
    """
    A stalled consumer must not make the producer grow RAM.

    The hand-off slot is a ``deque(maxlen=1)``: the bound is structural, so no
    future edit can turn it back into a list that grows.
    """
    from core.video_source import LiveSource

    source = LiveSource("0", name="BOUNDED")
    assert source._frames.maxlen == 1

    for n in range(500):
        with source._new_frame:
            if source._frames:
                source.stats.frames_dropped += 1
            source._frames.append(np.full((4, 4, 3), n % 255, np.uint8))
            source._frame_id += 1

    assert len(source._frames) == 1
    assert source.stats.frames_dropped == 499
    assert int(source._frames[0][0, 0, 0]) == 499 % 255, "kept a stale frame"


def test_frame_buffer_holds_one_frame_per_camera():
    from core.camera import FrameBuffer

    buffer = FrameBuffer.get()
    buffer.clear()
    buffer.open(9100)
    for n in range(200):
        buffer.publish(9100, np.zeros((4, 4, 3), np.uint8), bytes([n % 256]))

    assert buffer.tracked_cameras() == [9100]
    assert buffer.get_jpeg(9100) == bytes([199 % 256])
    assert buffer.sequence(9100) == 200      # counted, not accumulated
    buffer.clear()


def test_inference_queue_is_bounded_to_a_couple_of_batches():
    """Each entry holds a whole frame, so the queue must not become a backlog."""
    from core.config import settings
    from cv.detector import Detector

    detector = Detector.get()
    assert detector._queue.maxsize <= max(4, settings.INFERENCE_BATCH_MAX * 2)
    assert detector._queue.maxsize > 0


# --------------------------------------------------------------------------- #
# 4. Every background thread is a daemon
# --------------------------------------------------------------------------- #


def test_every_thread_this_project_starts_is_a_daemon():
    """
    A non-daemon thread keeps the process alive after the server is killed.

    One such thread anywhere — a camera loop, a notifier, the YOLO batcher —
    is enough to leave a stray Python process holding the webcam and a GPU
    context after Ctrl-C, which the next start then cannot open. This reads the
    source rather than the running process so it catches a thread that is only
    started on a path the tests do not exercise.
    """
    offenders = []
    for path in sorted(PROJECT.rglob("*.py")):
        rel = path.relative_to(PROJECT).as_posix()
        if rel.startswith(("tests/", ".venv/", "venv/")):
            continue
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name != "Thread":
                continue
            daemon = next((kw for kw in node.keywords if kw.arg == "daemon"), None)
            if daemon is None or not (
                isinstance(daemon.value, ast.Constant) and daemon.value.value is True
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"threads started without daemon=True: {offenders}"


# --------------------------------------------------------------------------- #
# 5. WebSocket clients are released on disconnect
# --------------------------------------------------------------------------- #


class _FakeSocket:
    """Minimal stand-in for a Starlette WebSocket."""

    def __init__(self) -> None:
        self.closed = False

    async def accept(self):
        return None

    async def close(self):
        self.closed = True


def test_disconnect_frees_the_client_its_queue_and_the_socket():
    """
    A closed browser tab must leave nothing behind.

    The fan-out dict is keyed by the socket object, so a leaked entry is also
    the last reference keeping that connection alive — the leak is the socket,
    not just a dictionary key.
    """
    from api.main import ConnectionManager

    async def _run():
        manager = ConnectionManager()
        socket = _FakeSocket()
        queue = await manager.connect(socket)
        assert manager.client_count == 1

        for _ in range(50):
            manager._fanout({"type": "alert", "data": {}})
        assert queue.qsize() == 50

        await manager.disconnect(socket)
        assert manager.client_count == 0
        assert queue.qsize() == 0, "queued payloads were not released"
        assert socket.closed is True, "the socket was never closed"

        await manager.disconnect(socket)          # idempotent
        assert manager.client_count == 0

    asyncio.run(_run())


def test_a_full_client_queue_never_blocks_the_publisher():
    """One stalled tab must not apply backpressure to a camera thread."""
    from api.main import ConnectionManager

    async def _run():
        manager = ConnectionManager()
        socket = _FakeSocket()
        await manager.connect(socket)

        started = time.perf_counter()
        for _ in range(5000):                 # far beyond the queue's maxsize
            manager._fanout({"type": "stats", "data": {}})
        elapsed = (time.perf_counter() - started) * 1000

        assert elapsed < 2000, f"fan-out blocked for {elapsed:.0f} ms"
        await manager.disconnect(socket)

    asyncio.run(_run())


def test_closing_a_websocket_removes_it_from_the_live_client_count(client):
    """End to end, through the real endpoint."""
    from api.main import ws_manager

    before = ws_manager.client_count
    with client.websocket_connect("/ws/alerts") as socket:
        assert socket.receive_json()["type"] == "connected"
        assert ws_manager.client_count == before + 1
    for _ in range(50):
        if ws_manager.client_count == before:
            break
        time.sleep(0.05)
    assert ws_manager.client_count == before, "client survived the disconnect"


# --------------------------------------------------------------------------- #
# 6. Database sessions are always returned
# --------------------------------------------------------------------------- #


def test_every_session_is_closed_in_a_finally():
    """
    A session that escapes without ``close()`` holds a pooled connection.

    Enough of them and every caller — including a camera thread sealing an
    event — blocks waiting for a connection that is never coming back. Checked
    in the source so a new call site cannot quietly introduce one.
    """
    offenders = []
    for path in sorted(PROJECT.rglob("*.py")):
        rel = path.relative_to(PROJECT).as_posix()
        if rel.startswith(("tests/", ".venv/", "venv/")) or rel in (
            "manage.py", "benchmark.py", "verify_system.py", "_track_bench.py",
        ):
            continue
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(node)
            if "SessionLocal()" not in body:
                continue
            guarded = any(
                isinstance(n, ast.Try) and n.finalbody for n in ast.walk(node)
            )
            if not (guarded and ".close()" in body):
                offenders.append(f"{rel}:{node.lineno} {node.name}()")
    assert not offenders, f"sessions not closed in a finally: {offenders}"


def test_the_pool_is_sized_for_the_threads_that_use_it(db):
    """
    Camera threads, the API threadpool and the background workers share this.

    SQLAlchemy's default of five plus ten overflow is a request/response
    assumption; exceeding it does not raise, it stalls every caller for the
    pool timeout first, which on a camera thread is indistinguishable from the
    pipeline hanging.
    """
    from core.database import engine

    pool = engine.pool
    if type(pool).__name__ != "QueuePool":
        pytest.skip("not a pooled backend")
    capacity = pool.size() + pool._max_overflow
    assert capacity >= 40, f"pool capacity is only {capacity}"
    assert pool._timeout <= 10, "a starved pool should fail fast, not hang"


def test_concurrent_sessions_do_not_starve_the_pool(db):
    """Thirty threads sealing at once must not raise or stall."""
    from core.database import SessionLocal
    from core.models import Alert

    errors: list[str] = []

    def _work() -> None:
        for _ in range(15):
            session = SessionLocal()
            try:
                session.query(Alert).count()
                time.sleep(0.005)
            except Exception as exc:         # pragma: no cover - reported below
                errors.append(type(exc).__name__)
            finally:
                session.close()

    threads = [threading.Thread(target=_work, daemon=True) for _ in range(30)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    elapsed = time.perf_counter() - started

    assert not errors, f"pool errors: {set(errors)}"
    assert elapsed < 30, f"pool serialised the work ({elapsed:.1f}s)"
