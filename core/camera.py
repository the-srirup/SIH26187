"""
Live camera processing.

Threading model — one camera, two threads:

    [capture thread]  decode as fast as the source emits, keep newest frame
            |
            v  (latest-frame handoff, older frames dropped)
    [analytics thread]  analyse -> rules -> events -> encode -> publish

The capture thread never waits for inference, and the analytics thread never
waits for the decoder.  If inference falls behind, frames are *dropped*, not
queued, so end-to-end latency stays flat instead of growing without bound —
this is the fix for the original build's frame backlog, where a single
serial loop read, inferred and slept in lockstep at a hard-capped 5 FPS.

The analytics thread also encodes the JPEG **once** and publishes the bytes.
Every MJPEG viewer then shares that one encode, instead of each client
re-encoding the same frame on its own timer.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.analytics import AnalysisResult, FrameAnalyzer, LowLightEnhancer
from core.config import settings
from core import detections
from core.database import SessionLocal
from core.evidence import ClipRecorder
from core.events import EventManager
from core.models import Camera, Rule
from cv import overlay as ov
from cv.detector import Detector
from cv.rules import Alert as RuleAlert

log = logging.getLogger("ibvap.camera")

_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc


# --------------------------------------------------------------------------- #
# Shared frame buffer — annotated frame + pre-encoded JPEG per camera
# --------------------------------------------------------------------------- #


class FrameBuffer:
    """
    Latest annotated frame *and* its JPEG bytes, per camera.

    Storing the encoded bytes here is what removes per-client encoding: with
    four cameras and three dashboard tabs the old code performed ~400 JPEG
    encodes per second of largely identical frames.  Now it performs one per
    produced frame, regardless of viewer count.

    ``seq`` lets a streaming client block until a genuinely new frame exists
    rather than polling on a timer.

    Removal is enforced here, not merely requested.  ``drop`` closes the camera
    id as well as clearing its entries, and a closed id refuses every later
    ``publish``.  Without that, dropping was a race the removal path could not
    win: a capture thread parked in a one-second ``read`` returns *after*
    ``stop()`` has cleared the buffer, publishes its offline card, and the
    entry — a stale JPEG for a camera that no longer exists — is resurrected
    with nothing left running to ever drop it again.  ``open`` re-arms an id,
    which only a starting processor does.
    """

    _instance: Optional["FrameBuffer"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._frames: dict[int, np.ndarray] = {}
        #: The unannotated frame. Analysis probes (ANPR, face) must run on this,
        #: never on the annotated one: the overlay burns the camera name and HUD
        #: text into the image, and an OCR probe pointed at the annotated frame
        #: cheerfully read the camera's own name back as a number plate.
        self._clean: dict[int, np.ndarray] = {}
        self._jpegs: dict[int, bytes] = {}
        self._seq: dict[int, int] = {}
        #: Camera ids that have been dropped and may not publish again until a
        #: processor explicitly reopens them.
        self._closed: set[int] = set()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @classmethod
    def get(cls) -> "FrameBuffer":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def publish(self, camera_id: int, frame: np.ndarray, jpeg: Optional[bytes] = None,
                clean: Optional[np.ndarray] = None) -> None:
        with self._cond:
            if camera_id in self._closed:
                # The camera was removed while this frame was in flight.
                return
            self._frames[camera_id] = frame
            if clean is not None:
                self._clean[camera_id] = clean
            if jpeg is not None:
                self._jpegs[camera_id] = jpeg
            self._seq[camera_id] = self._seq.get(camera_id, 0) + 1
            self._cond.notify_all()

    # Backwards-compatible alias used by older call sites / tests.
    def set(self, camera_id: int, frame: np.ndarray) -> None:
        self.publish(camera_id, frame)

    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(camera_id)

    def get_clean_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """
        The unannotated frame — what an analysis probe must be given.

        Note the explicit ``is None`` test: ``a or b`` on a numpy array calls
        ``__bool__`` on it, which raises for anything larger than one element.
        """
        with self._lock:
            frame = self._clean.get(camera_id)
            return frame if frame is not None else self._frames.get(camera_id)

    def get_jpeg(self, camera_id: int) -> Optional[bytes]:
        with self._lock:
            return self._jpegs.get(camera_id)

    def sequence(self, camera_id: int) -> int:
        with self._lock:
            return self._seq.get(camera_id, 0)

    def wait_for_jpeg(self, camera_id: int, last_seq: int, timeout: float = 2.0):
        """Block until a frame newer than ``last_seq`` is published."""
        deadline = time.time() + timeout
        with self._cond:
            while self._seq.get(camera_id, 0) <= last_seq:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            return self._jpegs.get(camera_id), self._seq.get(camera_id, 0)

    def is_closed(self, camera_id: int) -> bool:
        with self._lock:
            return camera_id in self._closed

    def open(self, camera_id: int) -> None:
        """Re-arm a previously dropped id — only a starting processor does this."""
        with self._cond:
            self._closed.discard(camera_id)

    def drop(self, camera_id: int) -> None:
        """
        Forget a camera's frames and refuse any further publish for it.

        Waiters are woken so an MJPEG generator blocked on this camera returns
        immediately instead of holding its connection for the full timeout.
        """
        with self._cond:
            self._closed.add(camera_id)
            self._frames.pop(camera_id, None)
            self._clean.pop(camera_id, None)
            self._jpegs.pop(camera_id, None)
            self._seq.pop(camera_id, None)
            self._cond.notify_all()

    def clear(self) -> int:
        """
        Drop every camera's frames — used by the hard reset.

        ``_closed`` is emptied rather than filled: a reset returns the buffer to
        its pristine state, and any id registered afterwards is a genuinely new
        camera that must be allowed to publish.
        """
        with self._cond:
            count = len(self._frames)
            self._frames.clear()
            self._clean.clear()
            self._jpegs.clear()
            self._seq.clear()
            self._closed.clear()
            self._cond.notify_all()
        return count

    def tracked_cameras(self) -> list[int]:
        with self._lock:
            return sorted(self._seq)


# --------------------------------------------------------------------------- #
# Camera processor
# --------------------------------------------------------------------------- #


class CameraState:
    """
    Explicit lifecycle states for one camera.

    Before this, "is the camera working?" had exactly two answers — a boolean
    ``is_online`` — which cannot distinguish a camera that is still opening its
    stream from one whose URL is wrong from one the operator stopped on purpose.
    All three rendered as the same red dot, so the dashboard could not tell an
    operator whether to wait, fix the URL, or do nothing.

    Transitions::

        CREATED -> CONNECTING -> CONNECTED -> PROCESSING
                        ^            |            |
                        |            v            v
                        +--------- ERROR <--------+
                                     |
                       STOPPING -> STOPPED
    """

    CREATED = "created"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PROCESSING = "processing"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"

    #: What the dashboard shows for each state.
    LABELS = {
        CREATED: ("Created", "grey"),
        CONNECTING: ("Connecting", "amber"),
        CONNECTED: ("Connected", "green"),
        PROCESSING: ("Processing", "green"),
        ERROR: ("Error", "red"),
        STOPPING: ("Stopping", "amber"),
        STOPPED: ("Stopped", "grey"),
    }


class CameraProcessor:
    """Full live pipeline for a single camera."""

    def __init__(
        self,
        camera_id: int,
        url: str,
        name: str = "",
        location: str = "",
        detector: Optional[Detector] = None,
    ) -> None:
        from core.video_source import LiveSource

        self.camera_id = int(camera_id)
        self.url = url
        self.name = name or f"CAM-{camera_id:02d}"
        self.location = location

        self.source = LiveSource(url, name=self.name)
        self.analyzer = FrameAnalyzer(
            source_id=str(camera_id), display_name=self.name, detector=detector
        )
        self.clips = ClipRecorder(source_id=str(camera_id))
        self.events = EventManager.get()

        #: The stop signal, and the only thing the analytics loop consults.
        #:
        #: An ``Event`` rather than a boolean flag, for two reasons that matter
        #: at teardown. It is atomic without a lock, so a thread can never read
        #: a half-written value; and, more usefully, every wait in the loop is
        #: ``_stop_event.wait(timeout)`` instead of ``time.sleep(timeout)``, so
        #: a sleeping thread wakes the instant stop is requested rather than
        #: serving out its sleep first. With a boolean, a camera parked in a
        #: back-off sleep kept the removal waiting for no reason.
        self._stop_event = threading.Event()
        #: Terminal. Once stopped, a processor can never be restarted — the
        #: manager builds a fresh one instead. Without this, ``add_camera`` and
        #: ``remove_camera`` racing on the same id could leave ``start()`` to
        #: run *after* ``stop()``, spawning threads on a processor nobody holds
        #: a reference to: a zombie that reconnects and seals events for a
        #: camera that has been removed, and that nothing can ever stop.
        self._stopped = False
        #: Guards the start/stop transition (not the analytics loop).
        self._lifecycle = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_id = -1
        #: Last source generation the analytics saw. A change means the feed
        #: restarted (file loop or reconnect) and per-track state is stale.
        self._last_generation = 0

        # Runtime metrics — all measured, none fabricated.
        self._fps = 0.0
        self._fps_window: list[float] = []
        self._frames_analysed = 0
        self._inference_ms = 0.0
        self._pipeline_ms = 0.0
        self._latency_ms = 0.0
        self._last_detections = 0
        self._person_count = 0
        self._vehicle_count = 0
        self._is_night = False
        self._scene = None
        #: Live zone / loiter occupancy from the most recent analysed frame.
        #: This is the continuous signal the dashboard holds a state on —
        #: entry and exit are events, being inside is a condition.
        self._zones: list = []
        self._online = False
        self._state = CameraState.CREATED
        self._state_since = time.time()
        self._state_detail = ""
        self._started_at = 0.0
        self._offline_announced = False
        self._last_status_write = 0.0
        #: Operator playback state. Held here as well as on the source because
        #: a live camera is paused at the publish, not at the decoder — see
        #: the comment in ``_run``.
        self._paused = False
        #: Replay bookkeeping for looping file sources. ``_lap`` counts passes
        #: over the footage; ``_replay_seen`` holds the signature of every event
        #: sealed so far, so a second pass can tell a repeat from something new.
        self._lap = 0
        self._replay_seen: set = set()
        self._replay_suppressed = 0

        self.reload_rules()

    # ------------------------------------------------------------------ #
    # Playback control
    # ------------------------------------------------------------------ #

    def set_playback(self, paused=None, speed=None) -> dict:
        """
        Pause / resume the tile, and set the review speed.

        Pause means two different things for the two kinds of source, and the
        difference is deliberate:

        * a **recording** is paused at the decoder, so it waits for the
          operator rather than racing on while they look at a frozen picture;
        * a **live camera** is paused only at the publish. The capture and the
          whole analytics pipeline keep running, so events are still detected
          and still sealed while the operator inspects the held frame. Freezing
          a real camera's analytics to look at it would be the surveillance
          equivalent of closing your eyes.

        Speed only applies where the source is paced exactly — a recording.
        """
        if paused is not None:
            self._paused = bool(paused)
        source = getattr(self, "source", None)
        if source is not None:
            # The source pauses its decoder only for a recording; for a live
            # feed it keeps running and the publish gate above does the work.
            source.set_playback(
                paused=(self._paused and source.playback_applies)
                if paused is not None else None,
                speed=speed,
            )
        return self.playback_state

    @property
    def playback_state(self) -> dict:
        source = getattr(self, "source", None)
        state = dict(source.playback_state) if source is not None else {
            "speed": 1.0, "speed_supported": False, "effective_fps": 0.0,
        }
        # The operator's intent, not the decoder's — they differ on a live feed.
        state["paused"] = bool(self._paused)
        return state

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #
    def reload_rules(self) -> None:
        """
        Load rules from the database into the analyzer's in-memory engine.

        Called on start and whenever a rule changes — never per frame.  The
        previous build queried SQLite once per frame *just to draw the fence*.
        """
        import json

        db = SessionLocal()
        try:
            rows = (
                db.query(Rule)
                .filter(Rule.camera_id == self.camera_id, Rule.is_active.is_(True))
                .all()
            )
            payload = []
            for row in rows:
                try:
                    geometry = json.loads(row.geometry) if row.geometry else []
                    params = json.loads(row.params) if row.params else {}
                except (json.JSONDecodeError, TypeError):
                    log.warning("Rule %s has malformed JSON — skipped", row.id)
                    continue
                payload.append({
                    "id": row.id,
                    "rule_type": row.rule_type,
                    "geometry": geometry,
                    "params": params,
                    "name": row.name or f"{row.rule_type}-{row.id}",
                })
            self.analyzer.set_rules(payload)
        except Exception as exc:
            log.exception("Failed to load rules for camera %d: %s", self.camera_id, exc)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _set_state(self, state: str, detail: str = "") -> None:
        """Record a lifecycle transition, logging only genuine changes."""
        if state == self._state and detail == self._state_detail:
            return
        previous = self._state
        self._state = state
        self._state_detail = detail
        self._state_since = time.time()
        log.info("CAMERA_STATE cam=%d (%s) %s -> %s%s",
                 self.camera_id, self.name, previous.upper(), state.upper(),
                 f" — {detail}" if detail else "")

    @property
    def state(self) -> str:
        return self._state

    def start(self) -> bool:
        """
        Bring the pipeline up. Returns False if this processor is already
        running or has been stopped for good.

        Refusing to start a stopped processor is the guarantee behind "removed
        means dead": a ``stop()`` that lands between the manager registering a
        processor and starting it used to be silently undone by the ``start()``
        that followed, leaving threads running for a camera the operator had
        just removed.
        """
        with self._lifecycle:
            if self._stopped:
                log.debug("Camera %d start ignored — processor already retired",
                          self.camera_id)
                return False
            if self._running and self._thread and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._started_at = time.time()
            self._set_state(CameraState.CONNECTING)
            # Re-arm the frame slot this camera publishes into; a previous
            # tenant of the same id may have closed it.
            FrameBuffer.get().open(self.camera_id)
            self.source.start()
            self._thread = threading.Thread(
                target=self._run, name=f"analytics-cam{self.camera_id}", daemon=True
            )
            self._thread.start()
        log.info("Camera %d (%s) started — source=%s", self.camera_id, self.name, self.url)
        return True

    def request_stop(self) -> bool:
        """
        Make this camera dead **immediately**, without waiting for any thread.

        Everything here takes microseconds and is what actually enforces the
        removal contract:

        * ``_stopped`` is terminal, so no later ``start()`` can revive it;
        * ``_stop_event`` is set, which both ends the analytics loop at its next
          turn and wakes it immediately out of any back-off wait;
        * the frame slot is *closed*, so a frame already in flight on a thread
          that has not noticed yet is discarded rather than published.

        The slow part — joining threads parked inside FFmpeg — is left to
        :meth:`stop`, which the caller runs outside any lock.  Returns False if
        the processor was already stopping, which makes stop idempotent and
        safe under concurrent removals of the same camera.
        """
        with self._lifecycle:
            if self._stopped:
                return False
            self._stopped = True
            self._stop_event.set()          # every wait in the loop aborts now
        self._set_state(CameraState.STOPPING)
        FrameBuffer.get().drop(self.camera_id)
        # Wake the capture thread's condition so a consumer blocked in read()
        # returns now instead of after its timeout.
        self.source.request_stop()
        return True

    def stop(self, join_timeout: float = 5.0) -> None:
        """
        Tear the camera down deterministically.

        Order is chosen so the *observable* effects are immediate and the slow
        part cannot delay them. :meth:`request_stop` runs first and takes
        microseconds, which is what actually guarantees the removal contract:
        no further frame can be published for this camera and no further event
        can be sealed against it, whether or not the worker threads have
        finished unwinding yet.

        Only then do we join. A capture thread parked inside an FFmpeg read
        cannot be interrupted from outside, so the join is bounded and a thread
        that overruns is reported rather than waited on indefinitely — it exits
        on its own once the bounded FFmpeg timeout expires, and it can no longer
        affect anything, because it has nowhere left to publish.

        Calling this twice is a no-op, not an error.
        """
        first = self.request_stop()

        # One budget for the whole teardown, shared between the two threads,
        # rather than a separate full timeout each: a camera whose capture
        # thread is parked in a native call used to be able to spend the source
        # timeout *and then* the analytics timeout before stop() returned.
        deadline = time.time() + max(0.0, join_timeout)
        self.source.stop(join_timeout=max(0.0, deadline - time.time()))
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - time.time()))
            if thread.is_alive():
                log.warning(
                    "CAMERA_STOP_SLOW cam=%d (%s): analytics thread still "
                    "unwinding after %.1fs — it is detached and can no longer "
                    "publish frames or events",
                    self.camera_id, self.name, join_timeout,
                )
        # Everything this camera holds goes back to the OS here, whether or not
        # the threads finished unwinding in time.
        self.cleanup()
        if first:
            self._set_online(False, persist=True)
        self._set_state(CameraState.STOPPED)
        # Deliberately no second FrameBuffer.drop() here. request_stop() already
        # closed the slot, so this would be redundant — and it would land after
        # a hard reset's FrameBuffer.clear() if this join overran, re-closing an
        # id the reset had just re-armed for whatever camera claims it next.
        if first:
            log.info("CAMERA_STOPPED cam=%d (%s)", self.camera_id, self.name)

    def cleanup(self) -> None:
        """
        Release every OS and memory resource this camera holds. Idempotent.

        Called at the end of :meth:`stop`, and safe to call again. What it
        frees, in order of how much it matters:

        * the OpenCV ``VideoCapture`` — a file handle on a webcam, a socket on
          an RTSP stream. Held open, a webcam stays locked against the next
          camera that wants it and an RTSP session stays established on the
          device;
        * the shared frame buffer entry — the annotated frame, the clean frame
          and the encoded JPEG, which for one 640x384 camera is about 1.5 MB
          that would otherwise sit in RAM for the life of the process;
        * the clip recorder's pre-roll buffer (four seconds of frames, ~44 MB)
          and any open ``VideoWriter``;
        * the analyzer's tracker and rule state.

        None of this is reclaimed by simply dropping the processor reference:
        ``VideoCapture`` and ``VideoWriter`` wrap native handles that are only
        released when told, and the frame buffer is a process-wide singleton
        that holds its own reference to the frames.
        """
        # 1. The capture device. LiveSource.cleanup is itself idempotent and
        #    refuses to release a handle its thread is still reading from.
        try:
            self.source.cleanup()
        except Exception as exc:
            log.warning("Camera %d: source cleanup failed: %s", self.camera_id, exc)

        # 2. Frames published for this camera, and the slot itself.
        try:
            FrameBuffer.get().drop(self.camera_id)
        except Exception as exc:
            log.warning("Camera %d: frame buffer drop failed: %s", self.camera_id, exc)

        # 3. Evidence writers and the pre-roll ring.
        try:
            self.clips.close()
        except Exception as exc:
            log.warning("Camera %d: clip recorder close failed: %s", self.camera_id, exc)

        # 4. Tracker, rule and per-track analyzer state.
        try:
            self.analyzer.reset_tracking()
        except Exception as exc:
            log.debug("Camera %d: analyzer reset failed: %s", self.camera_id, exc)

        log.debug("CAMERA_CLEANED cam=%d (%s)", self.camera_id, self.name)

    @property
    def _running(self) -> bool:
        """True while the analytics loop should keep going."""
        return not self._stop_event.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def threads_alive(self) -> bool:
        """True while any thread of this processor is still unwinding."""
        thread = self._thread
        return bool(
            (thread is not None and thread.is_alive()) or self.source.is_thread_alive
        )

    # ------------------------------------------------------------------ #
    # Analytics loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        buffer = FrameBuffer.get()
        min_interval = 1.0 / max(1, settings.TARGET_FPS)
        next_deadline = time.time()

        while self._running:
            frame, frame_id, captured_at = self.source.read(
                timeout=1.0, last_id=self._last_frame_id
            )

            if frame is None:
                self._handle_no_frame(buffer)
                continue

            self._last_frame_id = frame_id
            self._set_online(True)
            if self._state != CameraState.PROCESSING:
                self._set_state(CameraState.PROCESSING)

            # A looping video file or a reconnected stream is a discontinuity:
            # track ids are reissued and every object appears to teleport. Rule
            # state carried across that seam would fabricate fence crossings and
            # zone entries on the first frame of the new lap, which is exactly
            # the kind of phantom event that makes an operator stop trusting the
            # system. Drop tracking and rule state, keep the scene measurement
            # (the camera is still pointed at the same place).
            generation = self.source.generation
            if generation != self._last_generation:
                self._handle_source_restart(generation)

            # Cadence limiter. The capture thread keeps draining regardless, so
            # skipping here sheds load without ever building a backlog.
            now = time.time()
            if now < next_deadline:
                continue
            next_deadline = max(now, next_deadline) + min_interval

            try:
                result = self.analyzer.analyse(
                    frame, timestamp=now, fps=self._fps, online=True
                )
            except Exception as exc:
                log.exception("Camera %d analysis error: %s", self.camera_id, exc)
                self._stop_event.wait(0.25)      # interruptible back-off
                continue

            # A paused LIVE camera keeps analysing and keeps sealing events —
            # it is a surveillance system, and an operator freezing the picture
            # to look at something must never blind the post. Only the publish
            # is withheld, so the tile holds its last frame while the pipeline
            # runs on underneath. A paused FILE source really does stop, at the
            # decoder, because a recording under review should wait for you.
            if not self._paused:
                self._publish(buffer, result, captured_at)
            self._handle_alerts(result)
            self._update_metrics(result, captured_at)

        log.info("Camera %d analytics loop exited", self.camera_id)

    def _handle_source_restart(self, generation: int) -> None:
        """
        A looping file reached its end, or a dropped stream reconnected.

        Two things have to happen at the seam. Track ids are reissued and every
        object appears to teleport, so tracking and rule state are dropped —
        carrying them across would manufacture fence crossings on the first
        frame of the new lap. And the operator has to be told: a second pass
        over the same footage produces the same events again, and an event log
        that shows them with no seam between reads as a second incident.
        """
        first_lap = self._last_generation == 0
        self._last_generation = generation
        self.analyzer.reset_tracking()
        self._lap += 1

        if first_lap:
            return                      # the initial connection is not a replay

        replaying = self._is_replaying()
        suppressed = self._replay_suppressed
        self._replay_suppressed = 0

        detail = f"{self.name} restarted (pass {self._lap})"
        if replaying:
            detail += (" — this footage has been seen before, so repeated "
                       "events are recorded once and then suppressed")
            if suppressed:
                detail += f"; {suppressed} duplicate event(s) suppressed last pass"
        log.info("Camera %d (%s): source restarted — pass %d%s",
                 self.camera_id, self.name, self._lap,
                 f", {suppressed} duplicate event(s) suppressed" if suppressed else "")
        self._emit_system_event("source_restarted", detail)

    def _is_replaying(self) -> bool:
        """True once a looping file has begun a second pass over the same frames."""
        source = getattr(self, "source", None)
        if source is None or not settings.FILE_LOOP_SUPPRESS_REPEATS:
            return False
        return bool(getattr(source, "is_file", False)
                    and getattr(source, "loop_files", False)
                    and self._lap > 1)

    @staticmethod
    def _replay_signature(rule_alert, object_class: str) -> tuple:
        """
        What makes two events "the same event, seen again".

        Track ids cannot appear here: they are reissued at every loop, so the
        same car crossing the same line is a different track id on every pass.
        What does not change across a replay is which rule fired, what it
        reported, and what kind of object caused it.
        """
        return (
            str(getattr(rule_alert, "alert_type", "")),
            str(getattr(rule_alert, "rule_name", "")),
            (object_class or "").lower(),
        )

    def _should_seal(self, rule_alert, object_class: str) -> bool:
        """Seal this event, or recognise it as a replay of one already logged?"""
        signature = self._replay_signature(rule_alert, object_class)
        if not self._is_replaying():
            # First pass: remember everything so the next pass can recognise it.
            self._replay_seen.add(signature)
            return True
        if signature in self._replay_seen:
            self._replay_suppressed += 1
            return False
        # Genuinely new on a later pass — seal it, and remember it too.
        self._replay_seen.add(signature)
        return True

    def _handle_no_frame(self, buffer: "FrameBuffer") -> None:
        """No frame within the read timeout — decide whether we are offline."""
        if self._paused:
            # A paused file source stops decoding on purpose, so the read
            # timeout here is expected. Without this the camera would be
            # announced OFFLINE — and a camera_offline event sealed into the
            # audit log — every time an operator pressed pause.
            self._stop_event.wait(0.1)
            return
        if not self._running:
            # Being torn down is not being offline. request_stop() wakes the
            # source, so this runs once more on the way out; without this guard
            # it overwrote STOPPING with ERROR "no frames received" and the
            # operator briefly saw a camera they had just removed reported as
            # faulty.
            return
        if self.source.is_online:
            return

        # Starting up is not the same as going down. Opening an RTSP stream or
        # a large file can take a few seconds; announcing OFFLINE in that
        # window produced a false alert on every single restart.
        if not self.source.ever_connected:
            if time.time() - self._started_at < settings.CAMERA_TIMEOUT:
                self._set_state(CameraState.CONNECTING,
                                self.source.stats.last_error)
                self._stop_event.wait(0.2)
                return

        if self._online or not self._offline_announced:
            self._set_online(False, persist=True)
            self._set_state(CameraState.ERROR,
                            self.source.stats.last_error or "no frames received")
            self._offline_announced = True
            card = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), np.uint8)
            ov.draw_offline(card, self.name, self.source.stats.last_error)
            ok, encoded = cv2.imencode(
                ".jpg", card, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            buffer.publish(self.camera_id, card, encoded.tobytes() if ok else None)
            self._emit_system_event(
                "camera_offline",
                f"{self.name} lost signal: {self.source.stats.last_error or 'no frames received'}",
            )
        self._stop_event.wait(0.2)

    def _publish(self, buffer: "FrameBuffer", result: AnalysisResult, captured_at: float) -> None:
        if not self._running:
            return
        ok, encoded = cv2.imencode(
            ".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
        )
        buffer.publish(self.camera_id, result.frame,
                       encoded.tobytes() if ok else None, clean=result.raw_frame)
        if settings.EVIDENCE_ENABLED:
            self.clips.push(result.frame)

    def _handle_alerts(self, result: AnalysisResult) -> None:
        if not result.alerts or not self._running:
            # A camera being removed must not seal one last event: the row it
            # would reference is about to disappear, and the insert would fail
            # on the foreign key from inside a detached thread.
            return
        by_track = {d.track_id: d for d in result.detections}
        for rule_alert in result.alerts:
            det = by_track.get(rule_alert.track_id)
            object_class = det.class_name if det else ""
            # A looping file replays identical footage. Sealing the same events
            # on every pass buries the first, real occurrence under copies and
            # inflates the evidence chain with duplicates of itself.
            if not self._should_seal(rule_alert, object_class):
                continue
            self.events.record(
                camera_id=self.camera_id,
                rule_alert=rule_alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class=det.class_name if det else "",
                confidence=det.confidence if det else 0.0,
                camera_name=self.name,
                source_type="live",
            )

        # Face and ANPR produce their own event types on their own cadence.
        self._handle_face_events(result)
        self._handle_anpr_events(result)

    def _handle_face_events(self, result: AnalysisResult) -> None:
        if not result.faces:
            return
        recognizer = self.analyzer._face()
        if recognizer is None:
            return
        for match in result.faces:
            alert = recognizer.build_event(match, str(self.camera_id))
            if alert is None:
                continue
            payload = self.events.record(
                camera_id=self.camera_id,
                rule_alert=alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class="face",
                confidence=float(match.similarity if match.matched else match.det_score),
                camera_name=self.name,
                source_type="live",
            )
            # Structured row + face crop, keyed to the sealed alert. The crop is
            # cut from the *clean* frame: an evidence image with the HUD and
            # bounding boxes burned into it is not the face the camera saw.
            detections.record_face(
                match, camera_id=self.camera_id,
                alert_id=(payload or {}).get("id"),
                frame=result.raw_frame,
                source_type="live",
                threshold=recognizer._similarity_threshold,
            )

    def _handle_anpr_events(self, result: AnalysisResult) -> None:
        if not result.plates:
            return
        processor = self.analyzer._anpr_processor()
        if processor is None:
            return
        for plate in result.plates:
            alert = processor.build_event(plate, str(self.camera_id))
            if alert is None:
                continue
            payload = self.events.record(
                camera_id=self.camera_id,
                rule_alert=alert,
                frame=result.frame,
                clean_frame=result.raw_frame,
                clip_recorder=self.clips,
                object_class=plate.vehicle_class or "vehicle",
                confidence=float(plate.text_confidence),
                camera_name=self.name,
                source_type="live",
            )
            # Cut the crop from the frame the plate was actually read in.
            # The ANPR tick runs beside the pipeline now, so its boxes can
            # belong to a slightly earlier frame; cropping the current one
            # would save a picture of where the plate no longer is.
            detections.record_plate(
                plate, camera_id=self.camera_id,
                alert_id=(payload or {}).get("id"),
                frame=(result.plates_frame if result.plates_frame is not None
                       else result.raw_frame),
                source_type="live",
            )

    def _emit_system_event(self, alert_type: str, description: str) -> None:
        """Log a system-level condition (offline, error) into the audit chain."""
        if not self._running:
            return
        self.events.record(
            camera_id=self.camera_id,
            rule_alert=RuleAlert(
                rule_name="system", rule_type="system", track_id=0,
                alert_type=alert_type,  # type: ignore[arg-type]
                description=description,
                details={"camera": self.name, "url": self.url},
            ),
            frame=None,
            camera_name=self.name,
            source_type="live",
            capture_evidence=False,
        )

    # ------------------------------------------------------------------ #
    # Status / metrics
    # ------------------------------------------------------------------ #
    def _update_metrics(self, result: AnalysisResult, captured_at: float) -> None:
        now = time.time()
        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self._fps = (len(self._fps_window) - 1) / span

        self._frames_analysed += 1
        self._inference_ms = result.inference_ms
        self._pipeline_ms = result.total_ms
        if captured_at:
            self._latency_ms = max(0.0, (now - captured_at) * 1000.0)
        self._last_detections = len(result.detections)
        self._person_count = result.person_count
        self._vehicle_count = result.vehicle_count
        self._is_night = result.is_night
        self._scene = result.scene
        self._zones = result.zones

    def _set_online(self, online: bool, persist: bool = False) -> None:
        changed = online != self._online
        self._online = online
        if online:
            self._offline_announced = False

        now = time.time()
        if not (changed or persist):
            return
        # Throttle DB writes — status changes are rare, but a flapping RTSP
        # link must not turn into a write storm.
        if not changed and now - self._last_status_write < 30.0:
            return
        self._last_status_write = now

        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if cam and cam.is_online != online:
                cam.is_online = online
                db.commit()
                log.info("Camera %d (%s) is now %s", self.camera_id, self.name,
                         "ONLINE" if online else "OFFLINE")
        except Exception as exc:
            log.warning("Camera status write failed: %s", exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    @property
    def is_online(self) -> bool:
        return self._online and self.source.is_online

    def stats(self) -> dict:
        """Live runtime metrics — every value measured from this process."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "url": self.url,
            "online": self.is_online,
            "state": self._state,
            "state_label": CameraState.LABELS.get(self._state, (self._state, "grey"))[0],
            "state_colour": CameraState.LABELS.get(self._state, (self._state, "grey"))[1],
            "state_detail": self._state_detail,
            "state_seconds": round(time.time() - self._state_since, 1),
            "fps": round(self._fps, 1),
            "frames_analysed": self._frames_analysed,
            "inference_ms": round(self._inference_ms, 1),
            "pipeline_ms": round(self._pipeline_ms, 1),
            "latency_ms": round(self._latency_ms, 1),
            "detections": self._last_detections,
            "persons": self._person_count,
            "vehicles": self._vehicle_count,
            "night_mode": self._is_night,
            # Operator playback state, so a reloaded dashboard shows the pause
            # and speed that are actually in force rather than assuming 1x.
            "playback": self.playback_state,
            # The measurement behind the night decision, so the dashboard can
            # explain the state instead of merely asserting it.
            "scene": ({**self._scene.to_dict(),
                       "enter_threshold": settings.NIGHT_DARKNESS_ENTER,
                       "exit_threshold": settings.NIGHT_DARKNESS_EXIT}
                      if self._scene is not None else None),
            "uptime_seconds": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "rules": len(self.analyzer.rule_shapes),
            # Live per-rule observations, so a quiet zone can say why.
            "rule_diagnostics": self.analyzer.rules.diagnostics(),
            # Continuous occupancy, refreshed every analysed frame and pushed
            # to dashboards once a second. `occupied` means something is inside
            # the polygon right now; `breached` means it has been there past
            # the rule's dwell threshold.
            "zones": self._zones,
            "zones_occupied": sum(1 for z in self._zones if z.get("occupied")),
            "zones_breached": sum(1 for z in self._zones if z.get("breached")),
            "active_clips": self.clips.active_clips,
            "source": self.source.describe(),
        }


# --------------------------------------------------------------------------- #
# Camera manager
# --------------------------------------------------------------------------- #


class CameraManager:
    """
    Owns every :class:`CameraProcessor`.

    Cameras are fully independent — one failing source cannot stall another —
    while the expensive resource (the YOLO model) is shared.

    Two invariants make "remove" mean something here:

    **A stopped processor is never left in the dict.**  Registration, startup
    and the teardown signal all happen under one lock, so ``add_camera`` and
    ``remove_camera`` cannot interleave into the state where a processor is
    started *after* it was stopped — which produced a live pipeline that no
    dictionary referenced and nothing could ever stop.  Only the *joining*,
    which can block for seconds on a thread parked inside FFmpeg, happens
    outside the lock, so a slow camera never stalls the rest of the system.

    **A retired camera id is not started again.**  ``retire`` records a
    tombstone before the database row is touched.  Removal tears the runtime
    down first (so nothing can seal an event against a row that is about to
    disappear), which leaves a window in which a concurrent ``PUT`` or
    ``restart`` holding a stale read of that row would start a fresh processor
    for a camera the operator has just removed — a zombie nothing later
    removes, because the removal has already run.  The tombstone closes it.
    """

    _instance: Optional["CameraManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._cameras: dict[int, CameraProcessor] = {}
        #: Camera ids removed in this process. Never large: an id enters when a
        #: camera is retired and leaves only when that id is deliberately
        #: reusable again (re-registration, or a hard reset).
        self._retired: set[int] = set()
        #: Backgrounded teardowns still joining their threads.
        self._reapers: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._detector: Optional[Detector] = None

    @classmethod
    def get(cls) -> "CameraManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector.get()
        return self._detector

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def add_camera(self, camera: Camera) -> Optional[CameraProcessor]:
        """
        Build and start the pipeline for ``camera``.

        Returns the processor already running for this id if there is one, and
        ``None`` when the camera cannot be started — including when its id has
        been retired, which is a refusal rather than a failure.
        """
        camera_id = int(camera.id)
        url, name, location = camera.url, camera.name, (camera.location or "")
        # Reading the ORM attributes and loading the model happen before the
        # lock: the first can emit a lazy SELECT, the second can take seconds.
        detector = self.detector

        with self._lock:
            if camera_id in self._retired:
                log.info("Camera %d was removed — not starting it again", camera_id)
                return None
            existing = self._cameras.get(camera_id)
            if existing is not None and not existing.is_stopped:
                return existing
            try:
                proc = CameraProcessor(
                    camera_id=camera_id, url=url, name=name,
                    location=location, detector=detector,
                )
            except Exception as exc:
                log.exception("Cannot start camera %d (%s): %s", camera_id, name, exc)
                self._cameras.pop(camera_id, None)
                return None
            # Registered and started under the same lock, so a concurrent
            # removal is observed either before or after the pair, never
            # between them.
            self._cameras[camera_id] = proc
            proc.start()
        return proc

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def remove_camera(self, camera_id: int, join_timeout: float = 5.0,
                      background: bool = False) -> bool:
        """
        Stop a camera and forget it. Returns True if one was running.

        The dict entry and the "you are dead now" signal are delivered together
        under the lock; the bounded joins happen after it is released.

        ``background`` hands those joins to a reaper thread and returns at once.
        That is safe precisely because ``request_stop`` has already done
        everything that defines removal — the processor is out of the dict,
        tombstoned, unable to publish a frame or seal an event — so the join is
        bookkeeping, and making an operator's Remove click wait on a thread
        parked in a native call buys them nothing. Shutdown and the hard reset
        pass ``background=False``, because *they* genuinely must not return
        while a thread still holds a camera device.
        """
        camera_id = int(camera_id)
        with self._lock:
            proc = self._cameras.pop(camera_id, None)
            if proc is not None:
                proc.request_stop()          # microseconds — safe under the lock
        if proc is None:
            return False
        if background:
            self._reap(proc, join_timeout)
        else:
            proc.stop(join_timeout=join_timeout)
        return True

    def _reap(self, proc: CameraProcessor, join_timeout: float) -> None:
        """Finish a teardown off the caller's thread, tracked so we can wait."""
        def _join() -> None:
            try:
                proc.stop(join_timeout=join_timeout)
            except Exception as exc:
                log.warning("Error reaping camera %d: %s", proc.camera_id, exc)
            finally:
                with self._lock:
                    self._reapers.discard(thread)

        thread = threading.Thread(target=_join, name=f"reap-cam{proc.camera_id}",
                                  daemon=True)
        with self._lock:
            self._reapers.add(thread)
        thread.start()

    def wait_for_reapers(self, timeout: float = 10.0) -> bool:
        """
        Block until every backgrounded teardown has finished.

        Used by shutdown, by the hard reset and by tests — anywhere the next
        step needs the certainty that no camera thread is still running.
        """
        deadline = time.time() + max(0.0, timeout)
        with self._lock:
            pending = list(self._reapers)
        for thread in pending:
            thread.join(timeout=max(0.0, deadline - time.time()))
        with self._lock:
            return not any(t.is_alive() for t in self._reapers)

    def retire(self, camera_id: int, join_timeout: float = 5.0,
               background: bool = False) -> bool:
        """
        Remove a camera **and** refuse to start that id again in this process.

        This is what the database retirement path calls. ``release`` is the
        deliberate undo, for when the id genuinely becomes reusable.
        """
        camera_id = int(camera_id)
        with self._lock:
            self._retired.add(camera_id)
        removed = self.remove_camera(camera_id, join_timeout=join_timeout,
                                     background=background)
        FrameBuffer.get().drop(camera_id)
        return removed

    def release(self, camera_id: int) -> None:
        """Allow a previously retired id to be started again."""
        with self._lock:
            self._retired.discard(int(camera_id))

    def clear_retired(self) -> None:
        """Forget every tombstone — a hard reset starts from a blank slate."""
        with self._lock:
            self._retired.clear()

    def is_retired(self, camera_id: int) -> bool:
        with self._lock:
            return int(camera_id) in self._retired

    def is_running(self, camera_id: int) -> bool:
        """True only while a live, non-stopped processor is registered."""
        with self._lock:
            proc = self._cameras.get(int(camera_id))
        return proc is not None and not proc.is_stopped

    def get_camera(self, camera_id: int) -> Optional[CameraProcessor]:
        with self._lock:
            return self._cameras.get(int(camera_id))

    def list_cameras(self) -> list[CameraProcessor]:
        with self._lock:
            return list(self._cameras.values())

    def reload_camera_rules(self, camera_id: int) -> None:
        proc = self.get_camera(camera_id)
        if proc is not None and not proc.is_stopped:
            proc.reload_rules()

    def reload_all_rules(self) -> None:
        for proc in self.list_cameras():
            proc.reload_rules()

    def stats(self) -> list[dict]:
        return [proc.stats() for proc in self.list_cameras()]

    def aggregate(self) -> dict:
        """System-wide live counters for the dashboard's stat row."""
        cameras = self.list_cameras()
        online = [c for c in cameras if c.is_online]
        return {
            "cameras_total": len(cameras),
            "cameras_online": len(online),
            "persons_live": sum(c._person_count for c in online),
            "vehicles_live": sum(c._vehicle_count for c in online),
            "detections_live": sum(c._last_detections for c in online),
            "system_fps": round(sum(c._fps for c in online), 1),
            "avg_inference_ms": round(
                sum(c._inference_ms for c in online) / len(online), 1
            ) if online else 0.0,
            "avg_latency_ms": round(
                sum(c._latency_ms for c in online) / len(online), 1
            ) if online else 0.0,
            "night_mode": any(c._is_night for c in online),
            # System-wide continuous intrusion state, so the dashboard can hold
            # an alert condition without waiting for the next discrete event.
            "zones_occupied": sum(
                sum(1 for z in c._zones if z.get("occupied")) for c in online
            ),
            "zones_breached": sum(
                sum(1 for z in c._zones if z.get("breached")) for c in online
            ),
        }

    def stop_all(self, join_timeout: float = 5.0) -> int:
        """
        Stop every camera. Returns how many were running.

        Every processor is signalled *first*, under the lock, so the whole
        system goes dead in microseconds; only then are the threads joined, and
        those joins run concurrently.  Serially, each camera could contribute
        its full timeout — with the 44 sources this database accumulated, a
        shutdown or a hard reset would have taken minutes while the event loop
        waited, which is indistinguishable from the hang being reported.
        """
        with self._lock:
            processors = list(self._cameras.values())
            self._cameras.clear()
        for proc in processors:
            try:
                proc.request_stop()
            except Exception as exc:
                log.warning("Error signalling camera %d: %s", proc.camera_id, exc)

        if not processors:
            # A removal may still be finishing in the background; callers of
            # stop_all need "nothing is running" to be true when it returns.
            self.wait_for_reapers(timeout=join_timeout)
            return 0

        def _join(proc: CameraProcessor) -> None:
            try:
                proc.stop(join_timeout=join_timeout)
            except Exception as exc:
                log.warning("Error stopping camera %d: %s", proc.camera_id, exc)

        workers = [
            threading.Thread(target=_join, args=(proc,),
                             name=f"stop-cam{proc.camera_id}", daemon=True)
            for proc in processors
        ]
        for worker in workers:
            worker.start()
        # One timeout for the whole set, not one per camera.
        deadline = time.time() + join_timeout + 2.0
        for worker in workers:
            worker.join(timeout=max(0.1, deadline - time.time()))
        self.wait_for_reapers(timeout=max(0.1, deadline - time.time()))
        log.info("Stopped %d camera(s)", len(processors))
        return len(processors)


# Backwards-compatible re-export: older imports expect these from core.camera.
ClipWriter = ClipRecorder
__all__ = [
    "FrameBuffer", "CameraProcessor", "CameraManager", "CameraState",
    "ClipRecorder", "ClipWriter", "LowLightEnhancer",
]
