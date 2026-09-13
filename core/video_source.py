"""
Video ingestion — the front of the pipeline.

Two source types feed the *same* analytics core:

``LiveSource``
    RTSP / HTTP / webcam / looping file.  A dedicated capture thread drains
    the decoder continuously and keeps **only the newest frame**.  When
    inference cannot keep up, old frames are dropped rather than queued, so
    latency stays bounded instead of growing without limit — the single most
    important property of a real-time surveillance feed.

``FileSource``
    An uploaded MP4 analysed offline.  Here the opposite policy is correct:
    every frame (subject to a configurable stride) must be examined, so the
    reader is synchronous and never drops.

Both expose the same ``read()`` contract, which is what lets a live camera
and an uploaded video share one :class:`~core.analytics.FrameAnalyzer`.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.source")


@dataclass
class SourceStats:
    """What the dashboard needs to know about an ingest point."""

    connected: bool = False
    frames_read: int = 0
    frames_dropped: int = 0
    capture_fps: float = 0.0
    last_frame_at: float = 0.0
    last_error: str = ""
    reconnects: int = 0
    width: int = 0
    height: int = 0
    source_fps: float = 0.0


def _parse_source(url: str):
    """Webcam index, or a path/URL passed through untouched."""
    text = str(url).strip()
    if text.isdigit():
        return int(text)
    return text


#: FFmpeg open/read timeouts, in microseconds, pushed in via the environment
#: variable OpenCV's FFmpeg backend reads.
#:
#: Without this, ``cv2.VideoCapture`` on an unreachable RTSP host blocks the
#: calling thread for FFmpeg's default timeout — which on Windows can be
#: minutes, and for some transports is unbounded. That is the mechanism behind
#: "the site lags when I add a live camera": a mistyped or unreachable camera
#: URL parked a thread indefinitely, and because a stopped source only joins its
#: capture thread for a few seconds, every retry leaked another stuck thread and
#: another open socket. Bounding the open is what makes a bad URL a fast, clean
#: error instead of a slow resource leak.
_FFMPEG_OPEN_TIMEOUT_US = 8_000_000    # 8 s to establish
_FFMPEG_READ_TIMEOUT_US = 8_000_000    # 8 s without data before giving up

#: The same bounds in milliseconds, for OpenCV's *own* watchdog, which is a
#: separate mechanism from the FFmpeg options above and wins when it fires
#: first (see :func:`_capture_params`).
#:
#: The FFmpeg options above are necessary and are not sufficient, which is easy
#: to miss because they look like they cover it. OpenCV wraps every FFmpeg call
#: in an interrupt callback of its own, and that callback has separate
#: environment variables and its own 30-second default. Measured against an
#: unroutable host, a ``VideoCapture`` open returned after::
#:
#:     [WARN] _opencv_ffmpeg_interrupt_callback Stream timeout triggered
#:            after 30072.851000 ms
#:
#: — 30 s, not the 8 s configured right above it, because the interrupt fired
#: first. With ``RECONNECT_INTERVAL`` at 4 s that is a capture thread parked in
#: a 30-second syscall, waking briefly, and parking again, for as long as the
#: camera stays unreachable. A handful of mistyped or offline sources is then
#: a handful of threads each spending ~90% of its life inside an uninterruptible
#: native call — which is also why a removed camera's thread appeared to
#: outlive its removal by half a minute.
#:
#: Setting these through the environment does not work; they have to be passed
#: as open parameters. Both are kept because the FFmpeg-level options still
#: matter for transports the interrupt callback does not cover.
_FFMPEG_OPEN_TIMEOUT_MS = str(_FFMPEG_OPEN_TIMEOUT_US // 1000)
_FFMPEG_READ_TIMEOUT_MS = str(_FFMPEG_READ_TIMEOUT_US // 1000)


def _apply_ffmpeg_timeouts() -> None:
    """
    Bound how long a capture may block, before one is opened.

    Every variable is set only if the operator has not set it, so explicit
    tuning in the environment always wins.
    """
    import os

    if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join((
            "rtsp_transport;tcp",                      # UDP silently blackholes
            f"timeout;{_FFMPEG_READ_TIMEOUT_US}",      # newer FFmpeg
            f"stimeout;{_FFMPEG_OPEN_TIMEOUT_US}",     # older FFmpeg
            "reconnect;1",
            "reconnect_streamed;1",
            "reconnect_delay_max;4",
        ))


# Applied at import as well as before each open: OpenCV caches some of these
# configuration parameters the first time it reads them, which for a process
# that opens its first capture late is well after the value stopped mattering.
_apply_ffmpeg_timeouts()


def _webcam_backends() -> list[int]:
    """
    Backends to try for a local webcam index, best first.

    On Windows OpenCV defaults to Media Foundation, and MSMF is slow to open a
    camera: measured on this machine, ``VideoCapture(0)`` took 1803 ms with the
    default backend and 765 ms with DirectShow, and the first ``read()`` after
    it 781 ms against 532 ms. With a second handle already on the device —
    exactly what a remove-then-re-add does, because the old capture thread may
    still be unwinding — the gap is far wider: 1473 ms against 30 ms.

    That delay is the whole of "I added the webcam and the tile just sits
    there", so DirectShow is tried first and the default backend is kept as the
    fallback for any device DirectShow will not open. Elsewhere the default is
    already the right one and the list is left alone.
    """
    if sys.platform != "win32":
        return [0]
    dshow = getattr(cv2, "CAP_DSHOW", None)
    return [int(dshow), 0] if dshow is not None else [0]


def _capture_params() -> list:
    """
    Per-open timeouts for OpenCV's interrupt callback, or ``[]`` if unsupported.

    These are what actually bound the open; the FFmpeg options alone do not.
    ``CAP_PROP_*_TIMEOUT_MSEC`` exist from OpenCV 4.5, so the lookup is by
    ``getattr`` and an older build simply keeps its own default.
    """
    open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_prop is None or read_prop is None:
        return []
    return [int(open_prop), int(_FFMPEG_OPEN_TIMEOUT_MS),
            int(read_prop), int(_FFMPEG_READ_TIMEOUT_MS)]


def resolve_source_url(url: str) -> str:
    """
    Turn whatever the operator registered into something FFmpeg can open.

    This is the single junction where the source kinds converge.  An RTSP, HTTP
    or file URL is already openable and passes through untouched; a YouTube link
    is a *web page* and is resolved to a direct media URL first.  Everything
    downstream — decode, analytics, rules, ANPR, face, evidence — is identical
    for all of them, which is the point: there is one pipeline, not one per
    source type.
    """
    from core.youtube import YouTubeResolver, is_youtube_url

    text = str(url or "").strip()
    if not is_youtube_url(text):
        return text
    # Raises YouTubeError with an operator-facing message, which the capture
    # thread turns into the source's last_error.
    return YouTubeResolver.get().resolve(text).url


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    """
    Open a video source, returning ``None`` rather than raising.

    RTSP gets a short buffer so a reconnect does not replay several seconds of
    stale video before catching up, and a bounded timeout so an unreachable
    host fails in seconds rather than parking the thread.

    The timeout is passed as *open parameters*, not only through the
    environment. Measured against an unroutable host, the environment
    variables alone left the open taking 30.1 s — OpenCV's interrupt-callback
    default — while the same open with these parameters returned in 8.1 s. The
    difference is not academic: an unreachable camera retries forever, so it is
    the difference between a thread that is blocked most of the time and one
    that is blocked almost all of the time.
    """
    _apply_ffmpeg_timeouts()
    src = _parse_source(url)
    try:
        if isinstance(src, str) and src.lower().startswith(("rtsp", "http")):
            params = _capture_params()
            cap = (cv2.VideoCapture(src, cv2.CAP_FFMPEG, params) if params
                   else cv2.VideoCapture(src, cv2.CAP_FFMPEG))
        elif isinstance(src, int):
            cap = _open_webcam(src)
        else:
            cap = cv2.VideoCapture(src)
    except Exception as exc:
        log.warning("VideoCapture(%s) raised: %s", url, exc)
        return None

    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass  # not supported by every backend — harmless
    return cap


def _open_webcam(index: int) -> Optional[cv2.VideoCapture]:
    """Open a local camera index, preferring the backend that opens fastest."""
    for backend in _webcam_backends():
        try:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        except Exception as exc:
            log.debug("Webcam %d: backend %s raised %s", index, backend, exc)
            continue
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None


def probe_video(path: str | Path) -> dict:
    """Read an MP4's metadata without decoding it. Used to validate uploads."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {"valid": False, "error": "Unreadable or corrupt video file"}

    info = {
        "valid": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return {"valid": False, "error": "Video contains no decodable frames"}
    if info["width"] <= 0 or info["height"] <= 0:
        return {"valid": False, "error": "Video reports an invalid frame size"}

    fps = info["fps"] if 0 < info["fps"] < 240 else 25.0
    info["fps"] = fps
    info["duration_seconds"] = round(info["frame_count"] / fps, 2) if info["frame_count"] else 0.0
    return info


#: Bounds for the operator's playback speed control. Below 0.5x a review drags
#: and the pipeline idles; above 2x a file source decodes faster than the
#: analytics can consume, so frames would be shed rather than watched.
PLAYBACK_MIN_SPEED = 0.5
PLAYBACK_MAX_SPEED = 2.0


class LiveSource:
    """
    Threaded latest-frame capture for a continuous feed.

    The capture thread owns the ``VideoCapture`` exclusively; consumers only
    ever see the most recent decoded frame.  ``read()`` blocks until a *new*
    frame arrives (or the timeout expires), so the analytics loop is driven by
    the source instead of spinning on the newest frame repeatedly.
    """

    def __init__(self, url: str, name: str = "", loop_files: Optional[bool] = None) -> None:
        self.url = url
        self.name = name or url
        self.loop_files = settings.LOOP_FILE_SOURCES if loop_files is None else loop_files

        self.stats = SourceStats()
        self._cap: Optional[cv2.VideoCapture] = None
        #: The latest decoded frame, and *only* the latest.
        #:
        #: A ``deque(maxlen=1)`` rather than a list or a queue, so the bound is
        #: structural: if the analytics thread stalls — a slow inference, a
        #: blocked disk, a GC pause — the decoder keeps overwriting this one
        #: slot instead of stacking frames behind it. At 640x384x3 a frame is
        #: 737 KB, so an unbounded hand-off on a stalled 15 fps camera would
        #: grow RAM by ~11 MB per second and raise end-to-end latency by the
        #: whole backlog. Dropping is the right answer for surveillance: the
        #: operator wants the newest frame, never a queue of old ones.
        self._frames: deque = deque(maxlen=1)
        self._frame_id = 0
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        #: Set to stop the capture thread. Every wait in the loop is a wait on
        #: this, so a stopping source never sits out a sleep it no longer needs.
        self._stop_event = threading.Event()
        self._stop_event.set()              # not running until start() clears it
        self._thread: Optional[threading.Thread] = None
        self._last_open_attempt = 0.0
        self._fps_window: list[float] = []
        #: A file source is *paced* to its own frame rate. A real camera emits
        #: frames in real time; a decoder does not, and left unpaced a 12 fps
        #: clip was being decoded at ~1300 fps with 99% of frames thrown away —
        #: burning CPU and racing the footage past the analytics.
        self._is_file = False
        #: A webcam / capture card, opened by index. Such a device emits in real
        #: time by definition and cannot outrun it, which is what makes pacing
        #: it not merely pointless but harmful — see ``_try_open``.
        self._is_local_device = False
        self._frame_interval = 0.0
        #: The rate the capture loop is actually pacing to, which is not always
        #: what the source reports (see ``_try_open``).
        self._paced_rate = 0.0
        self._next_frame_due = 0.0
        #: Operator playback control. ``_speed`` multiplies the paced rate, so
        #: 2.0 decodes a recording twice as fast and 0.5 at half speed; it has
        #: no meaning for a live camera, which cannot outrun real time, so the
        #: capture loop only honours it where pacing is exact (see
        #: ``playback_applies``). ``_paused`` stops the decode entirely.
        self._speed = 1.0
        self._paused = False
        #: Set once a frame has ever arrived, so the pipeline can tell
        #: "still starting up" apart from "was up, now down".
        self._ever_connected = False
        #: What the URL actually resolved to (differs from ``url`` only for
        #: YouTube). Kept for diagnostics; never shown raw to the operator,
        #: since a signed playback URL is long and carries credentials.
        self._resolved_url = ""
        #: Incremented every time a finite file restarts from the beginning, and
        #: every time a dropped stream is reopened. Consumers watch this to know
        #: their tracking and rule state has become meaningless: at a loop seam
        #: every object jumps to a new position, which a fence rule holding the
        #: previous lap's trajectory would happily report as a crossing.
        self._generation = 0

    # -- lifecycle ------------------------------------------------------ #
    @property
    def _running(self) -> bool:
        """True while the capture thread should keep decoding."""
        return not self._stop_event.is_set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"capture-{self.name}", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """
        Signal the capture thread to finish, without waiting for it.

        Separated from :meth:`stop` so a caller holding a lock — the camera
        manager removing a source — can make the source dead instantly and do
        the joining afterwards, outside that lock.
        """
        self._stop_event.set()
        with self._new_frame:
            self.stats.connected = False
            self._new_frame.notify_all()

    def stop(self, join_timeout: float = 3.0) -> None:
        """
        Stop capture and release the decoder.

        The release is conditional on the capture thread actually having
        exited. ``cv2.VideoCapture.release()`` called while that thread is
        parked inside ``read()`` on the same handle is a use-after-free in
        native code, and a thread blocked on an unreachable RTSP host is
        exactly when it would happen. An overrunning thread releases its own
        handle on the way out (see the tail of ``_capture_loop``), so nothing
        leaks either way.
        """
        self.request_stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                log.warning(
                    "SOURCE_STOP_SLOW [%s]: capture thread still inside a "
                    "blocking read — it releases its own handle on exit",
                    self.name,
                )
        self.cleanup()

    @property
    def is_thread_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _invalidate_resolution(self) -> None:
        """Force the next attempt to re-resolve (YouTube signatures expire)."""
        try:
            from core.youtube import YouTubeResolver, is_youtube_url
            if is_youtube_url(self.url):
                YouTubeResolver.get().invalidate(self.url)
        except Exception:
            pass

    def _release(self) -> None:
        """Release the decoder handle. Only ever called from the owning thread."""
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                log.debug("[%s] release raised: %s", self.name, exc)

    def cleanup(self, wait: float = 0.0) -> None:
        """
        Guarantee the capture device is released, and drop the frame slot.

        Releasing an OpenCV capture is not optional bookkeeping — a webcam
        handle left open keeps the device locked against the next camera that
        wants it, and an RTSP handle keeps a session established on the
        recorder. But it is also not safe to do from just anywhere:
        ``VideoCapture.release()`` called while the capture thread is inside
        ``read()`` on the same handle is a use-after-free in native code, and a
        thread blocked on an unreachable host is exactly when that would
        happen.

        So the release is always explicit and always happens, from whichever of
        the two places can do it safely:

        * if the capture thread has exited, release here and now;
        * if it has not, it releases its own handle at the end of
          ``_capture_loop`` — which it reaches as soon as the native call
          returns, because ``_stop_event`` is already set. A short ``wait``
          lets a caller give it that chance; a watchdog then reports anything
          still holding on, rather than leaving it silent.

        Idempotent: calling it twice, or on a source that never started, does
        nothing the second time.
        """
        self._stop_event.set()
        with self._new_frame:
            self._new_frame.notify_all()

        thread = self._thread
        if wait > 0 and thread is not None and thread.is_alive():
            thread.join(timeout=wait)

        if thread is None or not thread.is_alive():
            self._release()                       # explicit, immediate
            self._thread = None
        elif self._cap is not None:
            # The owning thread still has it; it releases on its way out.
            log.info(
                "[%s] capture handle still owned by its thread — it is released "
                "when the pending read returns", self.name,
            )

        with self._new_frame:
            self._frames.clear()                  # ~737 KB per frame, freed now
            self.stats.connected = False
            self._new_frame.notify_all()

    # -- playback control ------------------------------------------------ #
    @property
    def is_file(self) -> bool:
        """Is this source a finite recording rather than a live feed?"""
        return bool(self._is_file)

    @property
    def playback_applies(self) -> bool:
        """
        Can this source honour a speed change?

        Only a source we pace exactly can: a recording is decoded as fast or as
        slowly as we ask. A webcam or an RTSP camera emits in real time by
        definition — asking it for 2x would just drop every other frame, and
        0.5x would build the backlog the pacing exists to prevent — so speed is
        reported as unavailable there rather than silently doing nothing.
        """
        return bool(self._is_file and not self._is_local_device)

    def _effective_interval(self) -> float:
        """Frame interval with the operator's speed multiplier applied."""
        if not self.playback_applies or self._speed <= 0:
            return self._frame_interval
        return self._frame_interval / self._speed

    def set_playback(self, paused: Optional[bool] = None,
                     speed: Optional[float] = None) -> dict:
        """Apply an operator's pause / speed request. Returns the new state."""
        with self._lock:
            if paused is not None:
                was = self._paused
                self._paused = bool(paused)
                # Resuming must not try to repay the whole pause as a burst of
                # frames, so the schedule restarts from now.
                if was and not self._paused:
                    self._next_frame_due = time.time()
            if speed is not None:
                self._speed = max(PLAYBACK_MIN_SPEED,
                                  min(PLAYBACK_MAX_SPEED, float(speed)))
            return self.playback_state

    @property
    def playback_state(self) -> dict:
        return {
            "paused": bool(self._paused),
            "speed": round(float(self._speed), 2),
            "speed_supported": self.playback_applies,
            "effective_fps": (round(self._paced_rate * self._speed, 1)
                              if self.playback_applies else round(self._paced_rate, 1)),
        }

    # -- capture thread ------------------------------------------------- #
    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None:
                if not self._try_open():
                    self._stop_event.wait(0.35)     # wakes instantly on stop
                    continue

            if self._paused:
                # Hold the decoder still. The wait is on the stop event, so a
                # paused source still tears down instantly rather than sitting
                # out a sleep. Nothing is read, so a recording resumes exactly
                # where the operator stopped it instead of having raced on.
                self._stop_event.wait(0.1)
                continue

            if self._frame_interval:
                # Emit at (or near) the source's own rate.
                #
                # This used to apply only to files, on the reasoning that a real
                # camera already emits in real time. Network streams do not.
                # A YouTube Live HLS feed hands FFmpeg whole buffered segments,
                # so this loop decoded it at 370-750 fps — measured — while the
                # analytics thread it was feeding fell to 1.5 fps. Nothing was
                # queueing and nothing was leaking: the capture thread was simply
                # burning every core it could reach decoding frames that were
                # thrown away microseconds later, and starving the rest of the
                # process. That is the "adding a live camera makes the site lag"
                # report, reproduced exactly.
                #
                # A live source is paced with headroom rather than pinned, so it
                # can still out-run real time briefly to regain the live edge
                # after a stall; a file is paced exactly, because playing a
                # recording faster than it was shot is never what is wanted.
                wait = self._next_frame_due - time.time()
                if wait > 0:
                    self._stop_event.wait(min(wait, 0.25))
                    continue
                self._next_frame_due += self._effective_interval()
                # Never accumulate a debt we cannot repay (e.g. after a stall).
                if time.time() - self._next_frame_due > 1.0:
                    self._next_frame_due = time.time()

            try:
                ok, frame = self._cap.read()
            except Exception as exc:
                log.warning("[%s] read raised: %s", self.name, exc)
                ok, frame = False, None

            if not ok or frame is None:
                if self._handle_read_failure():
                    continue
                self._stop_event.wait(0.05)
                continue

            now = time.time()
            self._track_fps(now)

            with self._new_frame:
                if self._frames:
                    # A frame the analytics loop never consumed is being
                    # discarded — that is the latency guarantee working, and
                    # the deque's maxlen is what enforces it.
                    self.stats.frames_dropped += 1
                self._frames.append(frame)
                self._frame_id += 1
                self._frame_ts = now
                self.stats.frames_read += 1
                self.stats.last_frame_at = now
                self.stats.connected = True
                self.stats.last_error = ""
                self._ever_connected = True
                self._new_frame.notify_all()

        # The owning thread's own release, on the way out. This is the path
        # that frees a handle cleanup() could not touch because this thread was
        # still inside a blocking read when stop was requested.
        self._release()
        with self._new_frame:
            self._frames.clear()
            self.stats.connected = False
            self._new_frame.notify_all()
        log.debug("[%s] capture thread exited and released its handle", self.name)

    def _try_open(self) -> bool:
        # Opening a capture is an uninterruptible native call of up to
        # _FFMPEG_OPEN_TIMEOUT_MS. Re-checking here — after the reconnect wait,
        # immediately before committing to it — is what keeps a source that was
        # stopped during that wait from entering one last blocking open that
        # nothing wants the answer to.
        if not self._running:
            return False
        now = time.time()
        if now - self._last_open_attempt < settings.RECONNECT_INTERVAL:
            return False
        self._last_open_attempt = now

        # Resolve here, on the capture thread, not at registration time: a
        # YouTube playback URL expires after a few hours, so a reconnect must
        # re-resolve rather than retry a dead signature. It also keeps a slow
        # network round trip off the request path entirely.
        try:
            target = resolve_source_url(self.url)
        except Exception as exc:
            self.stats.connected = False
            self.stats.last_error = str(exc)[:300]
            log.warning("[%s] cannot resolve source: %s", self.name, exc)
            return False
        self._resolved_url = target

        self._is_local_device = isinstance(_parse_source(target), int)
        cap = open_capture(target)
        if cap is None:
            if self.stats.connected or not self.stats.last_error:
                log.warning("CAMERA_OPEN_FAILED [%s] '%s'", self.name, self.url)
            self.stats.connected = False
            self.stats.last_error = f"Cannot open source: {self.url}"
            # A resolved YouTube URL that will not open is usually a stale
            # signature; drop it so the next attempt resolves afresh.
            self._invalidate_resolution()
            return False

        self._cap = cap
        self.stats.reconnects += 1
        # A reopened stream is a discontinuity like a loop seam.
        if self._ever_connected:
            self._generation += 1
        self.stats.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.stats.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.stats.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.stats.last_error = ""

        # A positive frame count means a finite file rather than a live feed.
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._is_file = total > 0
        # How fast to pull frames — which depends entirely on what kind of
        # source this is, and getting it wrong in either direction hurts.
        #
        # Too fast, and a buffered network stream becomes a CPU sink: a YouTube
        # Live HLS feed hands FFmpeg whole segments, and an unpaced loop decoded
        # it at 370-750 fps — measured — while the analytics it fed fell to 1.5.
        #
        # Too slow, and a real-time source backs up *inside the driver*. A
        # webcam delivers 30 fps whether we read them or not; pulling at 22.5
        # leaves the difference queued, and since that queue is FIFO every frame
        # we then read is the oldest one in it. Measured on this machine,
        # consuming a 30 fps webcam at 15 fps: 45 of 45 reads returned a
        # distinct frame and each returned in 0.2 ms instead of blocking ~33 ms
        # for the sensor — the signature of draining a backlog rather than
        # sitting at the live edge. ``CAP_PROP_BUFFERSIZE=1`` does not help;
        # DirectShow ignores it.
        #
        # So:
        reported = self.stats.source_fps
        if self._is_local_device:
            # A camera device cannot outrun real time, so there is nothing to
            # protect against — and ``read()`` blocking on the sensor is itself
            # perfect pacing. Anything we add here only inserts latency.
            rate = 0.0
        elif self._is_file:
            # A recording plays at exactly the rate it was shot at.
            rate = reported if 0 < reported < 240 else float(settings.TARGET_FPS)
        elif 0 < reported < 240:
            # A network camera, pacing with headroom so it can regain the live
            # edge after a stall — always above its own rate, never below.
            rate = reported * max(1.0, float(settings.LIVE_CAPTURE_HEADROOM))
        else:
            # A live source that will not say how fast it is. Cap the runaway
            # case, but stay above any ordinary camera rate so we can never
            # create the backlog described above.
            rate = max(
                float(settings.TARGET_FPS) * max(1.0, float(settings.LIVE_CAPTURE_HEADROOM)),
                30.0,
            )
            log.debug("[%s] source reports fps=%.1f — pacing at %.1f fps instead",
                      self.name, reported, rate)

        self._frame_interval = (1.0 / rate) if rate > 0 else 0.0
        self._paced_rate = rate
        self._next_frame_due = time.time()

        if self._is_local_device:
            pacing = " (local device — read at the sensor's own rate)"
        elif self._is_file and 0 < reported < 240:
            pacing = " (file, paced to source rate)"
        else:
            pacing = f" (paced to {self._paced_rate:.0f} fps)"
        log.info(
            "[%s] connected — %dx%d @ %s%s",
            self.name, self.stats.width, self.stats.height,
            f"{reported:.1f} fps" if reported > 0 else "unreported fps",
            pacing,
        )
        return True

    def _handle_read_failure(self) -> bool:
        """EOF on a finite file loops; anything else forces a reconnect."""
        cap = self._cap
        if cap is not None and self.loop_files:
            try:
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES) or 0
                if total > 0 and pos >= total - 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self._generation += 1
                        log.debug("[%s] video file looped (generation %d)",
                                  self.name, self._generation)
                        with self._new_frame:
                            self._frames.append(frame)
                            self._frame_id += 1
                            self._frame_ts = time.time()
                            self.stats.frames_read += 1
                            self.stats.last_frame_at = self._frame_ts
                            self._new_frame.notify_all()
                        return True
            except Exception:
                pass

        self._release()
        self.stats.connected = False
        self.stats.last_error = "Stream ended or dropped — reconnecting"
        self._invalidate_resolution()
        return False

    def _track_fps(self, now: float) -> None:
        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self.stats.capture_fps = (len(self._fps_window) - 1) / span

    # -- consumer API --------------------------------------------------- #
    def read(self, timeout: float = 1.0, last_id: int = -1):
        """
        Wait for a frame newer than ``last_id``.

        Returns ``(frame, frame_id, captured_at)`` or ``(None, last_id, 0.0)``
        on timeout.  The frame is handed over without copying; the capture
        thread never mutates a frame it has already published.
        """
        deadline = time.time() + timeout
        with self._new_frame:
            while self._running and self._frame_id <= last_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, last_id, 0.0
                self._new_frame.wait(remaining)
            if not self._frames:
                return None, last_id, 0.0
            # popleft on a maxlen=1 deque: take the one frame there is and
            # leave the slot empty for the next decode.
            frame, fid, ts = self._frames.popleft(), self._frame_id, self._frame_ts
            return frame, fid, ts

    @property
    def is_online(self) -> bool:
        if not self.stats.connected:
            return False
        return (time.time() - self.stats.last_frame_at) < settings.CAMERA_TIMEOUT

    @property
    def generation(self) -> int:
        """Bumped on every loop restart or reconnect — a tracking discontinuity."""
        return self._generation

    @property
    def kind(self) -> str:
        """``youtube`` / ``rtsp`` / ``http`` / ``webcam`` / ``file``."""
        from core.youtube import is_youtube_url

        text = str(self.url).strip()
        if is_youtube_url(text):
            return "youtube"
        if text.isdigit():
            return "webcam"
        low = text.lower()
        if low.startswith("rtsp"):
            return "rtsp"
        if low.startswith(("http://", "https://")):
            return "http"
        return "file"

    @property
    def youtube_info(self) -> Optional[dict]:
        """Resolved YouTube metadata, when this is a YouTube source."""
        try:
            from core.youtube import YouTubeResolver, is_youtube_url
            if not is_youtube_url(self.url):
                return None
            cached = YouTubeResolver.get().cached(self.url)
            return cached.describe() if cached else {"resolved": False}
        except Exception:
            return None

    @property
    def ever_connected(self) -> bool:
        """False until the first frame arrives — distinguishes 'starting up'
        from 'went down', so startup never raises a spurious OFFLINE alert."""
        return self._ever_connected

    def describe(self) -> dict:
        s = self.stats
        return {
            "url": self.url,
            "connected": s.connected,
            "online": self.is_online,
            "frames_read": s.frames_read,
            "frames_dropped": s.frames_dropped,
            "capture_fps": round(s.capture_fps, 2),
            "reconnects": max(0, s.reconnects - 1),
            "resolution": f"{s.width}x{s.height}" if s.width else "—",
            "source_fps": round(s.source_fps, 2),
            "paced": bool(self._frame_interval),
            "paced_fps": round(self._paced_rate, 1),
            "playback": self.playback_state,
            "local_device": bool(self._is_local_device),
            "is_file": bool(self._is_file),
            "loops": bool(self._is_file and self.loop_files),
            "generation": self._generation,
            "last_error": s.last_error,
            "kind": self.kind,
            "youtube": self.youtube_info,
        }


class FileSource:
    """
    Synchronous reader for offline analysis of an uploaded video.

    Never drops frames — an investigator analysing recorded footage needs
    complete coverage, not the lowest latency.  ``stride`` skips frames
    uniformly to keep long clips tractable while retaining even sampling.
    """

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        self.path = str(path)
        self.stride = max(1, int(stride))
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"Cannot open video file: {self.path}")

        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 0 < fps < 240 else 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count else 0.0
        self._index = 0

    def __iter__(self):
        """Yield ``(frame_index, media_time_seconds, frame)`` for kept frames."""
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            index = self._index
            self._index += 1
            if index % self.stride:
                continue
            yield index, index / self.fps, frame

    @property
    def position(self) -> int:
        return self._index

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "FileSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
