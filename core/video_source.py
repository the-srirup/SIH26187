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
import threading
import time
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


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    """
    Open a video source, returning ``None`` rather than raising.

    RTSP gets a short buffer so a reconnect does not replay several seconds of
    stale video before catching up.
    """
    src = _parse_source(url)
    try:
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
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
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_open_attempt = 0.0
        self._fps_window: list[float] = []
        #: A file source is *paced* to its own frame rate. A real camera emits
        #: frames in real time; a decoder does not, and left unpaced a 12 fps
        #: clip was being decoded at ~1300 fps with 99% of frames thrown away —
        #: burning CPU and racing the footage past the analytics.
        self._is_file = False
        self._frame_interval = 0.0
        self._next_frame_due = 0.0
        #: Set once a frame has ever arrived, so the pipeline can tell
        #: "still starting up" apart from "was up, now down".
        self._ever_connected = False

    # -- lifecycle ------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"capture-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._new_frame:
            self._new_frame.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release()

    def _release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # -- capture thread ------------------------------------------------- #
    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None:
                if not self._try_open():
                    time.sleep(0.35)
                    continue

            if self._frame_interval:
                # Emit at the file's own rate so a recorded clip behaves like
                # the camera that produced it.
                wait = self._next_frame_due - time.time()
                if wait > 0:
                    time.sleep(min(wait, 0.25))
                    continue
                self._next_frame_due += self._frame_interval
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
                time.sleep(0.05)
                continue

            now = time.time()
            self._track_fps(now)

            with self._new_frame:
                if self._frame is not None:
                    # A frame the analytics loop never consumed is being
                    # discarded — that is the latency guarantee working.
                    self.stats.frames_dropped += 1
                self._frame = frame
                self._frame_id += 1
                self._frame_ts = now
                self.stats.frames_read += 1
                self.stats.last_frame_at = now
                self.stats.connected = True
                self.stats.last_error = ""
                self._ever_connected = True
                self._new_frame.notify_all()

        self._release()
        with self._new_frame:
            self.stats.connected = False
            self._new_frame.notify_all()

    def _try_open(self) -> bool:
        now = time.time()
        if now - self._last_open_attempt < settings.RECONNECT_INTERVAL:
            return False
        self._last_open_attempt = now

        cap = open_capture(self.url)
        if cap is None:
            if self.stats.connected or not self.stats.last_error:
                log.warning("[%s] cannot open source '%s'", self.name, self.url)
            self.stats.connected = False
            self.stats.last_error = f"Cannot open source: {self.url}"
            return False

        self._cap = cap
        self.stats.reconnects += 1
        self.stats.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.stats.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.stats.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.stats.last_error = ""

        # A positive frame count means a finite file rather than a live feed.
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._is_file = total > 0
        if self._is_file and 0 < self.stats.source_fps < 240:
            self._frame_interval = 1.0 / self.stats.source_fps
        else:
            self._frame_interval = 0.0
        self._next_frame_due = time.time()

        log.info(
            "[%s] connected — %dx%d @ %.1f fps%s",
            self.name, self.stats.width, self.stats.height, self.stats.source_fps,
            " (file, paced to source rate)" if self._frame_interval else "",
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
                        with self._new_frame:
                            self._frame = frame
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
            if self._frame is None:
                return None, last_id, 0.0
            frame, fid, ts = self._frame, self._frame_id, self._frame_ts
            # Hand the frame off; a fresh one will replace it.
            self._frame = None
            return frame, fid, ts

    @property
    def is_online(self) -> bool:
        if not self.stats.connected:
            return False
        return (time.time() - self.stats.last_frame_at) < settings.CAMERA_TIMEOUT

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
            "last_error": s.last_error,
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
