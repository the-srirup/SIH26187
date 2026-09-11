"""
YouTube and YouTube Live ingestion.

A YouTube URL is not a video stream — it is a web page.  ``cv2.VideoCapture``
cannot open one, so the platform resolves it to a direct media URL first and
then hands *that* to the ordinary :class:`~core.video_source.LiveSource`.  From
the frame onwards nothing differs from an RTSP camera: same decoder, same
latest-frame buffer, same :class:`~core.analytics.FrameAnalyzer`, same rules,
tracking, ANPR, face stage, evidence and hash chain.  There is deliberately no
second analytics path for YouTube.

::

    https://youtu.be/XXXX
            |
            v   yt-dlp resolves formats (no download)
    https://...googlevideo.com/videoplayback?...   (progressive MP4 or HLS)
            |
            v
        LiveSource  ->  FrameAnalyzer  ->  rules/ANPR/face  ->  events

Two properties of resolved URLs drive the design:

* **They expire.** Google signs playback URLs with a deadline, typically a few
  hours. A source that resolved once at startup and cached the result forever
  would die mid-shift with an opaque decode error, so resolutions carry a TTL
  and a reconnect re-resolves rather than retrying a dead URL.
* **Resolution is slow** (a network round trip, sometimes seconds). It
  therefore never runs on the request thread — see the camera-registration
  path, which validates cheaply and resolves on the capture thread.

Only publicly accessible videos are supported.  Nothing here attempts to work
around age gates, private videos, purchases or regional restrictions; those are
reported to the operator as the errors they are.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("ibvap.youtube")

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    YTDLP_AVAILABLE = False
    log.info("yt-dlp not installed — YouTube sources unavailable "
             "(pip install yt-dlp)")


#: youtube.com/watch?v=, youtu.be/, /live/, /shorts/, /embed/ and m. variants.
_YOUTUBE_RE = re.compile(
    r"^https?://(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?|live/|shorts/|embed/|v/)|youtu\.be/)",
    re.IGNORECASE,
)

#: How long a resolved playback URL is trusted before it is resolved again.
#: Google's signatures usually last ~6 hours; re-resolving well inside that
#: window costs one request and avoids a mid-shift outage.
RESOLVE_TTL_SECONDS = 90 * 60

#: Cap the resolved stream's height. A 4K YouTube feed decoded in full is pure
#: waste when analytics run at 640x384 — it multiplies decode cost by an order
#: of magnitude for detail the resize discards immediately. 1080p still carries
#: far more than the analytics frame needs, and leaves ANPR and the face stage
#: real pixels to crop from.
MAX_HEIGHT = 1080


class _NullLogger:
    """
    Swallow yt-dlp's own console output.

    yt-dlp writes errors straight to stderr even under ``quiet``, which put raw
    extractor tracebacks into the server log for what are ordinary, handled
    conditions (a private video, a removed video). The message still reaches the
    operator — it is raised as :class:`YouTubeError` and surfaced on the camera
    tile — so printing it again unfiltered only buries the real logs.
    """

    def debug(self, msg):  # noqa: D102 - yt-dlp logger protocol
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.debug("yt-dlp: %s", msg)

    def error(self, msg):
        log.debug("yt-dlp: %s", msg)


class YouTubeError(RuntimeError):
    """A YouTube source could not be resolved, with an operator-facing reason."""


@dataclass
class ResolvedStream:
    """A direct media URL plus what we learned about the source."""

    url: str
    title: str = ""
    is_live: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    format_note: str = ""
    resolved_at: float = 0.0
    expires_at: float = 0.0

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    def describe(self) -> dict:
        return {
            "title": self.title,
            "is_live": self.is_live,
            "resolution": f"{self.width}x{self.height}" if self.width else "—",
            "fps": round(self.fps, 2),
            "format": self.format_note,
            "resolved_at": self.resolved_at,
            "expires_in_seconds": max(0, int(self.expires_at - time.time())),
        }


def is_youtube_url(url: str) -> bool:
    """True when ``url`` is a YouTube watch/live/shorts link."""
    return bool(_YOUTUBE_RE.match(str(url or "").strip()))


def available() -> bool:
    """True when YouTube ingestion can actually run."""
    return YTDLP_AVAILABLE


def _classify_error(message: str) -> str:
    """Turn a yt-dlp error into something an operator can act on."""
    low = message.lower()
    if "private" in low:
        return "This video is private and cannot be accessed."
    if "age" in low and ("confirm" in low or "restrict" in low or "sign in" in low):
        return "This video is age-restricted and cannot be accessed without sign-in."
    if "members-only" in low or "join this channel" in low:
        return "This video is members-only and cannot be accessed."
    if "unavailable" in low or "removed" in low or "does not exist" in low:
        return "This video is unavailable or has been removed."
    if "not available in your country" in low or "geo" in low:
        return "This video is not available from this location."
    if "live event will begin" in low or "premieres in" in low:
        return "This stream has not started yet."
    if "sign in to confirm" in low or "bot" in low:
        return ("YouTube is requiring sign-in for this request. Try a different "
                "public video, or use a direct RTSP/HTTP camera URL.")
    if "urlopen" in low or "timed out" in low or "network" in low or "resolve" in low:
        return "Could not reach YouTube — check the network connection."
    return message.strip()[:300]


def _pick_format(info: dict) -> tuple[str, dict]:
    """
    Choose a format OpenCV/FFmpeg can actually decode.

    Preference order matters.  A progressive MP4 (video+audio muxed, H.264) is
    the most reliable thing to hand FFmpeg.  A live stream has no progressive
    format at all, so its HLS manifest is used instead.  DASH video-only
    fragments are accepted last because FFmpeg handles them, but less
    predictably.
    """
    formats = info.get("formats") or []
    if not formats:
        # Some extractions return a ready-made url with no format list.
        if info.get("url"):
            return info["url"], info
        raise YouTubeError("No playable format was offered for this video.")

    def height_of(f) -> int:
        return int(f.get("height") or 0)

    def usable(f) -> bool:
        return bool(f.get("url")) and f.get("vcodec") not in (None, "none")

    live = bool(info.get("is_live"))
    candidates = [f for f in formats if usable(f)]

    if live:
        # HLS is what YouTube serves for live; m3u8 protocols only.
        hls = [f for f in candidates
               if "m3u8" in str(f.get("protocol") or "")
               and height_of(f) <= MAX_HEIGHT]
        if hls:
            return max(hls, key=height_of)["url"], max(hls, key=height_of)
        raise YouTubeError("This live stream offers no HLS format to read.")

    progressive = [f for f in candidates
                   if f.get("acodec") not in (None, "none")
                   and str(f.get("ext")) == "mp4"
                   and height_of(f) <= MAX_HEIGHT]
    if progressive:
        best = max(progressive, key=height_of)
        return best["url"], best

    video_only = [f for f in candidates
                  if height_of(f) <= MAX_HEIGHT
                  and "m3u8" not in str(f.get("protocol") or "")]
    if video_only:
        best = max(video_only, key=height_of)
        return best["url"], best

    best = max(candidates, key=height_of)
    return best["url"], best


def resolve(url: str, *, timeout: float = 20.0) -> ResolvedStream:
    """
    Resolve a YouTube URL to a direct media URL.

    Raises :class:`YouTubeError` with an operator-facing message for every
    failure mode — private, removed, age-gated, not yet started, offline — so
    the dashboard can say what is wrong instead of showing a dead tile.
    """
    if not YTDLP_AVAILABLE:
        raise YouTubeError(
            "YouTube support requires yt-dlp. Install it with: pip install yt-dlp"
        )
    url = str(url or "").strip()
    if not is_youtube_url(url):
        raise YouTubeError(f"Not a YouTube URL: {url}")

    options = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": timeout,
        "noplaylist": True,
        # Metadata only — nothing is ever written to disk.
        "skip_download": True,
        # Deliberately NO "format" selector. Passing one makes yt-dlp *itself*
        # fail the extraction when nothing matches ("Requested format is not
        # available"), which turned an ordinary video into a hard error. We want
        # the full format list and choose from it in _pick_format, where the
        # decision can fall back sensibly instead of raising.
        "logger": _NullLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise YouTubeError(_classify_error(str(exc))) from exc

    if not info:
        raise YouTubeError("YouTube returned no information for this URL.")
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        if not entries:
            raise YouTubeError("This playlist is empty.")
        info = entries[0]

    try:
        media_url, chosen = _pick_format(info)
    except YouTubeError:
        raise
    except Exception as exc:
        raise YouTubeError(f"Could not select a playable format: {exc}") from exc

    now = time.time()
    resolved = ResolvedStream(
        url=media_url,
        title=str(info.get("title") or "")[:200],
        is_live=bool(info.get("is_live")),
        width=int(chosen.get("width") or info.get("width") or 0),
        height=int(chosen.get("height") or info.get("height") or 0),
        fps=float(chosen.get("fps") or info.get("fps") or 0.0),
        format_note=str(chosen.get("format_note") or chosen.get("format_id") or ""),
        resolved_at=now,
        expires_at=now + RESOLVE_TTL_SECONDS,
    )
    log.info(
        "YOUTUBE_RESOLVED '%s' %s %dx%d @ %.1f fps (%s)",
        resolved.title[:60], "LIVE" if resolved.is_live else "VOD",
        resolved.width, resolved.height, resolved.fps, resolved.format_note,
    )
    return resolved


class YouTubeResolver:
    """
    Caching resolver shared by every YouTube source in the process.

    Two cameras pointed at the same stream should cost one resolution, and a
    reconnect must not stampede yt-dlp.  The per-URL lock serialises concurrent
    resolutions of the same link; different links resolve in parallel.
    """

    _instance: Optional["YouTubeResolver"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: dict[str, ResolvedStream] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @classmethod
    def get(cls) -> "YouTubeResolver":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _lock_for(self, url: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(url, threading.Lock())

    def resolve(self, url: str, *, force: bool = False) -> ResolvedStream:
        """Resolve ``url``, reusing a cached result until it nears expiry."""
        url = str(url).strip()
        if not force:
            cached = self._cache.get(url)
            if cached is not None and not cached.expired:
                return cached

        with self._lock_for(url):
            # Another thread may have resolved it while we waited.
            cached = self._cache.get(url)
            if not force and cached is not None and not cached.expired:
                return cached
            resolved = resolve(url)
            self._cache[url] = resolved
            return resolved

    def invalidate(self, url: str) -> None:
        self._cache.pop(str(url).strip(), None)

    def cached(self, url: str) -> Optional[ResolvedStream]:
        return self._cache.get(str(url).strip())


__all__ = [
    "YouTubeError", "ResolvedStream", "YouTubeResolver",
    "is_youtube_url", "available", "resolve", "MAX_HEIGHT",
]
