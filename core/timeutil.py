"""
Timezone handling for IBVAP.

Policy
------
* **Storage / transport** — every timestamp is a *timezone-aware* UTC
  ISO-8601 string (``2026-09-11T05:12:33.412000+00:00``).  UTC keeps the
  strings lexicographically sortable, makes range filters correct, and stays
  unambiguous across DST-free-but-offset regions.
* **Display** — every user-facing surface (dashboard, video overlay, event
  log, evidence metadata, API display fields) renders **Indian Standard
  Time, Asia/Kolkata (UTC+05:30)**.

Naive ``datetime`` objects are never produced by this module, and
``datetime.now()`` (machine-local) is never used anywhere in the codebase.
The IST offset is defined explicitly so the platform behaves identically on
a laptop in Delhi, a server in Frankfurt, or a container with ``TZ=UTC``.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

#: Matches a timezone offset whose "+" was turned into a space by URL decoding.
_PLUS_EATEN_BY_URL = re.compile(r"\s(\d{2}:?\d{2})$")

# India has never observed DST and has a single fixed offset, so a fixed
# ``timezone`` is exact and needs no tzdata package on Windows.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")
UTC = timezone.utc

#: e.g. ``11 Sep 2026, 12:35:42 AM IST``
DISPLAY_FORMAT = "%d %b %Y, %I:%M:%S %p IST"
#: e.g. ``12:35:42 AM IST`` — compact form for overlays and alert rows
TIME_FORMAT = "%I:%M:%S %p IST"
#: filesystem-safe IST stamp for evidence filenames
FILE_STAMP_FORMAT = "%Y%m%d_%H%M%S"


def now_utc() -> datetime:
    """Current instant as an aware UTC datetime."""
    return datetime.now(UTC)


def now_ist() -> datetime:
    """Current instant as an aware IST (Asia/Kolkata) datetime."""
    return datetime.now(UTC).astimezone(IST)


def to_ist(value: Union[datetime, str, float, int, None]) -> Optional[datetime]:
    """
    Convert anything timestamp-shaped to an aware IST datetime.

    Accepts aware/naive ``datetime`` (naive is assumed UTC — that is what the
    platform stores), ISO-8601 strings (``Z`` suffix supported), and POSIX
    epoch seconds.  Returns ``None`` when the value cannot be interpreted so
    callers can fall back gracefully instead of raising inside a render path.
    """
    dt = to_utc(value)
    return dt.astimezone(IST) if dt is not None else None


def to_utc(value: Union[datetime, str, float, int, None]) -> Optional[datetime]:
    """Convert anything timestamp-shaped to an aware UTC datetime (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # A "+" in a URL query string decodes to a space, so an IST
            # timestamp arrives as "2026-09-11T02:38:48 05:30". A space
            # immediately before a timezone offset can only have been a "+",
            # so repair it rather than silently dropping the client's filter.
            repaired = _PLUS_EATEN_BY_URL.sub(r"+\1", text)
            try:
                dt = datetime.fromisoformat(repaired)
            except ValueError:
                return None
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return None


def utc_iso(value: Union[datetime, str, float, None] = None) -> str:
    """Canonical storage representation: aware UTC ISO-8601."""
    dt = to_utc(value) if value is not None else now_utc()
    return (dt or now_utc()).isoformat()


def ist_iso(value: Union[datetime, str, float, None] = None) -> str:
    """Aware IST ISO-8601 — e.g. ``2026-09-11T00:35:42.123456+05:30``."""
    dt = to_ist(value) if value is not None else now_ist()
    return (dt or now_ist()).isoformat()


def fmt_ist(value: Union[datetime, str, float, None] = None,
            fmt: str = DISPLAY_FORMAT) -> str:
    """Human-readable IST string; returns ``"—"`` for unparseable input."""
    dt = to_ist(value) if value is not None else now_ist()
    if dt is None:
        return "—"
    return dt.strftime(fmt)


def fmt_ist_time(value: Union[datetime, str, float, None] = None) -> str:
    """Compact IST clock string — ``12:35:42 AM IST``."""
    return fmt_ist(value, TIME_FORMAT)


def file_stamp(value: Union[datetime, str, float, None] = None) -> str:
    """IST stamp suitable for evidence filenames — ``20260911_003542``."""
    dt = to_ist(value) if value is not None else now_ist()
    return (dt or now_ist()).strftime(FILE_STAMP_FORMAT)


def ist_hour(value: Union[datetime, str, float, None] = None) -> int:
    """Hour-of-day (0-23) in IST — used by the night-movement rule."""
    dt = to_ist(value) if value is not None else now_ist()
    return (dt or now_ist()).hour


def is_night(value: Union[datetime, str, float, None] = None,
             start_hour: int = 19, end_hour: int = 6) -> bool:
    """
    True when the instant falls inside the configured night window (IST).

    The window wraps midnight when ``start_hour > end_hour`` (the normal
    case, e.g. 19:00 → 06:00).
    """
    hour = ist_hour(value)
    if start_hour == end_hour:
        return False
    if start_hour > end_hour:          # wraps midnight
        return hour >= start_hour or hour < end_hour
    return start_hour <= hour < end_hour


def start_of_ist_day(value: Union[datetime, str, float, None] = None) -> datetime:
    """Midnight IST for the given instant, returned as an aware UTC datetime."""
    dt = to_ist(value) if value is not None else now_ist()
    dt = dt or now_ist()
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
