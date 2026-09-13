"""
Regression tests for the dashboard's start-up path.

These exist because of a defect that made the virtual-fence drawing tools
completely inert: ``renderAlertToggles`` declared ``const red`` inside its
``if (sound)`` block and then read it again inside ``if (notify)``, which is a
ReferenceError under strict mode.  ``restoreAlerting`` calls that function on
the third line of ``init``, so the exception propagated out of the
DOMContentLoaded handler and every statement below it never ran — including
the two ``addEventListener`` calls that wire the fence canvas.  Clicking the
canvas did nothing, and the page showed no error.

What made it easy to miss is that the faulty expression sits in the *false*
branch of a ternary guarded by ``Notification.permission === 'denied'``.  A
browser that had denied notifications skipped it and the dashboard worked; the
default state, which is what an operator actually starts in, crashed.  Both
states are therefore asserted below.

The JS is exercised for real in Node against a minimal DOM shim rather than
pattern-matched, so this catches *any* exception that escapes ``init`` — not
just this one identifier.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "js" / "init_smoke.mjs"
APP_JS = ROOT / "static" / "js" / "app.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node.js is not installed; the dashboard bundle cannot be executed",
)


def run_init(permission: str) -> dict:
    """Load static/js/app.js in Node and fire DOMContentLoaded."""
    proc = subprocess.run(
        ["node", str(HARNESS), str(APP_JS), permission],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
        # Explicit on Windows: under pytest's capture the inherited stdin
        # handle is invalid and CreateProcess fails with WinError 6.
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("permission", ["default", "granted", "denied"])
def test_init_survives_every_notification_permission(permission):
    result = run_init(permission)
    assert result["loaded"], "the dashboard bundle did not even parse"
    assert result["initCompleted"], (
        f"init() raised with Notification.permission={permission!r}: "
        f"{result['error']}.  Everything below the throwing line is now "
        f"unwired — the fence canvas included."
    )


@pytest.mark.parametrize("permission", ["default", "granted", "denied"])
def test_fence_canvas_is_wired_for_drawing(permission):
    """The drawing tools are dead unless init() reaches its listener wiring.

    Asserted for each permission state because the original bug only broke
    two of the three.
    """
    wired = run_init(permission)["wired"]
    assert "fence-canvas:click" in wired, (
        "no click listener on the fence canvas — clicking it cannot place a "
        "tripwire or zone point"
    )
    assert "fence-canvas:dblclick" in wired, (
        "no dblclick listener on the fence canvas — a polygon cannot be closed"
    )


def test_init_wires_the_rest_of_the_page_too():
    """A guard on the statements that sit below the canvas wiring."""
    wired = run_init("default")["wired"]
    for expected in ("document:keydown", "drop-zone:drop", "cam-dropzone:drop"):
        assert expected in wired, f"init() never reached the {expected} wiring"

# --------------------------------------------------------------------------- #
# Live-stream connection budget
# --------------------------------------------------------------------------- #
#
# A browser opens at most 6 concurrent HTTP/1.1 connections per origin, and an
# MJPEG tile holds one open for as long as it is on screen. With six tiles
# streaming, every other request the dashboard makes queues behind them and
# never runs. Measured against this server: with 5 streams open /health
# answered in 4 ms; with 6 it never answered at all, and closing one stream
# recovered it immediately. The dashboard therefore caps concurrent streams and
# refreshes the remaining tiles from snapshots.
#
# These guard the two things whose silent removal would bring the freeze back.

BROWSER_CONNECTION_LIMIT = 6


def _app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_stream_budget_leaves_connections_for_the_api():
    """The cap must sit strictly below the browser's per-origin limit."""
    src = _app_js()
    m = re.search(r"const\s+STREAM_BUDGET\s*=\s*(\d+)", src)
    assert m, "STREAM_BUDGET is gone — nothing caps concurrent MJPEG connections"
    budget = int(m.group(1))
    assert 0 < budget < BROWSER_CONNECTION_LIMIT, (
        f"STREAM_BUDGET={budget} leaves no connection free for API calls; the "
        f"browser allows only {BROWSER_CONNECTION_LIMIT} per origin, so the "
        f"dashboard would freeze once that many tiles are streaming"
    )


def test_tiles_beyond_the_budget_fall_back_to_snapshots():
    """The overflow path has to exist, or extra tiles would just go black."""
    src = _app_js()
    assert "goSnapshot" in src, "no snapshot fallback for tiles beyond the budget"
    assert "SNAPSHOT_INTERVAL_MS" in src, "snapshot refresh interval is gone"
    assert "/snapshot" in src, "snapshot endpoint is no longer requested"


def test_the_budget_is_actually_applied_when_tiles_change():
    """Guards the wiring: a budget nothing enforces is not a budget.

    ``renderCameras`` is what adds and removes tiles, so it must re-cut the
    allocation; otherwise a newly added camera just opens another stream.
    """
    src = _app_js()
    m = re.search(r"function renderCameras\(\)\s*\{(.*?)\n  \}", src, re.S)
    assert m, "renderCameras() not found"
    assert "rebalanceStreams()" in m.group(1), (
        "renderCameras() no longer re-cuts the stream budget, so adding a "
        "camera can push the page past the browser's connection limit"
    )
