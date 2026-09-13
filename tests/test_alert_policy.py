"""
What escalates, and what stays quiet.

Severity is not decoration here: it is what decides whether a siren sounds and
a desktop notification interrupts someone. Get it wrong in the loud direction
and the operator mutes the console; get it wrong in the quiet direction and an
intrusion passes unnoticed. So the grading is pinned by tests rather than left
to whoever last edited the table.

The rule being enforced: **routine traffic never escalates, intrusions always
do.** On a road-facing camera the routine types outnumber the security events
by an order of magnitude, so this is the difference between a system that gets
used and one that gets ignored.
"""
from __future__ import annotations

import pytest

from core.models import SEVERITY_ORDER
from cv.rules import (
    RED_ALERT_SEVERITIES, SEVERITY_BY_TYPE, is_red_alert, severity_for,
)

#: An intrusion, or something an operator must react to now.
MUST_ESCALATE = [
    "entry",            # crossed the fence onto the protected side
    "zone_presence",    # still inside a restricted zone past the threshold
    "watchlist_match",  # a face on the watchlist
    "enter",            # entered a restricted zone
    "loiter",           # dwelling in a monitored area
    "wrong_direction",  # moving against a one-way constraint
    "night_movement",   # movement in a scene measured as dark
    "camera_offline",   # a blind spot is a security condition
]

#: Routine observation. Recorded, displayed, and silent.
MUST_STAY_QUIET = [
    "human_detected",
    "vehicle_detected",
    "face_detected",
    "anpr_detection",
    "zone_exit",
    "exit",
]


@pytest.mark.parametrize("alert_type", MUST_ESCALATE)
def test_an_intrusion_escalates(alert_type):
    severity = severity_for(alert_type)
    assert is_red_alert(severity), (
        f"{alert_type} is graded {severity}, so it would raise no siren and no "
        f"notification — but it is an event an operator must act on"
    )


@pytest.mark.parametrize("alert_type", MUST_STAY_QUIET)
def test_routine_traffic_stays_quiet(alert_type):
    severity = severity_for(alert_type)
    assert not is_red_alert(severity), (
        f"{alert_type} is graded {severity}, so every occurrence would sound a "
        f"siren and raise a notification. On a road-facing camera that is a "
        f"siren per passing vehicle, and an operator who mutes the console"
    )


def test_the_highest_volume_types_are_the_quietest():
    """
    ``human_detected`` and ``vehicle_detected`` fire for *every* object seen.

    Measured on a 45-second run of a street-facing camera: 88 and 90 rows
    respectively, against a handful of genuine rule events. They belong at the
    noise floor, below even LOW, so a severity filter set to anything at all
    hides them.
    """
    for alert_type in ("human_detected", "vehicle_detected"):
        severity = severity_for(alert_type)
        assert SEVERITY_ORDER[severity] == SEVERITY_ORDER["INFO"], (
            f"{alert_type} is {severity}; it is the highest-volume event type "
            f"in the system and must sit at the noise floor"
        )


def test_a_sustained_intrusion_outranks_a_momentary_one():
    """
    Standing inside a restricted zone is worse than touching its boundary.

    ``enter`` fires the instant someone crosses in — which a person may do by
    accident. ``zone_presence`` fires only after they are still there past the
    dwell threshold, which they do not do by accident.
    """
    assert SEVERITY_ORDER[severity_for("zone_presence")] > \
           SEVERITY_ORDER[severity_for("enter")]


def test_crossing_inward_outranks_crossing_outward():
    """Entering the protected side is the intrusion; leaving it is context."""
    assert SEVERITY_ORDER[severity_for("entry")] > \
           SEVERITY_ORDER[severity_for("exit")]


def test_every_known_alert_type_is_graded():
    """An ungraded type falls to a default and escalates by accident."""
    from core.events import ALERT_TITLES

    ungraded = [t for t in ALERT_TITLES if t not in SEVERITY_BY_TYPE]
    assert not ungraded, f"alert types with no severity: {ungraded}"


def test_every_grade_is_a_severity_the_rest_of_the_system_knows():
    unknown = {t: s for t, s in SEVERITY_BY_TYPE.items() if s not in SEVERITY_ORDER}
    assert not unknown, f"unknown severities: {unknown}"


def test_red_alerts_are_the_top_two_grades():
    """The threshold must stay meaningful: red is the exception, not the rule."""
    assert set(RED_ALERT_SEVERITIES) == {"HIGH", "CRITICAL"}
    red = [t for t, s in SEVERITY_BY_TYPE.items() if is_red_alert(s)]
    quiet = [t for t, s in SEVERITY_BY_TYPE.items() if not is_red_alert(s)]
    assert red and quiet, "the grading must actually separate the two"


def test_the_threshold_is_published_to_the_dashboard(client):
    """
    The front end must not keep its own copy of the policy.

    If it did, retuning ``NOTIFY_MIN_SEVERITY`` on the server would silently
    fail to change what the operator's browser actually does.
    """
    info = client.get("/api/system/info").json()
    alerting = info.get("alerting")
    assert alerting, "/api/system/info does not publish the alerting policy"
    assert alerting["notify_min_severity"] in SEVERITY_ORDER
    assert set(alerting["red_alert_severities"]) == set(RED_ALERT_SEVERITIES)
    assert alerting["severity_by_type"]["human_detected"] == "INFO"
    assert alerting["severity_by_type"]["entry"] == "CRITICAL"


def test_a_sealed_event_carries_the_graded_severity(db, camera):
    """The log and the escalation policy must agree on every row."""
    from core.events import EventManager
    from core.models import Alert
    from cv.rules import Alert as RuleAlert

    for alert_type in ("entry", "human_detected"):
        EventManager.get().record(
            camera_id=camera.id,
            rule_alert=RuleAlert(rule_name="t", rule_type="line", track_id=1,
                                 alert_type=alert_type, description="x"),
            frame=None, camera_name="T", capture_evidence=False,
        )

    rows = {row.alert_type: row.severity for row in db.query(Alert).all()}
    assert rows["entry"] == "CRITICAL"
    assert rows["human_detected"] == "INFO"


@pytest.fixture
def client(db):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
