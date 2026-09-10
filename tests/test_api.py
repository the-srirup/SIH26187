"""
API surface tests using FastAPI's TestClient.

Cameras are never started here (``is_active=False``), so these run fast and
without a GPU: they verify routing, validation, serialisation and the IST
contract, not the video pipeline.
"""
import json

import pytest

pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(db):
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_camera(client):
    response = client.post("/api/cameras", data={
        "name": "API-CAM", "url": "samples/sample_border_scenario.mp4",
        "location": "Test", "is_active": "false",
    })
    assert response.status_code == 201
    return response.json()


# ------------------------------------------------------------------ health


def test_health_reports_ist(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "Asia/Kolkata" in body["timezone"]
    assert body["timestamp_ist"].endswith("IST")
    assert body["timestamp"].endswith("+00:00")


def test_time_endpoint_offset(client):
    body = client.get("/api/system/time").json()
    assert body["utc_offset"] == "+05:30"
    assert body["timezone"] == "Asia/Kolkata"
    assert "+05:30" in body["ist_iso"]


def test_root_redirects_to_dashboard(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert "/dashboard" in response.headers["location"]


# ----------------------------------------------------------------- cameras


def test_create_and_fetch_camera(client, api_camera):
    assert api_camera["name"] == "API-CAM"
    fetched = client.get(f"/api/cameras/{api_camera['id']}").json()
    assert fetched["id"] == api_camera["id"]
    assert fetched["stream_url"] == f"/stream/{api_camera['id']}"


def test_camera_requires_name_and_url(client):
    assert client.post("/api/cameras", data={"name": "  ", "url": "0"}).status_code == 400
    assert client.post("/api/cameras", data={"name": "x", "url": " "}).status_code == 400


def test_update_camera(client, api_camera):
    response = client.put(f"/api/cameras/{api_camera['id']}",
                          data={"location": "Sector 9"})
    assert response.status_code == 200
    assert response.json()["location"] == "Sector 9"


def test_delete_camera(client, api_camera):
    assert client.delete(f"/api/cameras/{api_camera['id']}").status_code == 200
    assert client.get(f"/api/cameras/{api_camera['id']}").status_code == 404


def test_unknown_camera_is_404(client):
    assert client.get("/api/cameras/999999").status_code == 404


def test_upload_pseudo_camera_hidden_by_default(client):
    from core.analysis import ensure_upload_camera

    ensure_upload_camera()
    default = client.get("/api/cameras").json()
    assert all(c["source_kind"] != "upload" for c in default)
    included = client.get("/api/cameras?include_uploads=true").json()
    assert any(c["source_kind"] == "upload" for c in included)


# ------------------------------------------------------------------- rules


def test_create_line_rule(client, api_camera):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "line", "name": "TRIPWIRE",
        "geometry": json.dumps([[10, 10], [600, 300]]), "params": "{}",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["rule_type"] == "line" and len(body["geometry"]) == 2


def test_create_zone_rule_with_params(client, api_camera):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "zone", "name": "ZONE",
        "geometry": json.dumps([[10, 10], [100, 10], [100, 100]]),
        "params": json.dumps({"presence_seconds": 8}),
    })
    assert response.status_code == 201
    assert response.json()["params"]["presence_seconds"] == 8


@pytest.mark.parametrize("rule_type,geometry", [
    ("line", [[1, 1]]),                 # too few points for a line
    ("zone", [[1, 1], [2, 2]]),         # too few points for a polygon
    ("loiter", [[1, 1]]),
])
def test_reject_insufficient_geometry(client, api_camera, rule_type, geometry):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": rule_type, "geometry": json.dumps(geometry), "params": "{}",
    })
    assert response.status_code == 400


def test_reject_unknown_rule_type(client, api_camera):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "wormhole", "geometry": json.dumps([[1, 1], [2, 2]]), "params": "{}",
    })
    assert response.status_code == 400


def test_reject_malformed_geometry_json(client, api_camera):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "line", "geometry": "{not json", "params": "{}",
    })
    assert response.status_code == 400


def test_reject_non_numeric_coordinates(client, api_camera):
    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "line", "geometry": json.dumps([["a", "b"], [1, 2]]), "params": "{}",
    })
    assert response.status_code == 400


def test_coordinates_are_clamped_to_the_frame(client, api_camera):
    from core.config import settings

    response = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "line",
        "geometry": json.dumps([[-500, -500], [99999, 99999]]), "params": "{}",
    })
    assert response.status_code == 201
    geometry = response.json()["geometry"]
    assert geometry[0] == [0.0, 0.0]
    assert geometry[1] == [float(settings.FRAME_WIDTH), float(settings.FRAME_HEIGHT)]


def test_toggle_and_delete_rule(client, api_camera):
    created = client.post(f"/api/cameras/{api_camera['id']}/rules", data={
        "rule_type": "line", "geometry": json.dumps([[1, 1], [2, 2]]), "params": "{}",
    }).json()

    disabled = client.put(f"/api/rules/{created['id']}", data={"is_active": "false"})
    assert disabled.json()["is_active"] is False

    assert client.delete(f"/api/rules/{created['id']}").status_code == 200
    assert client.delete(f"/api/rules/{created['id']}").status_code == 404


def test_clear_all_rules(client, api_camera):
    for _ in range(3):
        client.post(f"/api/cameras/{api_camera['id']}/rules", data={
            "rule_type": "line", "geometry": json.dumps([[1, 1], [2, 2]]), "params": "{}",
        })
    response = client.delete(f"/api/cameras/{api_camera['id']}/rules")
    assert response.json()["removed"] == 3
    assert client.get(f"/api/cameras/{api_camera['id']}/rules").json() == []


# ------------------------------------------------------------------ alerts


def _make_alert(db, camera_id, **kwargs):
    from core.events import EventManager
    from cv.rules import Alert as RuleAlert

    alert = RuleAlert(
        rule_name=kwargs.pop("rule_name", "fence"),
        rule_type="fence", track_id=kwargs.pop("track_id", 5),
        alert_type=kwargs.pop("alert_type", "entry"),
        description="test event",
    )
    return EventManager.get().record(
        camera_id=camera_id, rule_alert=alert, frame=None,
        object_class=kwargs.pop("object_class", "person"),
        confidence=0.88, camera_name="API-CAM", capture_evidence=False,
    )


def test_alert_listing_shape(client, api_camera, db):
    _make_alert(db, api_camera["id"])
    body = client.get("/api/alerts").json()
    assert body["total"] >= 1
    alert = body["alerts"][0]
    for field in ("id", "title", "icon", "severity", "timestamp",
                  "timestamp_ist", "analysis_kind", "hash", "prev_hash"):
        assert field in alert
    assert alert["timestamp_ist"].endswith("IST")


def test_alert_filters(client, api_camera, db):
    _make_alert(db, api_camera["id"], alert_type="entry", track_id=1)
    _make_alert(db, api_camera["id"], alert_type="loiter", track_id=2)

    entries = client.get("/api/alerts?alert_type=entry").json()
    assert all(a["alert_type"] == "entry" for a in entries["alerts"])

    by_track = client.get("/api/alerts?track_id=2").json()
    assert all(a["track_id"] == 2 for a in by_track["alerts"])

    critical = client.get("/api/alerts?severity=CRITICAL").json()
    assert all(a["severity"] == "CRITICAL" for a in critical["alerts"])


def test_alert_pagination(client, api_camera, db):
    for i in range(6):
        _make_alert(db, api_camera["id"], track_id=i)
    page1 = client.get("/api/alerts?limit=2&offset=0").json()
    page2 = client.get("/api/alerts?limit=2&offset=2").json()
    assert len(page1["alerts"]) == 2 and len(page2["alerts"]) == 2
    assert {a["id"] for a in page1["alerts"]} & {a["id"] for a in page2["alerts"]} == set()


def test_alert_time_range_filter_accepts_ist(client, api_camera, db):
    from core.timeutil import ist_iso, now_utc
    from datetime import timedelta

    _make_alert(db, api_camera["id"])
    past = ist_iso(now_utc() - timedelta(hours=1))
    future = ist_iso(now_utc() + timedelta(hours=1))
    assert client.get(f"/api/alerts?from_ts={past}&to_ts={future}").json()["total"] >= 1
    assert client.get(f"/api/alerts?from_ts={future}").json()["total"] == 0


def test_unknown_alert_is_404(client):
    assert client.get("/api/alerts/999999").status_code == 404


def test_missing_evidence_returns_404_not_500(client, api_camera, db):
    payload = _make_alert(db, api_camera["id"])
    assert client.get(f"/api/alerts/{payload['id']}/snapshot").status_code == 404
    assert client.get(f"/api/alerts/{payload['id']}/clip").status_code == 404


def test_alert_severity_is_assigned_by_type(client, api_camera, db):
    payload = _make_alert(db, api_camera["id"], alert_type="entry")
    assert payload["severity"] == "CRITICAL"
    payload = _make_alert(db, api_camera["id"], alert_type="loiter", track_id=9)
    assert payload["severity"] == "HIGH"


def test_rule_events_are_labelled_rule_based(client, api_camera, db):
    payload = _make_alert(db, api_camera["id"], alert_type="entry")
    assert payload["analysis_kind"] == "RULE-BASED EVENT ANALYSIS"


def test_detection_events_are_labelled_ai(client, api_camera, db):
    payload = _make_alert(db, api_camera["id"], alert_type="human_detected")
    assert payload["analysis_kind"] == "AI DETECTION"


# --------------------------------------------------------------- integrity


def test_integrity_verify_on_empty_log(client):
    body = client.get("/api/integrity/verify").json()
    assert body["valid"] is True
    assert body["headline"] == "INTEGRITY VERIFIED"


def test_integrity_verify_after_events(client, api_camera, db):
    for _ in range(4):
        _make_alert(db, api_camera["id"])
    body = client.get("/api/integrity/verify").json()
    assert body["valid"] is True and body["total_alerts"] >= 4


def test_integrity_names_itself_honestly(client):
    body = client.get("/api/integrity/verify").json()
    assert "hash chain" in body["scheme"].lower()
    assert "blockchain" not in body["scheme"].lower()


def test_checkpoint_endpoint(client, api_camera, db):
    _make_alert(db, api_camera["id"])
    created = client.post("/api/integrity/checkpoint").json()
    assert created["created"] is True

    listing = client.get("/api/integrity/checkpoints").json()
    assert len(listing["checkpoints"]) >= 1

    uid = created["checkpoint"]["checkpoint_uid"]
    assert client.get(f"/api/integrity/checkpoints/{uid}/verify").json()["valid"] is True


def test_certificate_runs_real_verification(client, api_camera, db):
    _make_alert(db, api_camera["id"])
    body = client.post("/api/integrity/certificate",
                       data={"issued_to": "Test"}).json()
    assert body["verification"]["valid"] is True
    assert body["issue_timestamp_ist"].endswith("IST")
    assert "non-blockchain" in body["scheme"].lower()


# ------------------------------------------------------------------- stats


def test_stats_reflect_real_events(client, api_camera, db):
    _make_alert(db, api_camera["id"], alert_type="entry")
    _make_alert(db, api_camera["id"], alert_type="loiter", track_id=3)
    body = client.get("/api/stats?hours=24").json()
    assert body["total_alerts"] >= 2
    assert "entry" in body["by_type"]
    assert body["today_since_ist"].endswith("IST")


def test_system_info_reports_real_device(client):
    body = client.get("/api/system/info").json()
    assert "detector" in body and "pipeline" in body
    assert body["timezone"].startswith("Asia/Kolkata")


# ------------------------------------------------------------------ upload


def test_upload_rejects_non_mp4_content(client):
    response = client.post("/api/analysis/upload", files={
        "file": ("evil.mp4", b"MZ\x90\x00 windows executable", "video/mp4")})
    assert response.status_code == 400
    assert "MP4" in response.json()["detail"]


def test_upload_rejects_bad_extension(client):
    response = client.post("/api/analysis/upload", files={
        "file": ("clip.avi", b"\x00" * 500, "video/avi")})
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post("/api/analysis/upload", files={
        "file": ("empty.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_analysis_listing_is_available(client):
    body = client.get("/api/analysis").json()
    assert "active" in body and "history" in body


def test_unknown_analysis_session_is_404(client):
    assert client.get("/api/analysis/deadbeef").status_code == 404


def test_cancel_unknown_session_is_404(client):
    assert client.post("/api/analysis/deadbeef/cancel").status_code == 404


# ------------------------------------------------------------------- misc


def test_stream_for_unknown_camera_is_404(client):
    assert client.get("/stream/999999").status_code == 404


def test_snapshot_without_frames_is_404(client, api_camera):
    assert client.get(f"/api/cameras/{api_camera['id']}/snapshot").status_code == 404


def test_watchlist_endpoint_shape(client):
    body = client.get("/api/watchlist").json()
    assert "entries" in body and "status" in body


def test_watchlist_threshold_validation(client):
    assert client.put("/api/watchlist/threshold", data={"threshold": "1.5"}).status_code == 400
    assert client.put("/api/watchlist/threshold", data={"threshold": "-0.1"}).status_code == 400
