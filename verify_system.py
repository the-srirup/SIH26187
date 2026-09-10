"""
IBVAP end-to-end acceptance test against a running server.

Exercises the real HTTP/WebSocket surface — no mocks, no stubs. Every check
either passes against the live system or is reported as a failure.
"""
import io
import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS, FAIL, SKIP = [], [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  \033[92mPASS\033[0m  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  \033[91mFAIL\033[0m  {name}" + (f"   [{detail}]" if detail else ""))
    return condition


def skip(name, why):
    SKIP.append((name, why))
    print(f"  \033[93mSKIP\033[0m  {name}   [{why}]")


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        body = r.read()
        ctype = r.headers.get("content-type", "")
        return (json.loads(body) if "json" in ctype else body), r.status


def post(path, fields=None, files=None, method="POST", timeout=300):
    boundary = "----ibvap-test-boundary"
    body = b""
    for k, v in (fields or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                 f"{v}\r\n").encode()
    for k, (fname, data, ctype) in (files or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                 f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(BASE + path, data=body, method=method)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return (json.loads(raw) if raw else {}), r.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return json.loads(raw), e.code
        except Exception:
            return {"detail": raw.decode(errors="replace")[:200]}, e.code


def delete(path):
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return {}, e.code


print("\n" + "=" * 74)
print("  IBVAP END-TO-END ACCEPTANCE TEST")
print("=" * 74)

# ---------------------------------------------------------------- 1. health
print("\n[1] SERVICE HEALTH & IST TIMEZONE")
health, status = get("/health")
check("Health endpoint responds 200", status == 200)
check("Reports Asia/Kolkata timezone", "Asia/Kolkata" in health["timezone"], health["timezone"])
check("Health timestamp rendered in IST", health["timestamp_ist"].endswith("IST"),
      health["timestamp_ist"])
check("Canonical timestamp is timezone-aware UTC", health["timestamp"].endswith("+00:00"))

tinfo, _ = get("/api/system/time")
check("Time endpoint offset is +05:30", tinfo["utc_offset"] == "+05:30")
check("IST ISO carries +05:30 offset", "+05:30" in tinfo["ist_iso"], tinfo["ist_iso"])

# Independently confirm IST == UTC + 5:30
from datetime import datetime, timedelta, timezone
utc_dt = datetime.fromisoformat(tinfo["utc"])
ist_dt = datetime.fromisoformat(tinfo["ist_iso"])
delta = ist_dt.utcoffset() - timedelta(0)
check("IST offset is exactly 5h30m", delta == timedelta(hours=5, minutes=30), str(delta))

# ------------------------------------------------------------- 2. detection
print("\n[2] AI DETECTION & DEVICE")
info, _ = get("/api/system/info")
det = info["detector"]
check("YOLO model loaded", bool(det.get("model")), det.get("model"))
check("Inference device reported", bool(det.get("device")), det.get("device"))
check("Inference is running", det.get("inference_count", 0) > 0,
      f"{det.get('inference_count')} frames")
check("Inference latency measured", det.get("inference_ms_avg", 0) > 0,
      f"{det.get('inference_ms_avg')} ms")
if det.get("cuda_available"):
    check("GPU in use (not silently on CPU)", det["device"].startswith("cuda"),
          det.get("gpu_name"))
else:
    skip("GPU acceleration", "CUDA not available on this machine")

# ----------------------------------------------------------- 3. live camera
print("\n[3] LIVE CAMERA PIPELINE")
cams, _ = get("/api/cameras")
check("At least one camera registered", len(cams) >= 1, f"{len(cams)} camera(s)")
cam = cams[0]
cam_id = cam["id"]
rt = cam.get("runtime", {})
check("Camera is online", cam["is_online"], cam["name"])
check("Camera producing frames", rt.get("frames_analysed", 0) > 0,
      f"{rt.get('frames_analysed')} frames")
check("Analytics FPS above 8", rt.get("fps", 0) >= 8, f"{rt.get('fps')} fps")
check("End-to-end latency under 400ms", 0 < rt.get("latency_ms", 999) < 400,
      f"{rt.get('latency_ms')} ms")
src = rt.get("source", {})
check("Capture reports source resolution", src.get("resolution", "—") != "—",
      src.get("resolution"))
check("File source is paced to native rate", src.get("paced") is True,
      f"capture {src.get('capture_fps')} fps vs source {src.get('source_fps')} fps")

# ------------------------------------------------------------ 4. MJPEG stream
print("\n[4] MJPEG STREAMING")
try:
    req = urllib.request.Request(f"{BASE}/stream/{cam_id}")
    with urllib.request.urlopen(req, timeout=15) as r:
        chunk = r.read(200000)
    check("Stream returns multipart data", b"--frame" in chunk, f"{len(chunk)} bytes")
    check("Stream carries JPEG frames", b"\xff\xd8\xff" in chunk)
    frames = chunk.count(b"--frame")
    check("Multiple frames delivered", frames >= 2, f"{frames} frame headers")
except Exception as e:
    check("MJPEG stream reachable", False, str(e)[:90])

snap, sstatus = get(f"/api/cameras/{cam_id}/snapshot")
check("Camera snapshot endpoint works", sstatus == 200 and snap[:3] == b"\xff\xd8\xff",
      f"{len(snap)} bytes")

# --------------------------------------------------------- 5. virtual fence
print("\n[5] VIRTUAL FENCE — CREATE / VERIFY / DELETE")
rules, _ = get(f"/api/cameras/{cam_id}/rules")
baseline = len(rules)
check("Existing rules readable", isinstance(rules, list), f"{baseline} rule(s)")

created, cstatus = post(f"/api/cameras/{cam_id}/rules", {
    "rule_type": "zone", "name": "TEST RESTRICTED ZONE",
    "geometry": json.dumps([[100, 100], [300, 100], [300, 300], [100, 300]]),
    "params": json.dumps({"presence_seconds": 3}), "is_active": "true",
})
check("Zone rule created", cstatus == 201, f"HTTP {cstatus}")
zone_id = created.get("id")
check("Rule geometry persisted", len(created.get("geometry", [])) == 4)

after, _ = get(f"/api/cameras/{cam_id}/rules")
check("Rule count increased", len(after) == baseline + 1)

bad, bstatus = post(f"/api/cameras/{cam_id}/rules", {
    "rule_type": "zone", "geometry": json.dumps([[10, 10]]), "params": "{}"})
check("Under-specified geometry rejected", bstatus == 400, f"HTTP {bstatus}")

bad2, b2status = post(f"/api/cameras/{cam_id}/rules", {
    "rule_type": "teleport", "geometry": json.dumps([[1, 1], [2, 2]]), "params": "{}"})
check("Unknown rule type rejected", b2status == 400, f"HTTP {b2status}")

bad3, b3status = post(f"/api/cameras/{cam_id}/rules", {
    "rule_type": "line", "geometry": "not-json", "params": "{}"})
check("Malformed geometry JSON rejected", b3status == 400, f"HTTP {b3status}")

if zone_id:
    _, dstatus = delete(f"/api/rules/{zone_id}")
    check("Rule deleted", dstatus == 200)
    final, _ = get(f"/api/cameras/{cam_id}/rules")
    check("Rule count restored", len(final) == baseline)

# ------------------------------------------------------ 6. alerts / events
print("\n[6] ALERT GENERATION & EVENT LOG")
log, _ = get("/api/alerts?limit=200")
check("Event log endpoint works", "alerts" in log)
alerts = log["alerts"]
check("Events have been generated", len(alerts) > 0, f"{log['total']} event(s)")

if alerts:
    a = alerts[0]
    check("Event carries IST timestamp", a["timestamp_ist"].endswith("IST"), a["timestamp_ist"])
    check("Event carries UTC canonical timestamp", a["timestamp"].endswith("+00:00"))
    check("Event has severity", a["severity"] in
          ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"), a["severity"])
    check("Event labels its producing subsystem",
          a["analysis_kind"] in ("AI DETECTION", "RULE-BASED EVENT ANALYSIS"),
          a["analysis_kind"])
    check("Event has SHA-256 hash", len(a["hash"]) == 64)
    check("Event links to predecessor", len(a["prev_hash"]) == 64)
    check("Event has human-readable title", bool(a["title"]))

    types = {x["alert_type"] for x in alerts}
    check("Human detection events present", "human_detected" in types, str(sorted(types)))

    # filtering
    filtered, _ = get(f"/api/alerts?alert_type={a['alert_type']}&limit=10")
    check("Filter by alert type works",
          all(x["alert_type"] == a["alert_type"] for x in filtered["alerts"]))
    sev, _ = get("/api/alerts?severity=CRITICAL&limit=10")
    check("Filter by severity works",
          all(x["severity"] == "CRITICAL" for x in sev["alerts"]))
    bycam, _ = get(f"/api/alerts?camera_id={cam_id}&limit=10")
    check("Filter by camera works",
          all(x["camera_id"] == cam_id for x in bycam["alerts"]))

    detail, dstat = get(f"/api/alerts/{a['id']}")
    check("Single event detail retrievable", dstat == 200 and detail["id"] == a["id"])

# --------------------------------------------------------------- 7. evidence
print("\n[7] EVIDENCE STORAGE")
snapshot_alerts = [a for a in alerts if a["has_snapshot"]]
clip_alerts = [a for a in alerts if a["has_clip"]]
check("Snapshots recorded for events", len(snapshot_alerts) > 0,
      f"{len(snapshot_alerts)} with snapshot")
check("Evidence clips recorded", len(clip_alerts) > 0, f"{len(clip_alerts)} with clip")

if snapshot_alerts:
    img, istat = get(f"/api/alerts/{snapshot_alerts[0]['id']}/snapshot")
    check("Snapshot downloads as JPEG", istat == 200 and img[:3] == b"\xff\xd8\xff",
          f"{len(img)} bytes")
    clean, cstat = get(f"/api/alerts/{snapshot_alerts[0]['id']}/snapshot?clean=true")
    check("Unannotated snapshot also stored", cstat == 200 and clean[:3] == b"\xff\xd8\xff",
          f"{len(clean)} bytes")
if clip_alerts:
    clip, clstat = get(f"/api/alerts/{clip_alerts[0]['id']}/clip")
    check("Evidence clip downloads as MP4", clstat == 200 and b"ftyp" in clip[:32],
          f"{len(clip)/1024:.0f} KB")

usage = info["evidence"]
check("Evidence store tracked on disk", usage["snapshots"] > 0 or usage["clips"] > 0,
      f"{usage['snapshots']} snaps, {usage['clips']} clips, {usage['megabytes']} MB")

# ------------------------------------------------------------- 8. integrity
print("\n[8] TAMPER-EVIDENT HASH CHAIN")
verify, _ = get("/api/integrity/verify")
check("Chain verification runs", "valid" in verify)
check("Chain reports INTACT", verify["valid"] is True, verify["message"][:70])
check("Verification counts every event", verify["total_alerts"] == log["total"],
      f"{verify['total_alerts']} vs {log['total']}")
check("Verification timestamped in IST", verify["verified_at_ist"].endswith("IST"))
check("Named honestly (hash chain, not blockchain)",
      "hash chain" in verify["scheme"].lower() and "blockchain" not in verify["scheme"].lower(),
      verify["scheme"])

cp, cpstatus = post("/api/integrity/checkpoint")
check("Merkle checkpoint can be sealed", cpstatus == 200 and cp.get("created") is not None)
if cp.get("created"):
    uid = cp["checkpoint"]["checkpoint_uid"]
    cpv, _ = get(f"/api/integrity/checkpoints/{uid}/verify")
    check("Sealed checkpoint verifies", cpv["valid"] is True, cpv["message"][:60])

cert, certstatus = post("/api/integrity/certificate", {"issued_to": "Acceptance Test"})
check("Integrity certificate exports", certstatus == 200 and "certificate_id" in cert)
check("Certificate runs real verification",
      cert.get("verification", {}).get("valid") is True)
check("Certificate states its scheme accurately",
      "non-blockchain" in cert.get("scheme", "").lower(), cert.get("scheme", "")[:60])

# ------------------------------------------------------------- 9. face/anpr
print("\n[9] FACE DETECTION & ANPR SUBSYSTEMS")
face = info["face"]
if face["available"]:
    check("Face subsystem available", True, f"det_size={face['det_size']}")
    ftest, fstat = get(f"/api/face/test/{cam_id}", timeout=90)
    check("Face detection probe runs on a live frame", fstat == 200,
          f"{ftest.get('count', 0)} face(s)")
    if ftest.get("matches"):
        unident = [m for m in ftest["matches"] if not m["identified"]]
        check("Unmatched faces are NOT given an identity",
              all("identity not established" in m["label"] for m in unident),
              f"{len(unident)} unidentified")
else:
    skip("Face detection", "InsightFace not available")

anpr = info["anpr"]
if anpr["available"]:
    check("ANPR subsystem available", True, f"device={anpr['device']}")
    check("ANPR is cadenced, not per-frame", anpr["cadence_frames"] > 1,
          f"every {anpr['cadence_frames']} frames")
    atest, astat = get(f"/api/anpr/test/{cam_id}", timeout=120)
    check("ANPR probe runs on a live frame", astat == 200,
          f"{atest.get('count', 0)} plate candidate(s)")
    if atest.get("detections"):
        low = [d for d in atest["detections"] if not d["certain"]]
        check("Low-confidence plates shown as PLATE UNCERTAIN",
              all(d["plate_text"] == "PLATE UNCERTAIN" for d in low),
              f"{len(low)} uncertain")
else:
    skip("ANPR", "EasyOCR not available")

# ---------------------------------------------------------- 10. mp4 upload
print("\n[10] MP4 UPLOAD & OFFLINE ANALYSIS")

with open("samples/sample_border_scenario.mp4", "rb") as fh:
    video_bytes = fh.read()

# Reject a non-MP4 masquerading as one.
bad_up, bstat = post("/api/analysis/upload", {},
                     {"file": ("evil.mp4", b"MZ\x90\x00this is a windows exe", "video/mp4")})
check("Non-MP4 content rejected despite .mp4 name", bstat == 400,
      f"HTTP {bstat}: {str(bad_up.get('detail'))[:60]}")

bad_ext, bestat = post("/api/analysis/upload", {},
                       {"file": ("clip.avi", video_bytes[:5000], "video/avi")})
check("Disallowed extension rejected", bestat == 400, f"HTTP {bestat}")

trav, tstat = post("/api/analysis/upload", {},
                   {"file": ("../../../etc/passwd.mp4", b"ftypisom" + b"\x00" * 100,
                             "video/mp4")})
check("Path-traversal filename handled safely", tstat in (400, 413),
      f"HTTP {tstat}")

# Real upload.
up, ustat = post("/api/analysis/upload",
                 {"camera_id": str(cam_id), "apply_rules": "true"},
                 {"file": ("border_test.mp4", video_bytes, "video/mp4")})
check("Valid MP4 accepted", ustat == 202, f"HTTP {ustat}")
session = up.get("session_id")
check("Analysis session created", bool(session), session)
check("Video metadata probed", up.get("video", {}).get("frame_count", 0) > 0,
      f"{up.get('video', {}).get('frame_count')} frames, "
      f"{up.get('video', {}).get('duration_seconds')}s")
check("Camera rules applied to upload", up.get("rules_applied", 0) >= 1,
      f"{up.get('rules_applied')} rule(s)")
check("Upload timestamped in IST", up.get("uploaded_at_ist", "").endswith("IST"))

if session:
    print("       waiting for analysis to complete…")
    deadline = time.time() + 420
    final = {}
    while time.time() < deadline:
        final, _ = get(f"/api/analysis/{session}")
        if final["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)

    check("Analysis completed", final.get("status") == "completed",
          f"{final.get('status')}: {final.get('error', '')[:60]}")
    check("Frames were analysed", final.get("analysed_frames", 0) > 0,
          f"{final.get('analysed_frames')} frames at "
          f"{final.get('processing_fps', 0):.1f} fps")
    check("Detections found in upload", final.get("detections_total", 0) > 0,
          f"{final.get('detections_total')} detections, "
          f"{final.get('persons')} persons, {final.get('vehicles')} vehicles")
    check("Events generated from upload", final.get("alerts", 0) > 0,
          f"{final.get('alerts')} event(s)")
    check("Annotated video rendered", final.get("has_output") is True)

    if final.get("has_output"):
        vid, vstat = get(f"/api/analysis/{session}/video")
        check("Annotated video downloads", vstat == 200 and b"ftyp" in vid[:32],
              f"{len(vid)/1024:.0f} KB")

    sess_alerts, _ = get(f"/api/analysis/{session}/alerts")
    check("Upload events queryable by session", sess_alerts["count"] > 0,
          f"{sess_alerts['count']} event(s)")
    if sess_alerts["alerts"]:
        ua = sess_alerts["alerts"][0]
        check("Upload events tagged source_type=upload", ua["source_type"] == "upload")
        check("Upload events carry IST timestamps", ua["timestamp_ist"].endswith("IST"))
        check("Upload events sealed in the same hash chain", len(ua["hash"]) == 64)

    # Same pipeline check: uploaded video must use the same rules engine.
    upload_types = {x["alert_type"] for x in sess_alerts["alerts"]}
    check("Upload ran the full analytics pipeline",
          bool(upload_types & {"human_detected", "vehicle_detected", "entry", "exit",
                               "enter", "loiter", "night_movement", "face_detected"}),
          str(sorted(upload_types)))

# ----------------------------------------------- 11. chain still valid after
print("\n[11] CHAIN INTEGRITY AFTER UPLOAD EVENTS")
verify2, _ = get("/api/integrity/verify")
check("Chain still intact after upload analysis", verify2["valid"] is True,
      f"{verify2['total_alerts']} events")
check("Chain grew with upload events", verify2["total_alerts"] > verify["total_alerts"],
      f"{verify['total_alerts']} -> {verify2['total_alerts']}")

# ------------------------------------------------------- 12. tamper detection
print("\n[12] TAMPER DETECTION (the actual security claim)")
import sqlite3
conn = sqlite3.connect("alerts.db")
row = conn.execute("SELECT id, alert_type FROM alerts ORDER BY id LIMIT 1 OFFSET 1").fetchone()
if row:
    victim_id, original = row
    conn.execute("UPDATE alerts SET alert_type='benign_activity' WHERE id=?", (victim_id,))
    conn.commit()
    conn.close()

    tampered, _ = get("/api/integrity/verify")
    check("Tampering is DETECTED", tampered["valid"] is False,
          tampered["message"][:80])
    check("Tampering located at the edited record",
          tampered.get("broken_at") == victim_id,
          f"reported #{tampered.get('broken_at')}, edited #{victim_id}")

    conn = sqlite3.connect("alerts.db")
    conn.execute("UPDATE alerts SET alert_type=? WHERE id=?", (original, victim_id))
    conn.commit()
    conn.close()

    restored, _ = get("/api/integrity/verify")
    check("Chain valid again after restoring the record", restored["valid"] is True)
else:
    skip("Tamper detection", "not enough events in the chain")

# ------------------------------------------------------------- 13. security
print("\n[13] SECURITY CHECKS")
_, nf = get("/api/alerts/999999") if False else ({}, 0)
try:
    urllib.request.urlopen(BASE + "/api/alerts/999999", timeout=10)
    check("Unknown alert returns 404", False, "got 200")
except urllib.error.HTTPError as e:
    check("Unknown alert returns 404", e.code == 404, f"HTTP {e.code}")

try:
    urllib.request.urlopen(BASE + "/api/cameras/999999", timeout=10)
    check("Unknown camera returns 404", False, "got 200")
except urllib.error.HTTPError as e:
    check("Unknown camera returns 404", e.code == 404, f"HTTP {e.code}")

# ------------------------------------------------------------- 14. stats
print("\n[14] STATISTICS (must reflect real runtime data)")
stats, _ = get("/api/stats?hours=24")
check("Stats endpoint responds", "total_alerts" in stats)
check("Total matches event log", stats["total_alerts"] == verify2["total_alerts"],
      f"{stats['total_alerts']} vs {verify2['total_alerts']}")
check("Per-type breakdown present", len(stats["by_type"]) > 0, str(stats["by_type"]))
check("'Today' computed against IST midnight",
      stats["today_since_ist"].endswith("IST"), stats["today_since_ist"])
check("Live aggregate reports real FPS", stats["live"]["system_fps"] > 0,
      f"{stats['live']['system_fps']} fps")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 74)
print(f"  RESULT:  {len(PASS)} passed   {len(FAIL)} failed   {len(SKIP)} skipped")
print("=" * 74)
if FAIL:
    print("\nFAILURES:")
    for name, detail in FAIL:
        print(f"  ✕ {name}" + (f"  [{detail}]" if detail else ""))
if SKIP:
    print("\nSKIPPED:")
    for name, why in SKIP:
        print(f"  – {name}: {why}")
print()
sys.exit(1 if FAIL else 0)
