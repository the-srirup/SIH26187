#!/usr/bin/env python
"""
Capability audit against the official problem statement.

Runs the real pipeline over real footage and reports, per PS capability,
whether it actually produced output — not whether the code for it exists.
Every number printed is measured in this run.

    python verify_capabilities.py                     # all clips, all capabilities
    python verify_capabilities.py --clip samples/x.mp4
    python verify_capabilities.py --webcam            # also prove face detection
    python verify_capabilities.py --seconds 30 --json report.json

Recorded footage cannot prove every capability: a border clip has no face
close enough to detect and no night frames, so those come back as warnings
about the *footage*, not about the platform. ``--webcam`` closes that gap by
running the same pipeline against the local camera, where a face is present
and at a usable size.

The PS capabilities, verbatim:

    * Human detection and tracking
    * Vehicle detection and classification
    * Face detection
    * Automatic Number Plate Recognition (ANPR)
    * Virtual fence intrusion detection
    * Suspicious activity detection
    * Night-time movement detection
    * Real-time alert generation and event logging
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parent))

TICK, CROSS, WARN = "PASS", "FAIL", "WARN"


def _fmt(status: str) -> str:
    return {"PASS": "  [PASS]", "FAIL": "  [FAIL]", "WARN": "  [WARN]"}[status]


class Audit:
    """Collects one verdict per capability, with the evidence behind it."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, capability: str, status: str, evidence: str) -> None:
        self.rows.append({"capability": capability, "status": status,
                          "evidence": evidence})
        print(f"{_fmt(status)} {capability:<44} {evidence}")

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    @property
    def warned(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "WARN"]


def run_clip(path: Path, seconds: float, audit: Audit) -> dict:
    """Drive the real FrameAnalyzer over a clip and tally what it produced."""
    import cv2
    import numpy as np

    from core.analytics import FrameAnalyzer
    from core.config import settings
    from cv.detector import Detector

    print(f"\n--- {path.name} ---")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        audit.record(f"open {path.name}", CROSS, "could not decode")
        return {}

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    analyzer = FrameAnalyzer(source_id=f"audit-{path.stem[:8]}",
                             display_name=path.stem[:18].upper(),
                             detector=Detector.get())

    # Rules that exercise every behavioural capability at once: a tripwire
    # across the middle, a zone over the lower half, a loiter area, and a
    # one-way direction rule.
    w, h = settings.FRAME_WIDTH, settings.FRAME_HEIGHT
    analyzer.set_rules([
        {"id": 1, "rule_type": "line", "name": "AUDIT-WIRE",
         "geometry": [[0, h // 2], [w, h // 2]], "params": {}},
        {"id": 2, "rule_type": "zone", "name": "AUDIT-ZONE",
         "geometry": [[0, h // 2], [w, h // 2], [w, h], [0, h]],
         "params": {"presence_seconds": 3}},
        {"id": 3, "rule_type": "loiter", "name": "AUDIT-LOITER",
         "geometry": [[0, 0], [w, 0], [w, h], [0, h]],
         "params": {"dwell_seconds": 4}},
        {"id": 4, "rule_type": "direction", "name": "AUDIT-ONEWAY",
         "geometry": [[w // 2, 0], [w // 2, h]], "params": {}},
    ])

    stats = {
        "clip": path.name, "resolution": f"{width}x{height}",
        "source_fps": round(float(source_fps), 1),
        "frames": 0, "persons": 0, "vehicles": 0,
        "vehicle_classes": Counter(), "track_ids": set(),
        "person_track_ids": set(), "faces": 0, "plates": 0,
        "plate_texts": Counter(), "alerts": Counter(),
        "night_frames": 0, "inference_ms": [], "total_ms": [],
        "zone_occupied_frames": 0,
    }

    budget = seconds * float(source_fps)
    started = time.perf_counter()
    frame_no = 0
    while frame_no < budget:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        media_time = frame_no / float(source_fps)
        try:
            result = analyzer.analyse(frame, timestamp=media_time)
        except Exception as exc:
            audit.record(f"pipeline on {path.name}", CROSS, f"raised {exc!r}")
            cap.release()
            return stats

        stats["frames"] += 1
        stats["persons"] = max(stats["persons"], result.person_count)
        stats["vehicles"] = max(stats["vehicles"], result.vehicle_count)
        stats["inference_ms"].append(result.inference_ms)
        stats["total_ms"].append(result.total_ms)
        if result.is_night:
            stats["night_frames"] += 1
        for det in result.detections:
            stats["track_ids"].add(det.track_id)
            if det.is_person:
                stats["person_track_ids"].add(det.track_id)
            else:
                stats["vehicle_classes"][det.class_name] += 1
        stats["faces"] += len(result.faces)
        for plate in result.plates:
            stats["plates"] += 1
            text = str(getattr(plate, "plate_text", "") or "").strip()
            if text:
                stats["plate_texts"][text] += 1
        for alert in result.alerts:
            stats["alerts"][alert.alert_type] += 1
        if any(z.get("occupied") for z in result.zones):
            stats["zone_occupied_frames"] += 1

    cap.release()
    stats["wall_seconds"] = round(time.perf_counter() - started, 1)
    stats["processing_fps"] = round(
        stats["frames"] / max(0.001, stats["wall_seconds"]), 1)
    return stats


def summarise(all_stats: list[dict], audit: Audit) -> None:
    """Turn per-clip tallies into one verdict per PS capability."""
    import statistics

    merged = {
        "frames": sum(s.get("frames", 0) for s in all_stats),
        "persons": sum(len(s.get("person_track_ids", ())) for s in all_stats),
        "tracks": sum(len(s.get("track_ids", ())) for s in all_stats),
        "faces": sum(s.get("faces", 0) for s in all_stats),
        "plates": sum(s.get("plates", 0) for s in all_stats),
        "night": sum(s.get("night_frames", 0) for s in all_stats),
        "occupied": sum(s.get("zone_occupied_frames", 0) for s in all_stats),
    }
    vehicle_classes = Counter()
    plate_texts = Counter()
    alerts = Counter()
    inference, total = [], []
    for s in all_stats:
        vehicle_classes.update(s.get("vehicle_classes", {}))
        plate_texts.update(s.get("plate_texts", {}))
        alerts.update(s.get("alerts", {}))
        inference.extend(s.get("inference_ms", []))
        total.extend(s.get("total_ms", []))

    print("\n" + "=" * 78)
    print("PS CAPABILITY AUDIT")
    print("=" * 78)

    audit.record(
        "Human detection and tracking",
        TICK if merged["persons"] else CROSS,
        f"{merged['persons']} distinct person track(s) across {merged['frames']} frames",
    )
    audit.record(
        "Vehicle detection and classification",
        TICK if len(vehicle_classes) >= 2 else (WARN if vehicle_classes else CROSS),
        (f"{sum(vehicle_classes.values())} detections across "
         f"{len(vehicle_classes)} class(es): "
         f"{', '.join(f'{k}x{v}' for k, v in vehicle_classes.most_common(6)) or 'none'}"),
    )
    audit.record(
        "Face detection",
        TICK if merged["faces"] else WARN,
        f"{merged['faces']} face observation(s)"
        + ("" if merged["faces"] else " — no face large enough in this footage"),
    )
    audit_anpr(plate_texts, merged["plates"], audit)
    crossings = alerts["entry"] + alerts["exit"]
    audit.record(
        "Virtual fence intrusion detection",
        TICK if crossings else CROSS,
        f"{crossings} crossing event(s) (entry={alerts['entry']}, exit={alerts['exit']})",
    )
    suspicious = (alerts["loiter"] + alerts["wrong_direction"]
                  + alerts["zone_presence"] + alerts["enter"])
    audit.record(
        "Suspicious activity detection",
        TICK if suspicious else CROSS,
        (f"{suspicious} event(s): loiter={alerts['loiter']}, "
         f"wrong_direction={alerts['wrong_direction']}, "
         f"zone_entry={alerts['enter']}, zone_presence={alerts['zone_presence']}"),
    )
    audit.record(
        "Night-time movement detection",
        TICK if (merged["night"] or alerts["night_movement"]) else WARN,
        (f"{merged['night']} frame(s) judged night, "
         f"{alerts['night_movement']} night-movement event(s)")
        + ("" if merged["night"] else " — all footage is daylight"),
    )
    audit.record(
        "Real-time alert generation",
        TICK if sum(alerts.values()) else CROSS,
        f"{sum(alerts.values())} rule event(s): "
        + ", ".join(f"{k}={v}" for k, v in alerts.most_common(8)),
    )
    audit.record(
        "Continuous zone occupancy signal",
        TICK if merged["occupied"] else WARN,
        f"occupied on {merged['occupied']} frame(s)",
    )

    if inference:
        inference.sort()
        total.sort()
        audit.record(
            "Real-time performance",
            TICK if statistics.median(total) < 100 else WARN,
            (f"inference median {statistics.median(inference):.1f} ms, "
             f"frame median {statistics.median(total):.1f} ms, "
             f"p95 {total[int(len(total) * 0.95)]:.1f} ms"),
        )


def audit_anpr(plate_texts, published: int, audit: Audit) -> None:
    """
    ANPR, judged on what the stage actually did — not only on what it output.

    A clip whose plates are genuinely illegible is indistinguishable, from the
    outside, from a reader that is broken. The two have completely different
    fixes, so they are reported differently: the stage's own counters say
    whether it searched, whether it localised anything, and whether OCR ran,
    and a synthetic plate of known text then proves the full path end to end
    regardless of what the footage happens to contain.
    """
    from cv.anpr import get_anpr_processor

    metrics = get_anpr_processor().get_metrics()
    searched = metrics.get("frames_searched", 0)
    candidates = metrics.get("candidates_found", 0)
    ocr_calls = metrics.get("ocr_call_count", 0)
    detail = (f"searched {searched} frame(s), {candidates} candidate(s), "
              f"{ocr_calls} OCR call(s)")

    if plate_texts:
        audit.record("ANPR on supplied footage", TICK,
                     f"{published} read(s): "
                     + ", ".join(f"{k}x{v}" for k, v in plate_texts.most_common(5)))
    elif not metrics.get("available"):
        audit.record("ANPR on supplied footage", CROSS,
                     "EasyOCR unavailable — ANPR cannot run at all")
    elif searched == 0:
        audit.record("ANPR on supplied footage", CROSS,
                     "the ANPR stage never ran on any frame")
    elif candidates == 0:
        audit.record("ANPR on supplied footage", CROSS,
                     f"{detail} — localisation found nothing on any vehicle")
    else:
        audit.record("ANPR on supplied footage", WARN,
                     f"{detail}, no legible plate in this footage")

    audit_anpr_synthetic(audit)


def audit_anpr_synthetic(audit: Audit) -> None:
    """Render a plate of known text and require the pipeline to read it back."""
    import cv2
    import numpy as np

    from core.config import settings
    from cv.anpr import get_anpr_processor

    processor = get_anpr_processor()
    if not processor.is_available():
        audit.record("ANPR on a known plate", CROSS, "EasyOCR unavailable")
        return

    expected = "MH12DE1433"
    frame_w, frame_h = 1920, 1080
    scene = np.full((frame_h, frame_w, 3), 118, np.uint8)
    cv2.rectangle(scene, (0, int(frame_h * 0.55)), (frame_w, frame_h), (92, 92, 96), -1)
    car_w = int(frame_w * 0.30)
    car_h = int(car_w * 0.78)
    car_x, car_y = (frame_w - car_w) // 2, int(frame_h * 0.42)
    cv2.rectangle(scene, (car_x, car_y), (car_x + car_w, car_y + car_h), (48, 52, 60), -1)

    plate_w, plate_h = 260, 60
    plate = np.full((plate_h, plate_w, 3), 245, np.uint8)
    cv2.rectangle(plate, (2, 2), (plate_w - 3, plate_h - 3), (20, 20, 20), 2)
    cv2.putText(plate, expected, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                (15, 15, 15), 3, cv2.LINE_AA)
    px = car_x + (car_w - plate_w) // 2
    py = car_y + car_h - plate_h - int(car_h * 0.10)
    scene[py:py + plate_h, px:px + plate_w] = plate

    class _Vehicle:
        bbox = (int(car_x * settings.FRAME_WIDTH / frame_w),
                int(car_y * settings.FRAME_HEIGHT / frame_h),
                int((car_x + car_w) * settings.FRAME_WIDTH / frame_w),
                int((car_y + car_h) * settings.FRAME_HEIGHT / frame_h))
        class_name = "car"
        track_id = 9001

    small = cv2.resize(scene, settings.frame_size, interpolation=cv2.INTER_AREA)
    reads = []
    for n in range(1, 8):
        for plate_det in processor.recognize_plates(
            small, vehicle_detections=[_Vehicle()], frame_number=n * 20,
            source_frame=scene, source_id="audit-synthetic",
        ):
            if plate_det.plate_text:
                reads.append(plate_det.plate_text.replace(" ", "").upper())

    if not reads:
        audit.record("ANPR on a known plate", CROSS,
                     f"rendered {expected} at {plate_w}x{plate_h} px — read nothing")
        return

    def similarity(a: str, b: str) -> float:
        return sum(1 for x, y in zip(a, b) if x == y) / float(max(len(a), len(b)))

    best = max(reads, key=lambda r: similarity(r, expected))
    score = similarity(best, expected)
    audit.record(
        "ANPR on a known plate",
        TICK if score >= 0.6 else WARN,
        f"rendered {expected} -> read {best} ({score:.0%} character match, "
        f"{len(set(reads))} distinct read(s))",
    )


def audit_webcam(seconds: float, audit: Audit) -> None:
    """
    Prove face detection and live capture against the local camera.

    Recorded border footage has no face at a detectable size, so the clip-based
    audit can only report that — it cannot distinguish "this footage has no
    close face" from "face detection is broken". A live camera does have one.
    """
    import cv2

    from core.analytics import FrameAnalyzer
    from cv.detector import Detector
    from cv.face import get_face_recognizer
    from core.video_source import open_capture

    recognizer = get_face_recognizer()
    if not getattr(recognizer, "_enabled", False):
        audit.record("Face detection (live camera)", CROSS,
                     "the face recogniser failed to initialise")
        return

    cap = open_capture("0")
    if cap is None:
        audit.record("Face detection (live camera)", WARN,
                     "no webcam available on this host")
        return

    analyzer = FrameAnalyzer(source_id="audit-webcam", display_name="WEBCAM",
                             detector=Detector.get(), enable_anpr=False)
    faces, frames, persons = 0, 0, 0
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            result = analyzer.analyse(frame, timestamp=time.time())
            persons = max(persons, result.person_count)
            faces += len(result.faces)
    finally:
        cap.release()

    audit.record(
        "Live capture (webcam)",
        TICK if frames else CROSS,
        f"{frames} frame(s) captured and analysed in {seconds:.0f}s",
    )
    if not persons:
        audit.record("Face detection (live camera)", WARN,
                     f"{frames} frame(s) analysed but nobody was in view")
        return
    audit.record(
        "Face detection (live camera)",
        TICK if faces else CROSS,
        f"{faces} face observation(s) with up to {persons} person(s) in view",
    )


def audit_event_log(audit: Audit) -> None:
    """The log is the other half of 'alert generation and event logging'."""
    from core.database import SessionLocal
    from core.hashchain import verify_chain
    from core.models import Alert

    db = SessionLocal()
    try:
        total = db.query(Alert).count()
        result = verify_chain(db)
        audit.record(
            "Event logging (tamper-evident)",
            TICK if result.valid else CROSS,
            f"{total} sealed event(s), SHA-256 chain "
            + ("verified" if result.valid else f"BROKEN: {result.message}"),
        )
    finally:
        db.close()


def audit_integration_surface(audit: Audit) -> None:
    """'Support integration with existing command and control systems.'"""
    from api.main import app

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    required = {
        "/api/alerts": "event pull",
        "/ws/alerts": "event push",
        "/stream/{camera_id}": "video relay",
        "/health": "liveness",
        "/api/integrity/verify": "evidence proof",
    }
    missing = [name for path, name in required.items() if path not in paths]
    audit.record(
        "C2 integration surface",
        TICK if not missing else WARN,
        f"{len(required) - len(missing)}/{len(required)} interfaces present"
        + (f" — missing: {', '.join(missing)}" if missing else ""),
    )

    from core.config import settings

    channels = []
    if settings.ALARM_ENABLED:
        channels.append("alarm/webhook")
    if settings.SMS_ENABLED:
        channels.append("SMS")
    audit.record(
        "Outbound escalation channels",
        TICK if channels else WARN,
        ", ".join(channels) if channels
        else "none enabled (ALARM_ENABLED / SMS_ENABLED are off)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", default=[],
                        help="footage to audit (repeatable); defaults to samples/ and videos/")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="seconds of each clip to analyse")
    parser.add_argument("--webcam", action="store_true",
                        help="also audit face detection against the local camera")
    parser.add_argument("--json", help="write the full report to this path")
    args = parser.parse_args()

    clips = [Path(c) for c in args.clip]
    if not clips:
        for folder in (Path("samples"), Path("videos")):
            if folder.is_dir():
                clips.extend(sorted(p for p in folder.glob("*.mp4") if p.is_file()))
    clips = [c for c in clips if c.is_file()]
    if not clips:
        print("No footage found. Pass --clip <file>.")
        return 1

    print("IBVAP capability audit — measured against the official problem statement")
    print(f"footage: {len(clips)} clip(s), {args.seconds:.0f}s each\n")

    audit = Audit()
    all_stats = []
    for clip in clips:
        stats = run_clip(clip, args.seconds, audit)
        if stats:
            all_stats.append(stats)
            print(f"    {stats['resolution']} @ {stats['source_fps']} fps · "
                  f"{stats['frames']} frames in {stats['wall_seconds']}s "
                  f"({stats['processing_fps']} fps) · "
                  f"{len(stats['track_ids'])} track(s) · "
                  f"{stats['faces']} face(s) · {stats['plates']} plate(s) · "
                  f"{sum(stats['alerts'].values())} event(s)")

    summarise(all_stats, audit)
    if args.webcam:
        print()
        audit_webcam(min(20.0, args.seconds), audit)
    audit_event_log(audit)
    audit_integration_surface(audit)

    print("\n" + "=" * 78)
    passed = sum(1 for r in audit.rows if r["status"] == TICK)
    print(f"{passed}/{len(audit.rows)} capabilities verified"
          f"   failures: {len(audit.failed)}   warnings: {len(audit.warned)}")
    for row in audit.failed:
        print(f"  FAIL  {row['capability']}: {row['evidence']}")
    for row in audit.warned:
        print(f"  WARN  {row['capability']}: {row['evidence']}")
    print("=" * 78)

    if args.json:
        payload = {
            "capabilities": audit.rows,
            "clips": [
                {k: (dict(v) if isinstance(v, Counter)
                     else sorted(v) if isinstance(v, set) else v)
                 for k, v in s.items() if k not in ("inference_ms", "total_ms")}
                for s in all_stats
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report written to {args.json}")

    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
