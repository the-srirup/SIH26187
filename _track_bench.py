"""
Controlled tracking-stability measurement.

Runs the same footage through the detector+tracker under different settings and
reports the metrics that matter for "detection is unpredictable":

* detections/frame            -- recall
* distinct track ids          -- lower is better for the same scene content
* ids per detection           -- the churn ratio; 1.0 would mean every
                                 detection got its own id (worst case)
* mean track lifetime         -- how long an identity survives
* fragmentation               -- tracks that lived < 5 frames (ghosts)

Also simulates FAST motion by sampling every Nth frame, which multiplies
inter-frame displacement exactly as a fast vehicle does.
"""
import argparse
import statistics
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, ".")

from core.config import settings  # noqa: E402


def run(video, stride, conf, new_track, match, imgsz, limit=400):
    # Apply the settings under test before constructing anything.
    settings.DEFAULT_CONFIDENCE = conf
    settings.NEW_TRACK_THRESH = new_track
    settings.MATCH_THRESH = match
    settings.INFERENCE_IMGSZ = imgsz

    from cv.detector import Detector, ObjectTracker

    det = Detector.get()
    tracker = ObjectTracker(frame_rate=max(1, int(12 / stride)))

    cap = cv2.VideoCapture(video)
    lifetimes = defaultdict(int)
    total_dets = 0
    frames = 0
    index = 0
    while frames < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride:
            index += 1
            continue
        index += 1
        frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        xyxy, c, cls, _ = det.raw_detect(frame)
        dets = tracker.update(xyxy, c, cls, det.names)
        frames += 1
        total_dets += len(dets)
        for d in dets:
            lifetimes[d.track_id] += 1
    cap.release()

    ids = len(lifetimes)
    lives = list(lifetimes.values()) or [0]
    return {
        "frames": frames,
        "dets_per_frame": round(total_dets / max(1, frames), 2),
        "ids": ids,
        "ids_per_det": round(ids / max(1, total_dets), 3),
        "mean_lifetime": round(statistics.mean(lives), 1),
        "ghosts_lt5": sum(1 for v in lives if v < 5),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="samples/sample_border_scenario.mp4")
    ap.add_argument("--limit", type=int, default=350)
    args = ap.parse_args()

    configs = [
        ("OLD  conf.30 new.50 match.85", 0.30, 0.50, 0.85, 640),
        ("NEW  conf.30 new.32 match.92", 0.30, 0.32, 0.92, 640),
    ]
    for stride, label in ((1, "normal motion (every frame)"),
                          (3, "FAST motion (every 3rd frame)")):
        print(f"\n=== {label} ===")
        print(f"{'config':32s} {'d/frm':>6} {'ids':>5} {'ids/det':>8} "
              f"{'lifetime':>9} {'ghosts':>7}")
        for name, conf, nt, mt, sz in configs:
            r = run(args.video, stride, conf, nt, mt, sz, args.limit)
            print(f"{name:32s} {r['dets_per_frame']:>6} {r['ids']:>5} "
                  f"{r['ids_per_det']:>8} {r['mean_lifetime']:>9} {r['ghosts_lt5']:>7}")
