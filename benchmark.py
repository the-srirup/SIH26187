#!/usr/bin/env python
"""
IBVAP performance benchmark.

Measures the *real* analytics pipeline — capture, preprocess, YOLO inference,
ByteTrack, rules engine, overlay, JPEG encode — at 1, 2 and 4 concurrent
camera streams, and reports hardware, CPU, RAM and GPU utilisation alongside.

Every number the README quotes comes from this script. Run it yourself:

    python benchmark.py                     # 1, 2, 4 cameras, 30 s each
    python benchmark.py --seconds 60        # longer sample
    python benchmark.py --streams 1 2 4 8   # custom stream counts
    python benchmark.py --device cpu        # force CPU to compare

The benchmark drives sources at full decode speed rather than the source's
native frame rate, so it measures the pipeline's ceiling rather than the
sample clip's 12 fps.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.analytics import FrameAnalyzer  # noqa: E402
from core.config import settings  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def hardware_report() -> dict:
    """Collect the machine facts the results must be read against."""
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "cpu": platform.processor() or "unknown",
    }
    if psutil:
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def gpu_utilisation() -> float | None:
    """GPU busy percentage via nvidia-smi, or None when unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


class StreamWorker(threading.Thread):
    """One simulated camera: decodes a clip on loop and runs full analytics."""

    def __init__(self, index: int, video: Path, duration: float, rules: list):
        super().__init__(daemon=True, name=f"bench-cam{index}")
        self.index = index
        self.video = video
        self.duration = duration
        self.rules = rules
        self.frame_times: list[float] = []
        self.inference_times: list[float] = []
        self.latencies: list[float] = []
        self.detections = 0
        self.events = 0
        self.error: str | None = None

    def run(self) -> None:
        try:
            analyzer = FrameAnalyzer(
                source_id=f"bench{self.index}",
                display_name=f"BENCH CAM {self.index:02d}",
                enable_face=False,        # measured separately; cadenced in production
                enable_anpr=False,
            )
            analyzer.set_rules(self.rules)

            cap = cv2.VideoCapture(str(self.video))
            if not cap.isOpened():
                self.error = f"cannot open {self.video}"
                return

            deadline = time.perf_counter() + self.duration
            while time.perf_counter() < deadline:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                captured = time.perf_counter()
                result = analyzer.analyse(frame, timestamp=time.time(), annotate=True)
                # Include the JPEG encode: it is part of what a viewer waits for.
                cv2.imencode(".jpg", result.frame,
                             [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
                done = time.perf_counter()

                self.frame_times.append((done - captured) * 1000)
                self.inference_times.append(result.inference_ms)
                self.latencies.append((done - captured) * 1000)
                self.detections += len(result.detections)
                self.events += len(result.alerts)
            cap.release()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def run_case(stream_count: int, video: Path, duration: float, rules: list) -> dict:
    """Run one benchmark case and return measured results."""
    if psutil:
        psutil.cpu_percent(interval=None)          # prime the sampler
        process = psutil.Process()
        rss_before = process.memory_info().rss

    workers = [StreamWorker(i, video, duration, rules) for i in range(stream_count)]
    gpu_samples: list[float] = []
    cpu_samples: list[float] = []

    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            if psutil:
                cpu_samples.append(psutil.cpu_percent(interval=0.5))
            else:
                time.sleep(0.5)
            util = gpu_utilisation()
            if util is not None:
                gpu_samples.append(util)

    monitor = threading.Thread(target=sampler, daemon=True)
    started = time.perf_counter()
    monitor.start()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    stop.set()
    monitor.join(timeout=2)
    wall = time.perf_counter() - started

    errors = [w.error for w in workers if w.error]
    all_frames = [t for w in workers for t in w.frame_times]
    all_inference = [t for w in workers for t in w.inference_times]

    per_stream_fps = [
        len(w.frame_times) / wall if wall else 0.0 for w in workers
    ]
    total_frames = sum(len(w.frame_times) for w in workers)

    result = {
        "streams": stream_count,
        "duration_s": round(wall, 1),
        "frames_total": total_frames,
        "fps_per_stream": round(statistics.mean(per_stream_fps), 1) if per_stream_fps else 0.0,
        "fps_aggregate": round(total_frames / wall, 1) if wall else 0.0,
        "inference_ms_median": round(statistics.median(all_inference), 2) if all_inference else 0.0,
        "pipeline_ms_median": round(statistics.median(all_frames), 2) if all_frames else 0.0,
        "pipeline_ms_p95": round(sorted(all_frames)[int(len(all_frames) * 0.95)], 2)
                           if all_frames else 0.0,
        "detections": sum(w.detections for w in workers),
        "events": sum(w.events for w in workers),
        "cpu_percent_mean": round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
        "cpu_percent_max": round(max(cpu_samples), 1) if cpu_samples else None,
        "gpu_percent_mean": round(statistics.mean(gpu_samples), 1) if gpu_samples else None,
        "gpu_percent_max": round(max(gpu_samples), 1) if gpu_samples else None,
        "errors": errors,
    }
    if psutil:
        result["process_rss_mb"] = round(process.memory_info().rss / (1024 ** 2), 1)
        result["process_rss_growth_mb"] = round(
            (process.memory_info().rss - rss_before) / (1024 ** 2), 1
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="IBVAP performance benchmark")
    parser.add_argument("--video", default="samples/sample_border_scenario.mp4")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="measurement window per case")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", default=None, help="override DEVICE (cpu / cuda:0)")
    parser.add_argument("--imgsz", type=int, default=None, help="override INFERENCE_IMGSZ")
    parser.add_argument("--json", default="", help="write results to this JSON file")
    args = parser.parse_args()

    if args.device:
        settings.DEVICE = args.device
    if args.imgsz:
        settings.INFERENCE_IMGSZ = args.imgsz

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        return 1

    print("=" * 78)
    print("  IBVAP PERFORMANCE BENCHMARK")
    print("=" * 78)

    hw = hardware_report()
    print("\nHARDWARE")
    for key, value in hw.items():
        print(f"  {key:<22} {value}")

    from cv.detector import Detector

    detector = Detector.get()
    metrics = detector.metrics()
    print("\nMODEL")
    print(f"  {'model':<22} {metrics['model']}")
    print(f"  {'inference device':<22} {metrics['device']}")
    print(f"  {'half precision':<22} {metrics['half']}")
    print(f"  {'inference resolution':<22} {metrics['imgsz']}px")
    print(f"  {'analytics resolution':<22} "
          f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
    print(f"  {'confidence threshold':<22} {settings.DEFAULT_CONFIDENCE}")

    # A tripwire so the rules engine is exercised, not bypassed.
    rules = [{
        "id": 1, "rule_type": "line", "name": "BENCH TRIPWIRE",
        "geometry": [[20, settings.FRAME_HEIGHT // 2],
                     [settings.FRAME_WIDTH - 20, settings.FRAME_HEIGHT // 2]],
        "params": {},
    }]

    print(f"\nMeasuring {args.seconds:.0f}s per case "
          f"(sources driven at full decode speed, not native fps)\n")

    results = []
    for count in args.streams:
        print(f"  running {count} stream(s)…", end=" ", flush=True)
        case = run_case(count, video, args.seconds, rules)
        results.append(case)
        print(f"{case['fps_per_stream']:.1f} fps/stream, "
              f"{case['fps_aggregate']:.1f} fps aggregate")
        if case["errors"]:
            print(f"    errors: {case['errors']}")
        time.sleep(2)                              # let clocks settle between cases

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    header = (f"{'Streams':>8} {'FPS/stream':>11} {'FPS total':>10} "
              f"{'Infer ms':>9} {'Pipe ms':>8} {'p95 ms':>8} "
              f"{'CPU %':>7} {'GPU %':>7} {'RSS MB':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['streams']:>8} {r['fps_per_stream']:>11.1f} {r['fps_aggregate']:>10.1f} "
              f"{r['inference_ms_median']:>9.2f} {r['pipeline_ms_median']:>8.2f} "
              f"{r['pipeline_ms_p95']:>8.2f} "
              f"{str(r['cpu_percent_mean'] or '—'):>7} "
              f"{str(r['gpu_percent_mean'] or '—'):>7} "
              f"{str(r.get('process_rss_mb', '—')):>8}")

    print("\nNOTES")
    print("  FPS/stream    analytics frames per second, per camera")
    print("  Infer ms      YOLO forward pass only")
    print("  Pipe ms       full frame cost: preprocess + infer + track + rules")
    print("                + overlay + JPEG encode")
    print("  A live camera is additionally capped by its own frame rate; these")
    print("  figures are the pipeline ceiling, measured with the source")
    print("  decoding as fast as it can.")

    if args.json:
        payload = {"hardware": hw, "model": metrics, "cases": results,
                   "settings": {"imgsz": settings.INFERENCE_IMGSZ,
                                "frame": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
                                "confidence": settings.DEFAULT_CONFIDENCE}}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
