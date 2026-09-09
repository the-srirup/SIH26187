"""
Object detection and multi-object tracking.

Uses Ultralytics YOLO (yolo11n.pt — nano variant, CPU-friendly) with
ByteTrack for persistent IDs.  A single ``Detector`` instance loads the
model once and exposes ``detect()`` and ``track()`` helpers.

The tracker writes results into a shared, thread-safe container that
the FastAPI layer reads for MJPEG streaming and WebSocket alerts.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import settings

log = logging.getLogger("ibvap.cv")


@dataclass
class Detection:
    """A single tracked detection in one frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    foot: tuple[int, int]            # bottom-centre point — where feet touch ground


@dataclass
class FrameResult:
    """All detections for one processed frame."""
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class Detector:
    """High-level wrapper around YOLO + ByteTrack."""

    def __init__(self, model_path: str = settings.MODEL_PATH) -> None:
        self.model = YOLO(model_path)
        log.info("YOLO model loaded from %s — classes=%d", model_path, len(self.model.names))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection (no tracking)."""
        results = self.model(frame, verbose=False)[0]
        return self._parse_boxes(results.boxes, track_ids=None, model_names=self.model.names)

    def track(self, frame: np.ndarray, tracker: str = settings.TRACKER) -> list[Detection]:
        """Run detection + ByteTrack, returning tracked objects with stable IDs."""
        results = self.model.track(
            frame, persist=True, tracker=tracker, verbose=False
        )[0]
        track_ids = self._extract_track_ids(results)
        return self._parse_boxes(results.boxes, track_ids, self.model.names)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_track_ids(results) -> dict[int, int]:
        """Map box-index → track_id from ByteTrack output."""
        ids: dict[int, int] = {}
        if results.boxes.id is None:
            return ids
        for i, tid in enumerate(results.boxes.id):
            ids[i] = int(tid)
        return ids

    @staticmethod
    def _parse_boxes(boxes, track_ids: Optional[dict[int, int]], model_names: dict) -> list[Detection]:
        """Convert raw YOLO boxes into typed Detection objects."""
        out: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        # Filter by minimum object area
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        for i in range(len(boxes)):
            if areas[i] < settings.MIN_OBJECT_AREA:
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            track_id = track_ids.get(i) if track_ids else i
            if track_id is None:
                continue
            class_id = int(cls[i])
            class_name = model_names.get(class_id, str(class_id)) if model_names else str(class_id)
            out.append(Detection(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf[i]),
                bbox=(x1, y1, x2, y2),
                foot=((x1 + x2) // 2, y2),
            ))
        return out

    @property
    def names(self) -> dict[int, str]:
        return self.model.names
