"""
Automatic Number Plate Recognition (ANPR) for IBVAP.

The critical gap in the previous implementation was that ANPR ran OCR over
the whole frame and merely returned text fragments; it did not actively
extract license-plate candidates. This module now performs a proper two-stop
edge pipeline:

1. Vehicle-aware plate localization:
   * Use YOLO vehicle detections (car/truck/bus/motorcycle) to crop likely
     plate areas.
   * Find high-contrast rectangular candidates using edge density, contour
     geometry, aspect-ratio and area constraints.
   * If YOLO misses a vehicle, the same candidate search runs on the full
     frame as a fallback.
2. OCR only the localized candidates with EasyOCR.

This makes ANPR materially extract alphanumerics from license plates instead
of transcribing arbitrary text from the frame.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.cv.anpr")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    log.info("EasyOCR available - ANPR functionality enabled")
except ImportError:
    EASYOCR_AVAILABLE = False
    log.warning("EasyOCR not installed - ANPR functionality disabled")


@dataclass
class PlateDetection:
    """Result of detecting a license plate in a frame."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    plate_text: str
    text_confidence: float
    frame_number: int
    timestamp: float
    vehicle_class: Optional[str] = None
    vehicle_track_id: Optional[int] = None


class ANPRProcessor:
    """
    License plate detection and OCR processor.

    Uses EasyOCR for recognition and OpenCV morphological/contour search for
    plate candidate localization. It is intentionally lazy: ``get_anpr_processor()``
    instantiates the processor, but EasyOCR models load only when enabled.
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike", "van"}
    PLATE_CHARS_RE = re.compile(r"[^A-Z0-9]+")
    INDIAN_PLATE_RE = re.compile(
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}[A-Z]?$"
    )
    LOOSE_PLATE_RE = re.compile(r"^[A-Z0-9]{6,12}$")

    def __init__(self, lang_list: Optional[list[str]] = None):
        """
        Initialize ANPR processor.

        Args:
            lang_list: List of languages for easyOCR (default: ['en'] for Latin chars,
                      would need ['hi', 'en'] for Indian plates with Devanagari).
        """
        self.lang_list = lang_list or [
            lang.strip() for lang in settings.ANPR_LANGUAGES.split(",") if lang.strip()
        ] or ["en"]
        self.reader = None
        self._initialized = False
        self._init_failed = False
        self._detection_cache: list[PlateDetection] = []
        self._last_inference_frame = -1
        self._ocr_call_count = 0
        self._ocr_ms_total = 0.0
        self._reader_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._event_debounce: dict[tuple, float] = {}
        self._gpu = False
        self._plates_read = 0
        self._plates_uncertain = 0

    def _init_reader(self) -> None:
        """
        Load the EasyOCR reader (~5-10 s, several hundred MB).

        Deliberately lazy: the previous build loaded this during application
        startup even on deployments that never see a vehicle, delaying the
        dashboard coming up.  It now loads on the first frame that actually
        contains a vehicle.
        """
        try:
            gpu = False
            try:
                import torch

                gpu = bool(torch.cuda.is_available())
            except Exception:
                pass
            self.reader = easyocr.Reader(self.lang_list, gpu=gpu, verbose=False)
            self._gpu = gpu
            self._initialized = True
            log.info(
                "ANPR reader ready — languages=%s device=%s",
                self.lang_list, "cuda" if gpu else "cpu",
            )
        except Exception as e:
            log.error("Failed to initialize EasyOCR: %s", e)
            self._initialized = False
            self._init_failed = True
            self.reader = None

    def is_available(self) -> bool:
        """
        True when OCR can run. Triggers the one-time model load if needed.

        Returns False (rather than raising) when EasyOCR is missing or failed
        to load, so the rest of the surveillance pipeline is unaffected.
        """
        if not (EASYOCR_AVAILABLE and settings.ANPR_ENABLED):
            return False
        if self._initialized:
            return True
        if self._init_failed:
            return False
        with self._init_lock:
            if not self._initialized and not self._init_failed:
                self._init_reader()
        return self._initialized and self.reader is not None

    def set_languages(self, lang_list: list[str]) -> bool:
        """Update the languages used for OCR."""
        if not EASYOCR_AVAILABLE:
            log.warning("Cannot set languages - EasyOCR not available")
            return False

        try:
            self.lang_list = lang_list or ["en"]
            self._init_failed = False
            self.reader = easyocr.Reader(self.lang_list, gpu=self._gpu, verbose=False)
            self._initialized = True
            log.info("ANPR languages updated to: %s", self.lang_list)
            return True
        except Exception as e:
            log.error("Failed to update ANPR languages: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def recognize_plates(
        self,
        frame: np.ndarray,
        vehicle_detections: Optional[list] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> list[PlateDetection]:
        """
        Actively localize and recognize license plates in the frame.

        Parameters
        ----------
        frame:
            BGR frame.
        vehicle_detections:
            YOLO detection objects. Each object should expose
            ``bbox``, ``class_name`` and ``track_id``.
        frame_number:
            Processing frame number (zero-based).
        timestamp:
            Epoch timestamp. Defaults to ``time.time()``.

        Returns
        -------
        list[PlateDetection]
            Detected plates, sorted by confidence.
        """
        if not self.is_available():
            return []

        ts = timestamp if timestamp is not None else time.time()
        processed = self.preprocess_for_indian_plates(frame)

        vehicle_boxes = self._vehicle_boxes(vehicle_detections)
        candidates: list[tuple[int, int, int, int, float, Optional[int], Optional[str]]] = []

        # Vehicle-aware localization. Crop each vehicle and find plate candidates.
        if vehicle_boxes:
            h, w = processed.shape[:2]
            for x1, y1, x2, y2, track_id, vehicle_class in vehicle_boxes:
                if x2 <= x1 or y2 <= y1:
                    continue

                # Expand crop slightly. ANPR should look for the lower part of
                # vehicle ROI, where most plates appear.
                crop_x1 = max(0, x1 - 6)
                crop_y1 = max(0, y1 + int((y2 - y1) * 0.45))
                crop_x2 = min(w, x2 + 6)
                crop_y2 = min(h, y2 + 6)
                crop = processed[crop_y1:crop_y2, crop_x1:crop_x2]

                local_candidates = self._find_plate_candidates(crop)
                for lx1, ly1, lx2, ly2, score in local_candidates:
                    candidates.append(
                        (
                            lx1 + crop_x1,
                            ly1 + crop_y1,
                            lx2 + crop_x1,
                            ly2 + crop_y1,
                            score,
                            track_id,
                            vehicle_class,
                        )
                    )

        # Full-frame fallback for missed vehicles or test images with plates.
        if not candidates:
            for fx1, fy1, fx2, fy2, score in self._find_plate_candidates(processed):
                candidates.append((fx1, fy1, fx2, fy2, score, None, None))

        if not candidates:
            self._detection_cache = []
            return []

        deduped = self._deduplicate_candidates(candidates)
        # OCR is by far the most expensive stage; only the strongest candidates
        # are read, bounding worst-case latency per analytics frame.
        deduped = sorted(deduped, key=lambda c: c[4], reverse=True)
        deduped = deduped[: max(1, int(settings.ANPR_MAX_PLATES_PER_TICK))]

        detections: list[PlateDetection] = []
        for x1, y1, x2, y2, score, track_id, vehicle_class in deduped:
            plate_text, text_conf = self._ocr_region(processed, (x1, y1, x2, y2))
            clean_text = self.normalize_plate_text(plate_text)
            if not clean_text:
                continue

            best = self._extract_most_plate_like(clean_text)
            if not best:
                continue

            # Reward a read that matches the Indian plate grammar; penalise a
            # string that is merely alphanumeric noise of the right length.
            structured = bool(self.INDIAN_PLATE_RE.match(best))
            adjusted = float(text_conf) * (1.0 if structured else 0.75)
            if adjusted >= settings.ANPR_CONFIDENCE_THRESHOLD:
                self._plates_read += 1
            else:
                self._plates_uncertain += 1

            detection = PlateDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                plate_text=self.format_indian_plate(best) if structured else best,
                text_confidence=adjusted,
                frame_number=frame_number,
                timestamp=ts,
                vehicle_class=vehicle_class,
                vehicle_track_id=track_id,
            )
            detections.append(detection)

        detections.sort(
            key=lambda d: (d.text_confidence * 0.6 + d.confidence * 0.4),
            reverse=True,
        )
        self._detection_cache = detections
        self._last_inference_frame = frame_number
        return detections

    def detect_and_recognize(self, frame: np.ndarray) -> list[PlateDetection]:
        """
        Backward-compatible wrapper used by the ANPR test endpoint.

        It runs the full-frame fallback path only, which is appropriate when
        no YOLO detections are available.
        """
        return self.recognize_plates(frame, vehicle_detections=None)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_for_indian_plates(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better license-plate localization.

        The returned image is kept as BGR because EasyOCR expects either RGB
        or BGR and internally converts.
        """
        if frame is None:
            return np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grey)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    # Plate localization
    # ------------------------------------------------------------------ #

    def _vehicle_boxes(
        self,
        vehicle_detections: Optional[list],
    ) -> list[tuple[int, int, int, int, Optional[int], Optional[str]]]:
        boxes: list[tuple[int, int, int, int, Optional[int], Optional[str]]] = []
        if not vehicle_detections:
            return boxes

        for det in vehicle_detections:
            class_name = (getattr(det, "class_name", "") or "").lower()
            if class_name not in self.VEHICLE_CLASSES:
                continue
            bbox = tuple(map(int, det.bbox))
            if len(bbox) != 4:
                continue
            boxes.append(
                (
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    getattr(det, "track_id", None),
                    class_name,
                )
            )
        return boxes

    def _find_plate_candidates(
        self,
        image_bgr: np.ndarray,
    ) -> list[tuple[int, int, int, int, float]]:
        """
        Find high-contrast rectangular regions that look like license plates.

        Returns ``(x1, y1, x2, y2, score)`` sorted best-first.
        """
        if image_bgr.size == 0:
            return []

        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 200)

        # Morphologically close short edge gaps (plate border and characters).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, float]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 50 or h < 15 or w * h < settings.ANPR_MIN_PLATE_AREA:
                continue

            aspect_ratio = w / float(h)
            if not (settings.ANPR_MIN_ASPECT_RATIO <= aspect_ratio <= settings.ANPR_MAX_ASPECT_RATIO):
                continue

            area = cv2.contourArea(contour)
            bbox_area = float(w * h)
            if area <= 0:
                continue

            # License plates are usually rectangular. Reject very sparse or
            # very filled blobs.
            extent = area / bbox_area
            if not (0.25 <= extent <= 0.95):
                continue

            roi_edges = edges[y : y + h, x : x + w]
            edge_density = float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
            if edge_density < 0.03:
                continue

            # Score rewards plate-like edges and penalizes extreme sizes.
            score = edge_density * extent
            if 3.0 <= aspect_ratio <= 5.5:
                score *= 1.25
            candidates.append((x, y, x + w, y + h, score))

        # Non-max suppression and score sorting.
        candidates.sort(key=lambda c: c[4], reverse=True)
        picked: list[tuple[int, int, int, int, float]] = []

        def _iou(a, b):
            ax1, ay1, ax2, ay2, _ = a
            bx1, by1, bx2, by2, _ = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(a_area + b_area - inter)

        for cand in candidates:
            if not any(_iou(cand, kept) > 0.55 for kept in picked):
                picked.append(cand[:5])
            if len(picked) >= 4:
                break

        return picked

    def _deduplicate_candidates(
        self,
        candidates: list[tuple[int, int, int, int, float, Optional[int], Optional[str]]],
    ) -> list[tuple[int, int, int, int, float, Optional[int], Optional[str]]]:
        """Deduplicate candidate boxes across full-frame and vehicle crops."""
        deduped: list[tuple[int, int, int, int, float, Optional[int], Optional[str]]] = []

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a[:4]
            bx1, by1, bx2, by2 = b[:4]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(a_area + b_area - inter)

        candidates = sorted(candidates, key=lambda c: c[4], reverse=True)
        for cand in candidates:
            if deduped and any(_iou(cand, kept) > 0.5 for kept in deduped):
                continue
            deduped.append(cand)
        return deduped

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #

    def _ocr_region(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        """Run EasyOCR on a localized plate candidate."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "", 0.0

        # Convert BGR->RGB because easyOCR documentation/pretrained models use
        # RGB image input. Some installations are tolerant, but do this
        # explicitly for consistency.
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        self._ocr_call_count += 1
        started = time.perf_counter()
        with self._reader_lock:
            try:
                results = self.reader.readtext(
                    roi_rgb,
                    detail=1,
                    paragraph=False,
                    width_ths=0.65,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
            except Exception as e:
                log.error("EasyOCR readtext failed: %s", e)
                return "", 0.0
            finally:
                self._ocr_ms_total += (time.perf_counter() - started) * 1000.0

        if not results:
            return "", 0.0

        # Keep every read and report the confidence honestly. Discarding
        # low-confidence reads here (as the previous build did) made it
        # impossible to distinguish "no plate" from "plate I could not read",
        # so a marginal plate silently vanished instead of surfacing as
        # PLATE UNCERTAIN.
        text_parts: list[str] = []
        confidences: list[float] = []
        for (_bbox, text, conf) in results:
            cleaned = text.strip().upper()
            if not cleaned:
                continue
            text_parts.append(cleaned)
            confidences.append(float(conf))

        if not text_parts:
            return "", 0.0

        # Weight by character count: a 9-character read at 0.6 is a better
        # plate candidate than a 1-character read at 0.95.
        total_chars = sum(len(t) for t in text_parts) or 1
        weighted = sum(c * len(t) for c, t in zip(confidences, text_parts)) / total_chars
        return "".join(text_parts), float(weighted)

    def normalize_plate_text(self, raw_text: str) -> str:
        """
        Normalize OCR output to uppercase alphanumeric plate characters.

        This intentionally does not aggressively map O->0 or I->1 because a
        little OCR ambiguity is safer to keep visible in the final string
        (and for later re-verification) than to silently change evidence.
        """
        if not raw_text:
            return ""
        return self.PLATE_CHARS_RE.sub("", raw_text).upper()

    def _extract_most_plate_like(self, clean_text: str) -> Optional[str]:
        """
        If OCR output includes extra text around the plate, pick the token
        that best matches a plausibly Indian alphanumeric plate pattern.
        """
        if not clean_text:
            return None

        if self.INDIAN_PLATE_RE.match(clean_text):
            return clean_text

        # OCR may concatenate all tokens as one string in small ROIs.
        # If the entire string is a plausible alphanumeric code, keep it.
        if self.LOOSE_PLATE_RE.match(clean_text):
            return clean_text

        # Try splitting into plausible tokens and prefer the longest one.
        tokens = re.findall(r"[A-Z0-9]{5,12}", clean_text)
        if not tokens:
            return None
        return max(tokens, key=len)

    # ------------------------------------------------------------------ #
    # Pipeline integration
    # ------------------------------------------------------------------ #

    def cached_detections(self) -> list[PlateDetection]:
        """Last OCR result set, reused between cadence ticks."""
        return list(self._detection_cache)

    @staticmethod
    def format_indian_plate(text: str) -> str:
        """
        Group a validated plate the way it is printed: ``MH12AB1234`` ->
        ``MH 12 AB 1234``.  Purely presentational; the raw string is kept in
        the event details so the evidence is not reshaped.
        """
        match = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})([A-Z]?)$", text)
        if not match:
            return text
        return " ".join(part for part in match.groups() if part)

    def build_event(self, plate: PlateDetection):
        """
        Build a debounced ANPR event, or ``None``.

        A read below ``ANPR_ALERT_CONFIDENCE`` is *not* logged as a plate
        number — writing a guessed registration into an evidentiary log is
        worse than logging nothing.  The overlay still shows it live as
        PLATE UNCERTAIN so the operator knows a plate was seen.
        """
        from cv.rules import Alert as RuleAlert

        if not plate.plate_text:
            return None
        if plate.text_confidence < settings.ANPR_ALERT_CONFIDENCE:
            return None

        track_id = int(plate.vehicle_track_id or 0)
        key = (plate.plate_text, track_id)
        now = time.time()
        last = self._event_debounce.get(key, 0.0)
        if now - last < settings.ANPR_ALERT_DEBOUNCE_SECONDS:
            return None
        self._event_debounce[key] = now
        if len(self._event_debounce) > 256:
            self._event_debounce = {
                k: v for k, v in self._event_debounce.items() if now - v < 600
            }

        verified = bool(self.INDIAN_PLATE_RE.match(plate.plate_text.replace(" ", "")))
        return RuleAlert(
            rule_name="anpr", rule_type="anpr", track_id=track_id,
            alert_type="anpr_detection",
            description=(
                f"Number plate read: {plate.plate_text} "
                f"({plate.text_confidence * 100:.0f}% OCR confidence"
                f"{', matches Indian plate format' if verified else ''})"
            ),
            details={
                "plate_text": plate.plate_text,
                "plate_raw": plate.plate_text.replace(" ", ""),
                "ocr_confidence": round(float(plate.text_confidence), 4),
                "localization_confidence": round(float(plate.confidence), 4),
                "format_verified": verified,
                "vehicle_class": plate.vehicle_class or "unknown",
                "bbox": list(plate.bbox),
            },
        )

    def get_metrics(self) -> dict:
        calls = max(1, self._ocr_call_count)
        return {
            "available": self.is_available(),
            "enabled": settings.ANPR_ENABLED,
            "easyocr_installed": EASYOCR_AVAILABLE,
            "languages": self.lang_list,
            "device": "cuda" if self._gpu else "cpu",
            "ocr_call_count": self._ocr_call_count,
            "ocr_ms_avg": round(self._ocr_ms_total / calls, 1),
            "plates_confident": self._plates_read,
            "plates_uncertain": self._plates_uncertain,
            "cadence_frames": settings.ANPR_EVERY_N_FRAMES,
            "confidence_threshold": settings.ANPR_CONFIDENCE_THRESHOLD,
            "alert_confidence": settings.ANPR_ALERT_CONFIDENCE,
            "last_inference_frame": self._last_inference_frame,
        }


# Global ANPR processor instance (lazy initialized)
_anpr_processor: Optional[ANPRProcessor] = None
_anpr_lock = None


def get_anpr_processor() -> ANPRProcessor:
    """Get or create the global ANPR processor instance."""
    global _anpr_processor, _anpr_lock  # noqa: PLW0602
    if _anpr_lock is None:
        import threading
        _anpr_lock = threading.Lock()

    with _anpr_lock:
        if _anpr_processor is None:
            _anpr_processor = ANPRProcessor()
        return _anpr_processor
