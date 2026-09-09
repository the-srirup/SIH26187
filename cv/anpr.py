"""
Automatic Number Plate Recognition (ANPR) for IBVAP.

Implements license plate detection and OCR using easyOCR.
Designed as a stretch goal similar to face recognition.
For production use with Indian license plates, would require
fine-tuning on Indian plate datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.cv.anpr")

# Optional import - easyOCR is a stretch dependency for ANPR
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


class ANPRProcessor:
    """
    License plate detection and OCR processor.

    Uses a two-stage approach:
    1. Plate detection (could use YOLO-plate or custom detector)
    2. OCR on detected plate region (using easyOCR)

    For Indian license plates, would require fine-tuning on
    Indian plate datasets for optimal performance.
    """

    def __init__(self, lang_list: list[str] = None):
        """
        Initialize ANPR processor.

        Args:
            lang_list: List of languages for easyOCR (default: ['en'] for Latin chars,
                      would need ['hi', 'en'] for Indian plates with Devanagari)
        """
        self.lang_list = lang_list or ['en']  # Default to English/Latin characters
        self.reader = None
        self._initialized = False

        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(self.lang_list)
                self._initialized = True
                log.info(f"ANPR processor initialized with languages: {self.lang_list}")
            except Exception as e:
                log.error(f"Failed to initialize EasyOCR: {e}")
                self._initialized = False
        else:
            log.warning("ANPR processor created but EasyOCR not available")

    def is_available(self) -> bool:
        """Check if ANPR functionality is available."""
        return self._initialized and EASYOCR_AVAILABLE

    def detect_and_recognize(self, frame: np.ndarray) -> list[PlateDetection]:
        """
        Detect license plates in frame and recognize text.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of PlateDetection objects
        """
        if not self.is_available():
            return []

        try:
            # For now, we'll use a simple approach:
            # Treat the entire frame as a potential plate region
            # In a production system, you would:
            # 1. Use a dedicated plate detector (YOLO-plate, Haar cascades, etc.)
            # 2. Extract plate regions
            # 3. Apply OCR to each region
            #
            # As a stretch goal implementation, we demonstrate the OCR capability
            # on the full frame, which would work for clear plate close-ups

            results = self.reader.readtext(frame)

            detections = []
            for (bbox, text, confidence) in results:
                # Filter by confidence threshold
                if confidence < settings.ANPR_CONFIDENCE_THRESHOLD:
                    continue

                # Convert bbox format
                # easyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                points = np.array(bbox)
                x1, y1 = points.min(axis=0).astype(int)
                x2, y2 = points.max(axis=0).astype(int)

                detection = PlateDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    plate_text=text.strip().upper(),
                    text_confidence=float(confidence),
                    frame_number=0,  # Would be set by caller
                    timestamp=0.0    # Would be set by caller
                )
                detections.append(detection)

            return detections

        except Exception as e:
            log.error(f"Error in ANPR processing: {e}")
            return []

    def preprocess_for_indian_plates(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better Indian license plate recognition.
        This would be enhanced with actual Indian plate training data.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Convert back to BGR for OCR
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    def set_languages(self, lang_list: list[str]):
        """Update the languages used for OCR."""
        if not EASYOCR_AVAILABLE:
            log.warning("Cannot set languages - EasyOCR not available")
            return False

        try:
            self.lang_list = lang_list
            self.reader = easyocr.Reader(self.lang_list)
            log.info(f"ANPR languages updated to: {self.lang_list}")
            return True
        except Exception as e:
            log.error(f"Failed to update ANPR languages: {e}")
            return False


# Global ANPR processor instance (lazy initialized)
_anpr_processor: Optional[ANPRProcessor] = None


def get_anpr_processor() -> ANPRProcessor:
    """Get or create the global ANPR processor instance."""
    global _anpr_processor
    if _anpr_processor is None:
        _anpr_processor = ANPRProcessor()
    return _anpr_processor