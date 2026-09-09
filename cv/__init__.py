"""
Computer vision package for IBVAP — detection, tracking, rules, and face recognition.
"""
from cv.detector import Detector, Detection, FrameResult
from cv.rules import (
    Alert,
    BaseRule,
    FenceRule,
    ZoneRule,
    LoiterRule,
    DirectionRule,
    RuleEngine,
)
from cv.face import FaceRecognizer, FaceMatch, WatchlistEntry, get_face_recognizer

__all__ = [
    # Detector
    "Detector",
    "Detection",
    "FrameResult",
    # Rules
    "Alert",
    "BaseRule",
    "FenceRule",
    "ZoneRule",
    "LoiterRule",
    "DirectionRule",
    "RuleEngine",
    # Face
    "FaceRecognizer",
    "FaceMatch",
    "WatchlistEntry",
    "get_face_recognizer",
]