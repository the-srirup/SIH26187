"""Tests for core modules."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_config_defaults():
    """Test that default settings are loaded."""
    from core.config import settings
    assert settings.PROJECT_NAME == "IBVAP"
    assert settings.VERSION == "1.0.0"
    assert "sqlite" in settings.DATABASE_URL
    assert settings.TARGET_FPS == 5


def test_chain_hash_consistency():
    """Hashing the same payload with same prev_hash always gives same result."""
    from core.hashchain import chain_hash
    payload = {"alert_type": "entry", "camera_id": 1}
    h1 = chain_hash(payload, "0" * 64)
    h2 = chain_hash(payload, "0" * 64)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_chain_hash_different_prev():
    """Different prev_hash gives different output."""
    from core.hashchain import chain_hash
    payload = {"alert_type": "entry"}
    h1 = chain_hash(payload, "a" * 64)
    h2 = chain_hash(payload, "b" * 64)
    assert h1 != h2


def test_chain_hash_different_payload():
    """Different payload with same prev_hash gives different output."""
    from core.hashchain import chain_hash
    h1 = chain_hash({"type": "entry"}, "prev")
    h2 = chain_hash({"type": "exit"}, "prev")
    assert h1 != h2


def test_frame_buffer():
    """FrameBuffer stores and retrieves frames per camera."""
    from core.camera import FrameBuffer
    buf = FrameBuffer()
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    buf.set(1, frame)
    retrieved = buf.get_frame(1)
    assert retrieved is not None
    assert retrieved.shape == frame.shape

    # Different camera should not see this frame
    assert buf.get_frame(2) is None


def test_low_light_enhancer():
    """CLAHE enhancer doesn't crash on normal frames."""
    from core.camera import LowLightEnhancer
    import numpy as np
    enhancer = LowLightEnhancer()
    # Bright frame should pass through
    bright = np.ones((480, 640, 3), dtype=np.uint8) * 200
    result = enhancer.maybe_enhance(bright)
    assert result.shape == bright.shape

    # Dark frame should be enhanced
    dark = np.ones((480, 640, 3), dtype=np.uint8) * 20
    result = enhancer.maybe_enhance(dark)
    assert result.shape == dark.shape


def test_models():
    """Test SQLAlchemy models exist and have correct attributes."""
    from core.models import Camera, Rule, Alert

    assert Camera.__tablename__ == "cameras"
    assert Rule.__tablename__ == "rules"
    assert Alert.__tablename__ == "alerts"

    # Check Alert has hash chain columns
    assert hasattr(Alert, "prev_hash")
    assert hasattr(Alert, "hash")


def test_detector_parse():
    """Detector returns correct Detection dataclass structure."""
    from cv.detector import Detection
    det = Detection(
        track_id=1, class_id=0, class_name="person",
        confidence=0.95, bbox=(100, 200, 300, 400), foot=(200, 400)
    )
    assert det.track_id == 1
    assert det.class_name == "person"
    assert len(det.bbox) == 4
    assert len(det.foot) == 2


def test_rules_engine():
    """RuleEngine can add rules and process detections."""
    from cv.rules import RuleEngine, FenceRule
    engine = RuleEngine(camera_id="test")
    rule = FenceRule("test_fence", 0, 0, 100, 100)
    engine.add_rule(rule)
    assert len(engine.get_rules()) == 1

    # Process empty detections — no alerts
    alerts = engine.update([])
    assert alerts == []


def test_face_recognizer():
    """FaceRecognizer singleton works."""
    from cv.face import FaceRecognizer, get_face_recognizer
    import numpy as np
    fr = get_face_recognizer()
    assert isinstance(fr, FaceRecognizer)

    # Test watchlist management
    fr.add_watchlist_from_embedding("test_person", np.array([0.1] * 512, dtype=np.float32))
    assert len(fr.get_watchlist()) == 1
    assert fr.get_watchlist()[0]["name"] == "test_person"

    fr.set_threshold(0.7)
    assert fr._similarity_threshold == 0.7

    fr.remove_watchlist_entry(1)
    assert len(fr.get_watchlist()) == 0


def test_detector_names():
    """Detector exposes class names."""
    from cv.detector import Detector
    det = Detector()
    names = det.names
    assert isinstance(names, dict)
    assert len(names) > 0  # COCO has 80 classes
    assert "person" in names.values() or "person" in str(names)
