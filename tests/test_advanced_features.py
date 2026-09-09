"""
Advanced feature tests for IBVAP hackathon readiness
"""
import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from api.main import (
    generate_ai_explanation,
    calculate_enhanced_confidence,
    assess_threat_level,
    get_recommended_actions,
    generate_anchor_proof
)
from core.hashchain import chain_hash
from core.models import Alert


class TestAIExplanations:
    """Test AI explanation generation."""

    def test_entry_explanation(self):
        alert = Alert(
            id=1,
            camera_id=1,
            alert_type="entry",
            object_class="person",
            track_id=123,
            confidence=0.85,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="a" * 64
        )

        explanation = generate_ai_explanation(alert)
        assert "crossing" in explanation.lower() or "entry" in explanation.lower()
        assert "person" in explanation
        assert "85.0%" in explanation or "0.85" in explanation

    def test_loiter_explanation(self):
        alert = Alert(
            id=2,
            camera_id=1,
            alert_type="loiter",
            object_class="vehicle",
            track_id=456,
            confidence=0.92,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="b" * 64
        )

        explanation = generate_ai_explanation(alert)
        assert "loiter" in explanation.lower() or "lingering" in explanation.lower()
        assert "vehicle" in explanation
        assert "92.0%" in explanation or "0.92" in explanation

    def test_unknown_alert_type(self):
        alert = Alert(
            id=3,
            camera_id=1,
            alert_type="unknown_type",
            object_class="animal",
            track_id=789,
            confidence=0.67,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="c" * 64
        )

        explanation = generate_ai_explanation(alert)
        assert "unknown_type" in explanation.lower()
        assert "animal" in explanation


class TestConfidenceCalculation:
    """Test enhanced confidence calculations."""

    def test_base_confidence_enhancement(self):
        # Create a timestamp for 2 PM UTC
        ts_day = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc).isoformat()
        alert = Alert(
            id=1,
            camera_id=1,
            alert_type="entry",
            object_class="person",
            track_id=1,
            confidence=0.75,
            timestamp=ts_day,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="d" * 64
        )

        enhanced_conf = calculate_enhanced_confidence(alert)
        # Should be slightly adjusted from base confidence (daytime + person class + entry type)
        assert 0.70 <= enhanced_conf <= 0.85
        assert enhanced_conf >= alert.confidence * 0.9  # Not too much lower

    def test_night_time_confidence_adjustment(self):
        # Create timestamps for day and night
        ts_day = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc).isoformat()  # 2 PM
        ts_night = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc).isoformat()   # 2 AM

        alert_day = Alert(
            id=1,
            camera_id=1,
            alert_type="entry",
            object_class="person",
            track_id=1,
            confidence=0.80,
            timestamp=ts_day,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="e" * 64
        )

        alert_night = Alert(
            id=2,
            camera_id=1,
            alert_type="entry",
            object_class="person",
            track_id=1,
            confidence=0.80,
            timestamp=ts_night,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="f" * 64
        )

        day_conf = calculate_enhanced_confidence(alert_day)
        night_conf = calculate_enhanced_confidence(alert_night)

        # Night confidence should be slightly lower due to lighting conditions
        assert night_conf <= day_conf


class TestThreatAssessment:
    """Test threat level assessment."""

    def test_critical_threat_wrong_direction(self):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc).isoformat()
        alert = Alert(
            id=1,
            camera_id=1,
            alert_type="wrong_direction",
            object_class="person",
            track_id=1,
            confidence=0.95,  # High confidence
            timestamp=ts,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="g" * 64
        )

        threat_level = assess_threat_level(alert)
        assert threat_level == "CRITICAL"

    def test_low_threat_loiter_low_confidence(self):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc).isoformat()
        alert = Alert(
            id=2,
            camera_id=1,
            alert_type="loiter",
            object_class="person",
            track_id=1,
            confidence=0.30,  # Low confidence
            timestamp=ts,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="h" * 64
        )

        threat_level = assess_threat_level(alert)
        assert threat_level == "LOW"

    def test_medium_threat_scenarios(self):
        # Entry with medium confidence during day
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc).isoformat()
        alert = Alert(
            id=3,
            camera_id=1,
            alert_type="entry",
            object_class="vehicle",
            track_id=1,
            confidence=0.60,
            timestamp=ts,
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="i" * 64
        )

        threat_level = assess_threat_level(alert)
        assert threat_level in ["MEDIUM", "HIGH"]  # Could go either way based on exact calculation


class TestRecommendedActions:
    """Test recommended action generation."""

    def test_entry_actions(self):
        alert = Alert(
            id=1,
            camera_id=1,
            alert_type="entry",
            object_class="person",
            track_id=1,
            confidence=0.80,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="j" * 64
        )

        actions = get_recommended_actions(alert)
        assert isinstance(actions, list)
        assert len(actions) > 0
        # Should contain entry-specific actions
        action_text = " ".join(actions).lower()
        assert any(word in action_text for word in ["identity", "intent", "accomplice", "vehicle"])

    def test_critical_threat_actions(self):
        # Create a wrong_direction alert with high confidence (should be CRITICAL)
        alert = Alert(
            id=2,
            camera_id=1,
            alert_type="wrong_direction",
            object_class="person",
            track_id=1,
            confidence=0.90,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path="",
            clip_path="",
            prev_hash="0" * 64,
            hash="k" * 64
        )

        actions = get_recommended_actions(alert)
        action_text = " ".join(actions).upper()
        # Should contain immediate response indicators for high threats
        assert any(indicator in action_text for indicator in ["IMMEDIATE", "INTERCEPTION", "LOCKDOWN"])


class TestBlockchainAnchoring:
    """Test blockchain anchoring functionality."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Initialize database for tests that need it."""
        import os
        # Use in-memory database for testing
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        from core.database import init_db
        init_db()
        yield
        # Cleanup
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]

    def test_anchor_proof_generation(self, setup_db):
        with patch('core.hashchain.latest_chain_hash') as mock_latest_hash:
            mock_latest_hash.return_value = "a" * 64  # Mock hash

            anchor_data = generate_anchor_proof()

            assert "anchor_id" in anchor_data
            assert "chain_tip" in anchor_data
            assert "timestamp" in anchor_data
            assert "block_height" in anchor_data
            assert anchor_data["chain_tip"] == "a" * 64
            assert isinstance(anchor_data["block_height"], int)
            assert anchor_data["block_height"] >= 1

    def test_anchor_data_structure(self, setup_db):
        with patch('core.hashchain.latest_chain_hash') as mock_latest_hash:
            mock_latest_hash.return_value = "b" * 64

            anchor_data = generate_anchor_proof()

            # Verify all required fields are present
            required_fields = [
                'timestamp', 'chain_tip', 'anchor_id',
                'block_height', 'merkle_root'
            ]

            for field in required_fields:
                assert field in anchor_data
                assert anchor_data[field] is not None

            # Verify timestamp is recent
            timestamp = datetime.fromisoformat(anchor_data['timestamp'].replace('Z', '+00:00'))
            time_diff = abs((datetime.now(timezone.utc) - timestamp).total_seconds())
            assert time_diff < 5  # Should be within 5 seconds


class TestHashChainIntegrity:
    """Test hash chain integrity functions."""

    def test_chain_hash_consistency(self):
        payload = {"test": "data", "number": 42}
        prev_hash = "0" * 64

        hash1 = chain_hash(payload, prev_hash)
        hash2 = chain_hash(payload, prev_hash)

        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters
        assert all(c in '0123456789abcdef' for c in hash1)  # Should be hex

    def test_chain_hash_sensitivity(self):
        payload1 = {"alert_type": "entry", "confidence": 0.95}
        payload2 = {"alert_type": "entry", "confidence": 0.96}  # Slight difference
        prev_hash = "0" * 64

        hash1 = chain_hash(payload1, prev_hash)
        hash2 = chain_hash(payload2, prev_hash)

        assert hash1 != hash2  # Should be sensitive to input changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])