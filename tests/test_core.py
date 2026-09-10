"""
Core platform behaviour: IST handling, the tamper-evident chain, evidence
retention, upload validation, and the database schema.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.config import settings
from core.hashchain import (
    GENESIS_HASH, chain_hash, compute_alert_payload, compute_merkle_root,
    create_checkpoint, latest_chain_hash, verify_chain, verify_checkpoint,
)
from core.models import Alert, Camera, Rule
from core.timeutil import (
    IST, fmt_ist, is_night, ist_hour, now_ist, now_utc, start_of_ist_day,
    to_ist, to_utc, utc_iso,
)


# --------------------------------------------------------------------- IST


def test_ist_offset_is_exactly_5h30():
    assert now_ist().utcoffset() == timedelta(hours=5, minutes=30)


def test_ist_conversion_matches_utc_instant():
    utc = datetime(2026, 9, 10, 19, 5, 42, tzinfo=timezone.utc)
    ist = to_ist(utc)
    assert (ist.hour, ist.minute, ist.day) == (0, 35, 11)


def test_fmt_ist_is_operator_readable():
    text = fmt_ist(datetime(2026, 9, 10, 19, 5, 42, tzinfo=timezone.utc))
    assert text == "11 Sep 2026, 12:35:42 AM IST"


def test_naive_datetime_is_treated_as_utc_not_machine_local():
    """The platform stores UTC; a naive value must not pick up the host's zone."""
    naive = datetime(2026, 9, 10, 19, 0, 0)
    assert to_utc(naive).tzinfo == timezone.utc
    assert to_ist(naive).hour == 0        # 19:00 UTC -> 00:30 IST next day


def test_timestamp_parsers_never_raise_on_junk():
    for value in ("", "not-a-date", None, "2026-13-45T99:99:99"):
        assert to_ist(value) is None or isinstance(to_ist(value), datetime)
    assert fmt_ist("garbage") == "—"


def test_iso_z_suffix_is_accepted():
    assert to_utc("2026-09-10T19:00:00Z").hour == 19


def test_url_decoded_plus_offset_is_repaired():
    """A "+" in a query string decodes to a space; an IST filter must still work."""
    proper = to_utc("2026-09-11T02:38:48+05:30")
    mangled = to_utc("2026-09-11T02:38:48 05:30")
    assert proper is not None and proper == mangled


def test_night_window_wraps_midnight():
    # 23:00 IST == 17:30 UTC
    assert is_night("2026-09-10T17:30:00+00:00", 19, 6) is True
    # 12:00 IST == 06:30 UTC
    assert is_night("2026-09-10T06:30:00+00:00", 19, 6) is False
    # 05:00 IST is still inside a 19:00->06:00 window
    assert is_night("2026-09-10T23:30:00+00:00", 19, 6) is True


def test_start_of_ist_day_is_1830_utc_previous_day():
    start = start_of_ist_day("2026-09-11T10:00:00+05:30")
    assert start.tzinfo == timezone.utc
    assert (start.hour, start.minute) == (18, 30)


def test_utc_iso_is_always_timezone_aware():
    assert utc_iso().endswith("+00:00")


# -------------------------------------------------------------- hash chain


def _seal(db, camera_id, alert_type="entry", **overrides):
    """Append one correctly-chained alert, the way EventManager does."""
    prev = latest_chain_hash(db)
    stamp = utc_iso()
    row = Alert(
        camera_id=camera_id, alert_type=alert_type, severity="HIGH",
        object_class="person", track_id=7, confidence=0.91,
        timestamp=stamp, timestamp_ist=fmt_ist(stamp),
        rule_name="fence", rule_type="fence", detector="rule_engine",
        source_type="live", session_id="", details_json="{}",
        description="test", snapshot_path="", clip_path="",
        prev_hash=prev, hash="",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    db.add(row)
    db.flush()
    row.hash = chain_hash(compute_alert_payload(row), prev)
    db.commit()
    return row


def test_empty_chain_is_valid(db):
    result = verify_chain(db)
    assert result.valid and result.total_alerts == 0
    assert result.chain_tip == GENESIS_HASH


def test_chain_verifies_after_appends(db, camera):
    for _ in range(6):
        _seal(db, camera.id)
    result = verify_chain(db)
    assert result.valid
    assert result.total_alerts == 6


def test_first_alert_links_to_genesis(db, camera):
    row = _seal(db, camera.id)
    assert row.prev_hash == GENESIS_HASH


def test_editing_a_field_breaks_the_chain(db, camera):
    rows = [_seal(db, camera.id) for _ in range(5)]
    victim = rows[2]
    victim.alert_type = "benign"
    db.commit()

    result = verify_chain(db)
    assert result.valid is False
    assert result.broken_at == victim.id
    assert "modified" in result.message


def test_deleting_a_record_breaks_the_chain(db, camera):
    rows = [_seal(db, camera.id) for _ in range(5)]
    db.delete(rows[1])
    db.commit()

    result = verify_chain(db)
    assert result.valid is False


def test_editing_evidence_path_breaks_the_chain(db, camera):
    """Evidence paths are inside the sealed payload — swapping one is detected."""
    rows = [_seal(db, camera.id) for _ in range(3)]
    rows[1].snapshot_path = "/evidence/some_other_image.jpg"
    db.commit()
    assert verify_chain(db).valid is False


def test_editing_timestamp_breaks_the_chain(db, camera):
    rows = [_seal(db, camera.id) for _ in range(3)]
    rows[0].timestamp = "2020-01-01T00:00:00+00:00"
    db.commit()
    assert verify_chain(db).valid is False


def test_chain_hash_is_deterministic():
    payload = {"id": 1, "alert_type": "entry", "confidence": 0.5}
    assert chain_hash(payload, GENESIS_HASH) == chain_hash(payload, GENESIS_HASH)


def test_chain_hash_depends_on_predecessor():
    payload = {"id": 1, "alert_type": "entry"}
    assert chain_hash(payload, GENESIS_HASH) != chain_hash(payload, "a" * 64)


def test_chain_hash_is_order_independent_for_dict_keys():
    a = chain_hash({"x": 1, "y": 2}, GENESIS_HASH)
    b = chain_hash({"y": 2, "x": 1}, GENESIS_HASH)
    assert a == b


# ----------------------------------------------------------------- merkle


def test_merkle_root_of_empty_list_is_stable():
    assert compute_merkle_root([]) == compute_merkle_root([])


def test_merkle_root_changes_with_content():
    assert compute_merkle_root(["a" * 64]) != compute_merkle_root(["b" * 64])


def test_merkle_root_does_not_mutate_caller_list():
    """Regression: the previous recursive implementation appended to its input."""
    hashes = ["a" * 64, "b" * 64, "c" * 64]
    before = list(hashes)
    compute_merkle_root(hashes)
    assert hashes == before


def test_merkle_root_handles_large_odd_input():
    assert len(compute_merkle_root([f"{i:064d}" for i in range(101)])) == 64


def test_checkpoint_seals_and_verifies(db, camera):
    for _ in range(4):
        _seal(db, camera.id)
    checkpoint = create_checkpoint(db)
    assert checkpoint is not None
    assert checkpoint.alert_count == 4

    result = verify_checkpoint(checkpoint.checkpoint_uid, db)
    assert result["valid"] is True


def test_checkpoint_detects_tampering_in_its_range(db, camera):
    rows = [_seal(db, camera.id) for _ in range(4)]
    checkpoint = create_checkpoint(db)
    rows[1].hash = "0" * 64
    db.commit()

    result = verify_checkpoint(checkpoint.checkpoint_uid, db)
    assert result["valid"] is False


def test_checkpoint_returns_none_when_nothing_new(db, camera):
    _seal(db, camera.id)
    assert create_checkpoint(db) is not None
    assert create_checkpoint(db) is None


# --------------------------------------------------------------- evidence


def test_evidence_path_guard_accepts_managed_paths():
    from core.evidence import is_safe_evidence_path

    settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    good = settings.SNAPSHOTS_DIR / "cam1_alert1.jpg"
    good.write_bytes(b"x")
    assert is_safe_evidence_path(good) is True


@pytest.mark.parametrize("hostile", [
    "C:/Windows/System32/config/SAM",
    "/etc/passwd",
    "../../../../etc/shadow",
    "",
])
def test_evidence_path_guard_rejects_paths_outside_the_store(hostile):
    """A tampered database row must not turn into arbitrary file disclosure."""
    from core.evidence import is_safe_evidence_path

    assert is_safe_evidence_path(hostile) is False


def test_evidence_traversal_out_of_snapshot_dir_is_rejected():
    from core.evidence import is_safe_evidence_path

    escape = settings.SNAPSHOTS_DIR / ".." / ".." / ".." / "secret.txt"
    assert is_safe_evidence_path(escape) is False


def test_evidence_sweep_removes_aged_files(tmp_root):
    import os
    import time as _time

    from core.evidence import sweep_evidence

    settings.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    old = settings.CLIPS_DIR / "ancient.mp4"
    old.write_bytes(b"0" * 1024)
    ancient = _time.time() - 90 * 86400
    os.utime(old, (ancient, ancient))

    fresh = settings.CLIPS_DIR / "recent.mp4"
    fresh.write_bytes(b"0" * 1024)

    sweep_evidence(max_mb=999999, max_age_days=30)
    assert not old.exists()
    assert fresh.exists()


# ------------------------------------------------------------ upload guard


def test_sanitize_filename_strips_directories():
    from core.analysis import sanitize_filename

    assert "/" not in sanitize_filename("../../../etc/passwd.mp4")
    assert "\\" not in sanitize_filename(r"C:\Windows\evil.mp4")
    assert sanitize_filename("../../../etc/passwd.mp4").endswith(".mp4")


def test_sanitize_filename_removes_null_bytes_and_odd_chars():
    from core.analysis import sanitize_filename

    cleaned = sanitize_filename("we\x00ird name;rm -rf.mp4")
    assert "\x00" not in cleaned and ";" not in cleaned and " " not in cleaned


def test_sanitize_filename_never_returns_empty():
    from core.analysis import sanitize_filename

    assert sanitize_filename("") and sanitize_filename("...")


def test_upload_rejects_non_mp4_extension(tmp_root):
    from core.analysis import UploadValidationError, validate_upload

    path = tmp_root / "clip.avi"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(UploadValidationError, match="Unsupported file type"):
        validate_upload(path, "clip.avi")


def test_upload_rejects_wrong_container_signature(tmp_root):
    """A renamed executable must be rejected on content, not just extension."""
    from core.analysis import UploadValidationError, validate_upload

    path = tmp_root / "evil.mp4"
    path.write_bytes(b"MZ\x90\x00" + b"\x00" * 200)
    with pytest.raises(UploadValidationError, match="not a valid MP4"):
        validate_upload(path, "evil.mp4")


def test_upload_rejects_empty_file(tmp_root):
    from core.analysis import UploadValidationError, validate_upload

    path = tmp_root / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(UploadValidationError, match="empty"):
        validate_upload(path, "empty.mp4")


def test_upload_accepts_the_sample_video(sample_video):
    from core.analysis import validate_upload

    info = validate_upload(sample_video, sample_video.name)
    assert info["valid"] and info["frame_count"] > 0 and info["fps"] > 0


def test_probe_rejects_a_corrupt_video(tmp_root):
    from core.video_source import probe_video

    path = tmp_root / "corrupt.mp4"
    path.write_bytes(b"ftypisom" + b"\xff" * 4096)
    assert probe_video(path)["valid"] is False


# ------------------------------------------------------------------- schema


def test_models_create_cleanly(db, camera):
    db.add(Rule(camera_id=camera.id, rule_type="line",
                geometry=json.dumps([[0, 0], [10, 10]]), params="{}",
                name="test", is_active=True))
    db.commit()
    assert db.query(Rule).count() == 1


def test_deleting_a_camera_cascades_to_its_rules_and_alerts(db, camera):
    db.add(Rule(camera_id=camera.id, rule_type="line",
                geometry="[[0,0],[1,1]]", params="{}", name="r"))
    db.commit()
    _seal(db, camera.id)
    db.delete(camera)
    db.commit()
    assert db.query(Rule).count() == 0
    assert db.query(Alert).count() == 0


def test_alert_stores_both_utc_and_ist(db, camera):
    row = _seal(db, camera.id)
    assert row.timestamp.endswith("+00:00")
    assert row.timestamp_ist.endswith("IST")


def test_migrations_are_idempotent():
    """init_db runs on every start; it must never fail on an existing schema."""
    from core.database import init_db

    for _ in range(3):
        init_db()
