"""
Shared pytest fixtures.

Every test runs against a throwaway SQLite database in a temp directory, so
the suite never touches a real deployment's alert log or evidence store.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the application at a scratch database *before* importing anything that
# builds an engine at import time.
_TMP = Path(tempfile.mkdtemp(prefix="ibvap_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["EVIDENCE_ENABLED"] = "false"
os.environ["FACE_ENABLED"] = "false"
os.environ["ANPR_ENABLED"] = "false"


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session", autouse=True)
def _redirect_storage(tmp_root):
    """Keep every artefact the tests create inside the temp directory."""
    from core.config import settings

    settings.ALERTS_DIR = tmp_root / "alerts"
    settings.SNAPSHOTS_DIR = tmp_root / "alerts" / "snapshots"
    settings.CLIPS_DIR = tmp_root / "clips"
    settings.VIDEOS_DIR = tmp_root / "videos"
    settings.PROCESSED_DIR = tmp_root / "videos" / "processed"
    settings.LOG_DIR = tmp_root / "logs"
    settings.ensure_dirs()
    yield


@pytest.fixture
def db():
    """A clean database for each test."""
    from core.database import Base, SessionLocal, engine, init_db

    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)


@pytest.fixture
def camera(db):
    from core.models import Camera
    from core.timeutil import utc_iso

    cam = Camera(name="TEST-CAM", url="0", location="Test Post",
                 is_active=False, is_online=False, created_at=utc_iso())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@pytest.fixture
def sample_video() -> Path:
    path = Path(__file__).resolve().parent.parent / "samples" / "sample_border_scenario.mp4"
    if not path.exists():
        pytest.skip("sample video not present")
    return path


class FakeDetection:
    """Minimal stand-in for cv.detector.Detection in rule/engine tests."""

    def __init__(self, track_id, x, y, class_name="person", confidence=0.9, age=10):
        self.track_id = track_id
        self.foot = (x, y)
        self.bbox = (x - 15, y - 60, x + 15, y)
        self.draw_bbox = self.bbox
        self.class_name = class_name
        self.class_id = 0 if class_name == "person" else 2
        self.confidence = confidence
        self.age = age

    @property
    def is_person(self):
        return self.class_id == 0

    @property
    def is_vehicle(self):
        return self.class_id != 0

    @property
    def label(self):
        return self.class_name.upper()


@pytest.fixture
def det_factory():
    return FakeDetection
