"""Shared test fixtures for IBVAP."""
from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Point to a temp database for tests
test_db = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{test_db}"

from core.database import SessionLocal, engine, Base


@pytest.fixture
def db_session():
    """Provide a fresh database session for each test."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.rollback()
    db.close()
    # Clean up
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def cleanup():
    """Auto-cleanup temp database after tests."""
    yield
    if os.path.exists(test_db):
        os.unlink(test_db)
