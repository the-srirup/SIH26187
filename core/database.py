"""
SQLite database setup.

SQLite is chosen because it is a single file with zero server process —
ideal for edge deployment at a Border Out Post.  Alert metadata can later be
replicated to PostgreSQL at sector headquarters; the schema stays identical
so the migration is trivial.

Two things matter for a multi-threaded video pipeline writing to SQLite:

* **WAL journal mode** — readers (the API) never block on the writer (camera
  threads), which is what previously made the dashboard stutter whenever an
  alert was being sealed.
* **A bounded write lock** — ``busy_timeout`` lets a concurrent writer wait
  instead of raising ``database is locked``.
"""
from __future__ import annotations

import logging

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.config import settings

log = logging.getLogger("ibvap.db")

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 15.0} if _is_sqlite else {},
    pool_pre_ping=True,
    echo=False,
)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        """Concurrency-friendly PRAGMAs applied to every new connection."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base shared by every model."""


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables from models (if they don't already exist) and migrate."""
    from core import models  # noqa: F401  — imported for side-effects

    Base.metadata.create_all(bind=engine)
    _ensure_schema_compatibility(engine)


#: Additive column migrations for databases created by an older build.
#: A production deployment should use Alembic; this keeps edge/hackathon
#: databases usable across source upgrades without forcing a reset.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE cameras ADD COLUMN is_online BOOLEAN DEFAULT 0",
    "ALTER TABLE cameras ADD COLUMN source_kind VARCHAR(20) DEFAULT 'live'",
    "ALTER TABLE cameras ADD COLUMN created_at VARCHAR(40) DEFAULT ''",
    "ALTER TABLE rules ADD COLUMN created_at VARCHAR(40) DEFAULT ''",
    "ALTER TABLE alerts ADD COLUMN severity VARCHAR(20) DEFAULT 'MEDIUM'",
    "ALTER TABLE alerts ADD COLUMN timestamp_ist VARCHAR(64) DEFAULT ''",
    "ALTER TABLE alerts ADD COLUMN rule_name VARCHAR(200) DEFAULT ''",
    "ALTER TABLE alerts ADD COLUMN rule_type VARCHAR(50) DEFAULT ''",
    "ALTER TABLE alerts ADD COLUMN detector VARCHAR(40) DEFAULT 'rule_engine'",
    "ALTER TABLE alerts ADD COLUMN source_type VARCHAR(20) DEFAULT 'live'",
    "ALTER TABLE alerts ADD COLUMN session_id VARCHAR(64) DEFAULT ''",
    "ALTER TABLE alerts ADD COLUMN details_json TEXT DEFAULT '{}'",
    "ALTER TABLE alerts ADD COLUMN description TEXT DEFAULT ''",
)


def _ensure_schema_compatibility(bound_engine: Engine) -> None:
    """Apply small, safe additive migrations for pre-existing databases."""
    if bound_engine.url.get_backend_name() != "sqlite":
        return

    with bound_engine.begin() as conn:
        for statement in _MIGRATIONS:
            try:
                conn.execute(text(statement))
            except Exception:
                # SQLite has no `ADD COLUMN IF NOT EXISTS`; a duplicate-column
                # error simply means this migration already ran.
                pass

    # Newly introduced tables (checkpoints, analysis_sessions) are handled by
    # create_all above; this call makes the intent explicit for older images.
    from core import models  # noqa: F401

    Base.metadata.create_all(bind=bound_engine)
    log.debug("Schema compatibility pass complete")
