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

#: Connection pool, sized for how many threads actually touch this database.
#:
#: SQLAlchemy's default for a SQLite file is ``QueuePool(pool_size=5,
#: max_overflow=10)`` — fifteen connections, and a 30-second wait before the
#: sixteenth caller raises ``TimeoutError``. That default assumes a request/
#: response server. This process is not one: every camera runs an analytics
#: thread that opens a session to seal an event, plus a capture thread; the API
#: adds up to forty Starlette threadpool workers; and there are background
#: tasks for checkpoints, evidence and analysis on top. Eight cameras and a
#: couple of dashboards can exceed fifteen concurrent sessions, and the symptom
#: when they do is the worst kind: not an error, but every caller stalling for
#: up to thirty seconds first.
#:
#: Pooling is kept rather than switched to ``NullPool``, because it is measured
#: at 0.355 ms per acquire/query/close against 1.595 ms without — SQLite
#: reconnects are cheap but not free, and this path runs on every sealed event.
#: The pool is simply made big enough, and made to fail fast instead of hanging
#: when something really is wrong.
_POOL_KWARGS = {
    "pool_size": 25,
    "max_overflow": 25,
    # Fail in five seconds with a clear error rather than freezing a camera
    # thread for thirty. Exhausting fifty connections is a bug to surface, not
    # a queue to wait in.
    "pool_timeout": 5.0,
    # Recycle idle connections so a long-running deployment never accumulates
    # handles the OS has quietly dropped.
    "pool_recycle": 3600,
} if _is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 15.0} if _is_sqlite else {},
    pool_pre_ping=True,
    echo=False,
    **_POOL_KWARGS,
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
    "ALTER TABLE cameras ADD COLUMN is_deleted BOOLEAN DEFAULT 0",
    "ALTER TABLE cameras ADD COLUMN deleted_at VARCHAR(40) DEFAULT ''",
)


def _ensure_schema_compatibility(bound_engine: Engine) -> None:
    """Apply small, safe additive migrations for pre-existing databases."""
    if bound_engine.url.get_backend_name() != "sqlite":
        return

    # Each statement gets its OWN transaction. Sharing one transaction meant a
    # single failure (which is the *normal* case — SQLite has no
    # ``ADD COLUMN IF NOT EXISTS``, so every already-applied migration raises)
    # left the connection needing a rollback, and every later statement was
    # refused. On an older database the first duplicate column therefore
    # silently skipped all remaining migrations, leaving the schema short of
    # columns the code expects.
    applied = 0
    for statement in _MIGRATIONS:
        try:
            with bound_engine.begin() as conn:
                conn.execute(text(statement))
            applied += 1
        except Exception:
            # A duplicate-column error simply means this migration already ran.
            continue
    if applied:
        log.info("Applied %d additive schema migration(s)", applied)

    # Newly introduced tables (checkpoints, analysis_sessions) are handled by
    # create_all above; this call makes the intent explicit for older images.
    from core import models  # noqa: F401

    Base.metadata.create_all(bind=bound_engine)
    log.debug("Schema compatibility pass complete")
