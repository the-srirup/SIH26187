"""
SQLite database setup.

SQLite is chosen because it is a single file with zero server process —
ideal for edge deployment at a Border Out Post.  Alert metadata can
later be replicated to PostgreSQL at sector headquarters; the schema
itself stays identical so the migration is trivial.
"""
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base shared by every model."""
    pass


def get_db():
    """FastAPI dependency that yields a DB session and closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables from models (if they don't already exist)."""
    from core import models  # noqa: F401  — imported for side-effects
    Base.metadata.create_all(bind=engine)
    _ensure_schema_compatibility(engine)


def _ensure_schema_compatibility(engine) -> None:
    """
    Apply tiny, safe SQLite migration shims for projects that were created
    before newer columns/tables were introduced.  A real production
    deployment should use Alembic; this function keeps local/hackathon
    databases usable after source upgrades without forcing a reset.
    """
    if not engine.url.get_backend_name() == "sqlite":
        return

    statements = [
        "ALTER TABLE cameras ADD COLUMN is_online BOOLEAN DEFAULT 1",
    ]

    with engine.begin() as conn:
        for statement in statements:
            try:
                conn.execute(text(statement))
            except Exception:
                # Column already exists. SQLite does not support IF NOT EXISTS
                # on ALTER TABLE ADD COLUMN, so ignore the duplicate-column error.
                pass
        # Create newly introduced watchlist table if it was not included in
        # an older database image. create_all above normally handles this, but
        # this explicit call makes the intent clear.
        from core import models  # noqa: F401
        models.Base.metadata.create_all(bind=engine)
