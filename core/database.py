"""
SQLite database setup.

SQLite is chosen because it is a single file with zero server process —
ideal for edge deployment at a Border Out Post.  Alert metadata can
later be replicated to PostgreSQL at sector headquarters; the schema
itself stays identical so the migration is trivial.
"""
from sqlalchemy import create_engine
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
