"""Database connection management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from keiba.config import DATABASE_URL
from keiba.db.schema import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(get_engine())
