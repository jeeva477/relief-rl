"""SQLAlchemy database setup for persistent hazard storage."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def create_engine_and_session(database_url: str) -> tuple[Engine, sessionmaker]:
    """Create an engine and session factory for a configured database."""
    if not database_url:
        raise ValueError("DATABASE_URL is required to initialize the database")

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    return engine, factory


def create_session_factory(database_url: str):
    """Backward-compatible helper returning only the session factory."""
    if not database_url:
        return None
    _, factory = create_engine_and_session(database_url)
    return factory
