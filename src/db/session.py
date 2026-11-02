"""SQLAlchemy engine configuration for database commands."""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine


def database_url_from_env() -> str:
    """Return the configured database URL or fail with an actionable error."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("La variable de entorno DATABASE_URL no está definida")
    return database_url


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create a synchronous SQLAlchemy engine without opening a connection."""
    return create_engine(
        database_url or database_url_from_env(),
        pool_pre_ping=True,
    )

