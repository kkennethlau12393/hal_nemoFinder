"""Async and sync SQLAlchemy engine/session factories."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings

# -- Async (FastAPI) -----------------------------------------------------------

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        yield session


# -- Sync (Celery / Alembic) --------------------------------------------------

_sync_engine: Engine | None = None


def get_sync_engine() -> Engine:
    """Return a lazily-created synchronous engine for Celery tasks."""
    global _sync_engine  # noqa: PLW0603
    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.SYNC_DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
    return _sync_engine
