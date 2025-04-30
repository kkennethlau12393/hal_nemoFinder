"""Database initialisation helpers — pgvector extension and table creation."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from src.db.base import Base  # noqa: F401 — ensure all models are imported
from src.db.session import async_engine


async def create_pgvector_extension(conn: AsyncConnection) -> None:
    """Enable the pgvector extension if it is not already present."""
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def create_tables(conn: AsyncConnection) -> None:
    """Create all tables declared on ``Base.metadata``."""
    await conn.run_sync(Base.metadata.create_all)


async def init_db() -> None:
    """Run full database bootstrap: extension + tables."""
    async with async_engine.begin() as conn:
        await create_pgvector_extension(conn)
        await create_tables(conn)
