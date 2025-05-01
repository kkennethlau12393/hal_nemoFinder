"""Top-level API router — aggregates all endpoint routers under /api/v1."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.endpoints import (
    analyze,
    auth,
    batches,
    claims,
    health,
    jobs,
    stats,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["System"])
api_router.include_router(analyze.router, tags=["Analysis"])
api_router.include_router(jobs.router, tags=["Jobs"])
api_router.include_router(batches.router, tags=["Jobs"])
api_router.include_router(claims.router, tags=["Claims"])
api_router.include_router(stats.router, tags=["System"])
api_router.include_router(auth.router)
