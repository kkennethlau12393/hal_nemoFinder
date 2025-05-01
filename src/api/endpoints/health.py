"""Health, liveness, and readiness endpoints.

Provides three entry points:

- ``GET /health`` — comprehensive, fan-out status of every dependency
  plus plugin load and calibration state.  Meant for human operators.
- ``GET /health/live`` — cheap liveness probe for Kubernetes.
- ``GET /health/ready`` — readiness probe; returns 503 on any failure.
"""

from __future__ import annotations

from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    CalibrationStatus,
    HealthResponse,
    LivenessResponse,
    PluginStatus,
    ReadinessResponse,
)
from src.config import settings
from src.db.session import get_session
from src.observability import get_logger
from src.observability.metrics import (
    CALIBRATION_ACCURACY,
    CALIBRATION_BRIER_SCORE,
    CALIBRATION_ECE,
    PLUGINS_LOADED,
    VERIFIERS_REGISTERED,
)
from src.verifiers.base import get_verifier_registry

logger = get_logger(__name__)

router = APIRouter()

#: Per-process timestamp of the most recent calibration run.  Updated
#: by the CLI via :func:`record_calibration_run`.
_LAST_CALIBRATION_RUN: datetime | None = None


def record_calibration_run(when: datetime | None = None) -> None:
    """Record that a calibration run happened, for ``/health`` surfacing."""
    global _LAST_CALIBRATION_RUN
    _LAST_CALIBRATION_RUN = when or datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Dependency probes
# ---------------------------------------------------------------------------


async def _check_database(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("health.database.failed")
        return False


async def _check_redis() -> bool:
    try:
        client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            return bool(await client.ping())
        finally:
            await client.aclose()
    except Exception:
        logger.exception("health.redis.failed")
        return False


async def _check_verifiers() -> dict[str, bool]:
    registry = get_verifier_registry()
    results: dict[str, bool] = {}
    for verifier in registry.get_all():
        try:
            results[verifier.name] = bool(await verifier.health_check())
        except Exception:
            logger.exception("health.verifier.failed", verifier=verifier.name)
            results[verifier.name] = False
    return results


def _gauge_value(gauge) -> float | None:
    """Return the current value of a single-sample :class:`Gauge`, if set."""
    try:
        val = gauge._value.get()  # type: ignore[attr-defined]
    except Exception:
        return None
    if val == 0.0:
        # ``0.0`` is a legitimate reading but also the default for a
        # gauge that has never been set.  We don't currently distinguish
        # the two — report None so operators aren't misled.
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Kubernetes liveness probe",
    description=(
        "Cheap liveness probe — returns 200 OK as long as the event loop "
        "is responding.  Kubernetes uses this to decide whether to "
        "restart the pod."
    ),
    tags=["System"],
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Kubernetes readiness probe",
    description=(
        "Checks every critical dependency: PostgreSQL, Redis, and that "
        "at least one verifier is registered.  Returns **503** if any "
        "check fails so the pod is removed from the Service's endpoint "
        "list until it recovers."
    ),
    tags=["System"],
    responses={503: {"description": "One or more dependencies are unavailable"}},
)
async def readiness(
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> ReadinessResponse:
    db_ok = await _check_database(session)
    redis_ok = await _check_redis()
    registry = get_verifier_registry()
    verifier_count = len(registry.get_all())

    failures: list[str] = []
    if not db_ok:
        failures.append("database")
    if not redis_ok:
        failures.append("redis")
    if verifier_count == 0:
        failures.append("verifiers")

    ready = not failures
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready,
        database=db_ok,
        redis=redis_ok,
        verifiers_registered=verifier_count,
        failures=failures,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=(
        "Verify that core infrastructure components are reachable and "
        "report plugin and calibration status."
    ),
    tags=["System"],
)
async def health(
    session: AsyncSession = Depends(get_session),
) -> HealthResponse:
    db_ok = await _check_database(session)
    redis_ok = await _check_redis()
    verifier_status = await _check_verifiers()

    plugins_loaded = int(_gauge_value(PLUGINS_LOADED) or 0)
    # Plugin error surfacing — pull from the plugin loader if available.
    plugin_errors: list[str] = []
    try:
        from src.plugins import get_last_plugin_report  # type: ignore

        report = get_last_plugin_report()
        if report is not None:
            plugin_errors = [f"{src}: {msg}" for src, msg in getattr(report, "errors", [])]
    except Exception:
        # Older plugin loader without accessor; tolerate silently.
        pass

    calibration = CalibrationStatus(
        last_run=_LAST_CALIBRATION_RUN,
        accuracy=_gauge_value(CALIBRATION_ACCURACY),
        brier_score=_gauge_value(CALIBRATION_BRIER_SCORE),
        ece=_gauge_value(CALIBRATION_ECE),
    )

    all_ok = db_ok and redis_ok and all(verifier_status.values())

    # Refresh the verifier-gauge for consistency with /metrics.
    try:
        VERIFIERS_REGISTERED.set(len(verifier_status))
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version="0.1.0",
        environment=str(getattr(settings, "ENVIRONMENT", "development")),
        database=db_ok,
        redis=redis_ok,
        verifiers=verifier_status,
        plugins=PluginStatus(loaded=plugins_loaded, errors=plugin_errors),
        calibration=calibration,
    )
