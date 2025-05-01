"""FastAPI application entry point for hal_nemoFinder."""

from __future__ import annotations

import time
import uuid

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.api.middleware.tenant import TenantContextMiddleware
from src.api.rate_limit import RateLimitMiddleware
from src.api.router import api_router
from src.audit.middleware import AuditLogMiddleware
from src.config import settings
from src.db.init_db import init_db
from src.observability import (
    bind_request_context,
    configure_logging,
    generate_metrics,
    get_logger,
    track_api_request,
    update_verifier_count,
)
from src.observability.metrics import METRICS_CONTENT_TYPE

# Configure structured logging as early as possible so imports and
# uvicorn's own log lines flow through the same pipeline.
configure_logging(settings)
logger = get_logger("hal_nemofinder")


# ---------------------------------------------------------------------------
# Tag metadata for Swagger UI grouping
# ---------------------------------------------------------------------------

TAGS_METADATA = [
    {
        "name": "Analysis",
        "description": (
            "Submit AI-generated drug-discovery text for hallucination analysis. "
            "The system extracts factual claims, verifies each against authoritative "
            "knowledge bases (ChEMBL, PubChem, UniProt, CrossRef), and produces "
            "a comprehensive report."
        ),
    },
    {
        "name": "Jobs",
        "description": (
            "Track the progress of analysis jobs. Poll for status, retrieve "
            "extracted claims, and download the final hallucination report."
        ),
    },
    {
        "name": "Claims",
        "description": (
            "Inspect individual factual claims and their verification evidence. "
            "Each claim is evaluated by one or more specialized verifiers that "
            "query authoritative scientific databases."
        ),
    },
    {
        "name": "System",
        "description": (
            "Operational endpoints for monitoring system health, connectivity "
            "to backing services, and aggregate usage statistics."
        ),
    },
]

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="hal_nemoFinder",
    description=(
        "**Hallucination Detection Framework for AI-Driven Drug Discovery**\n\n"
        "hal_nemoFinder automatically detects factual hallucinations in "
        "AI-generated pharmaceutical and biomedical text.  It:\n\n"
        "1. **Extracts** verifiable factual claims from free-text input\n"
        "2. **Classifies** each claim by type (molecular property, target "
        "interaction, clinical outcome, etc.)\n"
        "3. **Verifies** claims against authoritative knowledge bases "
        "(ChEMBL, PubChem, UniProt, CrossRef)\n"
        "4. **Aggregates** results into an overall hallucination severity "
        "score with per-claim evidence trails\n\n"
        "---\n\n"
        "**Quick start:** `POST /api/v1/analyze` with your text, then poll "
        "`GET /api/v1/jobs/{job_id}` until status is `completed`, and "
        "retrieve the report via `GET /api/v1/jobs/{job_id}/report`."
    ),
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RateLimitMiddleware,
    enabled=settings.RATE_LIMIT_ENABLED,
    rpm=settings.RATE_LIMIT_RPM,
    analyze_pm=settings.RATE_LIMIT_ANALYZE_PM,
    redis_url=settings.REDIS_URL,
)

# Audit logging middleware — records an entry for every authenticated
# state-changing request.  Failures are logged at CRITICAL level but
# never break the user-visible response.
app.add_middleware(AuditLogMiddleware)

# TenantContextMiddleware resolves X-API-Key / Bearer tokens once per
# request and stashes the resulting AuthContext on ``request.state.auth``.
# It must run *before* the audit middleware so audit records carry the
# resolved tenant.
app.add_middleware(TenantContextMiddleware)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Generate a request_id, bind contextvars, and record metrics.

    Combines three concerns that all need the same request-scoped data:

    - A UUID4 ``request_id`` bound into structlog's contextvars so every
      log line in the request carries it.
    - An ``X-Request-ID`` response header so clients can correlate.
    - Prometheus counters + histogram for the API itself.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    with bind_request_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    ):
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "request.error",
                method=request.method,
                path=request.url.path,
                duration_ms=round(elapsed * 1000, 2),
            )
            track_api_request(request.method, request.url.path, 500, elapsed)
            raise

        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id

        # Record API metrics.  Use the matched route template when
        # available to avoid a cardinality explosion from UUID paths.
        route_template = request.url.path
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            route_template = route.path
        track_api_request(
            request.method, route_template, response.status_code, elapsed
        )

        logger.info(
            "request.completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(api_router)


@app.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus metrics",
)
async def metrics() -> Response:
    """Expose the default Prometheus registry in text format."""
    return Response(content=generate_metrics(), media_type=METRICS_CONTENT_TYPE)


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise database tables and publish baseline gauges."""
    logger.info("startup.begin", version=app.version)
    await init_db()

    # ---- Audit log integrity check -----------------------------------
    if settings.AUDIT_ENABLED and settings.AUDIT_REQUIRE_INTEGRITY:
        try:
            from src.audit.recorder import get_audit_recorder
            from src.db.session import AsyncSessionLocal

            recorder = get_audit_recorder()
            async with AsyncSessionLocal() as session:
                report = await recorder.verify_integrity(session)
            if not report.valid:
                logger.critical(
                    "startup.audit.integrity_broken",
                    first_invalid=report.first_invalid_sequence,
                    errors=report.errors[:5],
                    total_checked=report.total_checked,
                )
                raise RuntimeError(
                    "Audit log integrity check failed — refusing to start. "
                    f"First invalid sequence: {report.first_invalid_sequence}"
                )
            logger.info(
                "startup.audit.integrity_ok",
                total_checked=report.total_checked,
            )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("startup.audit.integrity_check_failed")
            raise

    # Publish baseline gauges so /metrics is meaningful immediately.
    try:
        from src.verifiers.base import get_verifier_registry

        update_verifier_count(len(get_verifier_registry().get_all()))
    except Exception:
        logger.exception("startup.verifier_count.failed")

    logger.info("startup.complete")


# ---------------------------------------------------------------------------
# Custom OpenAPI schema
# ---------------------------------------------------------------------------


def custom_openapi():
    """Generate a customised OpenAPI schema with rich metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=TAGS_METADATA,
    )

    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/hal-nemofinder/assets/main/logo.png",
    }
    openapi_schema["info"]["contact"] = {
        "name": "hal_nemoFinder Contributors",
        "url": "https://github.com/hal-nemofinder/hal_nemoFinder",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the application via Uvicorn (used by the console_scripts entry point)."""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run()
