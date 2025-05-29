"""Observability primitives for hal_nemoFinder.

Exposes structured logging configuration and Prometheus metrics
helpers used across the FastAPI app, Celery workers, and CLI.
"""

from src.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from src.observability.metrics import (
    CLAIMS_TOTAL,
    JOB_DURATION_SECONDS,
    JOBS_TOTAL,
    METRICS_CONTENT_TYPE,
    VERIFIER_DURATION_SECONDS,
    VERIFIER_RUNS_TOTAL,
    generate_metrics,
    track_api_request,
    track_job,
    track_verifier_run,
    update_calibration_metrics,
    update_plugin_counts,
    update_verifier_count,
)

__all__ = [
    # Logging
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    # Metrics
    "CLAIMS_TOTAL",
    "JOB_DURATION_SECONDS",
    "JOBS_TOTAL",
    "METRICS_CONTENT_TYPE",
    "VERIFIER_DURATION_SECONDS",
    "VERIFIER_RUNS_TOTAL",
    "generate_metrics",
    "track_api_request",
    "track_job",
    "track_verifier_run",
    "update_calibration_metrics",
    "update_plugin_counts",
    "update_verifier_count",
]
