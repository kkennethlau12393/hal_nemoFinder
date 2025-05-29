"""Prometheus metrics for hal_nemoFinder.

All metrics are registered on the default registry so
``prometheus_client.generate_latest()`` will pick them up without any
extra plumbing.  The module exposes typed helper functions that the
rest of the codebase can call without having to know anything about
prometheus primitives.
"""

from __future__ import annotations

from typing import Any, Mapping

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = [
    "METRICS_CONTENT_TYPE",
    "CLAIMS_TOTAL",
    "VERIFIER_RUNS_TOTAL",
    "JOBS_TOTAL",
    "API_REQUESTS_TOTAL",
    "VERIFIER_DURATION_SECONDS",
    "JOB_DURATION_SECONDS",
    "API_REQUEST_DURATION_SECONDS",
    "VERIFIERS_REGISTERED",
    "PLUGINS_LOADED",
    "CALIBRATION_ACCURACY",
    "CALIBRATION_BRIER_SCORE",
    "CALIBRATION_ECE",
    "VERIFIER_DRIFT_SCORE",
    "SHADOW_DISAGREEMENTS_TOTAL",
    "COST_USD_TOTAL",
    "API_CALLS_TOTAL",
    "track_verifier_run",
    "track_api_request",
    "track_job",
    "update_calibration_metrics",
    "update_verifier_count",
    "update_plugin_counts",
    "generate_metrics",
]

METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

CLAIMS_TOTAL = Counter(
    "hal_nemofinder_claims_total",
    "Total number of claims analyzed, labelled by verdict and claim type.",
    ("verdict", "claim_type"),
)

VERIFIER_RUNS_TOTAL = Counter(
    "hal_nemofinder_verifier_runs_total",
    "Total number of verifier invocations, labelled by verifier, verdict, and outcome.",
    ("verifier", "verdict", "outcome"),
)

JOBS_TOTAL = Counter(
    "hal_nemofinder_jobs_total",
    "Total number of analysis jobs that reached a terminal state, by status.",
    ("status",),
)

API_REQUESTS_TOTAL = Counter(
    "hal_nemofinder_api_requests_total",
    "Total number of HTTP API requests, labelled by method, path, and status.",
    ("method", "path", "status"),
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

VERIFIER_DURATION_SECONDS = Histogram(
    "hal_nemofinder_verifier_duration_seconds",
    "Wall-clock duration of a verifier invocation, in seconds.",
    ("verifier",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

JOB_DURATION_SECONDS = Histogram(
    "hal_nemofinder_job_duration_seconds",
    "End-to-end analysis job duration, in seconds, bucketed by dominant claim type.",
    ("claim_type_bucket",),
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0),
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "hal_nemofinder_api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

VERIFIERS_REGISTERED = Gauge(
    "hal_nemofinder_verifiers_registered",
    "Number of verifier backends currently registered with the process.",
)

PLUGINS_LOADED = Gauge(
    "hal_nemofinder_plugins_loaded",
    "Number of third-party plugins currently loaded.",
)

CALIBRATION_ACCURACY = Gauge(
    "hal_nemofinder_calibration_accuracy",
    "Accuracy reported by the most recent regression / calibration run.",
)

CALIBRATION_BRIER_SCORE = Gauge(
    "hal_nemofinder_calibration_brier_score",
    "Brier score reported by the most recent regression / calibration run.",
)

CALIBRATION_ECE = Gauge(
    "hal_nemofinder_calibration_ece",
    "Expected Calibration Error reported by the most recent regression run.",
)

VERIFIER_DRIFT_SCORE = Gauge(
    "hal_nemofinder_verifier_drift_score",
    "Absolute accuracy delta between window and baseline for each verifier.",
    ("verifier",),
)

SHADOW_DISAGREEMENTS_TOTAL = Counter(
    "hal_nemofinder_shadow_disagreements_total",
    "Total number of disagreements between a champion and challenger verifier.",
    ("champion", "challenger"),
)

COST_USD_TOTAL = Counter(
    "hal_nemofinder_cost_usd_total",
    "Accumulated cost in USD, labelled by category (compute/api/storage).",
    ("category",),
)

API_CALLS_TOTAL = Counter(
    "hal_nemofinder_api_calls_total",
    "Total external API/KB client calls, labelled by client name.",
    ("client",),
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def track_verifier_run(
    verifier_name: str,
    verdict: str,
    duration: float,
    outcome: str = "ok",
) -> None:
    """Record a single verifier invocation.

    Parameters
    ----------
    verifier_name
        Canonical verifier name (e.g. ``"pubchem"``).
    verdict
        String verdict, typically from :class:`~src.models.enums.Verdict`.
    duration
        Wall-clock duration of the call, in seconds.
    outcome
        One of ``"ok"``, ``"error"``, ``"timeout"``.
    """
    VERIFIER_RUNS_TOTAL.labels(
        verifier=verifier_name,
        verdict=verdict,
        outcome=outcome,
    ).inc()
    VERIFIER_DURATION_SECONDS.labels(verifier=verifier_name).observe(max(duration, 0.0))


def track_api_request(method: str, path: str, status: int, duration: float) -> None:
    """Record one HTTP request's aggregate metrics."""
    API_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
    API_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(max(duration, 0.0))


def track_job(status: str, duration: float, claim_type: str = "mixed") -> None:
    """Record a terminal-state analysis job."""
    JOBS_TOTAL.labels(status=status).inc()
    JOB_DURATION_SECONDS.labels(claim_type_bucket=claim_type).observe(max(duration, 0.0))


def update_calibration_metrics(metrics: Mapping[str, Any]) -> None:
    """Publish the latest calibration / regression results as gauges.

    *metrics* is a mapping that may contain any of ``accuracy``,
    ``brier_score``, ``ece``.  Unknown keys are silently ignored so the
    CLI can pass its full metrics dict without having to filter.
    """
    if "accuracy" in metrics and metrics["accuracy"] is not None:
        CALIBRATION_ACCURACY.set(float(metrics["accuracy"]))
    if "brier_score" in metrics and metrics["brier_score"] is not None:
        CALIBRATION_BRIER_SCORE.set(float(metrics["brier_score"]))
    if "ece" in metrics and metrics["ece"] is not None:
        CALIBRATION_ECE.set(float(metrics["ece"]))


def update_verifier_count(count: int) -> None:
    """Publish the live verifier-registry size."""
    VERIFIERS_REGISTERED.set(max(count, 0))


def update_plugin_counts(count: int) -> None:
    """Publish the number of plugins successfully loaded."""
    PLUGINS_LOADED.set(max(count, 0))


def generate_metrics() -> bytes:
    """Render the current metrics snapshot in Prometheus text format."""
    return generate_latest()
