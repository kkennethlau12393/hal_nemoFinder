"""Celery application configuration for hal_nemoFinder.

Creates the Celery app instance, configures serialization, task routing,
time limits, and autodiscovery for the task modules.
"""

from __future__ import annotations

from celery import Celery

from src.config import settings

celery_app = Celery(
    "hal_nemofinder",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # -- Serialization ---------------------------------------------------------
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # -- Reliability -----------------------------------------------------------
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # -- Result storage --------------------------------------------------------
    result_expires=86400,  # 24 hours
    # -- Time limits -----------------------------------------------------------
    task_soft_time_limit=300,   # 5 minutes soft limit (raises SoftTimeLimitExceeded)
    task_hard_time_limit=360,   # 6 minutes hard limit (kills the worker)
    # -- Task routing ----------------------------------------------------------
    task_routes={
        "src.tasks.verification.verify_claim_task": {"queue": "verification"},
        "src.tasks.callbacks.aggregate_results_task": {"queue": "analysis"},
        "src.tasks.verification.run_analysis_task": {"queue": "analysis"},
        "src.tasks.webhooks.send_webhook_notification": {"queue": "notifications"},
    },
    # -- Retry / backoff defaults ----------------------------------------------
    task_default_retry_delay=10,
    task_max_retries=3,
    # -- Timezone --------------------------------------------------------------
    timezone="UTC",
    enable_utc=True,
)

# Autodiscover tasks inside the src.tasks package.
celery_app.autodiscover_tasks(["src.tasks"])

# Trigger verifier registration so the registry is populated when
# Celery workers import this module.
import src.verifiers  # noqa: E402, F401

# Load customer-provided plugins so workers have the same registry
# state as the API server.  Any errors are logged but never crash the
# worker boot sequence.
import logging  # noqa: E402

from src.plugins import load_plugins_from_settings  # noqa: E402

_plugin_logger = logging.getLogger("hal_nemofinder.celery")

try:
    _plugin_report = load_plugins_from_settings(settings)
    _plugin_logger.info(
        "Celery worker loaded %d custom verifiers, %d custom KB clients from plugins",
        len(_plugin_report.verifiers_loaded),
        len(_plugin_report.kb_clients_loaded),
    )
    if _plugin_report.errors:
        _plugin_logger.error(
            "Celery plugin load finished with %d error(s):",
            len(_plugin_report.errors),
        )
        for _source, _msg in _plugin_report.errors:
            _plugin_logger.error("  [%s] %s", _source, _msg)
except Exception:  # noqa: BLE001
    _plugin_logger.exception("Unexpected error while loading plugins in Celery worker")
