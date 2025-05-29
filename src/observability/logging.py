"""Structured logging for hal_nemoFinder.

This module centralises log configuration using :mod:`structlog`. The
same pipeline is used by the FastAPI application, Celery workers, and
CLI — so every log line, regardless of origin, comes out in the same
format and carries the same context keys (``request_id``, ``trace_id``,
``job_id``, ``service``, ``version``, ``environment``).

Usage
-----

.. code-block:: python

    from src.observability.logging import configure_logging, get_logger

    configure_logging(settings)          # once at startup
    log = get_logger(__name__)
    log.info("job.created", job_id=str(job.id))

Existing stdlib ``logger.info("msg %s", x)`` calls keep working because
the stdlib root logger is wired to emit through the same structlog
processors.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_request_context",
    "clear_request_context",
]


# ---------------------------------------------------------------------------
# Context vars — available even when structlog is not configured yet.
# ---------------------------------------------------------------------------

#: Per-request identifier, bound by the FastAPI middleware.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_CONFIGURED: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(settings: Any | None = None) -> None:
    """Configure :mod:`structlog` and the stdlib logging pipeline.

    Parameters
    ----------
    settings
        Application settings object.  Only attributes used here are
        ``LOG_LEVEL``, ``LOG_FORMAT``, ``ENVIRONMENT`` and ``VERSION``;
        each is also read from the environment if the settings object
        does not define it, so it's safe to call this with ``None`` from
        workers that do not have settings imported.

    The function is idempotent — calling it twice will not double-wrap
    handlers.
    """
    global _CONFIGURED

    log_level = _get_setting(settings, "LOG_LEVEL", "INFO").upper()
    log_format = _get_setting(settings, "LOG_FORMAT", "console").lower()
    environment = _get_setting(settings, "ENVIRONMENT", "development")
    version = _get_setting(settings, "VERSION", "0.1.0")
    service = "hal_nemofinder"

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
        _bind_service_fields(service=service, version=version, environment=environment),
    ]

    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # ---- Configure structlog -----------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level) if isinstance(log_level, str) else log_level
        ),
        cache_logger_on_first_use=True,
    )

    # ---- Configure stdlib root logger so existing logging.getLogger() works
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Replace existing handlers to avoid duplicate output when called
    # twice (e.g. uvicorn reload).
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Tame chatty loggers — uvicorn's access log already covers requests.
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(max(logging.getLevelName(log_level), logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a :class:`structlog.BoundLogger` for *name*.

    If :func:`configure_logging` has not been called yet, this will
    still return a usable logger — structlog falls back to a default
    configuration, which is the right behaviour for tests and for
    library code imported before ``configure_logging`` runs.
    """
    if not _CONFIGURED:
        # Best-effort lazy configuration so imports don't blow up the
        # test suite.
        try:
            configure_logging(None)
        except Exception:  # pragma: no cover — belt & braces
            pass
    return structlog.stdlib.get_logger(name)  # type: ignore[return-value]


@contextmanager
def bind_request_context(**kwargs: Any) -> Iterator[None]:
    """Bind request-scoped fields for the duration of the ``with`` block.

    Example
    -------
    .. code-block:: python

        with bind_request_context(request_id=rid, path=request.url.path):
            response = await call_next(request)
    """
    keys = tuple(kwargs.keys())
    token = None
    if "request_id" in kwargs:
        token = request_id_var.set(kwargs["request_id"])
    bind_contextvars(**kwargs)
    try:
        yield
    finally:
        unbind_contextvars(*keys)
        if token is not None:
            request_id_var.reset(token)


def clear_request_context() -> None:
    """Clear all contextvars bound via structlog."""
    clear_contextvars()
    request_id_var.set(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_setting(settings: Any | None, name: str, default: str) -> str:
    """Fetch a setting from *settings*, falling back to env, then default."""
    if settings is not None and hasattr(settings, name):
        val = getattr(settings, name)
        if val is not None:
            return str(val)
    return os.environ.get(name, os.environ.get(f"HAL_{name}", default))


def _bind_service_fields(service: str, version: str, environment: str) -> Any:
    """Return a structlog processor that stamps every event with service metadata."""

    def processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        event_dict.setdefault("service", service)
        event_dict.setdefault("version", version)
        event_dict.setdefault("environment", environment)
        return event_dict

    return processor
