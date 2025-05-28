"""ASGI middleware that records an audit entry for state-changing requests.

This middleware is intentionally permissive — an audit-write failure
must never break a user-facing request.  Instead, it logs a critical
message so operators are paged, and lets the response flow through.

Auth context (tenant_id, user_id, api_key_id) is expected to have been
populated on ``request.state.auth`` by the auth agent's middleware
earlier in the chain.  If ``request.state.auth`` is missing we fall
back to unauthenticated metadata and skip the record.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.audit.recorder import AuditRecorder, get_audit_recorder
from src.models.audit import AuditEventType
from src.observability import get_logger

logger = get_logger(__name__)

_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Paths whose audit entries would be noisy (healthchecks, metrics).
_IGNORED_PATHS = frozenset({"/health", "/healthz", "/metrics", "/ready"})


def _extract_ip(request: Request) -> str | None:
    """Return the effective client IP, honouring ``X-Forwarded-For``."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        # Take the first address in the list (the original client).
        return xff.split(",")[0].strip() or None
    if request.client is not None:
        return request.client.host
    return None


def _extract_auth(request: Request) -> dict[str, Any]:
    """Extract the actor triple from ``request.state.auth``.

    The auth agent populates ``request.state.auth`` as an arbitrary
    object (dataclass / dict / pydantic model).  This helper tolerates
    any of those shapes.
    """
    auth = getattr(request.state, "auth", None)
    if auth is None:
        return {}
    if isinstance(auth, dict):
        return {
            "user_id": auth.get("user_id"),
            "api_key_id": auth.get("api_key_id"),
            "tenant_id": auth.get("tenant_id"),
        }
    return {
        "user_id": getattr(auth, "user_id", None),
        "api_key_id": getattr(auth, "api_key_id", None),
        "tenant_id": getattr(auth, "tenant_id", None),
    }


def _event_type_for(request: Request) -> AuditEventType:
    """Best-effort mapping from request path to a canonical event type."""
    path = request.url.path
    method = request.method
    if "/analyze" in path and method == "POST":
        return AuditEventType.job_submitted
    if "/calibrate" in path:
        return AuditEventType.calibration_run
    return AuditEventType.config_changed


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Record an audit entry for every authenticated state-changing request.

    Reads/writes to the audit log happen on a background task so the
    request's response latency is not affected.  Failures during the
    background write are logged at CRITICAL severity — operators
    should alert on this log channel.
    """

    def __init__(self, app: Any, recorder: AuditRecorder | None = None) -> None:
        super().__init__(app)
        self._recorder = recorder

    @property
    def recorder(self) -> AuditRecorder | None:
        if self._recorder is None:
            try:
                self._recorder = get_audit_recorder()
            except Exception as exc:  # noqa: BLE001
                logger.critical(
                    "audit.middleware.recorder_init_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return None
        return self._recorder

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)

        if request.method not in _STATE_CHANGING_METHODS:
            return response
        if request.url.path in _IGNORED_PATHS:
            return response
        recorder = self.recorder
        if recorder is None or not recorder.enabled:
            return response

        auth = _extract_auth(request)
        # If there's no authenticated actor at all, skip the audit
        # write — anonymous requests to state-changing endpoints are
        # already rejected by the auth layer.
        if not any(auth.get(k) for k in ("user_id", "api_key_id")):
            return response

        # Fire-and-forget the audit write on a background task so the
        # user's response goes out immediately.
        asyncio.create_task(
            self._record_safely(
                recorder=recorder,
                event_type=_event_type_for(request),
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                tenant_id=auth.get("tenant_id"),
                actor={
                    "user_id": auth.get("user_id"),
                    "api_key_id": auth.get("api_key_id"),
                    "ip": _extract_ip(request),
                },
            )
        )
        return response

    async def _record_safely(
        self,
        *,
        recorder: AuditRecorder,
        event_type: AuditEventType,
        method: str,
        path: str,
        status_code: int,
        tenant_id: Any,
        actor: dict[str, Any],
    ) -> None:
        """Write the audit entry, swallowing any exception."""
        # Local import to avoid importing DB code at module load time,
        # which is important for the test suite (which may monkeypatch
        # the session factory before the middleware is imported).
        try:
            from src.db.session import AsyncSessionLocal
        except Exception:
            logger.critical("audit.middleware.session_factory_missing")
            return

        outcome = "success" if 200 <= status_code < 400 else "failure"
        try:
            async with AsyncSessionLocal() as session:
                await recorder.record(
                    event_type=event_type,
                    resource_type="api_request",
                    resource_id=f"{method} {path}",
                    action=method.lower(),
                    outcome=outcome,
                    session=session,
                    actor=actor,
                    tenant_id=tenant_id,
                    payload={
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                    },
                    commit=True,
                )
        except Exception as exc:  # noqa: BLE001
            # CRITICAL so operators page on this — audit failures are
            # compliance-significant even though we don't propagate
            # them to the API client.
            logger.critical(
                "audit.middleware.write_failed",
                method=method,
                path=path,
                error=str(exc),
                error_type=type(exc).__name__,
            )
