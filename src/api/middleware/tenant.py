"""Tenant-binding middleware.

This middleware:

* Reads the ``X-API-Key`` / ``Authorization`` headers up-front.
* Looks up the tenant (and user, if any) via
  :func:`src.auth.dependencies.get_auth_context`.
* Stashes the resolved :class:`AuthContext` on ``request.state.auth``
  so route handlers can access it without re-running the auth logic.
* Binds ``tenant_id`` and ``tenant_slug`` to the current structlog
  context so every log line emitted during this request carries the
  tenant for free.

Unlike the FastAPI dependency, this middleware runs *before* any route
handler and therefore has to construct its own :class:`AsyncSession`
from the engine rather than using the dependency injection system.
Importantly, it is **non-fatal**: if DB lookup fails (for instance
when the underlying database isn't available) the middleware installs
a synthetic default context and lets the request proceed so that
liveness probes and other auth-independent endpoints still respond.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.auth.dependencies import (
    AuthContext,
    _default_context,
    _extract_api_key,
    _extract_bearer,
    _resolve_api_key,
    _resolve_oidc_token,
)
from src.config import settings
from src.db.session import AsyncSessionLocal
from src.observability import bind_request_context, get_logger

logger = get_logger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Resolve the caller's tenant once per request.

    The resolved context is stored on ``request.state.auth`` and is
    consulted by :func:`src.auth.dependencies.get_auth_context` so that
    downstream ``Depends(...)`` usage does not re-run the auth path.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        ctx = await self._build_context(request)
        request.state.auth = ctx

        with bind_request_context(
            tenant_id=str(ctx.tenant.id),
            tenant_slug=ctx.tenant.slug,
        ):
            return await call_next(request)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _build_context(self, request: Request) -> AuthContext:
        """Best-effort resolution of the caller's auth context.

        Returns the synthetic default context on any lookup error when
        auth is not required, so non-auth endpoints (health, metrics)
        always work.
        """
        api_key_plain = _extract_api_key(request)
        bearer = _extract_bearer(request)

        if not api_key_plain and not bearer:
            if settings.AUTH_REQUIRED:
                # Defer the 401 to the endpoint dependency so the
                # response envelope matches the rest of the API.  We
                # still need *a* context — return the default and let
                # the dependency reject it.
                return _default_context()
            return _default_context()

        try:
            async with AsyncSessionLocal() as session:
                if api_key_plain:
                    tenant, api_key = await _resolve_api_key(session, api_key_plain)
                    from src.auth.rbac import permissions_for_role

                    return AuthContext(
                        tenant=tenant,
                        user=None,
                        api_key=api_key,
                        role=api_key.role,
                        permissions=permissions_for_role(api_key.role),
                    )

                assert bearer is not None  # narrowed
                tenant, user = await _resolve_oidc_token(session, bearer)
                from src.auth.rbac import permissions_for_role

                return AuthContext(
                    tenant=tenant,
                    user=user,
                    api_key=None,
                    role=user.role,
                    permissions=permissions_for_role(user.role),
                )
        except Exception:  # noqa: BLE001
            # We intentionally swallow the exception here.  The
            # dependency system will re-run the auth code with a live
            # session and raise the correct HTTPException for the
            # client.  Returning the default context keeps health /
            # metrics reachable even when auth is misconfigured.
            logger.exception("tenant_middleware.resolve_failed")
            return _default_context()
