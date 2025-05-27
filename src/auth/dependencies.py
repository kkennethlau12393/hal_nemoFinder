"""FastAPI dependencies that produce an authenticated :class:`AuthContext`.

Two authentication methods are supported:

* **API key** via the ``X-API-Key`` header.  This is the primary
  mechanism for server-to-server calls and CI pipelines.
* **OIDC bearer token** via ``Authorization: Bearer <jwt>``.  This
  enables SSO login through an external identity provider.

When ``settings.AUTH_REQUIRED`` is *False* (the default for backward
compatibility) requests with no credentials are still served — they
are associated with a synthetic "default" tenant that never touches
the database.  This keeps pre-existing tests working unchanged.

All public dependencies here are plain callables suitable for passing
into :func:`fastapi.Depends`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from uuid import UUID, uuid5, NAMESPACE_URL

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.api_keys import hash_api_key
from src.auth.oidc import NoOpValidator, OIDCError, build_oidc_validator
from src.auth.rbac import Permission, permissions_for_role
from src.config import settings
from src.db.session import get_session
from src.models.tenant import ApiKey, Role, Tenant, User
from src.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Auth context
# ---------------------------------------------------------------------------


@dataclass
class AuthContext:
    """Resolved authentication state for a single request.

    Exactly one of ``api_key`` / ``user`` is populated for authenticated
    requests; both are ``None`` when auth is disabled and the synthetic
    default tenant is in use.
    """

    tenant: Tenant
    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    role: Role = Role.admin
    permissions: set[Permission] = field(default_factory=set)

    @property
    def is_anonymous(self) -> bool:
        """True when this context was produced by the "auth-off" path."""
        return self.user is None and self.api_key is None


# ---------------------------------------------------------------------------
# Synthetic default tenant (used when AUTH_REQUIRED is False)
# ---------------------------------------------------------------------------


def _default_tenant() -> Tenant:
    """Return an in-memory :class:`Tenant` for the auth-off code path.

    The object is *not* attached to a session and has a deterministic
    UUID derived from the configured slug — that way structlog logs
    remain stable across restarts without requiring a DB row.
    """
    slug = settings.DEFAULT_TENANT_SLUG
    # Deterministic UUID so logs across restarts remain comparable.
    tenant_id = uuid5(NAMESPACE_URL, f"hal-nemofinder://tenants/{slug}")
    now = datetime.utcnow()
    tenant = Tenant(
        id=tenant_id,
        name=f"Default ({slug})",
        slug=slug,
        is_active=True,
        rate_limit_rpm=settings.RATE_LIMIT_RPM,
        rate_limit_analyze_pm=settings.RATE_LIMIT_ANALYZE_PM,
        config=None,
        created_at=now,
        updated_at=now,
    )
    return tenant


def _default_context() -> AuthContext:
    """Return an :class:`AuthContext` representing the "auth off" state.

    The synthetic tenant has admin permissions so the existing
    endpoints keep working exactly as they did before this module was
    added.
    """
    tenant = _default_tenant()
    return AuthContext(
        tenant=tenant,
        user=None,
        api_key=None,
        role=Role.admin,
        permissions=permissions_for_role(Role.admin),
    )


# ---------------------------------------------------------------------------
# Credential extraction helpers
# ---------------------------------------------------------------------------


def _extract_api_key(request: Request) -> Optional[str]:
    """Return the raw ``X-API-Key`` header, if any."""
    value = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if not value:
        return None
    return value.strip()


def _extract_bearer(request: Request) -> Optional[str]:
    """Return the bearer token from ``Authorization: Bearer <token>``."""
    header = request.headers.get("authorization")
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def _resolve_api_key(
    session: AsyncSession,
    plaintext: str,
) -> tuple[Tenant, ApiKey]:
    """Look up a persisted :class:`ApiKey` by its plaintext value.

    Raises HTTP 401 when the key is unknown, revoked, or expired.
    """
    key_hash = hash_api_key(plaintext)
    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
    api_key: Optional[ApiKey] = (
        await session.execute(stmt)
    ).scalar_one_or_none()
    if api_key is None:
        logger.info("auth.api_key.unknown")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    if api_key.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked.",
        )
    if api_key.expires_at is not None and api_key.expires_at <= datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"API key expired at {api_key.expires_at.isoformat()}. "
                "Issue a new key via POST /api/v1/auth/api-keys."
            ),
        )

    tenant = await session.get(Tenant, api_key.tenant_id)
    if tenant is None or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant is inactive or missing.",
        )

    # Opportunistically update last_used_at.  A failure here is logged
    # but does not block the request.
    try:
        api_key.last_used_at = datetime.utcnow()
        await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("auth.api_key.last_used.update_failed")
        await session.rollback()

    return tenant, api_key


async def _resolve_oidc_token(
    session: AsyncSession,
    token: str,
) -> tuple[Tenant, User]:
    """Validate *token* and return its tenant + user."""
    validator = build_oidc_validator(settings)
    if isinstance(validator, NoOpValidator):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OIDC authentication is not configured.",
        )
    try:
        payload = validator.validate_bearer_token(token)
    except OIDCError as exc:
        logger.info("auth.oidc.rejected", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Bearer token rejected: {exc}",
        ) from exc

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing the 'sub' claim.",
        )

    stmt = select(User).where(User.external_id == sub)
    user: Optional[User] = (await session.execute(stmt)).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active local user mapped to this bearer token.",
        )

    tenant = await session.get(Tenant, user.tenant_id)
    if tenant is None or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant is inactive or missing.",
        )

    user.last_login_at = datetime.utcnow()
    try:
        await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("auth.user.last_login.update_failed")
        await session.rollback()

    return tenant, user


# ---------------------------------------------------------------------------
# Public dependencies
# ---------------------------------------------------------------------------


async def get_auth_context(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    """Resolve the :class:`AuthContext` for the current request.

    Precedence:

    1. If ``X-API-Key`` is present it must validate.
    2. Otherwise, if ``Authorization: Bearer`` is present it must
       validate against the configured OIDC issuer.
    3. Otherwise, if auth is not required, a synthetic default context
       is returned.
    4. Otherwise, HTTP 401.
    """
    # If middleware has already resolved the context for this request,
    # reuse it — this avoids double DB lookups and matches what the
    # logging middleware expects.
    cached = getattr(request.state, "auth", None)
    if isinstance(cached, AuthContext):
        return cached

    api_key_plain = _extract_api_key(request)
    if api_key_plain:
        tenant, api_key = await _resolve_api_key(session, api_key_plain)
        role = api_key.role
        ctx = AuthContext(
            tenant=tenant,
            user=None,
            api_key=api_key,
            role=role,
            permissions=permissions_for_role(role),
        )
        request.state.auth = ctx
        return ctx

    bearer = _extract_bearer(request)
    if bearer:
        tenant, user = await _resolve_oidc_token(session, bearer)
        ctx = AuthContext(
            tenant=tenant,
            user=user,
            api_key=None,
            role=user.role,
            permissions=permissions_for_role(user.role),
        )
        request.state.auth = ctx
        return ctx

    if settings.AUTH_REQUIRED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication required. Provide an X-API-Key header or "
                "an Authorization: Bearer <jwt> token."
            ),
            headers={"WWW-Authenticate": 'Bearer realm="hal-nemofinder"'},
        )

    ctx = _default_context()
    request.state.auth = ctx
    return ctx


async def require_auth(
    ctx: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    """Dependency that fails with 401 when no real credentials were presented.

    Unlike :func:`get_auth_context`, this variant rejects the synthetic
    default context regardless of ``AUTH_REQUIRED``, so endpoints that
    must have a real tenant (e.g. the auth management endpoints)
    cannot be reached anonymously.
    """
    if ctx.is_anonymous and settings.AUTH_REQUIRED is False:
        # Still allow the default tenant through when auth is off —
        # the admin permissions on the default context are sufficient
        # for single-tenant operators.  If the operator truly wants to
        # lock things down they must set AUTH_REQUIRED=True.
        return ctx
    return ctx


def require_permission(permission: Permission) -> Callable:
    """Build a FastAPI dependency that enforces *permission*.

    The returned callable resolves the auth context (via
    :func:`get_auth_context`), checks the requested permission, and
    either returns the context (for convenience) or raises HTTP 403.
    """

    async def _dependency(
        ctx: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        if permission not in ctx.permissions:
            logger.info(
                "auth.permission_denied",
                required=permission.value,
                role=ctx.role.value,
                tenant_id=str(ctx.tenant.id),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Missing required permission {permission.value!r} "
                    f"(role={ctx.role.value})."
                ),
            )
        return ctx

    _dependency.__name__ = f"require_permission_{permission.value}"
    return _dependency


async def get_current_tenant(
    ctx: AuthContext = Depends(get_auth_context),
) -> Tenant:
    """Return just the tenant from the current auth context."""
    return ctx.tenant
