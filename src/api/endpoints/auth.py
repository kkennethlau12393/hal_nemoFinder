"""Auth management endpoints — tenants, users, API keys.

All endpoints live under ``/api/v1/auth`` and require appropriate
permissions (``MANAGE_TENANT``, ``MANAGE_USERS``, ``MANAGE_API_KEYS``)
via the :func:`src.auth.require_permission` dependency factory.

One deliberate exception is :func:`create_tenant`: the very first
tenant must exist before any API key can be issued, so this endpoint
additionally accepts ``X-Bootstrap-Key`` (matched against
``settings.AUTH_BOOTSTRAP_KEY``) as an alternative to a real admin
credential.  Once a tenant + admin user + admin API key exist the
bootstrap key should be deleted from the environment.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    AuthContextResponse,
    TenantCreateRequest,
    TenantResponse,
    UserCreateRequest,
    UserResponse,
)
from src.auth.api_keys import extract_prefix, generate_api_key
from src.auth.dependencies import (
    AuthContext,
    get_auth_context,
    require_permission,
)
from src.auth.rbac import Permission, permissions_for_role
from src.config import settings
from src.db.session import get_session
from src.models.tenant import ApiKey, Role, Tenant, User
from src.observability import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_matches(presented: Optional[str]) -> bool:
    """Return True if *presented* matches the configured bootstrap key."""
    expected = settings.AUTH_BOOTSTRAP_KEY
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected, presented)


async def _ensure_tenant(session: AsyncSession, tenant_id: UUID) -> Tenant:
    """Load a tenant or raise HTTP 404."""
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found.",
        )
    return tenant


def _enforce_same_tenant(ctx: AuthContext, tenant_id: UUID) -> None:
    """Guard a per-tenant resource from cross-tenant access."""
    if ctx.tenant.id != tenant_id and Permission.MANAGE_TENANT not in ctx.permissions:
        # Hide cross-tenant existence: treat like "not found".
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found.",
        )


# ---------------------------------------------------------------------------
# Tenant endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tenants",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a tenant",
    description=(
        "Create a new tenant.  Requires an authenticated caller with "
        "``MANAGE_TENANT`` permission, or a valid ``X-Bootstrap-Key`` "
        "header matching the configured ``AUTH_BOOTSTRAP_KEY``."
    ),
)
async def create_tenant(
    payload: TenantCreateRequest,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
    x_bootstrap_key: Optional[str] = Header(default=None, alias="X-Bootstrap-Key"),
) -> TenantResponse:
    authorised = (
        Permission.MANAGE_TENANT in ctx.permissions
        or _bootstrap_matches(x_bootstrap_key)
    )
    if not authorised:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant creation requires MANAGE_TENANT or a bootstrap key.",
        )

    existing = (
        await session.execute(select(Tenant).where(Tenant.slug == payload.slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant with slug {payload.slug!r} already exists.",
        )

    tenant = Tenant(
        name=payload.name,
        slug=payload.slug,
        is_active=True,
        rate_limit_rpm=payload.rate_limit_rpm,
        rate_limit_analyze_pm=payload.rate_limit_analyze_pm,
        config=payload.config,
    )
    session.add(tenant)
    await session.commit()
    await session.refresh(tenant)
    logger.info(
        "auth.tenant.created",
        tenant_id=str(tenant.id),
        slug=tenant.slug,
    )
    return TenantResponse.model_validate(tenant)


@router.get(
    "/tenants/me",
    response_model=TenantResponse,
    summary="Get the current tenant",
)
async def get_my_tenant(
    ctx: AuthContext = Depends(get_auth_context),
) -> TenantResponse:
    return TenantResponse.model_validate(ctx.tenant)


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tenants/{tenant_id}/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user in a tenant",
)
async def create_user(
    tenant_id: UUID,
    payload: UserCreateRequest,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_USERS)),
) -> UserResponse:
    _enforce_same_tenant(ctx, tenant_id)
    tenant = await _ensure_tenant(session, tenant_id)

    dup = (
        await session.execute(
            select(User).where(
                User.tenant_id == tenant.id, User.email == payload.email
            )
        )
    ).scalar_one_or_none()
    if dup is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User {payload.email!r} already exists in this tenant.",
        )

    user = User(
        tenant_id=tenant.id,
        email=payload.email,
        external_id=payload.external_id,
        role=payload.role,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info(
        "auth.user.created",
        tenant_id=str(tenant.id),
        user_id=str(user.id),
        role=user.role.value,
    )
    return UserResponse.model_validate(user)


@router.get(
    "/tenants/{tenant_id}/users",
    response_model=list[UserResponse],
    summary="List users in a tenant",
)
async def list_users(
    tenant_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_USERS)),
) -> list[UserResponse]:
    _enforce_same_tenant(ctx, tenant_id)
    await _ensure_tenant(session, tenant_id)
    stmt = (
        select(User)
        .where(User.tenant_id == tenant_id)
        .order_by(User.created_at.desc())
    )
    users = (await session.execute(stmt)).scalars().all()
    return [UserResponse.model_validate(u) for u in users]


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate a user",
)
async def deactivate_user(
    user_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_USERS)),
) -> None:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    _enforce_same_tenant(ctx, user.tenant_id)
    user.is_active = False
    await session.commit()
    logger.info("auth.user.deactivated", user_id=str(user.id))


# ---------------------------------------------------------------------------
# API key endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue a new API key",
    description=(
        "Create a new API key for the authenticated tenant.  The "
        "plaintext key is returned **exactly once** in the response "
        "body; it is never logged or returned by subsequent reads."
    ),
)
async def issue_api_key(
    payload: ApiKeyCreateRequest,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_API_KEYS)),
) -> ApiKeyCreateResponse:
    if ctx.is_anonymous:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A real tenant is required to issue API keys.",
        )

    user: Optional[User] = None
    if payload.user_id is not None:
        user = await session.get(User, payload.user_id)
        if user is None or user.tenant_id != ctx.tenant.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {payload.user_id} not found in this tenant.",
            )

    plaintext, key_hash = generate_api_key()
    expires_at = (
        datetime.utcnow() + timedelta(days=payload.expires_in_days)
        if payload.expires_in_days
        else None
    )
    api_key = ApiKey(
        tenant_id=ctx.tenant.id,
        user_id=user.id if user is not None else None,
        name=payload.name,
        key_prefix=extract_prefix(plaintext),
        key_hash=key_hash,
        role=payload.role,
        scopes=payload.scopes,
        expires_at=expires_at,
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)
    logger.info(
        "auth.api_key.issued",
        tenant_id=str(api_key.tenant_id),
        key_prefix=api_key.key_prefix,
        role=api_key.role.value,
    )
    base = ApiKeyResponse.model_validate(api_key)
    return ApiKeyCreateResponse(**base.model_dump(), plaintext_key=plaintext)


@router.get(
    "/api-keys",
    response_model=list[ApiKeyResponse],
    summary="List API keys for the current tenant",
)
async def list_api_keys(
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_API_KEYS)),
) -> list[ApiKeyResponse]:
    stmt = (
        select(ApiKey)
        .where(ApiKey.tenant_id == ctx.tenant.id)
        .order_by(ApiKey.created_at.desc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [ApiKeyResponse.model_validate(r) for r in rows]


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
)
async def revoke_api_key(
    key_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(require_permission(Permission.MANAGE_API_KEYS)),
) -> None:
    api_key = await session.get(ApiKey, key_id)
    if api_key is None or api_key.tenant_id != ctx.tenant.id:
        # 404 to prevent cross-tenant probing.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found.",
        )
    api_key.revoked_at = datetime.utcnow()
    await session.commit()
    logger.info(
        "auth.api_key.revoked",
        tenant_id=str(api_key.tenant_id),
        key_prefix=api_key.key_prefix,
    )


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=AuthContextResponse,
    summary="Describe the current caller",
)
async def whoami(
    ctx: AuthContext = Depends(get_auth_context),
) -> AuthContextResponse:
    return AuthContextResponse(
        tenant=TenantResponse.model_validate(ctx.tenant),
        user=UserResponse.model_validate(ctx.user) if ctx.user is not None else None,
        api_key_prefix=ctx.api_key.key_prefix if ctx.api_key is not None else None,
        role=ctx.role,
        permissions=sorted(p.value for p in ctx.permissions),
        anonymous=ctx.is_anonymous,
    )
