"""Tenant, User, and API key models for multi-tenant auth.

These tables support hal-nemoFinder's production deployment story:

* :class:`Tenant` — the isolation unit.  Every resource that belongs to a
  customer (jobs, reports, claims, API keys, users) is scoped to a tenant
  via a foreign key.  Two tenants can never see each other's data.
* :class:`User` — a human operator within a tenant, identified either by
  e-mail (for local accounts) or by an OIDC ``sub`` claim for SSO.
* :class:`ApiKey` — a bearer credential used by machines (and optionally
  humans) to authenticate against the API.  Only the SHA-256 hash of the
  plaintext is ever persisted; the plaintext is shown exactly once at
  creation time.
* :class:`Role` — a string enum describing what the bearer may do.  Roles
  expand to permissions via :mod:`src.auth.rbac`.

All columns on existing tables that reference tenants are deliberately
nullable so that pre-auth installations continue to function while a
customer migrates to the new multi-tenant model.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.models.job import AnalysisJob


class Role(str, enum.Enum):
    """Coarse-grained role assigned to users and API keys.

    Roles are intentionally few; fine-grained access is expressed through
    :class:`src.auth.rbac.Permission` which is derived from the role.
    """

    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class Tenant(UUIDMixin, TimestampMixin, Base):
    """An isolated customer environment.

    A tenant owns users, API keys, analysis jobs, reports, and any
    per-customer configuration overrides.  Deleting a tenant cascades to
    all of its users and API keys; analysis jobs are intentionally kept
    because historical audit trails may outlive an organisation.
    """

    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    rate_limit_analyze_pm: Mapped[int] = mapped_column(
        Integer, nullable=False, default=10
    )
    #: Opaque per-tenant configuration blob, e.g. ``{"verifier_exclusions":
    #: ["drugbank"], "default_claim_types": [...]}``.
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # ---- Relationships -----------------------------------------------------
    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    users: Mapped[list["User"]] = relationship(
        "User",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    jobs: Mapped[list["AnalysisJob"]] = relationship(
        "AnalysisJob",
        back_populates="tenant",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Tenant slug={self.slug!r} name={self.name!r}>"


class User(UUIDMixin, TimestampMixin, Base):
    """A human operator who belongs to a single :class:`Tenant`.

    Users are identified either by local e-mail address or, for SSO
    deployments, by the ``sub`` claim (``external_id``) of the OIDC
    provider.  The pair ``(tenant_id, email)`` is unique so different
    tenants may use the same e-mail address.
    """

    __tablename__ = "users"

    tenant_id: Mapped[UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="user_role", native_enum=True),
        nullable=False,
        default=Role.viewer,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ---- Relationships -----------------------------------------------------
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="users")
    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Per-tenant uniqueness on email: two tenants may share an address.
        # A plain UniqueConstraint is used instead of a partial index so
        # the constraint participates in SQLAlchemy's ON CONFLICT story.
        {"sqlite_autoincrement": True},
    )


class ApiKey(UUIDMixin, TimestampMixin, Base):
    """A bearer credential granting access to a tenant's resources.

    The plaintext key follows the format ``hal_<32_base64url_chars>`` and
    is never stored — only its SHA-256 digest lives in ``key_hash``.  The
    first twelve characters (``hal_<8_chars>``) are stored as
    ``key_prefix`` so operators can identify a key from server logs
    without compromising it.

    An API key is always bound to a tenant.  When ``user_id`` is NULL
    the key is a *service account* key — typically used by CI pipelines
    or server-to-server integrations.  The effective role is the one
    recorded on the key row; it does not automatically inherit from the
    associated user because the user's role may change independently.
    """

    __tablename__ = "api_keys"

    tenant_id: Mapped[UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="api_key_role", native_enum=True),
        nullable=False,
        default=Role.viewer,
    )
    #: Optional list of endpoint glob patterns this key is allowed to
    #: hit, e.g. ``["/api/v1/analyze", "/api/v1/jobs/*"]``.  ``None``
    #: means the role's default permissions apply.
    scopes: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ---- Relationships -----------------------------------------------------
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="api_keys")
    user: Mapped[Optional[User]] = relationship("User", back_populates="api_keys")

    # ---- Helpers -----------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """Return True if the key can still be used to authenticate."""
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at <= datetime.utcnow():
            return False
        return True

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ApiKey prefix={self.key_prefix!r} tenant={self.tenant_id}>"
