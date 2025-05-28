"""Immutable audit log model with HMAC integrity chain.

This module defines :class:`AuditLogEntry`, a tamper-evident append-only
table used to support 21 CFR Part 11 compliance.  Each row contains an
``integrity_hash`` computed from the previous row's hash plus the core
fields of the current row — any modification, deletion, or reordering
of rows will cause the chain verification to fail.

Append-only enforcement
-----------------------
Application code must never update or delete rows in ``audit_log``.
For Postgres deployments, a row-level trigger is registered by the
Alembic migration that raises an exception on UPDATE or DELETE.  For
SQLite (tests), enforcement is purely application-level — the
:class:`AuditRecorder` never issues updates or deletes.
"""

from __future__ import annotations

import enum
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Enum as SAEnum,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDMixin


class AuditEventType(str, enum.Enum):
    """Enumerated set of audit-log event types.

    Values are dotted strings so they read naturally in log pipelines.
    """

    job_submitted = "job.submitted"
    job_completed = "job.completed"
    job_failed = "job.failed"
    report_generated = "report.generated"
    claim_verified = "claim.verified"
    verdict_recorded = "verdict.recorded"
    user_created = "user.created"
    user_deactivated = "user.deactivated"
    api_key_issued = "api_key.issued"
    api_key_revoked = "api_key.revoked"
    tenant_created = "tenant.created"
    config_changed = "config.changed"
    calibration_run = "calibration.run"
    plugin_loaded = "plugin.loaded"
    login = "auth.login"
    permission_denied = "auth.permission_denied"


#: Sentinel prev_hash for the very first entry in the chain.
GENESIS_PREV_HASH = "0" * 64


class AuditLogEntry(UUIDMixin, TimestampMixin, Base):
    """A single immutable audit-log row.

    The combination of ``sequence`` + ``prev_hash`` + ``integrity_hash``
    forms a hash chain: verifying the chain from the beginning confirms
    that no row has been added, removed, or modified.
    """

    __tablename__ = "audit_log"

    # ----- Ordering for integrity chain -----------------------------------
    sequence: Mapped[int] = mapped_column(
        BigInteger,
        autoincrement=True,
        unique=True,
        nullable=False,
        index=True,
    )

    # ----- Core event fields ----------------------------------------------
    event_type: Mapped[AuditEventType] = mapped_column(
        SAEnum(AuditEventType, name="audit_event_type", native_enum=True),
        nullable=False,
    )
    # UUID columns use the native PG UUID type on Postgres and fall back
    # to CHAR(32) on SQLite (used by the test suite).  Kept as soft
    # references rather than hard FKs so this table does not acquire
    # dependencies on the auth agent's schema definition order.
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True).with_variant(String(36), "sqlite"),
        nullable=True,
        index=True,
    )
    actor_user_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True).with_variant(String(36), "sqlite"),
        nullable=True,
    )
    actor_api_key_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True).with_variant(String(36), "sqlite"),
        nullable=True,
    )
    actor_ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)

    # ----- Context payload (sanitized) ------------------------------------
    payload: Mapped[Optional[dict]] = mapped_column(
        JSONB().with_variant(Text(), "sqlite"),
        nullable=True,
    )

    # ----- Integrity chain -------------------------------------------------
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    integrity_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )

    __table_args__ = (
        Index("ix_audit_tenant_time", "tenant_id", "created_at"),
        Index("ix_audit_type_time", "event_type", "created_at"),
    )

    # ------------------------------------------------------------------
    # App-level immutability guard
    # ------------------------------------------------------------------

    def __setattr__(self, key: str, value: Any) -> None:
        # Allow initial assignment (when SQLAlchemy is hydrating the
        # instance) but disallow any post-flush mutation of core
        # integrity fields.  This is a cheap defence-in-depth layer;
        # the authoritative guarantee is the Postgres trigger.
        _frozen = {
            "sequence",
            "event_type",
            "tenant_id",
            "actor_user_id",
            "actor_api_key_id",
            "actor_ip",
            "resource_type",
            "resource_id",
            "action",
            "outcome",
            "payload",
            "prev_hash",
            "integrity_hash",
        }
        if key in _frozen and key in self.__dict__ and self.__dict__[key] is not None:
            raise AttributeError(
                f"AuditLogEntry.{key} is immutable once set (append-only table)"
            )
        super().__setattr__(key, value)
