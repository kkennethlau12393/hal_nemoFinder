"""AnalysisJob and BatchJob models."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin
from src.models.enums import JobStatus

if TYPE_CHECKING:
    from src.models.claim import Claim
    from src.models.report import Report
    from src.models.tenant import Tenant, User


class BatchStatus(str, enum.Enum):
    """Terminal/intermediate states a :class:`BatchJob` can occupy."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    partial = "partial"
    failed = "failed"


class AnalysisJob(UUIDMixin, TimestampMixin, Base):
    """A single analysis request dispatched to a Celery worker."""

    __tablename__ = "analysis_jobs"

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=True),
        default=JobStatus.pending,
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ---- Webhook notification ---------------------------------------------
    webhook_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    webhook_secret: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # ---- Batch membership --------------------------------------------------
    batch_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ---- Multi-tenant scoping ---------------------------------------------
    #: Tenant that owns this job.  Nullable for backward compatibility
    #: with rows created before multi-tenancy was introduced.  Post-auth
    #: API requests always populate this column.
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("tenants.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    #: The user (if any) who submitted this job.  NULL when the request
    #: authenticated via a service-account API key or when auth was off.
    submitted_by_user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ---- Relationships -----------------------------------------------------
    claims: Mapped[list["Claim"]] = relationship(
        "Claim", back_populates="job", cascade="all, delete-orphan"
    )
    report: Mapped[Optional["Report"]] = relationship(
        "Report", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    batch: Mapped[Optional["BatchJob"]] = relationship(
        "BatchJob", back_populates="jobs"
    )
    tenant: Mapped[Optional["Tenant"]] = relationship(
        "Tenant", back_populates="jobs"
    )
    submitted_by: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys="AnalysisJob.submitted_by_user_id",
    )


class BatchJob(UUIDMixin, TimestampMixin, Base):
    """A group of :class:`AnalysisJob` rows created via ``/analyze/batch``."""

    __tablename__ = "batch_jobs"

    status: Mapped[BatchStatus] = mapped_column(
        Enum(BatchStatus, name="batch_status", native_enum=True),
        default=BatchStatus.pending,
    )
    total_jobs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_jobs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_jobs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Tenant that owns this batch.  Nullable for backward compatibility.
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("tenants.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    jobs: Mapped[list[AnalysisJob]] = relationship(
        "AnalysisJob",
        back_populates="batch",
        cascade="save-update, merge",
    )
