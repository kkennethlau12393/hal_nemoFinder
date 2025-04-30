"""Report model — aggregated hallucination analysis report for a job."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import Enum, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin
from src.models.enums import Severity

if TYPE_CHECKING:
    from src.models.job import AnalysisJob


class Report(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "reports"

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[Severity] = mapped_column(
        Enum(Severity, name="severity", native_enum=True),
        nullable=False,
    )
    claim_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    verified_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    refuted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unverifiable_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    partial_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_breakdown: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    report_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    job: Mapped[AnalysisJob] = relationship("AnalysisJob", back_populates="report")
