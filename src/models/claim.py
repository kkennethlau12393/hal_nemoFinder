"""Claim model — represents a factual claim extracted from LLM output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import Enum, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin
from src.models.enums import ClaimType

if TYPE_CHECKING:
    from src.models.job import AnalysisJob
    from src.models.verification import VerificationResult


class Claim(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "claims"

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[ClaimType] = mapped_column(
        Enum(ClaimType, name="claim_type", native_enum=True),
        nullable=False,
    )
    source_span: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    job: Mapped[AnalysisJob] = relationship("AnalysisJob", back_populates="claims")
    verification_results: Mapped[list[VerificationResult]] = relationship(
        "VerificationResult", back_populates="claim", cascade="all, delete-orphan"
    )
