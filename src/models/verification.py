"""Verification models — VerificationResult and EvidenceRecord."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin
from src.models.enums import Verdict

if TYPE_CHECKING:
    from src.models.claim import Claim


class VerificationResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "verification_results"

    claim_id: Mapped[UUID] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
    )
    verifier_name: Mapped[str] = mapped_column(String(255), nullable=False)
    verdict: Mapped[Verdict] = mapped_column(
        Enum(Verdict, name="verdict", native_enum=True),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    claim: Mapped[Claim] = relationship("Claim", back_populates="verification_results")
    evidence_records: Mapped[list[EvidenceRecord]] = relationship(
        "EvidenceRecord",
        back_populates="verification_result",
        cascade="all, delete-orphan",
    )


class EvidenceRecord(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "evidence_records"

    verification_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("verification_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_db: Mapped[str] = mapped_column(String(255), nullable=False)
    query_used: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    retrieval_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    kb_version: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    verification_result: Mapped[VerificationResult] = relationship(
        "VerificationResult", back_populates="evidence_records"
    )
