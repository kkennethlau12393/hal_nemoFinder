"""Statistics endpoint — aggregate metrics across all analysis jobs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import StatsResponse
from src.db.session import get_session
from src.models.claim import Claim
from src.models.enums import JobStatus, Verdict
from src.models.job import AnalysisJob
from src.models.report import Report
from src.models.verification import VerificationResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregate statistics",
    description=(
        "Return system-wide statistics including total and completed job "
        "counts, the average hallucination score across all reports, and "
        "per-claim-type hallucination (refuted) rates."
    ),
    tags=["System"],
)
async def get_stats(
    session: AsyncSession = Depends(get_session),
) -> StatsResponse:
    # --- Job counts ---------------------------------------------------------
    job_stats_stmt = select(
        func.count(AnalysisJob.id).label("total_jobs"),
        func.count(
            case(
                (AnalysisJob.status == JobStatus.completed, AnalysisJob.id),
            )
        ).label("completed_jobs"),
    )
    job_row = (await session.execute(job_stats_stmt)).one()

    # --- Average hallucination score from reports ---------------------------
    avg_stmt = select(func.avg(Report.overall_score))
    avg_score = (await session.execute(avg_stmt)).scalar() or 0.0

    # --- Hallucination rate by claim type -----------------------------------
    # Rate = refuted / total for each claim type
    rate_stmt = (
        select(
            Claim.claim_type,
            func.count(VerificationResult.id).label("total"),
            func.count(
                case(
                    (VerificationResult.verdict == Verdict.refuted, VerificationResult.id),
                )
            ).label("refuted"),
        )
        .join(VerificationResult, VerificationResult.claim_id == Claim.id)
        .group_by(Claim.claim_type)
    )
    rate_rows = (await session.execute(rate_stmt)).all()

    hallucination_rate_by_type: dict[str, float] = {}
    for claim_type, total, refuted in rate_rows:
        rate = (refuted / total) if total > 0 else 0.0
        hallucination_rate_by_type[claim_type.value] = round(rate, 4)

    return StatsResponse(
        total_jobs=job_row.total_jobs,
        completed_jobs=job_row.completed_jobs,
        avg_hallucination_score=round(float(avg_score), 4),
        hallucination_rate_by_type=hallucination_rate_by_type,
    )
