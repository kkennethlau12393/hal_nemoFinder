"""Job retrieval endpoints — list, detail, report, and claims for a job."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.schemas import (
    ClaimResponse,
    JobResponse,
    ReportResponse,
    VerdictSummary,
)
from src.auth.dependencies import AuthContext, get_auth_context
from src.auth.rbac import Permission
from src.db.session import get_session
from src.models.claim import Claim
from src.models.enums import JobStatus
from src.models.job import AnalysisJob
from src.models.report import Report
from src.models.verification import VerificationResult

logger = logging.getLogger(__name__)

router = APIRouter()


def _enforce_read_jobs(ctx: AuthContext) -> None:
    """Require the ``READ_JOBS`` permission for the current caller."""
    if Permission.READ_JOBS not in ctx.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role {ctx.role.value!r} is not permitted to read jobs."
            ),
        )


def _tenant_scope(ctx: AuthContext):
    """Return a SQLAlchemy predicate scoping a query to the caller's tenant.

    When the context is anonymous (auth disabled, synthetic default
    tenant) the predicate is ``None`` so queries return every row as
    they did before auth was introduced.
    """
    if ctx.is_anonymous:
        return None
    return AnalysisJob.tenant_id == ctx.tenant.id


# ---------------------------------------------------------------------------
# GET /jobs — paginated list
# ---------------------------------------------------------------------------


@router.get(
    "/jobs",
    response_model=list[JobResponse],
    summary="List analysis jobs",
    description=(
        "Return a paginated list of analysis jobs ordered by creation time "
        "(newest first).  Each entry includes the current status and the "
        "number of extracted claims."
    ),
    tags=["Jobs"],
)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> list[JobResponse]:
    _enforce_read_jobs(ctx)
    stmt = (
        select(
            AnalysisJob,
            func.count(Claim.id).label("claim_count"),
        )
        .outerjoin(Claim, Claim.job_id == AnalysisJob.id)
        .group_by(AnalysisJob.id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    scope = _tenant_scope(ctx)
    if scope is not None:
        stmt = stmt.where(scope)
    rows = (await session.execute(stmt)).all()
    return [
        JobResponse(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            claim_count=count,
        )
        for job, count in rows
    ]


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — single job detail
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get analysis job status",
    description=(
        "Retrieve the current status of a single analysis job including the "
        "number of claims extracted so far."
    ),
    tags=["Jobs"],
)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> JobResponse:
    _enforce_read_jobs(ctx)
    stmt = (
        select(
            AnalysisJob,
            func.count(Claim.id).label("claim_count"),
        )
        .outerjoin(Claim, Claim.job_id == AnalysisJob.id)
        .where(AnalysisJob.id == job_id)
        .group_by(AnalysisJob.id)
    )
    scope = _tenant_scope(ctx)
    if scope is not None:
        stmt = stmt.where(scope)
    row = (await session.execute(stmt)).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    job, count = row
    return JobResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        claim_count=count,
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/report — full report
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}/report",
    response_model=ReportResponse,
    summary="Get hallucination report",
    description=(
        "Retrieve the complete hallucination analysis report for a finished "
        "job.  The report includes an overall hallucination score, severity "
        "rating, per-claim verdicts, and evidence summaries.\n\n"
        "Returns **404** if the job does not exist, or **409** if the job "
        "has not yet completed."
    ),
    tags=["Jobs"],
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job has not completed yet"},
    },
)
async def get_report(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> ReportResponse:
    _enforce_read_jobs(ctx)
    # Verify the job exists and belongs to the caller's tenant.
    job = await session.get(AnalysisJob, job_id)
    if job is None or (
        not ctx.is_anonymous and job.tenant_id != ctx.tenant.id
    ):
        # Return 404 rather than 403 on cross-tenant access so tenants
        # cannot probe for the existence of each other's jobs.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} has not completed yet (current status: {job.status.value}).",
        )

    # Fetch report with claims and their verdicts
    stmt = (
        select(Report)
        .where(Report.job_id == job_id)
    )
    report = (await session.execute(stmt)).scalar_one_or_none()
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report for job {job_id} not found.",
        )

    # Load claims with verdict summaries
    claims_stmt = (
        select(Claim)
        .where(Claim.job_id == job_id)
        .options(selectinload(Claim.verification_results))
    )
    claims = (await session.execute(claims_stmt)).scalars().all()

    claim_responses = [
        ClaimResponse(
            id=claim.id,
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            source_span=claim.source_span,
            extraction_confidence=claim.extraction_confidence,
            verdicts=[
                VerdictSummary(
                    verifier_name=vr.verifier_name,
                    verdict=vr.verdict,
                    confidence=vr.confidence,
                    reasoning=vr.reasoning,
                )
                for vr in claim.verification_results
            ],
        )
        for claim in claims
    ]

    return ReportResponse(
        id=report.id,
        job_id=report.job_id,
        overall_score=report.overall_score,
        severity=report.severity,
        claim_count=report.claim_count,
        verified_count=report.verified_count,
        refuted_count=report.refuted_count,
        unverifiable_count=report.unverifiable_count,
        partial_count=report.partial_count,
        section_breakdown=report.section_breakdown,
        claims=claim_responses,
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/claims — claims with verdict summaries
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}/claims",
    response_model=list[ClaimResponse],
    summary="List claims for a job",
    description=(
        "Return all factual claims extracted from the job's input text, each "
        "annotated with verdict summaries from every verifier that evaluated it."
    ),
    tags=["Jobs"],
    responses={
        404: {"description": "Job not found"},
    },
)
async def list_job_claims(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> list[ClaimResponse]:
    _enforce_read_jobs(ctx)
    job = await session.get(AnalysisJob, job_id)
    if job is None or (
        not ctx.is_anonymous and job.tenant_id != ctx.tenant.id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    stmt = (
        select(Claim)
        .where(Claim.job_id == job_id)
        .options(selectinload(Claim.verification_results))
        .order_by(Claim.created_at)
    )
    claims = (await session.execute(stmt)).scalars().all()

    return [
        ClaimResponse(
            id=claim.id,
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            source_span=claim.source_span,
            extraction_confidence=claim.extraction_confidence,
            verdicts=[
                VerdictSummary(
                    verifier_name=vr.verifier_name,
                    verdict=vr.verdict,
                    confidence=vr.confidence,
                    reasoning=vr.reasoning,
                )
                for vr in claim.verification_results
            ],
        )
        for claim in claims
    ]
