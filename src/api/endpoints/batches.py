"""Batch job endpoints — status polling and aggregate reporting."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.schemas import (
    BatchReportResponse,
    BatchStatusResponse,
    ClaimResponse,
    ReportResponse,
    VerdictSummary,
)
from src.auth.dependencies import AuthContext, get_auth_context
from src.auth.rbac import Permission
from src.db.session import get_session
from src.models.claim import Claim
from src.models.enums import JobStatus
from src.models.job import AnalysisJob, BatchJob, BatchStatus
from src.models.report import Report
from src.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _enforce_read_batches(ctx: AuthContext) -> None:
    """Require the ``READ_JOBS`` permission to inspect batches."""
    if Permission.READ_JOBS not in ctx.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role {ctx.role.value!r} is not permitted to read batches."
            ),
        )


async def _recompute_batch_status(
    session: AsyncSession, batch: BatchJob
) -> BatchJob:
    """Refresh the ``BatchJob``'s counters from its member jobs.

    Counters are updated opportunistically on every status/read so we
    don't need a periodic sweeper; this keeps the datamodel simple and
    always-consistent from the caller's perspective.
    """
    counts_stmt = (
        select(AnalysisJob.status, func.count(AnalysisJob.id))
        .where(AnalysisJob.batch_id == batch.id)
        .group_by(AnalysisJob.status)
    )
    rows = (await session.execute(counts_stmt)).all()
    counts: dict[JobStatus, int] = {status_val: n for status_val, n in rows}

    completed = counts.get(JobStatus.completed, 0)
    failed = counts.get(JobStatus.failed, 0)
    total = batch.total_jobs
    terminal = completed + failed

    if terminal == 0:
        new_status = BatchStatus.pending if total > 0 else BatchStatus.completed
    elif terminal < total:
        new_status = BatchStatus.processing
    elif failed == total:
        new_status = BatchStatus.failed
    elif failed > 0:
        new_status = BatchStatus.partial
    else:
        new_status = BatchStatus.completed

    changed = (
        batch.completed_jobs != completed
        or batch.failed_jobs != failed
        or batch.status != new_status
    )
    if changed:
        batch.completed_jobs = completed
        batch.failed_jobs = failed
        batch.status = new_status
        await session.commit()
        await session.refresh(batch)
    return batch


@router.get(
    "/batches/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Get batch job status",
    description="Return progress counters for an existing batch submission.",
    tags=["Jobs"],
    responses={404: {"description": "Batch not found"}},
)
async def get_batch(
    batch_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> BatchStatusResponse:
    _enforce_read_batches(ctx)
    batch = await session.get(BatchJob, batch_id)
    if batch is None or (
        not ctx.is_anonymous and batch.tenant_id != ctx.tenant.id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )
    batch = await _recompute_batch_status(session, batch)

    pending = max(batch.total_jobs - batch.completed_jobs - batch.failed_jobs, 0)
    progress = (
        round((batch.completed_jobs + batch.failed_jobs) / batch.total_jobs, 4)
        if batch.total_jobs
        else 1.0
    )
    return BatchStatusResponse(
        id=batch.id,
        status=batch.status,
        total_jobs=batch.total_jobs,
        completed_jobs=batch.completed_jobs,
        failed_jobs=batch.failed_jobs,
        pending_jobs=pending,
        progress=progress,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
    )


@router.get(
    "/batches/{batch_id}/report",
    response_model=BatchReportResponse,
    summary="Aggregate report for a completed batch",
    description=(
        "Return a combined report that includes every individual job's "
        "report.  Available only when every member job has reached a "
        "terminal state (either ``completed`` or ``failed``)."
    ),
    tags=["Jobs"],
    responses={
        404: {"description": "Batch not found"},
        409: {"description": "Batch has not finished yet"},
    },
)
async def get_batch_report(
    batch_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> BatchReportResponse:
    _enforce_read_batches(ctx)
    batch = await session.get(BatchJob, batch_id)
    if batch is None or (
        not ctx.is_anonymous and batch.tenant_id != ctx.tenant.id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )
    batch = await _recompute_batch_status(session, batch)

    if batch.status not in (BatchStatus.completed, BatchStatus.partial, BatchStatus.failed):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Batch {batch_id} has not finished yet "
                f"(current status: {batch.status.value})."
            ),
        )

    jobs_filter = AnalysisJob.batch_id == batch_id
    if not ctx.is_anonymous:
        jobs_filter = jobs_filter & (AnalysisJob.tenant_id == ctx.tenant.id)
    jobs_stmt = (
        select(AnalysisJob)
        .where(jobs_filter)
        .options(
            selectinload(AnalysisJob.report),
            selectinload(AnalysisJob.claims).selectinload(Claim.verification_results),
        )
    )
    jobs = (await session.execute(jobs_stmt)).scalars().all()

    report_responses: list[ReportResponse] = []
    severity_counts: dict[str, int] = {}
    score_total = 0.0
    scored = 0

    for job in jobs:
        if job.report is None:
            continue
        rep: Report = job.report
        claim_responses = [
            ClaimResponse(
                id=c.id,
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                source_span=c.source_span,
                extraction_confidence=c.extraction_confidence,
                verdicts=[
                    VerdictSummary(
                        verifier_name=vr.verifier_name,
                        verdict=vr.verdict,
                        confidence=vr.confidence,
                        reasoning=vr.reasoning,
                    )
                    for vr in c.verification_results
                ],
            )
            for c in job.claims
        ]
        report_responses.append(
            ReportResponse(
                id=rep.id,
                job_id=rep.job_id,
                overall_score=rep.overall_score,
                severity=rep.severity,
                claim_count=rep.claim_count,
                verified_count=rep.verified_count,
                refuted_count=rep.refuted_count,
                unverifiable_count=rep.unverifiable_count,
                partial_count=rep.partial_count,
                section_breakdown=rep.section_breakdown,
                claims=claim_responses,
            )
        )
        severity_counts[rep.severity.value] = severity_counts.get(rep.severity.value, 0) + 1
        score_total += rep.overall_score
        scored += 1

    avg_score = round(score_total / scored, 4) if scored else 0.0

    return BatchReportResponse(
        batch_id=batch.id,
        total_jobs=batch.total_jobs,
        completed_jobs=batch.completed_jobs,
        failed_jobs=batch.failed_jobs,
        average_score=avg_score,
        severity_counts=severity_counts,
        reports=report_responses,
    )
