"""Analysis submission endpoints — single-text and batch."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchResponse,
)
from src.auth.dependencies import AuthContext, get_auth_context
from src.auth.rbac import Permission
from src.db.session import get_session
from src.models.enums import JobStatus
from src.models.job import AnalysisJob, BatchJob, BatchStatus
from src.observability import get_logger
from src.tasks.celery_app import celery_app

logger = get_logger(__name__)

router = APIRouter()

#: Reasonable upper bound for input text (500 KB).
MAX_INPUT_LENGTH = 500_000

#: Maximum number of texts allowed per batch request.
MAX_BATCH_SIZE = 500


def _check_write_permission(ctx: AuthContext) -> None:
    """Enforce ``WRITE_JOBS`` on the analyse endpoints.

    Separated from the dependency chain so the error message matches
    the 403 schema used elsewhere in the module.
    """
    if Permission.WRITE_JOBS not in ctx.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role {ctx.role.value!r} is not permitted to submit analysis jobs."
            ),
        )


def _validate_text(text: str) -> str:
    """Return a cleaned-and-validated version of *text* or raise ``HTTPException``."""
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text must be non-empty.",
        )
    if len(text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input text exceeds maximum length of {MAX_INPUT_LENGTH:,} characters.",
        )
    return text.strip()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit text for hallucination analysis",
    description=(
        "Accept a block of AI-generated text (e.g. a drug-discovery report) "
        "and queue it for hallucination detection.  The system will extract "
        "factual claims, verify each one against authoritative databases, and "
        "produce an aggregated report.\n\n"
        "Returns a `job_id` that can be polled via `GET /api/v1/jobs/{job_id}`. "
        "If a `webhook_url` is provided the service will POST a notification "
        "to it when the job reaches a terminal state."
    ),
    tags=["Analysis"],
)
async def submit_analysis(
    payload: AnalyzeRequest,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> AnalyzeResponse:
    _check_write_permission(ctx)
    cleaned = _validate_text(payload.text)

    job = AnalysisJob(
        status=JobStatus.pending,
        input_text=cleaned,
        config=payload.config,
        webhook_url=str(payload.webhook_url) if payload.webhook_url else None,
        webhook_secret=payload.webhook_secret,
        tenant_id=ctx.tenant.id if not ctx.is_anonymous else None,
        submitted_by_user_id=ctx.user.id if ctx.user is not None else None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    logger.info("analysis.job.created", job_id=str(job.id))

    celery_app.send_task(
        "src.tasks.verification.run_analysis_task",
        args=[str(job.id)],
    )

    return AnalyzeResponse(
        job_id=str(job.id),
        status=JobStatus.pending.value,
        message="Analysis job queued successfully.",
    )


@router.post(
    "/analyze/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a batch of texts for hallucination analysis",
    description=(
        "Accept up to 500 input texts and create one :class:`AnalysisJob` per "
        "text, grouped under a single batch identifier.  Each text is "
        "dispatched to its own Celery task.  Clients can poll the returned "
        "``progress_url`` to monitor the batch as a whole."
    ),
    tags=["Analysis"],
)
async def submit_batch_analysis(
    payload: BatchAnalyzeRequest,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> BatchResponse:
    _check_write_permission(ctx)
    if len(payload.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE} texts.",
        )

    cleaned_texts = [_validate_text(t) for t in payload.texts]

    # --- Create the batch row first so jobs can reference it ---------------
    batch = BatchJob(
        status=BatchStatus.pending,
        total_jobs=len(cleaned_texts),
        completed_jobs=0,
        failed_jobs=0,
        tenant_id=ctx.tenant.id if not ctx.is_anonymous else None,
    )
    session.add(batch)
    await session.flush()  # assigns batch.id without committing

    webhook_url = str(payload.webhook_url) if payload.webhook_url else None
    jobs: list[AnalysisJob] = []
    for text in cleaned_texts:
        jobs.append(
            AnalysisJob(
                status=JobStatus.pending,
                input_text=text,
                config=payload.config,
                webhook_url=webhook_url,
                webhook_secret=payload.webhook_secret,
                batch_id=batch.id,
                tenant_id=ctx.tenant.id if not ctx.is_anonymous else None,
                submitted_by_user_id=ctx.user.id if ctx.user is not None else None,
            )
        )
    session.add_all(jobs)
    await session.commit()
    for job in jobs:
        await session.refresh(job)

    logger.info(
        "analysis.batch.created",
        batch_id=str(batch.id),
        total_jobs=len(jobs),
    )

    # --- Dispatch Celery tasks (one per job) -------------------------------
    # A true Celery ``group`` would couple this request to the worker's
    # broker; instead we send individual tasks which is simpler and
    # equivalent for our progress-tracking needs.
    for job in jobs:
        celery_app.send_task(
            "src.tasks.verification.run_analysis_task",
            args=[str(job.id)],
        )

    return BatchResponse(
        batch_id=batch.id,
        job_ids=[job.id for job in jobs],
        total_jobs=len(jobs),
        progress_url=f"/api/v1/batches/{batch.id}",
    )
