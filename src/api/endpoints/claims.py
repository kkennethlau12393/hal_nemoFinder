"""Claim detail endpoint — single claim with full verification evidence."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.schemas import EvidenceRecordResponse, VerificationDetailResponse
from src.auth.dependencies import AuthContext, get_auth_context
from src.auth.rbac import Permission
from src.db.session import get_session
from src.models.claim import Claim
from src.models.job import AnalysisJob
from src.models.verification import VerificationResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/claims/{claim_id}",
    response_model=list[VerificationDetailResponse],
    summary="Get claim verification details",
    description=(
        "Retrieve every verification result for a single claim, including the "
        "verifier name, verdict, confidence score, reasoning, and all underlying "
        "evidence records fetched from authoritative knowledge bases."
    ),
    tags=["Claims"],
    responses={
        404: {"description": "Claim not found"},
    },
)
async def get_claim_detail(
    claim_id: UUID,
    session: AsyncSession = Depends(get_session),
    ctx: AuthContext = Depends(get_auth_context),
) -> list[VerificationDetailResponse]:
    if Permission.READ_REPORTS not in ctx.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role {ctx.role.value!r} is not permitted to read claims."
            ),
        )
    # Ensure claim exists and belongs to the caller's tenant.
    claim = await session.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found.",
        )
    if not ctx.is_anonymous:
        parent_job = await session.get(AnalysisJob, claim.job_id)
        if parent_job is None or parent_job.tenant_id != ctx.tenant.id:
            # 404 rather than 403 to prevent cross-tenant probing.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {claim_id} not found.",
            )

    # Load verification results with evidence records
    stmt = (
        select(VerificationResult)
        .where(VerificationResult.claim_id == claim_id)
        .options(selectinload(VerificationResult.evidence_records))
        .order_by(VerificationResult.created_at)
    )
    results = (await session.execute(stmt)).scalars().all()

    return [
        VerificationDetailResponse(
            id=vr.id,
            verifier_name=vr.verifier_name,
            verdict=vr.verdict,
            confidence=vr.confidence,
            reasoning=vr.reasoning,
            evidence=vr.evidence,
            evidence_records=[
                EvidenceRecordResponse(
                    source_db=er.source_db,
                    query_used=er.query_used,
                    retrieval_timestamp=er.retrieval_timestamp,
                    kb_version=er.kb_version,
                )
                for er in vr.evidence_records
            ],
        )
        for vr in results
    ]
