"""Core Celery tasks for claim verification and analysis orchestration.

Tasks
-----
verify_claim_task
    Run a single verifier against a single claim.
run_analysis_task
    Orchestrate an end-to-end analysis job: extract, classify, fan-out
    verification, and trigger aggregation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from celery import chord
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.claim_classifier import ClaimClassifier
from src.core.claim_extractor import ClaimExtractor
from src.models.claim import Claim
from src.models.enums import JobStatus, Verdict
from src.models.job import AnalysisJob
from src.models.verification import EvidenceRecord, VerificationResult
from src.tasks.celery_app import celery_app
from src.verifiers.base import VerificationOutput, get_verifier_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sync_session() -> Session:
    """Create a new synchronous SQLAlchemy session for Celery workers."""
    from src.db.session import get_sync_engine

    from sqlalchemy.orm import sessionmaker

    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def _update_job_status(
    session: Session,
    job_id: str,
    status: JobStatus,
    error_message: str | None = None,
) -> None:
    """Update an AnalysisJob's status (and optionally error_message) in place."""
    job = session.get(AnalysisJob, uuid.UUID(job_id))
    if job is None:
        raise ValueError(f"AnalysisJob {job_id} not found")
    job.status = status
    if error_message is not None:
        job.error_message = error_message
    session.commit()


# ---------------------------------------------------------------------------
# verify_claim_task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="src.tasks.verification.verify_claim_task",
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(OSError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=120,
    acks_late=True,
)
def verify_claim_task(self, claim_id: str, verifier_name: str) -> dict:
    """Run a single verifier on a single claim and persist results.

    Parameters
    ----------
    claim_id : str
        UUID of the :class:`Claim` row.
    verifier_name : str
        Key in the verifier registry identifying which verifier to use.

    Returns
    -------
    dict
        Serialised verification result containing ``claim_id``,
        ``verifier_name``, ``verdict``, ``confidence``, and ``reasoning``.
    """
    session = _get_sync_session()
    try:
        # ---- Load claim from DB ------------------------------------------
        claim = session.get(Claim, uuid.UUID(claim_id))
        if claim is None:
            raise ValueError(f"Claim {claim_id} not found in database")

        # ---- Resolve verifier --------------------------------------------
        registry = get_verifier_registry()
        try:
            verifier = registry.get_by_name(verifier_name)
        except KeyError:
            available = [v.name for v in registry.get_all()]
            raise ValueError(
                f"Verifier {verifier_name!r} not found in registry. "
                f"Available: {available}"
            )

        # ---- Run verifier (async -> sync bridge) -------------------------
        context: dict[str, Any] = {
            "claim_type": claim.claim_type.value,
            "source_span": claim.source_span,
            "job_id": str(claim.job_id),
        }
        try:
            output: VerificationOutput = asyncio.run(
                verifier.verify(
                    claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    context=context,
                )
            )
        except SoftTimeLimitExceeded:
            logger.warning(
                "Verifier %s timed out for claim %s", verifier_name, claim_id
            )
            output = VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Verification timed out after soft time limit ({verifier_name})",
                evidence={},
                source_db=verifier_name,
            )

        # ---- Persist VerificationResult ----------------------------------
        vr = VerificationResult(
            claim_id=uuid.UUID(claim_id),
            verifier_name=verifier_name,
            verdict=output.verdict,
            confidence=output.confidence,
            reasoning=output.reasoning,
            evidence=output.evidence,
        )
        session.add(vr)
        session.flush()  # populate vr.id

        # ---- Persist EvidenceRecord --------------------------------------
        if output.evidence or output.source_db:
            er = EvidenceRecord(
                verification_result_id=vr.id,
                source_db=output.source_db or verifier_name,
                query_used=claim.claim_text,
                raw_response=output.evidence,
                retrieval_timestamp=datetime.now(timezone.utc),
            )
            session.add(er)

        session.commit()

        result = {
            "claim_id": claim_id,
            "verifier_name": verifier_name,
            "verification_result_id": str(vr.id),
            "verdict": output.verdict.value,
            "confidence": output.confidence,
            "reasoning": output.reasoning,
        }
        logger.info(
            "Verified claim %s with %s -> %s (%.2f)",
            claim_id,
            verifier_name,
            output.verdict.value,
            output.confidence,
        )
        return result

    except SoftTimeLimitExceeded:
        raise  # let Celery handle the time-limit
    except (OSError, ConnectionError):
        raise  # autoretry handles these
    except Exception:
        logger.exception(
            "Unrecoverable error verifying claim %s with %s",
            claim_id,
            verifier_name,
        )
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# run_analysis_task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="src.tasks.verification.run_analysis_task",
    max_retries=3,
    default_retry_delay=15,
    autoretry_for=(OSError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=120,
    acks_late=True,
)
def run_analysis_task(self, job_id: str) -> dict:
    """Orchestrate an end-to-end hallucination analysis.

    Workflow
    --------
    1. Update job status to *extracting*.
    2. Load the job and extract claims with :class:`ClaimExtractor`.
    3. Classify each claim with :class:`ClaimClassifier`.
    4. Persist all claims to the database.
    5. Update job status to *verifying*.
    6. Fan out ``verify_claim_task`` for every (claim, verifier) pair
       via a Celery :func:`chord`, with :func:`aggregate_results_task`
       as the callback.

    Parameters
    ----------
    job_id : str
        UUID of the :class:`AnalysisJob`.

    Returns
    -------
    dict
        Job status summary with ``job_id``, ``status``, and ``claim_count``.
    """
    session = _get_sync_session()
    try:
        # ---- Step 1: Update status to extracting -------------------------
        _update_job_status(session, job_id, JobStatus.extracting)
        logger.info("Job %s: status -> extracting", job_id)

        # ---- Step 2: Load job and extract claims -------------------------
        job = session.get(AnalysisJob, uuid.UUID(job_id))
        if job is None:
            raise ValueError(f"AnalysisJob {job_id} not found")

        extractor = ClaimExtractor()
        extracted_claims = extractor.extract(job.input_text)

        if not extracted_claims:
            logger.warning("Job %s: no claims extracted", job_id)
            _update_job_status(session, job_id, JobStatus.completed)
            return {
                "job_id": job_id,
                "status": JobStatus.completed.value,
                "claim_count": 0,
            }

        # ---- Step 3: Classify claims -------------------------------------
        classifier = ClaimClassifier()
        claim_types = [classifier.classify(c) for c in extracted_claims]

        # ---- Step 4: Persist claims to DB --------------------------------
        db_claims: list[Claim] = []
        for ec, ct in zip(extracted_claims, claim_types):
            claim = Claim(
                job_id=uuid.UUID(job_id),
                claim_text=ec.claim_text,
                claim_type=ct,
                source_span={
                    "start": ec.source_span.start,
                    "end": ec.source_span.end,
                    "context": ec.source_span.context,
                },
                extraction_confidence=ec.confidence,
            )
            session.add(claim)
            db_claims.append(claim)

        session.flush()  # populate claim IDs

        # ---- Step 5: Update status to verifying --------------------------
        _update_job_status(session, job_id, JobStatus.verifying)
        logger.info(
            "Job %s: status -> verifying (%d claims)", job_id, len(db_claims)
        )

        # ---- Step 6: Fan out verification via chord ----------------------
        registry = get_verifier_registry()

        verification_signatures = []
        for claim in db_claims:
            matching_verifiers = registry.get_verifiers_for_type(claim.claim_type)
            for verifier in matching_verifiers:
                verification_signatures.append(
                    verify_claim_task.s(str(claim.id), verifier.name)
                )

        if not verification_signatures:
            # No verifiers match any claim — skip straight to aggregation.
            logger.warning(
                "Job %s: no matching verifiers for %d claims",
                job_id,
                len(db_claims),
            )
            from src.tasks.callbacks import aggregate_results_task

            aggregate_results_task.delay([], job_id)
        else:
            from src.tasks.callbacks import aggregate_results_task

            callback = aggregate_results_task.s(job_id)
            chord(verification_signatures)(callback)
            logger.info(
                "Job %s: dispatched chord with %d verification tasks",
                job_id,
                len(verification_signatures),
            )

        session.commit()

        return {
            "job_id": job_id,
            "status": JobStatus.verifying.value,
            "claim_count": len(db_claims),
        }

    except SoftTimeLimitExceeded:
        session.rollback()
        _update_job_status(session, job_id, JobStatus.failed, "Analysis task timed out")
        raise
    except (OSError, ConnectionError):
        session.rollback()
        raise  # autoretry handles these
    except Exception as exc:
        logger.exception("Job %s: analysis failed", job_id)
        session.rollback()
        try:
            _update_job_status(session, job_id, JobStatus.failed, str(exc))
        except Exception:
            logger.exception("Job %s: failed to update status to failed", job_id)
        raise
    finally:
        session.close()
