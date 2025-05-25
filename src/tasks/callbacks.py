"""Celery callback tasks for post-verification aggregation.

Tasks
-----
aggregate_results_task
    Chord callback that aggregates verification results and produces
    the final hallucination report for an analysis job.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.audit.sync_recorder import get_sync_audit_recorder
from src.models.audit import AuditEventType
from src.models.claim import Claim
from src.models.enums import JobStatus, Severity, Verdict
from src.models.job import AnalysisJob
from src.models.report import Report
from src.models.verification import VerificationResult
from src.observability import track_job
from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _dispatch_webhook(session: Session, job_id: str, event_type: str) -> None:
    """Queue a webhook notification if the job has a webhook_url configured.

    Silently ignored if the job has no webhook URL or if Celery dispatch
    fails for any reason — webhook delivery must never prevent the
    primary aggregation pipeline from completing.
    """
    try:
        job = session.get(AnalysisJob, uuid.UUID(job_id))
        if job is None or not job.webhook_url:
            return
        celery_app.send_task(
            "src.tasks.webhooks.send_webhook_notification",
            args=[job_id, event_type],
        )
        logger.info("Job %s: webhook %s dispatched", job_id, event_type)
    except Exception:  # noqa: BLE001
        logger.exception("Job %s: failed to dispatch webhook", job_id)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sync_session() -> Session:
    """Create a new synchronous SQLAlchemy session for Celery workers."""
    from sqlalchemy.orm import sessionmaker

    from src.db.session import get_sync_engine

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


def _compute_severity(overall_score: float) -> Severity:
    """Derive a :class:`Severity` from the overall verification score.

    The score is on a 0.0-1.0 scale where higher means more hallucination.
    """
    if overall_score <= 0.1:
        return Severity.clean
    if overall_score <= 0.3:
        return Severity.minor
    if overall_score <= 0.6:
        return Severity.major
    return Severity.critical


# ---------------------------------------------------------------------------
# aggregate_results_task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="src.tasks.callbacks.aggregate_results_task",
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(OSError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=120,
    acks_late=True,
)
def aggregate_results_task(
    self, results: list[dict], job_id: str
) -> dict:
    """Aggregate verification results and produce the final report.

    This task is designed to be used as a Celery chord callback.  After all
    ``verify_claim_task`` tasks complete, the list of their return values
    is passed as *results*.

    Parameters
    ----------
    results : list[dict]
        Return values from each completed ``verify_claim_task``.
    job_id : str
        UUID of the :class:`AnalysisJob`.

    Returns
    -------
    dict
        Report summary with ``job_id``, ``overall_score``, ``severity``,
        and per-verdict counts.
    """
    session = _get_sync_session()
    try:
        # ---- Step 1: Update status to aggregating ------------------------
        _update_job_status(session, job_id, JobStatus.aggregating)
        logger.info("Job %s: status -> aggregating", job_id)

        # ---- Step 2: Load all verification results from DB ---------------
        claims = (
            session.execute(
                select(Claim).where(Claim.job_id == uuid.UUID(job_id))
            )
            .scalars()
            .all()
        )
        claim_ids = [c.id for c in claims]

        if not claim_ids:
            logger.warning("Job %s: no claims found during aggregation", job_id)
            # Produce a clean report with zero claims.
            report = Report(
                job_id=uuid.UUID(job_id),
                overall_score=0.0,
                severity=Severity.clean,
                claim_count=0,
                verified_count=0,
                refuted_count=0,
                unverifiable_count=0,
                partial_count=0,
                section_breakdown={},
                report_metadata={"note": "No claims extracted"},
            )
            session.add(report)
            _update_job_status(session, job_id, JobStatus.completed)
            session.commit()
            try:
                track_job(JobStatus.completed.value, 0.0, claim_type="empty")
            except Exception:
                logger.exception("Job %s: metrics track_job failed", job_id)
            _dispatch_webhook(session, job_id, "job.completed")
            return {
                "job_id": job_id,
                "overall_score": 0.0,
                "severity": Severity.clean.value,
                "claim_count": 0,
            }

        verification_results = (
            session.execute(
                select(VerificationResult).where(
                    VerificationResult.claim_id.in_(claim_ids)
                )
            )
            .scalars()
            .all()
        )

        # ---- Step 3: Compute aggregate scores ----------------------------
        verdict_counts: dict[str, int] = {
            Verdict.verified.value: 0,
            Verdict.refuted.value: 0,
            Verdict.unverifiable.value: 0,
            Verdict.partially_supported.value: 0,
        }

        for vr in verification_results:
            verdict_val = vr.verdict.value if hasattr(vr.verdict, "value") else vr.verdict
            if verdict_val in verdict_counts:
                verdict_counts[verdict_val] += 1

        total_results = len(verification_results)
        claim_count = len(claims)

        # Overall hallucination score: proportion of results that are
        # refuted or unverifiable, weighted by confidence.
        if total_results > 0:
            hallucination_weight = sum(
                vr.confidence
                for vr in verification_results
                if vr.verdict in (Verdict.refuted, Verdict.unverifiable)
            )
            partial_weight = sum(
                vr.confidence * 0.5
                for vr in verification_results
                if vr.verdict == Verdict.partially_supported
            )
            total_confidence = sum(vr.confidence for vr in verification_results)
            if total_confidence > 0:
                overall_score = round(
                    (hallucination_weight + partial_weight) / total_confidence, 4
                )
            else:
                overall_score = 0.0
        else:
            overall_score = 0.0

        severity = _compute_severity(overall_score)

        # Build per-claim breakdown for the section_breakdown field.
        section_breakdown: dict[str, Any] = {}
        for claim in claims:
            claim_results = [
                vr for vr in verification_results if vr.claim_id == claim.id
            ]
            section_breakdown[str(claim.id)] = {
                "claim_text": claim.claim_text[:200],
                "claim_type": claim.claim_type.value,
                "results": [
                    {
                        "verifier": vr.verifier_name,
                        "verdict": vr.verdict.value,
                        "confidence": vr.confidence,
                    }
                    for vr in claim_results
                ],
            }

        # ---- Step 4: Persist Report --------------------------------------
        report = Report(
            job_id=uuid.UUID(job_id),
            overall_score=overall_score,
            severity=severity,
            claim_count=claim_count,
            verified_count=verdict_counts[Verdict.verified.value],
            refuted_count=verdict_counts[Verdict.refuted.value],
            unverifiable_count=verdict_counts[Verdict.unverifiable.value],
            partial_count=verdict_counts[Verdict.partially_supported.value],
            section_breakdown=section_breakdown,
            report_metadata={
                "total_verification_results": total_results,
                "chord_results_received": len(results),
            },
        )
        session.add(report)

        # ---- Step 5: Mark job completed ----------------------------------
        _update_job_status(session, job_id, JobStatus.completed)

        # ---- Audit: report generated + per-verdict records --------------
        # The audit entries share the transaction with the report commit
        # so they either all land together or roll back together.
        try:
            audit = get_sync_audit_recorder()
            if audit.enabled:
                tenant_id = getattr(
                    session.get(AnalysisJob, uuid.UUID(job_id)),
                    "tenant_id",
                    None,
                )
                audit.record(
                    session=session,
                    event_type=AuditEventType.report_generated,
                    resource_type="report",
                    resource_id=str(report.id),
                    action="create",
                    outcome="success",
                    tenant_id=tenant_id,
                    payload={
                        "job_id": job_id,
                        "overall_score": overall_score,
                        "severity": severity.value,
                        "claim_count": claim_count,
                    },
                )
                verdict_events = [
                    {
                        "event_type": AuditEventType.verdict_recorded,
                        "resource_type": "verification_result",
                        "resource_id": str(vr.id),
                        "action": "create",
                        "outcome": "success",
                        "tenant_id": tenant_id,
                        "payload": {
                            "claim_id": str(vr.claim_id),
                            "verifier": vr.verifier_name,
                            "verdict": (
                                vr.verdict.value
                                if hasattr(vr.verdict, "value")
                                else str(vr.verdict)
                            ),
                            "confidence": vr.confidence,
                        },
                    }
                    for vr in verification_results
                ]
                audit.record_batch(session, verdict_events)
        except Exception:
            # Audit write failure must never break the aggregation.
            logger.exception(
                "Job %s: audit write failed during aggregation", job_id
            )

        session.commit()

        summary = {
            "job_id": job_id,
            "report_id": str(report.id),
            "overall_score": overall_score,
            "severity": severity.value,
            "claim_count": claim_count,
            "verified_count": verdict_counts[Verdict.verified.value],
            "refuted_count": verdict_counts[Verdict.refuted.value],
            "unverifiable_count": verdict_counts[Verdict.unverifiable.value],
            "partial_count": verdict_counts[Verdict.partially_supported.value],
        }
        logger.info(
            "Job %s: completed — score=%.4f severity=%s claims=%d",
            job_id,
            overall_score,
            severity.value,
            claim_count,
        )

        # Metrics: record a terminal-state job. Duration is best-effort
        # (we don't have a submission timestamp plumbed through the
        # chord) so we fall back to 0 — the counter is still useful.
        try:
            dominant_type = "mixed"
            if claims:
                type_counts: dict[str, int] = {}
                for c in claims:
                    t = c.claim_type.value if hasattr(c.claim_type, "value") else str(c.claim_type)
                    type_counts[t] = type_counts.get(t, 0) + 1
                dominant_type = max(type_counts, key=type_counts.get)
            track_job(JobStatus.completed.value, 0.0, claim_type=dominant_type)
        except Exception:
            logger.exception("Job %s: metrics track_job failed", job_id)

        # Fire the customer webhook if configured.
        _dispatch_webhook(session, job_id, "job.completed")

        return summary

    except SoftTimeLimitExceeded:
        session.rollback()
        _update_job_status(session, job_id, JobStatus.failed, "Aggregation task timed out")
        raise
    except (OSError, ConnectionError):
        session.rollback()
        raise  # autoretry handles these
    except Exception as exc:
        logger.exception("Job %s: aggregation failed", job_id)
        session.rollback()
        try:
            _update_job_status(session, job_id, JobStatus.failed, str(exc))
        except Exception:
            logger.exception(
                "Job %s: failed to update status to failed", job_id
            )
        try:
            track_job(JobStatus.failed.value, 0.0, claim_type="unknown")
        except Exception:
            logger.exception("Job %s: metrics track_job failed", job_id)
        try:
            _dispatch_webhook(session, job_id, "job.failed")
        except Exception:
            logger.exception("Job %s: webhook dispatch failed", job_id)
        raise
    finally:
        session.close()
