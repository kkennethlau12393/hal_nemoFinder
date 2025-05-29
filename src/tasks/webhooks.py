"""Webhook delivery task.

Customers can supply ``webhook_url`` and ``webhook_secret`` on each
analysis request.  When the job reaches a terminal state we POST a
JSON envelope describing the event.  The payload is signed with
HMAC-SHA256 if a secret was supplied, which the customer can verify
before trusting the message.

Delivery is retried with exponential backoff on transient network
errors and 5xx responses.  Successful delivery (2xx) or a permanent
4xx error terminates the task.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.job import AnalysisJob
from src.models.report import Report
from src.observability import get_logger
from src.tasks.celery_app import celery_app

logger = get_logger(__name__)

#: Hard upper bound on how long we're willing to wait on the customer
#: endpoint before considering the delivery a failure.
WEBHOOK_TIMEOUT_SECONDS = 10

#: HTTP header used to carry the HMAC-SHA256 signature.
SIGNATURE_HEADER = "X-Hal-Signature"


def _get_sync_session() -> Session:
    """Construct a new synchronous SQLAlchemy session."""
    from sqlalchemy.orm import sessionmaker

    from src.db.session import get_sync_engine

    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Return the hex-encoded HMAC-SHA256 of *payload_bytes* under *secret*."""
    digest = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def _build_payload(job: AnalysisJob, event_type: str, report: Report | None) -> dict[str, Any]:
    """Assemble the JSON envelope delivered to the customer endpoint."""
    payload: dict[str, Any] = {
        "event_type": event_type,
        "job_id": str(job.id),
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if event_type == "job.completed" and report is not None:
        payload["summary"] = {
            "report_id": str(report.id),
            "overall_score": report.overall_score,
            "severity": (
                report.severity.value
                if hasattr(report.severity, "value")
                else str(report.severity)
            ),
            "claim_count": report.claim_count,
            "verified_count": report.verified_count,
            "refuted_count": report.refuted_count,
            "unverifiable_count": report.unverifiable_count,
            "partial_count": report.partial_count,
        }
    if event_type == "job.failed":
        payload["error"] = job.error_message
    return payload


@celery_app.task(
    bind=True,
    name="src.tasks.webhooks.send_webhook_notification",
    max_retries=3,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    acks_late=True,
)
def send_webhook_notification(self, job_id: str, event_type: str) -> dict[str, Any]:
    """Deliver a webhook notification for *job_id*.

    Parameters
    ----------
    job_id
        UUID string of the :class:`AnalysisJob` whose state changed.
    event_type
        Either ``"job.completed"`` or ``"job.failed"``.
    """
    session = _get_sync_session()
    try:
        job = session.get(AnalysisJob, uuid.UUID(job_id))
        if job is None:
            logger.warning("webhook.job.not_found", job_id=job_id)
            return {"delivered": False, "reason": "job-not-found"}
        if not job.webhook_url:
            logger.debug("webhook.skip.no_url", job_id=job_id)
            return {"delivered": False, "reason": "no-webhook-url"}

        report = session.execute(
            select(Report).where(Report.job_id == uuid.UUID(job_id))
        ).scalar_one_or_none()

        payload = _build_payload(job, event_type, report)
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "hal_nemofinder-webhook/1.0",
            "X-Hal-Event": event_type,
            "X-Hal-Delivery": str(uuid.uuid4()),
        }
        if job.webhook_secret:
            headers[SIGNATURE_HEADER] = _sign_payload(job.webhook_secret, payload_bytes)

        logger.info(
            "webhook.delivery.attempt",
            job_id=job_id,
            event_type=event_type,
            webhook_url=job.webhook_url,
            attempt=self.request.retries + 1,
        )

        try:
            response = requests.post(
                job.webhook_url,
                data=payload_bytes,
                headers=headers,
                timeout=WEBHOOK_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "webhook.delivery.network_error",
                job_id=job_id,
                error=str(exc),
            )
            raise self.retry(exc=exc)

        # --- Success: any 2xx counts ---------------------------------
        if 200 <= response.status_code < 300:
            logger.info(
                "webhook.delivery.success",
                job_id=job_id,
                status_code=response.status_code,
                event_type=event_type,
            )
            return {
                "delivered": True,
                "status_code": response.status_code,
                "attempt": self.request.retries + 1,
            }

        # --- Transient failure: 5xx -> retry --------------------------
        if response.status_code >= 500:
            logger.warning(
                "webhook.delivery.server_error",
                job_id=job_id,
                status_code=response.status_code,
            )
            raise self.retry(
                exc=RuntimeError(f"webhook returned {response.status_code}")
            )

        # --- Permanent failure: 4xx (except 408/429) ------------------
        if response.status_code in (408, 429):
            logger.warning(
                "webhook.delivery.throttled",
                job_id=job_id,
                status_code=response.status_code,
            )
            raise self.retry(
                exc=RuntimeError(f"webhook throttled with {response.status_code}")
            )

        logger.error(
            "webhook.delivery.permanent_failure",
            job_id=job_id,
            status_code=response.status_code,
        )
        return {
            "delivered": False,
            "status_code": response.status_code,
            "reason": "permanent-4xx",
        }
    finally:
        session.close()
