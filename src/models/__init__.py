"""SQLAlchemy models for hal_nemoFinder."""

from src.models.audit import AuditEventType, AuditLogEntry
from src.models.base import Base, TimestampMixin, UUIDMixin
from src.models.claim import Claim
from src.models.enums import ClaimType, JobStatus, Severity, Verdict
from src.models.job import AnalysisJob, BatchJob, BatchStatus
from src.models.report import Report
from src.models.tenant import ApiKey, Role, Tenant, User
from src.models.verification import EvidenceRecord, VerificationResult

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDMixin",
    "AnalysisJob",
    "BatchJob",
    "BatchStatus",
    "Claim",
    "VerificationResult",
    "EvidenceRecord",
    "Report",
    "JobStatus",
    "ClaimType",
    "Verdict",
    "Severity",
    # Multi-tenant auth models
    "Tenant",
    "User",
    "ApiKey",
    "Role",
    # Audit log
    "AuditLogEntry",
    "AuditEventType",
]
