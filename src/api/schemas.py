"""Pydantic v2 schemas for API request/response serialization."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from src.models.enums import ClaimType, JobStatus, Severity, Verdict
from src.models.job import BatchStatus
from src.models.tenant import Role


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for ``POST /api/v1/analyze``."""

    text: str
    config: dict | None = None
    webhook_url: Optional[HttpUrl] = Field(
        default=None,
        description=(
            "Optional URL to POST a completion notification to when the job "
            "reaches a terminal state."
        ),
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        max_length=256,
        description=(
            "Optional shared secret used to sign the webhook payload with "
            "HMAC-SHA256. Sent as the ``X-Hal-Signature`` header."
        ),
    )


class BatchAnalyzeRequest(BaseModel):
    """Request body for ``POST /api/v1/analyze/batch``."""

    texts: list[str] = Field(..., min_length=1, max_length=500)
    config: dict | None = None
    webhook_url: Optional[HttpUrl] = None
    webhook_secret: Optional[str] = Field(default=None, max_length=256)

    @field_validator("texts")
    @classmethod
    def _texts_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("All texts must be non-empty.")
        return cleaned


class BatchResponse(BaseModel):
    """Response body for a newly-created batch job."""

    batch_id: UUID
    job_ids: list[UUID]
    total_jobs: int
    progress_url: str


class BatchStatusResponse(BaseModel):
    """Progress report for an existing batch."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: BatchStatus
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    progress: float
    created_at: datetime
    updated_at: datetime


class BatchReportResponse(BaseModel):
    """Aggregate report across every job in a completed batch."""

    batch_id: UUID
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_score: float
    severity_counts: dict[str, int]
    reports: list["ReportResponse"]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class VerdictSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    verifier_name: str
    verdict: Verdict
    confidence: float
    reasoning: str | None


class EvidenceRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_db: str
    query_used: str
    retrieval_timestamp: datetime
    kb_version: str | None


class VerificationDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    verifier_name: str
    verdict: Verdict
    confidence: float
    reasoning: str | None
    evidence: dict | None
    evidence_records: list[EvidenceRecordResponse]


class ClaimResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    claim_text: str
    claim_type: ClaimType
    source_span: dict | None
    extraction_confidence: float
    verdicts: list[VerdictSummary]


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    claim_count: int | None = None


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    overall_score: float
    severity: Severity
    claim_count: int
    verified_count: int
    refuted_count: int
    unverifiable_count: int
    partial_count: int
    section_breakdown: dict | None
    claims: list[ClaimResponse]


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe payload."""

    status: str = "alive"


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe payload — ``ready`` is false if any dep fails."""

    ready: bool
    database: bool
    redis: bool
    verifiers_registered: int
    failures: list[str] = Field(default_factory=list)


class PluginStatus(BaseModel):
    loaded: int
    errors: list[str] = Field(default_factory=list)


class CalibrationStatus(BaseModel):
    last_run: Optional[datetime] = None
    accuracy: Optional[float] = None
    brier_score: Optional[float] = None
    ece: Optional[float] = None


class HealthResponse(BaseModel):
    """Comprehensive health report returned by ``GET /health``."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    version: str
    environment: str
    database: bool
    redis: bool
    verifiers: dict[str, bool]
    plugins: PluginStatus
    calibration: CalibrationStatus


class StatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_jobs: int
    completed_jobs: int
    avg_hallucination_score: float
    hallucination_rate_by_type: dict


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Auth / multi-tenancy schemas
# ---------------------------------------------------------------------------


class TenantCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/tenants``."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="Lowercase kebab-case identifier, e.g. 'acme-research'.",
    )
    rate_limit_rpm: int = Field(60, ge=1, le=100_000)
    rate_limit_analyze_pm: int = Field(10, ge=1, le=10_000)
    config: dict | None = None


class TenantResponse(BaseModel):
    """Public representation of a :class:`src.models.tenant.Tenant`."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    is_active: bool
    rate_limit_rpm: int
    rate_limit_analyze_pm: int
    config: dict | None = None
    created_at: datetime
    updated_at: datetime


class UserCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/tenants/{tenant_id}/users``."""

    email: str = Field(..., min_length=3, max_length=320)
    role: Role = Role.viewer
    external_id: str | None = Field(
        default=None,
        max_length=255,
        description="OIDC 'sub' claim for SSO users.",
    )


class UserResponse(BaseModel):
    """Public representation of a :class:`src.models.tenant.User`."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: UUID
    email: str
    external_id: str | None = None
    role: Role
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ApiKeyCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/api-keys``."""

    name: str = Field(..., min_length=1, max_length=255)
    role: Role = Role.viewer
    user_id: UUID | None = None
    scopes: list[str] | None = None
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    """Representation of an API key *without* the plaintext secret."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID | None
    name: str
    key_prefix: str
    role: Role
    scopes: list[str] | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    created_at: datetime


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response returned exactly once when a new API key is issued.

    The ``plaintext_key`` field is populated only on creation — it is
    never returned by subsequent reads and is never logged.
    """

    plaintext_key: str = Field(
        ...,
        description=(
            "The plaintext API key.  Returned exactly once; store it now "
            "or rotate the key."
        ),
    )


class AuthContextResponse(BaseModel):
    """Response payload for ``GET /api/v1/auth/me``."""

    tenant: TenantResponse
    user: UserResponse | None = None
    api_key_prefix: str | None = None
    role: Role
    permissions: list[str]
    anonymous: bool


# Resolve forward references declared with string literals above.
BatchReportResponse.model_rebuild()
