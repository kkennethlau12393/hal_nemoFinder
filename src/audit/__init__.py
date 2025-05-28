"""Audit logging subsystem.

This package provides an append-only, HMAC-chained audit log for
21 CFR Part 11 compliance.  Application code should import
:class:`AuditRecorder` and the :class:`AuditEventType` enum from here.
"""

from src.audit.recorder import (
    AuditRecorder,
    IntegrityReport,
    canonical_json,
    compute_integrity_hash,
    get_audit_recorder,
)
from src.audit.sync_recorder import SyncAuditRecorder, get_sync_audit_recorder
from src.models.audit import AuditEventType, AuditLogEntry, GENESIS_PREV_HASH

__all__ = [
    "AuditRecorder",
    "AuditEventType",
    "AuditLogEntry",
    "IntegrityReport",
    "GENESIS_PREV_HASH",
    "canonical_json",
    "compute_integrity_hash",
    "get_audit_recorder",
    "SyncAuditRecorder",
    "get_sync_audit_recorder",
]
