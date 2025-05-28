"""Audit recorder — persists events and verifies HMAC integrity chain.

The recorder is the single entry point application code uses to write
audit events.  Every write:

1. Takes a row lock on the tail of the ``audit_log`` table so
   concurrent writers see a consistent ``(sequence, prev_hash)`` pair.
2. Computes ``integrity_hash = HMAC-SHA256(key, canonical_json(payload))``
   where ``payload`` includes the previous row's hash — producing a
   tamper-evident chain.
3. Inserts the new row.  The row is never subsequently updated or
   deleted.

Integrity verification walks the chain from the beginning, recomputing
each HMAC and confirming it matches the stored ``integrity_hash``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.audit import (
    GENESIS_PREV_HASH,
    AuditEventType,
    AuditLogEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical JSON / HMAC helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def canonical_json(data: Mapping[str, Any]) -> str:
    """Return a deterministic, stable JSON representation of *data*.

    Keys are sorted, whitespace is stripped, and known non-JSON types
    (UUID, datetime, Enum) are normalised.  The output is deterministic
    for the same logical input across Python versions and processes,
    which is a prerequisite for HMAC chain verification.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _core_fields(entry: AuditLogEntry) -> dict[str, Any]:
    """Extract the fields that participate in the integrity hash."""
    return {
        "sequence": entry.sequence,
        "event_type": (
            entry.event_type.value
            if isinstance(entry.event_type, AuditEventType)
            else entry.event_type
        ),
        "tenant_id": str(entry.tenant_id) if entry.tenant_id else None,
        "actor_user_id": str(entry.actor_user_id) if entry.actor_user_id else None,
        "actor_api_key_id": (
            str(entry.actor_api_key_id) if entry.actor_api_key_id else None
        ),
        "actor_ip": entry.actor_ip,
        "resource_type": entry.resource_type,
        "resource_id": entry.resource_id,
        "action": entry.action,
        "outcome": entry.outcome,
        "payload": entry.payload,
    }


def compute_integrity_hash(
    key: str | bytes,
    prev_hash: str,
    core_fields: Mapping[str, Any],
) -> str:
    """Compute the HMAC-SHA256 integrity hash for an entry.

    Parameters
    ----------
    key:
        HMAC secret key (``AUDIT_HMAC_KEY``).  Strings are encoded as
        UTF-8.
    prev_hash:
        The ``integrity_hash`` of the preceding row, or
        :data:`~src.models.audit.GENESIS_PREV_HASH` for the first row.
    core_fields:
        Mapping returned by :func:`_core_fields`.
    """
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key
    payload = {"prev_hash": prev_hash, "core": dict(core_fields)}
    message = canonical_json(payload).encode("utf-8")
    return hmac.new(key_bytes, message, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# IntegrityReport
# ---------------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """Result of walking the audit-log chain."""

    valid: bool
    total_checked: int
    first_invalid_sequence: Optional[int] = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AuditRecorder
# ---------------------------------------------------------------------------


class AuditRecorderError(RuntimeError):
    """Raised when the recorder cannot be configured correctly."""


class AuditRecorder:
    """Writes and verifies entries in the audit log.

    The recorder is stateless apart from its HMAC key; it can safely
    be instantiated once at process start and shared across requests.
    Concurrency safety relies on row-level locks obtained via
    ``SELECT ... FOR UPDATE`` against the tail of the table.
    """

    def __init__(self, hmac_key: str | bytes, *, enabled: bool = True) -> None:
        if enabled and not hmac_key:
            raise AuditRecorderError(
                "AuditRecorder requires a non-empty HMAC key when enabled"
            )
        self._hmac_key = hmac_key
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def record(
        self,
        *,
        event_type: AuditEventType | str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        session: AsyncSession,
        actor: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[UUID | str] = None,
        commit: bool = False,
    ) -> AuditLogEntry:
        """Persist a single audit event.

        ``actor`` is an optional mapping that may contain any of:
        ``user_id``, ``api_key_id``, ``ip``.  It accepts the same shape
        the middleware passes in, so callers outside the middleware can
        supply ``{"user_id": ..., "ip": ...}`` directly.

        If ``commit`` is True the recorder flushes and commits the
        session; otherwise the caller is expected to commit.  For
        request-scoped sessions the default (False) is the right
        choice so the audit entry participates in the request's
        transaction.
        """
        if not self._enabled:
            # No-op when auditing is globally disabled — the caller
            # still gets back a dummy entry object to keep call-sites
            # simple.
            return AuditLogEntry(
                sequence=0,
                event_type=event_type,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                outcome=outcome,
                prev_hash=GENESIS_PREV_HASH,
                integrity_hash="",
                payload=dict(payload) if payload else None,
            )

        entry = await self._build_entry(
            session=session,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            actor=actor,
            payload=payload,
            tenant_id=tenant_id,
        )
        session.add(entry)
        await session.flush()
        if commit:
            await session.commit()
        return entry

    async def record_batch(
        self,
        events: Iterable[Mapping[str, Any]],
        session: AsyncSession,
        *,
        commit: bool = False,
    ) -> list[AuditLogEntry]:
        """Append a batch of events as a single chain extension.

        This is the hot-path for verification callbacks that need to
        write one entry per verifier result.  The sequence + prev_hash
        bookkeeping is computed in-memory after a single tail lookup,
        producing ``len(events)`` inserts inside one flush.
        """
        events_list = list(events)
        if not events_list:
            return []

        if not self._enabled:
            return []

        tail_seq, tail_hash = await self._tail(session)
        entries: list[AuditLogEntry] = []

        for offset, raw in enumerate(events_list, start=1):
            next_seq = tail_seq + offset
            actor = raw.get("actor") or {}
            tenant_id = raw.get("tenant_id")
            core = {
                "sequence": next_seq,
                "event_type": self._coerce_event_type(raw["event_type"]).value,
                "tenant_id": str(tenant_id) if tenant_id else None,
                "actor_user_id": (
                    str(actor.get("user_id")) if actor.get("user_id") else None
                ),
                "actor_api_key_id": (
                    str(actor.get("api_key_id")) if actor.get("api_key_id") else None
                ),
                "actor_ip": actor.get("ip"),
                "resource_type": raw["resource_type"],
                "resource_id": str(raw["resource_id"]),
                "action": raw["action"],
                "outcome": raw["outcome"],
                "payload": raw.get("payload"),
            }
            integrity = compute_integrity_hash(self._hmac_key, tail_hash, core)

            entry = AuditLogEntry(
                sequence=next_seq,
                event_type=self._coerce_event_type(raw["event_type"]),
                tenant_id=_coerce_uuid(tenant_id),
                actor_user_id=_coerce_uuid(actor.get("user_id")),
                actor_api_key_id=_coerce_uuid(actor.get("api_key_id")),
                actor_ip=actor.get("ip"),
                resource_type=raw["resource_type"],
                resource_id=str(raw["resource_id"]),
                action=raw["action"],
                outcome=raw["outcome"],
                payload=raw.get("payload"),
                prev_hash=tail_hash,
                integrity_hash=integrity,
            )
            entries.append(entry)
            tail_hash = integrity

        session.add_all(entries)
        await session.flush()
        if commit:
            await session.commit()
        return entries

    async def verify_integrity(
        self,
        session: AsyncSession,
        limit: Optional[int] = None,
        *,
        start_sequence: int = 0,
        start_prev_hash: str = GENESIS_PREV_HASH,
    ) -> IntegrityReport:
        """Walk the audit-log chain verifying every HMAC.

        Parameters
        ----------
        session:
            Active async session.
        limit:
            Optional cap on the number of rows to verify.  Useful for
            incremental nightly jobs that verify the tail of the chain.
        start_sequence:
            Resume verification at this sequence (exclusive).  Combined
            with ``start_prev_hash`` this supports checkpointed
            verification over very large chains.
        start_prev_hash:
            The ``integrity_hash`` of the row immediately preceding
            ``start_sequence``, or :data:`GENESIS_PREV_HASH` at the
            start of the chain.
        """
        stmt = (
            select(AuditLogEntry)
            .where(AuditLogEntry.sequence > start_sequence)
            .order_by(AuditLogEntry.sequence.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        rows = result.scalars().all()

        errors: list[str] = []
        first_invalid: Optional[int] = None
        prev_hash = start_prev_hash
        expected_seq = start_sequence + 1
        total = 0

        for row in rows:
            total += 1
            if row.sequence != expected_seq:
                errors.append(
                    f"sequence gap at {row.sequence} "
                    f"(expected {expected_seq})"
                )
                if first_invalid is None:
                    first_invalid = row.sequence
            if row.prev_hash != prev_hash:
                errors.append(
                    f"prev_hash mismatch at sequence {row.sequence}: "
                    f"stored={row.prev_hash[:12]}... expected={prev_hash[:12]}..."
                )
                if first_invalid is None:
                    first_invalid = row.sequence
            recomputed = compute_integrity_hash(
                self._hmac_key, prev_hash, _core_fields(row)
            )
            if recomputed != row.integrity_hash:
                errors.append(
                    f"integrity_hash mismatch at sequence {row.sequence}"
                )
                if first_invalid is None:
                    first_invalid = row.sequence
            prev_hash = row.integrity_hash
            expected_seq = row.sequence + 1

        return IntegrityReport(
            valid=not errors,
            total_checked=total,
            first_invalid_sequence=first_invalid,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _tail(self, session: AsyncSession) -> tuple[int, str]:
        """Fetch ``(sequence, integrity_hash)`` of the latest row.

        Takes a ``FOR UPDATE`` lock on Postgres so concurrent writers
        serialise on the tail.  Falls back gracefully on SQLite (used
        by the tests) where ``FOR UPDATE`` is a no-op.
        """
        stmt = (
            select(AuditLogEntry.sequence, AuditLogEntry.integrity_hash)
            .order_by(AuditLogEntry.sequence.desc())
            .limit(1)
        )
        try:
            stmt = stmt.with_for_update()
        except Exception:  # pragma: no cover — safety net
            pass

        row = (await session.execute(stmt)).first()
        if row is None:
            return 0, GENESIS_PREV_HASH
        return int(row[0]), str(row[1])

    async def _build_entry(
        self,
        *,
        session: AsyncSession,
        event_type: AuditEventType | str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        actor: Optional[Mapping[str, Any]],
        payload: Optional[Mapping[str, Any]],
        tenant_id: Optional[UUID | str],
    ) -> AuditLogEntry:
        tail_seq, tail_hash = await self._tail(session)
        next_seq = tail_seq + 1
        actor = actor or {}
        core = {
            "sequence": next_seq,
            "event_type": self._coerce_event_type(event_type).value,
            "tenant_id": str(tenant_id) if tenant_id else None,
            "actor_user_id": (
                str(actor.get("user_id")) if actor.get("user_id") else None
            ),
            "actor_api_key_id": (
                str(actor.get("api_key_id")) if actor.get("api_key_id") else None
            ),
            "actor_ip": actor.get("ip"),
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "action": action,
            "outcome": outcome,
            "payload": dict(payload) if payload else None,
        }
        integrity = compute_integrity_hash(self._hmac_key, tail_hash, core)
        return AuditLogEntry(
            sequence=next_seq,
            event_type=self._coerce_event_type(event_type),
            tenant_id=_coerce_uuid(tenant_id),
            actor_user_id=_coerce_uuid(actor.get("user_id")),
            actor_api_key_id=_coerce_uuid(actor.get("api_key_id")),
            actor_ip=actor.get("ip"),
            resource_type=resource_type,
            resource_id=str(resource_id),
            action=action,
            outcome=outcome,
            payload=dict(payload) if payload else None,
            prev_hash=tail_hash,
            integrity_hash=integrity,
        )

    @staticmethod
    def _coerce_event_type(value: AuditEventType | str) -> AuditEventType:
        if isinstance(value, AuditEventType):
            return value
        try:
            return AuditEventType(value)
        except ValueError as exc:
            raise AuditRecorderError(f"Unknown audit event type: {value!r}") from exc


def _coerce_uuid(value: Any) -> Optional[UUID]:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_recorder_singleton: Optional[AuditRecorder] = None


def get_audit_recorder() -> AuditRecorder:
    """Return the process-wide :class:`AuditRecorder` singleton.

    Reads ``AUDIT_ENABLED`` / ``AUDIT_HMAC_KEY`` from the active
    settings on first call.  If auditing is disabled the returned
    recorder is a no-op that still satisfies the interface, so
    caller code can unconditionally ``await recorder.record(...)``.
    """
    global _recorder_singleton  # noqa: PLW0603
    if _recorder_singleton is not None:
        return _recorder_singleton

    from src.config import settings

    enabled = bool(getattr(settings, "AUDIT_ENABLED", False))
    key = getattr(settings, "AUDIT_HMAC_KEY", None)

    if enabled and not key:
        raise AuditRecorderError(
            "AUDIT_ENABLED is true but AUDIT_HMAC_KEY is unset. "
            "Generate one with `openssl rand -hex 32`."
        )

    _recorder_singleton = AuditRecorder(
        hmac_key=key or "", enabled=enabled and bool(key)
    )
    return _recorder_singleton


def reset_audit_recorder() -> None:
    """Test helper — clears the cached singleton."""
    global _recorder_singleton  # noqa: PLW0603
    _recorder_singleton = None
