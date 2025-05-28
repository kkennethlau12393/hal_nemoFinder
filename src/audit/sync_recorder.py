"""Synchronous audit recorder for Celery / Alembic contexts.

The async :class:`~src.audit.recorder.AuditRecorder` is the preferred
interface for request-scoped code.  Celery tasks, however, run against
a synchronous SQLAlchemy session and cannot easily ``await`` into the
async machinery.  This module provides a thin sync wrapper that shares
the same hashing logic so the chain remains consistent regardless of
which worker appended the entry.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.audit.recorder import (
    _core_fields,
    _coerce_uuid,
    compute_integrity_hash,
)
from src.models.audit import (
    GENESIS_PREV_HASH,
    AuditEventType,
    AuditLogEntry,
)

logger = logging.getLogger(__name__)


class SyncAuditRecorder:
    """Synchronous analogue of :class:`AuditRecorder`."""

    def __init__(self, hmac_key: str | bytes, *, enabled: bool = True) -> None:
        self._hmac_key = hmac_key
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        *,
        session: Session,
        event_type: AuditEventType | str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        actor: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[UUID | str] = None,
    ) -> AuditLogEntry | None:
        if not self._enabled:
            return None
        try:
            return self._do_record(
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
        except Exception:  # noqa: BLE001
            # Never let an audit write bring down a Celery task.
            logger.critical(
                "audit.sync.write_failed",
                exc_info=True,
                extra={
                    "event_type": str(event_type),
                    "resource_type": resource_type,
                    "resource_id": str(resource_id),
                },
            )
            return None

    def record_batch(
        self,
        session: Session,
        events: Iterable[Mapping[str, Any]],
    ) -> list[AuditLogEntry]:
        if not self._enabled:
            return []
        events_list = list(events)
        if not events_list:
            return []

        try:
            tail_seq, tail_hash = self._tail(session)
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
                        str(actor.get("user_id"))
                        if actor.get("user_id") else None
                    ),
                    "actor_api_key_id": (
                        str(actor.get("api_key_id"))
                        if actor.get("api_key_id") else None
                    ),
                    "actor_ip": actor.get("ip"),
                    "resource_type": raw["resource_type"],
                    "resource_id": str(raw["resource_id"]),
                    "action": raw["action"],
                    "outcome": raw["outcome"],
                    "payload": raw.get("payload"),
                }
                integrity = compute_integrity_hash(
                    self._hmac_key, tail_hash, core
                )
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
            session.flush()
            return entries
        except Exception:  # noqa: BLE001
            logger.critical("audit.sync.batch_write_failed", exc_info=True)
            session.rollback()
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_record(
        self,
        *,
        session: Session,
        event_type: AuditEventType | str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        actor: Optional[Mapping[str, Any]],
        payload: Optional[Mapping[str, Any]],
        tenant_id: Optional[UUID | str],
    ) -> AuditLogEntry:
        tail_seq, tail_hash = self._tail(session)
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
                str(actor.get("api_key_id"))
                if actor.get("api_key_id") else None
            ),
            "actor_ip": actor.get("ip"),
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "action": action,
            "outcome": outcome,
            "payload": dict(payload) if payload else None,
        }
        integrity = compute_integrity_hash(self._hmac_key, tail_hash, core)
        entry = AuditLogEntry(
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
        session.add(entry)
        session.flush()
        return entry

    def _tail(self, session: Session) -> tuple[int, str]:
        stmt = (
            select(AuditLogEntry.sequence, AuditLogEntry.integrity_hash)
            .order_by(AuditLogEntry.sequence.desc())
            .limit(1)
        )
        try:
            stmt = stmt.with_for_update()
        except Exception:  # pragma: no cover
            pass
        row = session.execute(stmt).first()
        if row is None:
            return 0, GENESIS_PREV_HASH
        return int(row[0]), str(row[1])

    @staticmethod
    def _coerce_event_type(value: AuditEventType | str) -> AuditEventType:
        if isinstance(value, AuditEventType):
            return value
        return AuditEventType(value)


_sync_recorder_singleton: Optional[SyncAuditRecorder] = None


def get_sync_audit_recorder() -> SyncAuditRecorder:
    """Return the process-wide :class:`SyncAuditRecorder` singleton."""
    global _sync_recorder_singleton  # noqa: PLW0603
    if _sync_recorder_singleton is not None:
        return _sync_recorder_singleton

    from src.config import settings

    enabled = bool(getattr(settings, "AUDIT_ENABLED", False))
    key = getattr(settings, "AUDIT_HMAC_KEY", None)
    _sync_recorder_singleton = SyncAuditRecorder(
        hmac_key=key or "",
        enabled=enabled and bool(key),
    )
    return _sync_recorder_singleton


def reset_sync_audit_recorder() -> None:
    """Test helper — clears the cached singleton."""
    global _sync_recorder_singleton  # noqa: PLW0603
    _sync_recorder_singleton = None
