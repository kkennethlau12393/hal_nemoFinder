"""Cost tracking for hal-nemoFinder verifications.

The :class:`CostTracker` models three sources of cost:

1. **Compute** -- wall-clock seconds multiplied by a configurable
   ``$/CPU-hour`` rate (default ``$0.10``, matching a typical cloud
   vCPU on-demand price).
2. **API calls** -- each knowledge-base client reports a per-call
   dollar cost.  Most public sources are free; the few commercial
   providers (DrugBank) have non-zero per-call costs.
3. **Storage** -- an amortised fixed fee per record stored in the
   audit log + evidence store.  Callers can set their own rate.

The tracker is process-local.  Per-job costs are attributable by
calling :meth:`start_job` / :meth:`end_job` around a verification run;
totals aggregate across all jobs for the process lifetime.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["CostTracker", "CostReport"]


#: Default dollar-per-call rates for each knowledge-base client.  Most
#: public sources are free; commercial ones are non-zero.  Customers
#: override any of these via :class:`CostTracker`.
_DEFAULT_API_COSTS: dict[str, float] = {
    "pubchem": 0.0,
    "chembl": 0.0,
    "crossref": 0.0,
    "uniprot": 0.0,
    "clinicaltrials": 0.0,
    "drugbank": 0.001,
    "pubmed": 0.0,
    "faers": 0.0,
    "pdb": 0.0,
}

#: Default amortised cost per stored audit/evidence record (USD).
_DEFAULT_STORAGE_COST_PER_RECORD: float = 0.000_001


@dataclass(slots=True)
class CostReport:
    """Snapshot of costs over a window (job or lifetime).

    Attributes
    ----------
    compute_cost_usd, api_cost_usd, storage_cost_usd : float
        Per-category dollar totals.
    total_cost_usd : float
        Sum of the three categories.
    api_call_counts : dict[str, int]
        Per-client call counts contributing to ``api_cost_usd``.
    cpu_seconds : float
        Total CPU-seconds billed to compute.
    storage_records : int
        Number of audit/evidence records billed to storage.
    recorded_at : datetime
        UTC timestamp the report was generated.
    """

    compute_cost_usd: float
    api_cost_usd: float
    storage_cost_usd: float
    total_cost_usd: float
    api_call_counts: dict[str, int] = field(default_factory=dict)
    cpu_seconds: float = 0.0
    storage_records: int = 0
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict view of the report."""
        return {
            "compute_cost_usd": self.compute_cost_usd,
            "api_cost_usd": self.api_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "api_call_counts": dict(self.api_call_counts),
            "cpu_seconds": self.cpu_seconds,
            "storage_records": self.storage_records,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass(slots=True)
class _Bucket:
    """Per-job or global accumulator for a cost breakdown."""

    cpu_seconds: float = 0.0
    api_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    storage_records: int = 0


class CostTracker:
    """Accumulate compute / API / storage costs for verification work.

    Parameters
    ----------
    settings : Settings | None
        If provided, the CPU-hour rate is read from
        ``settings.COST_CPU_PER_HOUR``.  When ``None``, the default
        ``$0.10 / CPU-hour`` is used.
    api_cost_overrides : dict[str, float] | None
        Override the built-in per-client prices.
    storage_cost_per_record : float
        Dollar cost charged per stored audit/evidence record.
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        api_cost_overrides: dict[str, float] | None = None,
        storage_cost_per_record: float = _DEFAULT_STORAGE_COST_PER_RECORD,
    ) -> None:
        cpu_rate = 0.10
        if settings is not None and hasattr(settings, "COST_CPU_PER_HOUR"):
            try:
                cpu_rate = float(settings.COST_CPU_PER_HOUR)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Could not read COST_CPU_PER_HOUR from settings; "
                    "falling back to 0.10",
                    exc_info=True,
                )
        self._cpu_cost_per_hour = cpu_rate
        self._storage_cost_per_record = float(storage_cost_per_record)
        self._api_costs: dict[str, float] = dict(_DEFAULT_API_COSTS)
        if api_cost_overrides:
            self._api_costs.update(api_cost_overrides)
        self._global = _Bucket()
        self._jobs: dict[str, _Bucket] = {}

    # -- Recording ----------------------------------------------------------

    def record_api_call(
        self, client: str, count: int = 1, job_id: str | None = None
    ) -> None:
        """Record *count* API calls against *client* (and optional *job_id*)."""
        if count <= 0:
            return
        name = str(client).strip().lower() or "unknown"
        self._global.api_calls[name] += int(count)
        if job_id is not None:
            self._bucket_for_job(job_id).api_calls[name] += int(count)

        self._emit_api_counter(name, int(count))
        per_call = self._api_costs.get(name, 0.0)
        if per_call > 0.0:
            self._emit_cost_counter("api", per_call * int(count))

    def record_compute(
        self,
        duration_seconds: float,
        cpu_count: int = 1,
        job_id: str | None = None,
    ) -> None:
        """Record *duration_seconds* of compute (billed at cpu_count CPUs)."""
        if duration_seconds <= 0 or cpu_count <= 0:
            return
        cpu_secs = float(duration_seconds) * int(cpu_count)
        self._global.cpu_seconds += cpu_secs
        if job_id is not None:
            self._bucket_for_job(job_id).cpu_seconds += cpu_secs

        cost = cpu_secs * self._cpu_cost_per_hour / 3600.0
        if cost > 0.0:
            self._emit_cost_counter("compute", cost)

    def record_storage(
        self, records: int = 1, job_id: str | None = None
    ) -> None:
        """Record *records* audit/evidence rows stored."""
        if records <= 0:
            return
        self._global.storage_records += int(records)
        if job_id is not None:
            self._bucket_for_job(job_id).storage_records += int(records)

        cost = int(records) * self._storage_cost_per_record
        if cost > 0.0:
            self._emit_cost_counter("storage", cost)

    # -- Reporting ----------------------------------------------------------

    def get_total_cost(self) -> CostReport:
        """Return a :class:`CostReport` for all recorded activity."""
        return self._report_from_bucket(self._global)

    def get_cost_per_job(self, job_id: str) -> CostReport:
        """Return a :class:`CostReport` for a single job, empty if unknown."""
        bucket = self._jobs.get(job_id)
        if bucket is None:
            return CostReport(
                compute_cost_usd=0.0,
                api_cost_usd=0.0,
                storage_cost_usd=0.0,
                total_cost_usd=0.0,
            )
        return self._report_from_bucket(bucket)

    def known_jobs(self) -> list[str]:
        """Return the job ids currently tracked."""
        return list(self._jobs.keys())

    # -- Internals ----------------------------------------------------------

    def _bucket_for_job(self, job_id: str) -> _Bucket:
        bucket = self._jobs.get(job_id)
        if bucket is None:
            bucket = _Bucket()
            self._jobs[job_id] = bucket
        return bucket

    def _report_from_bucket(self, bucket: _Bucket) -> CostReport:
        compute_cost = (
            bucket.cpu_seconds * self._cpu_cost_per_hour / 3600.0
        )
        api_cost = sum(
            self._api_costs.get(name, 0.0) * count
            for name, count in bucket.api_calls.items()
        )
        storage_cost = bucket.storage_records * self._storage_cost_per_record
        total = compute_cost + api_cost + storage_cost
        return CostReport(
            compute_cost_usd=round(compute_cost, 6),
            api_cost_usd=round(api_cost, 6),
            storage_cost_usd=round(storage_cost, 6),
            total_cost_usd=round(total, 6),
            api_call_counts=dict(bucket.api_calls),
            cpu_seconds=round(bucket.cpu_seconds, 4),
            storage_records=bucket.storage_records,
        )

    @staticmethod
    def _emit_cost_counter(category: str, delta_usd: float) -> None:
        try:
            from src.observability.metrics import COST_USD_TOTAL

            COST_USD_TOTAL.labels(category=category).inc(float(delta_usd))
        except Exception:  # noqa: BLE001
            logger.debug("COST_USD_TOTAL emit skipped", exc_info=True)

    @staticmethod
    def _emit_api_counter(client: str, count: int) -> None:
        try:
            from src.observability.metrics import API_CALLS_TOTAL

            API_CALLS_TOTAL.labels(client=client).inc(int(count))
        except Exception:  # noqa: BLE001
            logger.debug("API_CALLS_TOTAL emit skipped", exc_info=True)
