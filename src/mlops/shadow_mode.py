"""Shadow-mode / champion-vs-challenger A/B testing for verifiers.

:class:`ShadowModeRouter` wraps two :class:`~src.verifiers.base.BaseVerifier`
instances -- a "champion" and a "challenger" -- and runs them side by
side.  Only the champion's verdict is returned to the caller; the
challenger runs in shadow and its output is recorded via a
:class:`ShadowRecorder` for offline analysis.

This is the mechanism operators use to validate a new verifier against
production traffic without risking user-visible regressions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models.enums import ClaimType
from src.verifiers.base import BaseVerifier, VerificationOutput

logger = logging.getLogger(__name__)

__all__ = [
    "ShadowRecord",
    "ShadowRecorder",
    "ShadowComparisonReport",
    "ShadowModeRouter",
]


@dataclass(slots=True)
class ShadowRecord:
    """A single champion/challenger comparison event.

    Attributes
    ----------
    claim_text : str
        The claim that was verified.
    claim_type : ClaimType
        Claim classification.
    champion_verdict : str
        Champion's verdict value.
    challenger_verdict : str
        Challenger's verdict value.
    champion_confidence, challenger_confidence : float
        Self-reported confidence scores.
    champion_duration_ms, challenger_duration_ms : float
        Wall-clock runtime in milliseconds.
    agreement : bool
        ``True`` if the two verdicts match exactly.
    error : str
        Non-empty when the challenger raised; the champion's verdict is
        still returned.
    timestamp : datetime
        UTC event timestamp.
    """

    claim_text: str
    claim_type: ClaimType
    champion_verdict: str
    challenger_verdict: str
    champion_confidence: float
    challenger_confidence: float
    champion_duration_ms: float
    challenger_duration_ms: float
    agreement: bool
    error: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ShadowRecorder:
    """Bounded in-memory store of :class:`ShadowRecord` events.

    A real deployment will typically replace this with a database-backed
    implementation; the in-memory version is sufficient for tests and
    short-lived analysis sessions.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._records: list[ShadowRecord] = []

    def record(self, record: ShadowRecord) -> None:
        """Append *record*, evicting the oldest entry if at capacity."""
        if len(self._records) >= self._capacity:
            # Drop the oldest item — cheap for typical capacities.
            self._records.pop(0)
        self._records.append(record)

    @property
    def records(self) -> list[ShadowRecord]:
        """Return a shallow copy of all recorded events."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


@dataclass(slots=True)
class ShadowComparisonReport:
    """Aggregate comparison report for a champion/challenger pair.

    Attributes
    ----------
    champion_name, challenger_name : str
        The two verifiers being compared.
    total_comparisons : int
        Number of shadow records analysed.
    agreement_count, disagreement_count : int
        Split of ``total_comparisons`` by verdict agreement.
    agreement_rate : float
        ``agreement_count / total_comparisons`` (0 when no data).
    champion_verdicts, challenger_verdicts : dict[str, int]
        Verdict distribution for each side.
    confusion_matrix : dict[tuple[str, str], int]
        Counts keyed by ``(champion_verdict, challenger_verdict)``.
    mean_champion_duration_ms, mean_challenger_duration_ms : float
        Mean per-call runtime (milliseconds).
    """

    champion_name: str
    challenger_name: str
    total_comparisons: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    champion_verdicts: dict[str, int] = field(default_factory=dict)
    challenger_verdicts: dict[str, int] = field(default_factory=dict)
    confusion_matrix: dict[tuple[str, str], int] = field(default_factory=dict)
    mean_champion_duration_ms: float = 0.0
    mean_challenger_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict view of the report."""
        return {
            "champion_name": self.champion_name,
            "challenger_name": self.challenger_name,
            "total_comparisons": self.total_comparisons,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "agreement_rate": self.agreement_rate,
            "champion_verdicts": dict(self.champion_verdicts),
            "challenger_verdicts": dict(self.challenger_verdicts),
            "confusion_matrix": {
                f"{a}->{b}": v for (a, b), v in self.confusion_matrix.items()
            },
            "mean_champion_duration_ms": self.mean_champion_duration_ms,
            "mean_challenger_duration_ms": self.mean_challenger_duration_ms,
        }


class ShadowModeRouter:
    """Run *champion* and *challenger* verifiers side by side.

    The caller always receives the champion's :class:`VerificationOutput`;
    the challenger runs concurrently and its result is recorded (or its
    failure is logged, never surfaced).

    Parameters
    ----------
    champion : BaseVerifier
        The production verifier whose verdict is returned to the caller.
    challenger : BaseVerifier
        The candidate verifier being evaluated.
    sampling_rate : float
        Fraction of requests in ``[0, 1]`` for which the challenger is
        actually invoked.  ``1.0`` means "shadow every request".
    recorder : ShadowRecorder | None
        Store for :class:`ShadowRecord` events.  A new in-memory
        recorder is created if omitted.
    seed : int | None
        Optional RNG seed used for sampling decisions (useful in tests).
    """

    def __init__(
        self,
        champion: BaseVerifier,
        challenger: BaseVerifier,
        sampling_rate: float = 1.0,
        recorder: ShadowRecorder | None = None,
        seed: int | None = None,
    ) -> None:
        if not (0.0 <= sampling_rate <= 1.0):
            raise ValueError("sampling_rate must be in [0, 1]")
        self._champion = champion
        self._challenger = challenger
        self._sampling_rate = float(sampling_rate)
        self._recorder = recorder or ShadowRecorder()
        self._rng = random.Random(seed)

    # -- Public API ---------------------------------------------------------

    @property
    def recorder(self) -> ShadowRecorder:
        """The underlying :class:`ShadowRecorder`."""
        return self._recorder

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        """Return the champion verdict; run the challenger in shadow.

        If *sampling_rate* is less than 1.0 the challenger is skipped on
        some fraction of calls.  An error raised by the challenger is
        logged and recorded as a synthetic :class:`ShadowRecord` with
        the ``error`` field populated — it never propagates to the
        caller.
        """
        should_shadow = self._rng.random() < self._sampling_rate

        champion_t0 = time.perf_counter()
        champion_task = asyncio.create_task(
            self._champion.verify(claim_text, claim_type, context)
        )

        challenger_task: asyncio.Task[VerificationOutput] | None = None
        challenger_t0: float | None = None
        if should_shadow:
            challenger_t0 = time.perf_counter()
            challenger_task = asyncio.create_task(
                self._challenger.verify(claim_text, claim_type, context)
            )

        try:
            champion_output = await champion_task
        except BaseException:
            if challenger_task is not None:
                challenger_task.cancel()
            raise
        champion_duration_ms = (time.perf_counter() - champion_t0) * 1000.0

        if challenger_task is None:
            return champion_output

        challenger_output: VerificationOutput | None
        challenger_error = ""
        try:
            challenger_output = await challenger_task
        except BaseException as exc:  # noqa: BLE001
            logger.warning(
                "Shadow challenger %r raised on claim; champion result unaffected: %s",
                self._challenger.name,
                exc,
            )
            challenger_output = None
            challenger_error = f"{type(exc).__name__}: {exc}"

        challenger_duration_ms = (
            (time.perf_counter() - (challenger_t0 or time.perf_counter())) * 1000.0
        )

        champion_verdict = champion_output.verdict.value
        challenger_verdict = (
            challenger_output.verdict.value if challenger_output else "error"
        )
        agreement = (
            challenger_output is not None
            and champion_output.verdict == challenger_output.verdict
        )

        self._recorder.record(
            ShadowRecord(
                claim_text=claim_text,
                claim_type=claim_type,
                champion_verdict=champion_verdict,
                challenger_verdict=challenger_verdict,
                champion_confidence=float(champion_output.confidence),
                challenger_confidence=(
                    float(challenger_output.confidence)
                    if challenger_output
                    else 0.0
                ),
                champion_duration_ms=champion_duration_ms,
                challenger_duration_ms=challenger_duration_ms,
                agreement=agreement,
                error=challenger_error,
            )
        )

        if not agreement:
            self._emit_disagreement_metric()

        return champion_output

    def get_comparison_report(self) -> ShadowComparisonReport:
        """Compute a :class:`ShadowComparisonReport` over all recorded events."""
        records = self._recorder.records
        total = len(records)
        if total == 0:
            return ShadowComparisonReport(
                champion_name=self._champion.name,
                challenger_name=self._challenger.name,
                total_comparisons=0,
                agreement_count=0,
                disagreement_count=0,
                agreement_rate=0.0,
            )

        champ_counts: Counter[str] = Counter()
        chall_counts: Counter[str] = Counter()
        conf: dict[tuple[str, str], int] = defaultdict(int)
        agreement_count = 0
        sum_champ_dur = 0.0
        sum_chall_dur = 0.0

        for r in records:
            champ_counts[r.champion_verdict] += 1
            chall_counts[r.challenger_verdict] += 1
            conf[(r.champion_verdict, r.challenger_verdict)] += 1
            if r.agreement:
                agreement_count += 1
            sum_champ_dur += r.champion_duration_ms
            sum_chall_dur += r.challenger_duration_ms

        return ShadowComparisonReport(
            champion_name=self._champion.name,
            challenger_name=self._challenger.name,
            total_comparisons=total,
            agreement_count=agreement_count,
            disagreement_count=total - agreement_count,
            agreement_rate=round(agreement_count / total, 4),
            champion_verdicts=dict(champ_counts),
            challenger_verdicts=dict(chall_counts),
            confusion_matrix=dict(conf),
            mean_champion_duration_ms=round(sum_champ_dur / total, 4),
            mean_challenger_duration_ms=round(sum_chall_dur / total, 4),
        )

    # -- Internals ----------------------------------------------------------

    def _emit_disagreement_metric(self) -> None:
        """Best-effort increment of the Prometheus disagreement counter."""
        try:
            from src.observability.metrics import SHADOW_DISAGREEMENTS_TOTAL

            SHADOW_DISAGREEMENTS_TOTAL.labels(
                champion=self._champion.name,
                challenger=self._challenger.name,
            ).inc()
        except Exception:  # noqa: BLE001
            logger.debug("Shadow disagreement metric emit skipped", exc_info=True)
