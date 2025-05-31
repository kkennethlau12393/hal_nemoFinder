"""Drift detection for per-verifier performance over time.

A :class:`DriftDetector` maintains a rolling window of
``(predicted_verdict, actual_label)`` observations for each verifier.
When the accuracy on the most recent window diverges meaningfully from
the historical baseline, :meth:`check` returns a :class:`DriftReport`.

The statistical test is a plain 2x2 chi-square test for proportions --
"fraction-correct in the recent window" versus "fraction-correct in the
baseline".  The chi-square CDF is approximated via Wilson-Hilferty so
the module does not depend on ``scipy``.

A Prometheus gauge ``hal_nemofinder_verifier_drift_score`` is updated
every time :meth:`check` runs, labelled by verifier name.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque

from src.models.enums import Verdict

logger = logging.getLogger(__name__)

__all__ = ["DriftDetector", "DriftReport"]


@dataclass(slots=True)
class _Observation:
    """A single labelled prediction kept in the rolling buffer."""

    predicted_verdict: Verdict
    actual_label: bool
    confidence: float
    timestamp: datetime

    @property
    def correct(self) -> bool:
        """Whether the prediction matched the label.

        Positive class is ``Verdict.refuted`` (= predicted hallucination).
        """
        predicted_positive = self.predicted_verdict == Verdict.refuted
        return predicted_positive == self.actual_label


@dataclass(slots=True)
class DriftReport:
    """Structured drift signal for a single verifier.

    Attributes
    ----------
    verifier_name : str
        Name of the drifting verifier.
    window_accuracy, baseline_accuracy : float
        Accuracy on the recent window vs. the historical baseline.
    delta : float
        ``window_accuracy - baseline_accuracy`` (negative = regression).
    p_value : float
        Chi-square p-value for the observed difference.
    method : str
        Statistical test used (``"chi_square"`` or ``"ks"``).
    alert_level : str
        ``"info"``, ``"warning"``, or ``"critical"``.
    recommendation : str
        Human-readable suggested action.
    recorded_at : datetime
        UTC timestamp at which the report was generated.
    """

    verifier_name: str
    window_accuracy: float
    baseline_accuracy: float
    delta: float
    p_value: float
    method: str
    alert_level: str
    recommendation: str
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict representation."""
        return {
            "verifier_name": self.verifier_name,
            "window_accuracy": self.window_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "delta": self.delta,
            "p_value": self.p_value,
            "method": self.method,
            "alert_level": self.alert_level,
            "recommendation": self.recommendation,
            "recorded_at": self.recorded_at.isoformat(),
        }


class DriftDetector:
    """Detect per-verifier accuracy drift over a rolling window.

    Parameters
    ----------
    window_size : int
        Number of most-recent observations considered the "current" window.
    baseline_size : int
        Number of observations used as the historical baseline.  The
        buffer therefore retains ``window_size + baseline_size`` items.
    significance_threshold : float
        p-value below which the detector fires a drift alert.
    method : str
        Statistical test: ``"chi_square"`` (default) or ``"ks"`` (a
        two-sample test on the confidence distribution).
    """

    def __init__(
        self,
        window_size: int = 500,
        baseline_size: int = 2000,
        significance_threshold: float = 0.05,
        method: str = "chi_square",
    ) -> None:
        if window_size <= 0 or baseline_size <= 0:
            raise ValueError("window_size and baseline_size must be positive")
        if method not in {"chi_square", "ks"}:
            raise ValueError(f"unknown drift method: {method!r}")
        self._window_size = int(window_size)
        self._baseline_size = int(baseline_size)
        self._alpha = float(significance_threshold)
        self._method = method
        #: Per-verifier rolling observation buffer.  Newest items at the
        #: right; capacity = window_size + baseline_size.
        self._buffers: dict[str, Deque[_Observation]] = {}

    # -- Recording ----------------------------------------------------------

    def record(
        self,
        verifier_name: str,
        predicted_verdict: Verdict,
        actual_label: bool,
        confidence: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Append an observation for *verifier_name*.

        Parameters
        ----------
        verifier_name : str
            The verifier that produced the prediction.
        predicted_verdict : Verdict
            The verifier's verdict (positive class = ``refuted``).
        actual_label : bool
            Ground-truth hallucination label.
        confidence : float
            Verifier self-reported confidence in [0, 1].
        timestamp : datetime | None
            Event time (defaults to ``datetime.now(timezone.utc)``).
        """
        buf = self._buffers.get(verifier_name)
        if buf is None:
            buf = deque(maxlen=self._window_size + self._baseline_size)
            self._buffers[verifier_name] = buf
        buf.append(
            _Observation(
                predicted_verdict=predicted_verdict,
                actual_label=bool(actual_label),
                confidence=float(confidence),
                timestamp=timestamp or datetime.now(timezone.utc),
            )
        )

    # -- Querying -----------------------------------------------------------

    def check(self, verifier_name: str) -> DriftReport | None:
        """Return a :class:`DriftReport` if drift is detected for *verifier_name*.

        Returns ``None`` if the buffer does not yet contain enough data
        (at least ``window_size`` recent + 2 baseline points) or if the
        test does not reach :attr:`significance_threshold`.
        """
        buf = self._buffers.get(verifier_name)
        if buf is None:
            return None

        n_total = len(buf)
        if n_total < self._window_size + 2:
            return None

        obs = list(buf)
        window = obs[-self._window_size :]
        baseline = obs[: -self._window_size]
        if not baseline:
            return None

        window_acc = _accuracy(window)
        baseline_acc = _accuracy(baseline)

        if self._method == "chi_square":
            p_value = _chi_square_p_value(window, baseline)
        else:
            p_value = _ks_p_value(
                [o.confidence for o in window],
                [o.confidence for o in baseline],
            )

        delta = round(window_acc - baseline_acc, 4)

        # Always update the Prometheus gauge (even when no alert fires) so
        # dashboards can chart the absolute delta over time.
        _update_drift_gauge(verifier_name, abs(delta))

        if p_value >= self._alpha:
            return None

        alert_level, recommendation = _classify_alert(delta, p_value)
        return DriftReport(
            verifier_name=verifier_name,
            window_accuracy=round(window_acc, 4),
            baseline_accuracy=round(baseline_acc, 4),
            delta=delta,
            p_value=round(p_value, 6),
            method=self._method,
            alert_level=alert_level,
            recommendation=recommendation,
        )

    def check_all(self) -> dict[str, DriftReport]:
        """Run :meth:`check` for every verifier with recorded data."""
        reports: dict[str, DriftReport] = {}
        for name in list(self._buffers.keys()):
            r = self.check(name)
            if r is not None:
                reports[name] = r
        return reports

    # -- Introspection ------------------------------------------------------

    def observation_count(self, verifier_name: str) -> int:
        """Return the number of buffered observations for *verifier_name*."""
        buf = self._buffers.get(verifier_name)
        return len(buf) if buf is not None else 0

    def verifiers(self) -> list[str]:
        """Return all verifier names with at least one recorded observation."""
        return list(self._buffers.keys())


# ---------------------------------------------------------------------------
# Statistical helpers (stdlib only)
# ---------------------------------------------------------------------------


def _accuracy(observations: list[_Observation]) -> float:
    """Fraction-correct over *observations*.  Returns 0 for empty input."""
    if not observations:
        return 0.0
    hits = sum(1 for o in observations if o.correct)
    return hits / len(observations)


def _chi_square_p_value(
    window: list[_Observation], baseline: list[_Observation]
) -> float:
    """2x2 chi-square test on ``[correct, incorrect] x [window, baseline]``.

    Returns the upper-tail p-value using the Wilson-Hilferty
    approximation (no scipy dependency).
    """
    w_correct = sum(1 for o in window if o.correct)
    w_wrong = len(window) - w_correct
    b_correct = sum(1 for o in baseline if o.correct)
    b_wrong = len(baseline) - b_correct

    row_totals = [w_correct + w_wrong, b_correct + b_wrong]
    col_totals = [w_correct + b_correct, w_wrong + b_wrong]
    n = row_totals[0] + row_totals[1]
    if n == 0 or any(t == 0 for t in col_totals):
        return 1.0

    observed = [[w_correct, w_wrong], [b_correct, b_wrong]]
    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            expected = row_totals[i] * col_totals[j] / n
            if expected <= 0:
                continue
            diff = observed[i][j] - expected
            chi2 += (diff * diff) / expected

    # 2x2 table -> 1 degree of freedom.
    return _chi2_sf(chi2, df=1)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution.

    Uses the Wilson-Hilferty cube-root transformation which is
    well-behaved for df >= 1 and accurate to ~1e-3 for typical drift
    p-values.  Degenerate inputs (x <= 0) return 1.0.
    """
    if x <= 0.0:
        return 1.0
    if df <= 0:
        return 1.0
    # z ~ N(0, 1) under H0, per Wilson-Hilferty.
    third = 1.0 / 3.0
    mean = 1.0 - 2.0 / (9.0 * df)
    var = 2.0 / (9.0 * df)
    z = ((x / df) ** third - mean) / math.sqrt(var)
    return _standard_normal_sf(z)


def _standard_normal_sf(z: float) -> float:
    """Upper-tail P(Z > z) via the error-function identity."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _ks_p_value(sample_a: list[float], sample_b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov test.

    Uses the asymptotic formula
    ``p ~= 2 * sum_{k=1..inf} (-1)^(k-1) exp(-2 k^2 lambda^2)``
    where ``lambda = (sqrt(n_eff) + 0.12 + 0.11 / sqrt(n_eff)) * D``.
    """
    if not sample_a or not sample_b:
        return 1.0
    a = sorted(sample_a)
    b = sorted(sample_b)
    na = len(a)
    nb = len(b)
    i = j = 0
    fa = fb = 0.0
    d = 0.0
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
            fa = i / na
        else:
            j += 1
            fb = j / nb
        d = max(d, abs(fa - fb))

    n_eff = (na * nb) / (na + nb)
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d

    # Truncated alternating series — 101 terms is comfortably enough.
    total = 0.0
    for k in range(1, 101):
        total += ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
    p = 2.0 * total
    return max(0.0, min(1.0, p))


def _classify_alert(delta: float, p_value: float) -> tuple[str, str]:
    """Map ``(delta, p_value)`` to a severity and human recommendation."""
    magnitude = abs(delta)
    direction = "improved" if delta > 0 else "degraded"
    if magnitude >= 0.15 or p_value < 0.001:
        level = "critical"
        rec = (
            f"Verifier has {direction} by {magnitude:.1%} (p={p_value:.4f}). "
            "Pause the verifier and retrain/recalibrate immediately."
        )
    elif magnitude >= 0.05:
        level = "warning"
        rec = (
            f"Verifier has {direction} by {magnitude:.1%} (p={p_value:.4f}). "
            "Schedule a recalibration run and review recent evidence."
        )
    else:
        level = "info"
        rec = (
            f"Minor drift detected ({magnitude:.2%}). "
            "No immediate action required; continue monitoring."
        )
    return level, rec


# ---------------------------------------------------------------------------
# Prometheus integration (optional)
# ---------------------------------------------------------------------------


def _update_drift_gauge(verifier_name: str, score: float) -> None:
    """Best-effort update of the ``hal_nemofinder_verifier_drift_score`` gauge.

    Wrapped in a try/except so that unit tests running without the
    observability module installed still work.
    """
    try:
        from src.observability.metrics import VERIFIER_DRIFT_SCORE

        VERIFIER_DRIFT_SCORE.labels(verifier=verifier_name).set(
            max(0.0, float(score))
        )
    except Exception:  # noqa: BLE001
        logger.debug("Drift gauge update skipped", exc_info=True)
