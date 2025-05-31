"""Bootstrap confidence intervals for Bayesian hallucination scores.

Quantifies uncertainty due to verifier disagreement by resampling the
set of :class:`VerificationOutput` objects with replacement and running
the :class:`BayesianAggregator` on each bootstrap replicate.

A claim where every verifier agrees produces a tight CI; a claim where
verifiers disagree produces a wide CI.  The CI can be used to flag
"borderline" claims whose posterior straddles the decision threshold
even though the point estimate is on one side of it.

The core path is pure-Python (``random.choices``) and does not require
``numpy`` or ``scipy``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.aggregator import BayesianAggregator
    from src.verifiers.base import VerificationOutput


__all__ = ["BootstrapConfidenceInterval", "CIResult"]


@dataclass(slots=True)
class CIResult:
    """Summary of a bootstrap confidence interval on a posterior.

    Attributes
    ----------
    point_estimate : float
        Posterior P(hallucination) computed on the full set (no resampling).
    lower : float
        2.5th percentile of the bootstrap distribution.
    upper : float
        97.5th percentile of the bootstrap distribution.
    median : float
        50th percentile of the bootstrap distribution.
    std : float
        Sample standard deviation of the bootstrap distribution.
    width : float
        ``upper - lower``.  Useful shorthand for "how uncertain is this?".
    n_bootstrap : int
        Number of bootstrap replicates drawn.
    """

    point_estimate: float
    lower: float
    upper: float
    median: float
    std: float
    width: float
    n_bootstrap: int

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Return ``True`` if the CI does not straddle *threshold*.

        A CI that sits entirely above or entirely below the decision
        threshold is "significant" in the sense that bootstrap sampling
        never flips the verdict: verifier agreement is strong enough.
        """
        return self.lower > threshold or self.upper < threshold

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe dict representation of the interval."""
        return {
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "median": self.median,
            "std": self.std,
            "width": self.width,
            "n_bootstrap": self.n_bootstrap,
        }


class BootstrapConfidenceInterval:
    """Non-parametric bootstrap CI on the Bayesian hallucination posterior.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap replicates.  1000 is a reasonable default;
        100 is enough for a rough gauge.
    seed : int | None
        Optional RNG seed for reproducible intervals (handy in tests).
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int | None = None) -> None:
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")
        self._n = int(n_bootstrap)
        self._rng = random.Random(seed)

    # -- Public API ---------------------------------------------------------

    def compute(
        self,
        results: "list[VerificationOutput]",
        aggregator: "BayesianAggregator",
    ) -> CIResult:
        """Compute the bootstrap CI for the Bayesian posterior on *results*.

        Parameters
        ----------
        results : list[VerificationOutput]
            Per-verifier outputs for a single claim.
        aggregator : BayesianAggregator
            The aggregator to run on each bootstrap replicate.

        Returns
        -------
        CIResult
            Point estimate plus the 2.5/50/97.5 percentiles of the
            bootstrap distribution.
        """
        if not results:
            return CIResult(
                point_estimate=aggregator.posterior_probability([]),
                lower=0.0,
                upper=0.0,
                median=0.0,
                std=0.0,
                width=0.0,
                n_bootstrap=0,
            )

        point = float(aggregator.posterior_probability(results))

        samples: list[float] = []
        n = len(results)
        for _ in range(self._n):
            resample = self._rng.choices(results, k=n)
            samples.append(float(aggregator.posterior_probability(resample)))

        samples.sort()
        lower = _percentile(samples, 2.5)
        upper = _percentile(samples, 97.5)
        median = _percentile(samples, 50.0)
        std = _stddev(samples)

        return CIResult(
            point_estimate=round(point, 6),
            lower=round(lower, 6),
            upper=round(upper, 6),
            median=round(median, 6),
            std=round(std, 6),
            width=round(upper - lower, 6),
            n_bootstrap=self._n,
        )


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list.

    *q* is expressed in ``[0, 100]``.  Mirrors ``numpy.percentile`` with
    the default ``linear`` interpolation mode.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(100.0, float(q)))
    idx = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _stddev(values: list[float]) -> float:
    """Sample standard deviation (ddof=1).  Returns 0 for ``n<2``."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    ss = sum((v - mean) ** 2 for v in values)
    return math.sqrt(ss / (n - 1))
