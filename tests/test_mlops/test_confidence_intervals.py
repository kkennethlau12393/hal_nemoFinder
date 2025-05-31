"""Tests for :mod:`src.core.confidence_intervals`."""

from __future__ import annotations

from src.core.aggregator import BayesianAggregator
from src.core.confidence_intervals import BootstrapConfidenceInterval, CIResult
from src.models.enums import Verdict
from src.verifiers.base import VerificationOutput


def _vo(source: str, verdict: Verdict, confidence: float = 0.9) -> VerificationOutput:
    return VerificationOutput(
        verdict=verdict,
        confidence=confidence,
        reasoning="",
        evidence={},
        source_db=source,
    )


def test_ci_structure_and_percentile_ordering() -> None:
    agg = BayesianAggregator()
    results = [
        _vo("chemical", Verdict.refuted),
        _vo("pubchem", Verdict.refuted),
        _vo("crossref", Verdict.refuted),
    ]
    tool = BootstrapConfidenceInterval(n_bootstrap=200, seed=7)
    ci = tool.compute(results, agg)

    assert isinstance(ci, CIResult)
    assert ci.n_bootstrap == 200
    assert 0.0 <= ci.lower <= ci.median <= ci.upper <= 1.0
    assert ci.width == round(ci.upper - ci.lower, 6)
    assert 0.0 <= ci.point_estimate <= 1.0


def test_ci_width_is_tighter_on_agreement() -> None:
    """High-agreement verifiers should yield a tighter CI than a disagreeing mix."""
    agg = BayesianAggregator()

    agree = [
        _vo("chemical", Verdict.refuted, 0.95),
        _vo("pubchem", Verdict.refuted, 0.95),
        _vo("crossref", Verdict.refuted, 0.95),
        _vo("uniprot", Verdict.refuted, 0.95),
    ]
    disagree = [
        _vo("chemical", Verdict.refuted, 0.95),
        _vo("pubchem", Verdict.verified, 0.95),
        _vo("crossref", Verdict.refuted, 0.95),
        _vo("uniprot", Verdict.verified, 0.95),
    ]

    tool = BootstrapConfidenceInterval(n_bootstrap=500, seed=42)
    agree_ci = tool.compute(agree, agg)
    disagree_ci = tool.compute(disagree, agg)

    assert disagree_ci.width >= agree_ci.width
    # The tight-agreement CI should refuse to straddle 0.5.
    assert agree_ci.is_significant(0.5) is True


def test_significance_detection() -> None:
    agg = BayesianAggregator()
    results = [_vo("chemical", Verdict.refuted, 0.99)] * 5
    tool = BootstrapConfidenceInterval(n_bootstrap=300, seed=1)
    ci = tool.compute(results, agg)
    assert ci.is_significant(threshold=0.5) is True


def test_aggregate_claim_with_ci_roundtrip() -> None:
    agg = BayesianAggregator()
    results = [
        _vo("chemical", Verdict.refuted, 0.9),
        _vo("pubchem", Verdict.refuted, 0.8),
    ]
    verdict, ci = agg.aggregate_claim_with_ci(results, n_bootstrap=100, seed=0)
    assert verdict.verdict in {
        Verdict.refuted,
        Verdict.partially_supported,
        Verdict.verified,
    }
    assert ci.n_bootstrap == 100
    assert 0.0 <= ci.lower <= ci.upper <= 1.0


def test_empty_results_returns_zero_width() -> None:
    agg = BayesianAggregator()
    tool = BootstrapConfidenceInterval(n_bootstrap=50, seed=0)
    ci = tool.compute([], agg)
    assert ci.width == 0.0
    assert ci.n_bootstrap == 0
