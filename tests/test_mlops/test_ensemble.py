"""Tests for :mod:`src.core.ensemble`."""

from __future__ import annotations

from src.core.aggregator import BayesianAggregator
from src.core.ensemble import (
    EnsembleAggregator,
    MajorityVoteAggregator,
    StackingAggregator,
    WeightedVoteAggregator,
)
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


def test_majority_vote_picks_mode() -> None:
    agg = MajorityVoteAggregator()
    results = [
        _vo("chemical", Verdict.refuted, 0.9),
        _vo("pubchem", Verdict.refuted, 0.8),
        _vo("crossref", Verdict.verified, 0.7),
    ]
    out = agg.aggregate_claim(results)
    assert out.verdict == Verdict.refuted


def test_majority_tie_break_by_confidence() -> None:
    agg = MajorityVoteAggregator()
    results = [
        _vo("chemical", Verdict.refuted, 0.4),
        _vo("pubchem", Verdict.verified, 0.95),
    ]
    out = agg.aggregate_claim(results)
    assert out.verdict == Verdict.verified


def test_weighted_vote_respects_reliability() -> None:
    agg = WeightedVoteAggregator()
    # 'statistical' has sensitivity 0.93; 'consistency' has 0.70.
    results = [
        _vo("statistical", Verdict.refuted, 0.9),
        _vo("consistency", Verdict.verified, 0.9),
    ]
    out = agg.aggregate_claim(results)
    assert out.verdict == Verdict.refuted


def test_ensemble_majority_requires_two_refuted() -> None:
    """EnsembleAggregator in 'majority' mode needs >=2 sub-aggregators to agree."""
    ensemble = EnsembleAggregator(strictness="majority")

    # Case 1: two confident refuted votes + one soft verified -> all three
    # sub-aggregators should call refuted -> ensemble says refuted.
    strong_refute = [
        _vo("chemical", Verdict.refuted, 0.99),
        _vo("pubchem", Verdict.refuted, 0.99),
        _vo("crossref", Verdict.refuted, 0.99),
    ]
    out = ensemble.aggregate_claim(strong_refute)
    assert out.verdict == Verdict.refuted

    # Case 2: a single weak refuted vote should NOT win because the
    # ensemble requires >=2 refuted votes to stick.
    weak_mixed = [
        _vo("chemical", Verdict.refuted, 0.3),
        _vo("pubchem", Verdict.verified, 0.95),
        _vo("crossref", Verdict.verified, 0.95),
    ]
    out = ensemble.aggregate_claim(weak_mixed)
    assert out.verdict != Verdict.refuted


def test_ensemble_unanimous_strictness() -> None:
    ensemble = EnsembleAggregator(strictness="unanimous")
    mixed = [
        _vo("chemical", Verdict.refuted, 0.9),
        _vo("pubchem", Verdict.verified, 0.9),
    ]
    out = ensemble.aggregate_claim(mixed)
    # With no unanimity across the 3 sub-aggregators we fall back to
    # partially_supported.
    assert out.verdict == Verdict.partially_supported


def test_stacking_mode_over_multiple_bases() -> None:
    stacker = StackingAggregator(
        bases=[
            BayesianAggregator(),
            MajorityVoteAggregator(),
            WeightedVoteAggregator(),
        ]
    )
    results = [
        _vo("chemical", Verdict.refuted, 0.95),
        _vo("pubchem", Verdict.refuted, 0.95),
        _vo("crossref", Verdict.refuted, 0.95),
    ]
    out = stacker.aggregate_claim(results)
    assert out.verdict == Verdict.refuted


def test_majority_with_empty_results_is_unverifiable() -> None:
    out = MajorityVoteAggregator().aggregate_claim([])
    assert out.verdict == Verdict.unverifiable
