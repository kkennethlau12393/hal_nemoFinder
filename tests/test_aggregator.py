"""Tests for src.core.aggregator.ResultAggregator."""

from __future__ import annotations

import pytest

from src.core.aggregator import (
    AggregatedVerdict,
    ClaimInfo,
    ReportData,
    ResultAggregator,
)
from src.models.enums import ClaimType, Severity, Verdict
from src.verifiers.base import VerificationOutput


def _vo(
    verdict: Verdict,
    confidence: float = 0.8,
    reasoning: str = "test",
    source_db: str = "test_db",
) -> VerificationOutput:
    """Shortcut to build a VerificationOutput."""
    return VerificationOutput(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        source_db=source_db,
    )


def _claim_info(
    text: str = "test claim",
    claim_type: ClaimType = ClaimType.molecular_property,
    extraction_confidence: float = 0.8,
) -> ClaimInfo:
    """Shortcut to build a ClaimInfo."""
    return ClaimInfo(
        claim_text=text,
        claim_type=claim_type,
        extraction_confidence=extraction_confidence,
    )


# -------------------------------------------------------------------------
# Per-claim aggregation
# -------------------------------------------------------------------------


class TestAggregateAllVerified:
    """All verifier results are 'verified' -> overall verified."""

    def test_aggregate_all_verified(self, aggregator: ResultAggregator) -> None:
        results = [
            _vo(Verdict.verified, 0.9),
            _vo(Verdict.verified, 0.85),
        ]
        agg = aggregator.aggregate_claim(results)
        assert agg.verdict == Verdict.verified
        assert agg.confidence > 0.0
        assert len(agg.verifier_results) == 2


class TestAggregateAnyRefuted:
    """One refuted result with high confidence -> overall refuted."""

    def test_aggregate_any_refuted(self, aggregator: ResultAggregator) -> None:
        results = [
            _vo(Verdict.verified, 0.9),
            _vo(Verdict.refuted, 0.85),  # above default threshold of 0.7
        ]
        agg = aggregator.aggregate_claim(results)
        assert agg.verdict == Verdict.refuted


class TestAggregateMixed:
    """Verified + partially_supported -> partially_supported."""

    def test_aggregate_mixed(self, aggregator: ResultAggregator) -> None:
        results = [
            _vo(Verdict.verified, 0.9),
            _vo(Verdict.partially_supported, 0.6),
        ]
        agg = aggregator.aggregate_claim(results)
        assert agg.verdict == Verdict.partially_supported


class TestAggregateAllUnverifiable:
    """All unverifiable -> unverifiable."""

    def test_aggregate_all_unverifiable(self, aggregator: ResultAggregator) -> None:
        results = [
            _vo(Verdict.unverifiable, 0.0),
            _vo(Verdict.unverifiable, 0.1),
        ]
        agg = aggregator.aggregate_claim(results)
        assert agg.verdict == Verdict.unverifiable


# -------------------------------------------------------------------------
# Job-level aggregation
# -------------------------------------------------------------------------


class TestJobScoreNoHallucinations:
    """All verified claims should produce a score near 0.0."""

    def test_job_score_no_hallucinations(self, aggregator: ResultAggregator) -> None:
        claims_with_verdicts = [
            (
                _claim_info("claim 1"),
                AggregatedVerdict(
                    verdict=Verdict.verified,
                    confidence=0.9,
                    verifier_results=[_vo(Verdict.verified)],
                ),
            ),
            (
                _claim_info("claim 2"),
                AggregatedVerdict(
                    verdict=Verdict.verified,
                    confidence=0.85,
                    verifier_results=[_vo(Verdict.verified)],
                ),
            ),
        ]
        report = aggregator.aggregate_job(claims_with_verdicts)
        assert isinstance(report, ReportData)
        assert report.hallucination_score == pytest.approx(0.0, abs=0.01)
        assert report.total_claims == 2


class TestJobScoreAllHallucinated:
    """All refuted claims should produce a score near 1.0."""

    def test_job_score_all_hallucinated(self, aggregator: ResultAggregator) -> None:
        claims_with_verdicts = [
            (
                _claim_info("claim 1"),
                AggregatedVerdict(
                    verdict=Verdict.refuted,
                    confidence=0.95,
                    verifier_results=[_vo(Verdict.refuted, 0.95)],
                ),
            ),
            (
                _claim_info("claim 2"),
                AggregatedVerdict(
                    verdict=Verdict.refuted,
                    confidence=0.90,
                    verifier_results=[_vo(Verdict.refuted, 0.90)],
                ),
            ),
        ]
        report = aggregator.aggregate_job(claims_with_verdicts)
        assert report.hallucination_score > 0.8
        assert report.severity in (Severity.major, Severity.critical)


class TestSeverityLevels:
    """Test that severity thresholds map correctly."""

    def test_severity_clean(self, aggregator: ResultAggregator) -> None:
        """Score < 0.1 -> clean."""
        assert aggregator._severity_from_score(0.05) == Severity.clean

    def test_severity_minor(self, aggregator: ResultAggregator) -> None:
        """0.1 <= score < 0.3 -> minor."""
        assert aggregator._severity_from_score(0.2) == Severity.minor

    def test_severity_major(self, aggregator: ResultAggregator) -> None:
        """0.3 <= score < 0.6 -> major."""
        assert aggregator._severity_from_score(0.45) == Severity.major

    def test_severity_critical(self, aggregator: ResultAggregator) -> None:
        """Score >= 0.6 -> critical."""
        assert aggregator._severity_from_score(0.75) == Severity.critical
