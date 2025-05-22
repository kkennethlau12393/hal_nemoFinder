"""Tests for src.verifiers.statistical.StatisticalVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.statistical import StatisticalVerifier

pytestmark = pytest.mark.asyncio


def _make_verifier() -> StatisticalVerifier:
    return StatisticalVerifier()


class TestImpossiblePValue:
    """A negative p-value must be flagged as impossible."""

    async def test_negative_p_value_refuted(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "The drug showed a statistically significant effect "
                "(p = -0.03) versus placebo."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert result.confidence >= 0.9
        assert any(
            not f["passed"] and f["check"] == "p_value_sanity"
            for f in result.evidence["findings"]
        )


class TestConfidenceIntervalInconsistency:
    """A point estimate outside its own CI must be refuted."""

    async def test_point_outside_ci_refuted(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "The adjusted hazard ratio was HR 0.45 (95% CI: 0.60 to 0.80)."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert any(
            f["check"] == "ci_consistency" and not f["passed"]
            for f in result.evidence["findings"]
        )


class TestZeroAdverseEventsNexovarinCase:
    """The Nexovarin case: zero AEs across 15,000 patients is implausible."""

    async def test_zero_aes_in_15k_patients_refuted(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Nexovarin demonstrated an exceptional safety profile with "
                "zero adverse events reported across a cohort of 15,000 patients."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert result.confidence >= 0.9
        binomial_findings = [
            f for f in result.evidence["findings"] if f["check"] == "binomial"
        ]
        assert any(not f["passed"] for f in binomial_findings)


class TestMultipleTestingWithoutCorrection:
    """Many p-values reported with no correction mentioned should warn."""

    async def test_many_p_values_no_correction_partial(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Across endpoints we observed p = 0.04, p = 0.03, p = 0.02, "
                "p = 0.045, p = 0.048, p = 0.01 and p = 0.006."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.partially_supported
        assert any(
            f["check"] == "multiple_testing" and not f["passed"]
            for f in result.evidence["findings"]
        )


class TestPlausibleHbA1c:
    """A 1.2% HbA1c reduction is clinically plausible → verified."""

    async def test_plausible_hba1c_verified(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "In the phase 3 trial, the investigational agent produced an "
                "HbA1c reduction of 1.2% at 24 weeks."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.verified
        assert any(
            f["check"] == "clinical_metrics" and f["passed"]
            for f in result.evidence["findings"]
        )


class TestImplausibleHbA1c:
    """A 4% HbA1c reduction in monotherapy is implausible → refuted."""

    async def test_implausible_hba1c_refuted(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Monotherapy with the compound achieved an HbA1c reduction "
                "of 4.0% in treatment-naive patients."
            ),
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert result.confidence >= 0.9
        assert any(
            f["check"] == "clinical_metrics" and not f["passed"]
            for f in result.evidence["findings"]
        )
