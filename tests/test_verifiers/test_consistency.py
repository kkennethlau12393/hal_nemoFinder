"""Tests for src.verifiers.consistency.ConsistencyVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.consistency import ConsistencyVerifier

pytestmark = pytest.mark.asyncio


def _make_verifier() -> ConsistencyVerifier:
    """Build a ConsistencyVerifier with embedding service disabled."""
    verifier = ConsistencyVerifier()
    verifier._embedding = None
    return verifier


class TestNoContradictions:
    """Multiple consistent claims should pass the consistency check."""

    async def test_no_contradictions(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text="Imatinib has a molecular weight of 493.6 Da.",
            claim_type=ClaimType.molecular_property,
            context={
                "all_claims": [
                    "Imatinib has a molecular weight of 493.6 Da.",
                    "Imatinib targets BCR-ABL kinase.",
                    "Imatinib was approved by the FDA in 2001.",
                ]
            },
        )
        # With multiple non-contradicting claims, verdict should be verified.
        assert result.verdict == Verdict.verified

    async def test_single_claim_unverifiable(self) -> None:
        """A single claim cannot be checked for cross-claim consistency."""
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text="Imatinib has a molecular weight of 493.6 Da.",
            claim_type=ClaimType.molecular_property,
            context={"all_claims": ["Imatinib has a molecular weight of 493.6 Da."]},
        )
        assert result.verdict == Verdict.unverifiable


class TestContradictoryMolecularWeights:
    """Two claims about the same molecule with different MWs should be refuted."""

    async def test_contradictory_molecular_weights(self) -> None:
        verifier = _make_verifier()
        claim_a = "Imatinib has a molecular weight of 493.6 Da."
        claim_b = "Imatinib has a molecular weight of 350.0 Da."

        result = await verifier.verify(
            claim_text=claim_a,
            claim_type=ClaimType.molecular_property,
            context={"all_claims": [claim_a, claim_b]},
        )
        assert result.verdict == Verdict.refuted
        assert result.confidence > 0.5


class TestContradictoryMechanisms:
    """Same target described as both agonist and antagonist should be refuted."""

    async def test_contradictory_mechanisms(self) -> None:
        verifier = _make_verifier()
        claim_a = "Imatinib is an agonist of EGFR with high potency."
        claim_b = "Imatinib is an antagonist of EGFR and blocks signalling."

        result = await verifier.verify(
            claim_text=claim_a,
            claim_type=ClaimType.mechanism_of_action,
            context={"all_claims": [claim_a, claim_b]},
        )
        assert result.verdict == Verdict.refuted
        assert "conflict" in result.reasoning.lower() or "contradict" in result.reasoning.lower()


class TestUnrelatedClaims:
    """Claims about completely different entities should not flag contradictions."""

    async def test_unrelated_claims(self) -> None:
        verifier = _make_verifier()
        claim_a = "Imatinib has a molecular weight of 493.6 Da."
        claim_b = "Aspirin was first synthesised in 1897."

        result = await verifier.verify(
            claim_text=claim_a,
            claim_type=ClaimType.molecular_property,
            context={"all_claims": [claim_a, claim_b]},
        )
        assert result.verdict == Verdict.verified
