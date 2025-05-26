"""Tests for src.core.claim_classifier.ClaimClassifier."""

from __future__ import annotations

from src.core.claim_classifier import ClaimClassifier
from src.core.claim_extractor import ExtractedClaim, SourceSpan
from src.models.enums import ClaimType


def _make_claim(text: str) -> ExtractedClaim:
    """Helper to build an ExtractedClaim with minimal boilerplate."""
    return ExtractedClaim(
        claim_text=text,
        source_span=SourceSpan(start=0, end=len(text), context=text),
        confidence=0.5,
    )


class TestClassifyMolecularProperty:
    """Claims containing SMILES or molecular property keywords."""

    def test_classify_molecular_property(
        self, claim_classifier: ClaimClassifier
    ) -> None:
        claim = _make_claim(
            "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has a molecular weight of 180.16 Da."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.molecular_property


class TestClassifyCitation:
    """Claims containing DOIs or PMIDs."""

    def test_classify_citation(self, claim_classifier: ClaimClassifier) -> None:
        claim = _make_claim(
            "This was first reported in 10.1021/jm501000a by Smith et al."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.citation


class TestClassifyClinicalOutcome:
    """Claims containing NCT IDs or clinical trial language."""

    def test_classify_clinical(self, claim_classifier: ClaimClassifier) -> None:
        claim = _make_claim(
            "The phase III trial NCT01234567 demonstrated significant efficacy."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.clinical_outcome


class TestClassifyPharmacokinetic:
    """Claims about PK parameters."""

    def test_classify_pharmacokinetic(
        self, claim_classifier: ClaimClassifier
    ) -> None:
        claim = _make_claim(
            "The compound demonstrated an oral bioavailability of 85% in rats."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.pharmacokinetic


class TestClassifyTargetInteraction:
    """Claims about receptor binding or target interactions."""

    def test_classify_target_interaction(
        self, claim_classifier: ClaimClassifier
    ) -> None:
        claim = _make_claim(
            "The compound binds to EGFR with a Kd of 5 nM and shows high selectivity."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.target_interaction


class TestClassifyMechanismOfAction:
    """Claims about signalling pathways and mechanisms."""

    def test_classify_mechanism(self, claim_classifier: ClaimClassifier) -> None:
        claim = _make_claim(
            "The drug inhibits the PI3K/AKT pathway leading to apoptosis."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.mechanism_of_action


class TestClassifyGeneralBiomedical:
    """Vague biomedical claims that don't match any specific rule."""

    def test_classify_general(self, claim_classifier: ClaimClassifier) -> None:
        claim = _make_claim(
            "The treatment was well tolerated in elderly patients."
        )
        result = claim_classifier.classify(claim)
        assert result == ClaimType.general_biomedical
