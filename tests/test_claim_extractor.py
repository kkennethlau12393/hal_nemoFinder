"""Tests for src.core.claim_extractor.ClaimExtractor."""

from __future__ import annotations

from src.core.claim_extractor import ClaimExtractor, ExtractedClaim


class TestExtractSmilesClaim:
    """Extraction of SMILES-containing sentences."""

    def test_extract_smiles_claim(self, claim_extractor: ClaimExtractor) -> None:
        text = "Aspirin has the SMILES notation CC(=O)Oc1ccccc1C(O)=O and is widely used."
        claims = claim_extractor.extract(text)

        assert len(claims) >= 1
        smiles_claim = claims[0]
        assert isinstance(smiles_claim, ExtractedClaim)
        assert "CC(=O)Oc1ccccc1C(O)=O" in smiles_claim.claim_text
        # Source span should cover the claim location
        assert smiles_claim.source_span.start >= 0
        assert smiles_claim.source_span.end > smiles_claim.source_span.start


class TestExtractDoiClaim:
    """Extraction of DOI-containing sentences."""

    def test_extract_doi_claim(self, claim_extractor: ClaimExtractor) -> None:
        text = "This finding was reported in 10.1021/jm501000a by Smith et al."
        claims = claim_extractor.extract(text)

        assert len(claims) >= 1
        doi_claim = claims[0]
        assert "10.1021/jm501000a" in doi_claim.claim_text
        assert doi_claim.confidence > 0.3  # base_confidence + doi boost


class TestExtractPmidClaim:
    """Extraction of PMID-containing sentences."""

    def test_extract_pmid_claim(self, claim_extractor: ClaimExtractor) -> None:
        text = "The crystal structure was deposited (PMID: 12345678) and analysed."
        claims = claim_extractor.extract(text)

        assert len(claims) >= 1
        pmid_claim = claims[0]
        assert "PMID" in pmid_claim.claim_text
        assert pmid_claim.confidence > 0.3


class TestExtractNctClaim:
    """Extraction of clinical trial NCT ID sentences."""

    def test_extract_nct_claim(self, claim_extractor: ClaimExtractor) -> None:
        text = "The phase III trial NCT01234567 demonstrated significant efficacy."
        claims = claim_extractor.extract(text)

        assert len(claims) >= 1
        nct_claim = claims[0]
        assert "NCT01234567" in nct_claim.claim_text
        assert nct_claim.confidence > 0.3


class TestExtractNumericalProperty:
    """Extraction of claims containing numerical molecular properties."""

    def test_extract_numerical_property(self, claim_extractor: ClaimExtractor) -> None:
        text = "The compound has a molecular weight of 180.2 g/mol and shows good solubility."
        claims = claim_extractor.extract(text)

        assert len(claims) >= 1
        mw_claim = claims[0]
        assert "180.2" in mw_claim.claim_text
        assert mw_claim.confidence > 0.3


class TestExtractMultipleClaims:
    """Extraction of multiple distinct claims from a paragraph."""

    def test_extract_multiple_claims(self, claim_extractor: ClaimExtractor) -> None:
        text = (
            "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has a molecular weight of 180.16 g/mol. "
            "This was reported in 10.1021/jm501000a by the authors. "
            "The clinical trial NCT01234567 showed efficacy in phase II."
        )
        claims = claim_extractor.extract(text)

        # Should find at least 3 distinct sentences with claims
        assert len(claims) >= 3

        claim_texts = [c.claim_text for c in claims]
        # Each sentence should produce a separate claim
        assert any("CC(=O)Oc1ccccc1C(O)=O" in t for t in claim_texts)
        assert any("10.1021/jm501000a" in t for t in claim_texts)
        assert any("NCT01234567" in t for t in claim_texts)


class TestExtractNoClaims:
    """Plain text without verifiable claims should yield no results."""

    def test_extract_no_claims(self, claim_extractor: ClaimExtractor) -> None:
        text = "The sky is blue and water is wet."
        claims = claim_extractor.extract(text)

        assert len(claims) == 0


class TestConfidenceHigherForSpecificClaims:
    """Claims with concrete identifiers should score higher confidence."""

    def test_confidence_higher_for_specific_claims(
        self, claim_extractor: ClaimExtractor
    ) -> None:
        # Concrete SMILES claim should get high confidence
        smiles_text = "The molecule CC(=O)Oc1ccccc1C(O)=O has a molecular weight of 180.16 g/mol."
        smiles_claims = claim_extractor.extract(smiles_text)
        assert len(smiles_claims) >= 1
        smiles_confidence = smiles_claims[0].confidence

        # Vague claim with just a keyword should get lower confidence
        vague_text = "The receptor was studied for its binding properties."
        vague_claims = claim_extractor.extract(vague_text)
        assert len(vague_claims) >= 1
        vague_confidence = vague_claims[0].confidence

        assert smiles_confidence > vague_confidence
