"""Tests for src.verifiers.citation.CitationVerifier."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.citation import CitationVerifier

pytestmark = pytest.mark.asyncio


def _make_verifier() -> CitationVerifier:
    """Build a CitationVerifier with external clients disabled."""
    verifier = CitationVerifier()
    verifier._crossref = None
    verifier._embedding = None
    return verifier


class TestValidDoiFormat:
    """A well-formed DOI should be at least partially_supported."""

    async def test_valid_doi_format(self) -> None:
        verifier = _make_verifier()
        # With no CrossRef client, the DOI cannot be validated externally,
        # so the verifier cannot confirm it.  It should still recognise the
        # DOI was extracted.
        result = await verifier.verify(
            claim_text="As reported in 10.1038/s41586-021-03819-2 the compound was effective.",
            claim_type=ClaimType.citation,
            context={},
        )
        # DOI extracted but CrossRef unavailable -> unverifiable or partially_supported
        assert result.verdict in (
            Verdict.partially_supported,
            Verdict.unverifiable,
        )
        assert "10.1038/s41586-021-03819-2" in str(result.evidence.get("extracted_dois", []))

    async def test_valid_doi_with_mock_crossref(self) -> None:
        """When CrossRef returns metadata, a valid DOI should be partially_supported."""
        verifier = _make_verifier()
        mock_crossref = AsyncMock()
        mock_crossref.validate_doi = AsyncMock(return_value={
            "doi": "10.1038/s41586-021-03819-2",
            "title": "A Nature paper",
            "valid": True,
        })
        verifier._crossref = mock_crossref

        result = await verifier.verify(
            claim_text="As reported in 10.1038/s41586-021-03819-2 the compound was effective.",
            claim_type=ClaimType.citation,
            context={},
        )
        assert result.verdict in (Verdict.verified, Verdict.partially_supported)
        assert result.confidence > 0.0
        mock_crossref.validate_doi.assert_awaited_once()


class TestInvalidDoiFormat:
    """A fake DOI that cannot be validated."""

    async def test_invalid_doi_format(self) -> None:
        verifier = _make_verifier()
        mock_crossref = AsyncMock()
        mock_crossref.validate_doi = AsyncMock(return_value=None)
        verifier._crossref = mock_crossref

        result = await verifier.verify(
            claim_text="Refer to 10.fake/not-real-doi for details.",
            claim_type=ClaimType.citation,
            context={},
        )
        # CrossRef returns None -> invalid DOI
        assert result.verdict in (Verdict.refuted, Verdict.unverifiable)


class TestPmidExtraction:
    """PMID should be extracted and format-validated."""

    async def test_pmid_extraction(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text="The binding affinity was reported (PMID: 12345678) in a landmark study.",
            claim_type=ClaimType.citation,
            context={},
        )
        # PMID has valid format -> at least partially_supported
        assert result.verdict in (Verdict.partially_supported, Verdict.unverifiable)
        evidence = result.evidence
        assert "12345678" in evidence.get("valid_pmids", [])


class TestNoIdentifiers:
    """Claim without any DOI or PMID should be unverifiable."""

    async def test_no_identifiers(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text="The treatment was effective in clinical studies.",
            claim_type=ClaimType.citation,
            context={},
        )
        assert result.verdict == Verdict.unverifiable
        assert result.confidence == 0.0
