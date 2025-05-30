"""Smoke tests for LiteratureRetrievalVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.literature import LiteratureRetrievalVerifier

pytestmark = pytest.mark.asyncio


def _v() -> LiteratureRetrievalVerifier:
    verifier = LiteratureRetrievalVerifier()
    verifier._pubmed = None  # force fallback path
    return verifier


class TestLiteratureKnown:
    async def test_imatinib_abl1_match(self) -> None:
        result = await _v().verify(
            claim_text=(
                "Imatinib was shown to inhibit the BCR-ABL tyrosine kinase "
                "by binding the inactive conformation of ABL1."
            ),
            claim_type=ClaimType.mechanism_of_action,
            context={},
        )
        assert result.verdict in (Verdict.verified, Verdict.partially_supported)
        assert result.evidence.get("matched_papers")


class TestLiteratureRefute:
    async def test_unanchored_unsupported(self) -> None:
        result = await _v().verify(
            claim_text=(
                "It has been reported that xylophonase zymogenates "
                "paraflugen through quantum tunnelling."
            ),
            claim_type=ClaimType.general_biomedical,
            context={},
        )
        assert result.verdict in (Verdict.refuted, Verdict.partially_supported, Verdict.unverifiable)


class TestLiteratureUnverifiable:
    async def test_empty_tokens(self) -> None:
        result = await _v().verify(
            claim_text="    ",
            claim_type=ClaimType.general_biomedical,
            context={},
        )
        assert result.verdict == Verdict.unverifiable
