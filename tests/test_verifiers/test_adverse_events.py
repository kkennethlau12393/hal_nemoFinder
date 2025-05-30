"""Smoke tests for AdverseEventVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.adverse_events import AdverseEventVerifier

pytestmark = pytest.mark.asyncio


def _v() -> AdverseEventVerifier:
    verifier = AdverseEventVerifier()
    verifier._faers = None  # force fallback data
    return verifier


class TestAEKnown:
    async def test_aspirin_bleeding_confirmed(self) -> None:
        result = await _v().verify(
            claim_text="Aspirin can cause gastrointestinal bleeding.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.verified


class TestAERefute:
    async def test_metformin_no_ae_refuted(self) -> None:
        result = await _v().verify(
            claim_text="Metformin has no adverse events in clinical use.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted


class TestAEUnverifiable:
    async def test_unknown_drug(self) -> None:
        result = await _v().verify(
            claim_text="Xyzzymab produces rainbow-coloured side effects.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.unverifiable
