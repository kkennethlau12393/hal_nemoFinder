"""Smoke tests for DrugInteractionVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict
from src.verifiers.drug_interaction import DrugInteractionVerifier

pytestmark = pytest.mark.asyncio


class TestDDIKnown:
    async def test_warfarin_ibuprofen_major(self) -> None:
        verifier = DrugInteractionVerifier()
        result = await verifier.verify(
            claim_text="Warfarin and ibuprofen interact, increasing bleeding risk.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.verified


class TestDDIRefute:
    async def test_denied_interaction_refuted(self) -> None:
        verifier = DrugInteractionVerifier()
        result = await verifier.verify(
            claim_text="There is no known interaction between warfarin and ibuprofen.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.refuted


class TestDDIUnverifiable:
    async def test_single_drug(self) -> None:
        verifier = DrugInteractionVerifier()
        result = await verifier.verify(
            claim_text="Warfarin is a potent anticoagulant.",
            claim_type=ClaimType.clinical_outcome,
            context={},
        )
        assert result.verdict == Verdict.unverifiable
