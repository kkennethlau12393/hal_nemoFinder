"""Patent verifier — validates patent numbers and disclosed compounds.

Extracts US/EP/WO patent numbers from the claim text using
:meth:`~src.knowledge.patents.PatentClient.extract_patents` and checks
each one against the curated :class:`~src.knowledge.patents.PatentClient`
table.  Reports REFUTED for numbers that do not exist in the curated
database and PARTIALLY_SUPPORTED when the claim associates the patent
with a compound the record does not mention.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge.patents import PatentClient, PatentRecord
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


@register_verifier
class PatentVerifier(BaseVerifier):
    """Verify patent-citation and patent-content claims."""

    name = "patent"
    supported_claim_types = [
        ClaimType.citation,
        ClaimType.general_biomedical,
    ]

    def __init__(self) -> None:
        self._client = PatentClient()

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("PatentVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during patent verification: {exc}",
                source_db="patent",
            )

    def _verify_impl(self, claim_text: str) -> VerificationOutput:
        numbers = PatentClient.extract_patents(claim_text)
        evidence: dict[str, Any] = {"patent_numbers": numbers}
        if not numbers:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No patent numbers extracted from claim.",
                evidence=evidence,
                source_db="patent",
            )

        records: list[tuple[str, PatentRecord | None]] = [
            (n, self._client.get_patent(n)) for n in numbers
        ]
        evidence["records"] = [
            {
                "number": n,
                "found": r is not None,
                "compound": r.compound if r else None,
                "title": r.title if r else None,
            }
            for n, r in records
        ]

        unknown = [n for n, r in records if r is None]
        if unknown:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.9,
                reasoning=(
                    f"Patent number(s) {unknown} are not present in the curated "
                    "patent database; likely fabricated."
                ),
                evidence=evidence,
                source_db="patent",
            )

        lowered = claim_text.lower()
        mismatches: list[str] = []
        matches: list[str] = []
        for n, r in records:
            assert r is not None
            if r.compound.lower() in lowered:
                matches.append(f"{n}->{r.compound}")
                continue
            keyword_hit = any(kw in lowered for kw in r.keywords)
            if keyword_hit:
                matches.append(f"{n}->{r.compound} (kw)")
            else:
                mismatches.append(f"{n}: record={r.compound}, claim does not mention it")

        if mismatches and not matches:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.55,
                reasoning=(
                    "Patent number(s) resolve, but the compound/topic mentioned "
                    f"in the claim does not match the record: {mismatches}."
                ),
                evidence=evidence,
                source_db="patent",
            )
        return VerificationOutput(
            verdict=Verdict.verified,
            confidence=0.88,
            reasoning=f"All patent numbers verified: {matches}.",
            evidence=evidence,
            source_db="patent",
        )

    async def health_check(self) -> bool:
        return len(self._client.all_patents()) > 0
