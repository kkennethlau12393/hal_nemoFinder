"""Drug-drug interaction verifier.

Extracts 2+ drug names from a claim, looks up the curated DDI table,
and verifies the stated interaction (or lack thereof) against the
authoritative entry.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.ddi import DrugInteraction, DrugInteractionClient
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


_NEGATION_RE = re.compile(
    r"\b(no (?:known )?(?:clinically (?:significant|relevant) )?interaction|"
    r"do(?:es)? not interact|is safe (?:to )?(?:co[- ]?administer|combine)|"
    r"no drug(?:-| )drug interaction|safely combined)\b",
    re.IGNORECASE,
)

_SEVERITY_WORDS: dict[str, str] = {
    "contraindicated": "contraindicated",
    "avoid": "contraindicated",
    "major": "major",
    "severe": "major",
    "dangerous": "major",
    "moderate": "moderate",
    "minor": "minor",
    "mild": "minor",
}


def _claimed_severity(text: str) -> str | None:
    lowered = text.lower()
    for k, v in _SEVERITY_WORDS.items():
        if k in lowered:
            return v
    return None


@register_verifier
class DrugInteractionVerifier(BaseVerifier):
    """Verify drug-drug interaction claims against a curated table."""

    name = "drug_interaction"
    supported_claim_types = [
        ClaimType.clinical_outcome,
        ClaimType.pharmacokinetic,
    ]

    def __init__(self) -> None:
        self._client = DrugInteractionClient()

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("DrugInteractionVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during DDI verification: {exc}",
                source_db="ddi",
            )

    def _verify_impl(self, claim_text: str) -> VerificationOutput:
        drugs = self._client.extract_drugs(claim_text)
        evidence: dict[str, Any] = {"drugs": drugs}

        if len(drugs) < 2:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="Fewer than two recognised drugs in claim; no DDI lookup possible.",
                evidence=evidence,
                source_db="ddi",
            )

        # Check every drug pair for a curated interaction.
        found: list[DrugInteraction] = []
        checked: list[tuple[str, str]] = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                a, b = drugs[i], drugs[j]
                checked.append((a, b))
                interaction = self._client.lookup(a, b)
                if interaction is not None:
                    found.append(interaction)

        evidence["pairs_checked"] = checked
        evidence["interactions_found"] = [
            {
                "drug_a": i.drug_a,
                "drug_b": i.drug_b,
                "severity": i.severity,
                "mechanism": i.mechanism,
                "effect": i.effect,
            }
            for i in found
        ]

        claim_says_no_ddi = bool(_NEGATION_RE.search(claim_text))
        claimed_sev = _claimed_severity(claim_text)

        # Case 1: Claim denies an interaction that exists in the DB.
        if claim_says_no_ddi and found:
            worst = max(found, key=lambda i: _SEV_ORDER.get(i.severity, 0))
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.92,
                reasoning=(
                    f"Claim denies a DDI between {worst.drug_a} and {worst.drug_b}, "
                    f"but curated DB lists a {worst.severity} interaction "
                    f"({worst.mechanism}) causing {worst.effect}."
                ),
                evidence=evidence,
                source_db="ddi",
            )

        # Case 2: Claim affirms an interaction that exists — verify severity if claimed.
        if found:
            if claimed_sev is not None:
                # Compare claimed vs curated severity.
                mismatched = [
                    i for i in found if i.severity != claimed_sev
                ]
                if mismatched and all(
                    abs(_SEV_ORDER.get(i.severity, 0) - _SEV_ORDER.get(claimed_sev, 0)) >= 2
                    for i in mismatched
                ):
                    return VerificationOutput(
                        verdict=Verdict.refuted,
                        confidence=0.8,
                        reasoning=(
                            f"Claim labels the interaction as {claimed_sev}, but "
                            f"curated DB records: "
                            f"{[(i.drug_a, i.drug_b, i.severity) for i in mismatched]}."
                        ),
                        evidence=evidence,
                        source_db="ddi",
                    )
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.88,
                reasoning=(
                    f"Curated DB confirms {len(found)} interaction(s) among "
                    f"{drugs}: {[(i.drug_a, i.drug_b, i.severity) for i in found]}."
                ),
                evidence=evidence,
                source_db="ddi",
            )

        # Case 3: Claim asserts an interaction that is absent from the DB.
        if not claim_says_no_ddi:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.4,
                reasoning=(
                    f"No curated interaction found for pairs {checked}; "
                    "claim cannot be confirmed from this table."
                ),
                evidence=evidence,
                source_db="ddi",
            )

        # Case 4: Claim denies a non-existent interaction — consistent.
        return VerificationOutput(
            verdict=Verdict.verified,
            confidence=0.6,
            reasoning=(
                f"No DDI listed for pairs {checked}, consistent with the claim."
            ),
            evidence=evidence,
            source_db="ddi",
        )

    async def health_check(self) -> bool:
        return len(self._client.all_interactions()) > 0


_SEV_ORDER: dict[str, int] = {
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "contraindicated": 4,
}
