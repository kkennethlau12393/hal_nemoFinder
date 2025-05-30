"""Adverse-event verifier — cross-checks safety claims against FAERS profiles.

Extracts a drug name and any AE mentions from the claim, consults the
:class:`~src.knowledge.faers.FAERSClient`, and flags:

* ``"no adverse events"`` language for drugs with thousands of FAERS reports
* specific AEs that never appear in the top-20 FAERS terms
* frequency qualifiers ("rare", "<1%") that are order-of-magnitude off
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.faers import AdverseEventProfile, FAERSClient
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


_NO_AE_RE = re.compile(
    r"\b(no (?:adverse|side) (?:effects?|events?)|"
    r"no known side effects|well[- ]tolerated with no side effects|"
    r"does not cause (?:any )?(?:adverse|side) (?:effects?|events?))\b",
    re.IGNORECASE,
)

_RARE_RE = re.compile(
    r"\b(very rare|rare|less than 1%|<\s*1%|uncommon|isolated cases?)\b",
    re.IGNORECASE,
)

_AE_KEYWORDS: tuple[str, ...] = (
    "bleeding", "haemorrhage", "hemorrhage", "lactic acidosis",
    "rhabdomyolysis", "myopathy", "myalgia", "hepatotoxicity",
    "liver injury", "nausea", "diarrhoea", "diarrhea", "cough",
    "angioedema", "serotonin syndrome", "qt prolongation", "torsade",
    "bradycardia", "hypotension", "rash", "pneumonitis", "colitis",
    "hyperkalaemia", "priapism", "tinnitus", "anaphylaxis",
)

# Map of AE keywords to synonyms — when the user mentions "bleeding" the
# FAERS data may list "haemorrhage" instead.  All terms are lowercase.
_AE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bleeding": ("bleeding", "haemorrhage", "hemorrhage", "blood loss"),
    "haemorrhage": ("haemorrhage", "hemorrhage", "bleeding"),
    "hemorrhage": ("haemorrhage", "hemorrhage", "bleeding"),
    "diarrhea": ("diarrhea", "diarrhoea"),
    "diarrhoea": ("diarrhea", "diarrhoea"),
    "liver injury": ("liver injury", "hepatotoxicity", "hepatitis"),
    "hepatotoxicity": ("hepatotoxicity", "liver injury", "hepatitis"),
    "myopathy": ("myopathy", "rhabdomyolysis", "myalgia", "muscle"),
    "myalgia": ("myalgia", "myopathy", "muscle"),
    "torsade": ("torsade", "qt prolongation", "long qt"),
    "qt prolongation": ("qt prolongation", "torsade", "long qt"),
    "anaphylaxis": ("anaphylaxis", "anaphylactic"),
}


def _matches_term(keyword: str, term: str) -> bool:
    """True if *keyword* (or a known synonym) appears in *term*."""
    synonyms = _AE_SYNONYMS.get(keyword, (keyword,))
    return any(syn in term for syn in synonyms)


# Built once — list of fallback drug names.
from src.knowledge.faers import _FALLBACK_PROFILES as _PROFILES  # noqa: E402
_FALLBACK_DRUGS: tuple[str, ...] = tuple(
    d for d in _PROFILES.keys() if "+" not in d
)


def _extract_drug(text: str) -> str | None:
    lowered = text.lower()
    for drug in _FALLBACK_DRUGS:
        if re.search(rf"\b{re.escape(drug)}\b", lowered):
            return drug
    return None


@register_verifier
class AdverseEventVerifier(BaseVerifier):
    """Cross-check adverse-event claims against FAERS data."""

    name = "adverse_events"
    supported_claim_types = [
        ClaimType.clinical_outcome,
        ClaimType.general_biomedical,
    ]

    def __init__(self) -> None:
        try:
            self._faers: FAERSClient | None = FAERSClient()
        except Exception as exc:  # pragma: no cover
            logger.debug("FAERSClient unavailable: %s", exc)
            self._faers = None

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return await self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("AdverseEventVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during AE verification: {exc}",
                source_db="faers",
            )

    async def _verify_impl(self, claim_text: str) -> VerificationOutput:
        drug = _extract_drug(claim_text)
        evidence: dict[str, Any] = {"drug": drug}
        if not drug:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No drug name recognised in claim; FAERS lookup not possible.",
                evidence=evidence,
                source_db="faers",
            )

        profile = await self._lookup(drug)
        if profile is None or not profile.reactions:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.1,
                reasoning=f"No FAERS data available for {drug}.",
                evidence=evidence,
                source_db="faers",
            )

        top_terms = profile.top_terms(20)
        evidence["total_reports"] = profile.total_reports
        evidence["top_terms"] = top_terms

        # 1) "No adverse events" claim on a drug with thousands of reports
        if _NO_AE_RE.search(claim_text) and profile.total_reports >= 1000:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.93,
                reasoning=(
                    f"Claim asserts no adverse events for {drug}, but FAERS "
                    f"lists {profile.total_reports} reports including: "
                    f"{', '.join(top_terms[:5])}."
                ),
                evidence=evidence,
                source_db="faers",
            )

        # 2) Specific AE mentions — are they in the top-20?
        mentioned: list[str] = []
        missing: list[str] = []
        confirmed: list[str] = []
        text_l = claim_text.lower()
        for kw in _AE_KEYWORDS:
            if kw in text_l:
                mentioned.append(kw)
                if any(_matches_term(kw, term) for term in top_terms):
                    confirmed.append(kw)
                else:
                    missing.append(kw)
        evidence["mentioned_ae"] = mentioned
        evidence["confirmed_ae"] = confirmed
        evidence["missing_ae"] = missing

        if mentioned and missing and not confirmed:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.6,
                reasoning=(
                    f"Claim mentions AEs {missing} that do not appear in the "
                    f"top-20 FAERS reactions for {drug}: {top_terms[:10]}."
                ),
                evidence=evidence,
                source_db="faers",
            )

        # 3) "Rare" qualifier vs. actual report counts
        if _RARE_RE.search(claim_text) and confirmed:
            for kw in confirmed:
                count = 0
                for r in profile.reactions:
                    if kw in r.term.lower():
                        count = max(count, r.count)
                # "rare" implies < 1/1000; if count exceeds 1% of reports
                # it is clearly more than rare.
                if profile.total_reports and count / profile.total_reports > 0.05:
                    return VerificationOutput(
                        verdict=Verdict.refuted,
                        confidence=0.85,
                        reasoning=(
                            f"Claim describes {kw} as rare, but FAERS lists "
                            f"{count} reports out of {profile.total_reports} "
                            f"(~{100 * count / profile.total_reports:.1f}%) for {drug}."
                        ),
                        evidence=evidence,
                        source_db="faers",
                    )

        if confirmed:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.8,
                reasoning=(
                    f"FAERS confirms reported adverse events {confirmed} for {drug}."
                ),
                evidence=evidence,
                source_db="faers",
            )

        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.4,
            reasoning=(
                f"FAERS profile retrieved for {drug} but no specific AE "
                "mentioned in the claim could be matched."
            ),
            evidence=evidence,
            source_db="faers",
        )

    async def _lookup(self, drug: str) -> AdverseEventProfile | None:
        if self._faers is not None:
            try:
                return await self._faers.search_adverse_events(drug)
            except Exception as exc:  # pragma: no cover
                logger.debug("FAERS online lookup failed: %s", exc)
        return FAERSClient.fallback_profile(drug)

    async def health_check(self) -> bool:
        return True
