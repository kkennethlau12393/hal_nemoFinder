"""Metabolism / CYP substrate verifier.

Maps ~30 well-known drugs to their primary CYP450 metabolising enzyme
and optional inhibitor classification.  The verifier compares the
claimed CYP against the curated mapping and flags mismatches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated CYP metabolism table (~30 drugs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CYPProfile:
    drug: str
    primary_cyps: tuple[str, ...]  # metabolised by
    inhibits: tuple[str, ...] = field(default_factory=tuple)
    induces: tuple[str, ...] = field(default_factory=tuple)


_PROFILES: dict[str, CYPProfile] = {
    "warfarin":     CYPProfile("warfarin",     ("CYP2C9",)),
    "clopidogrel":  CYPProfile("clopidogrel",  ("CYP2C19",)),
    "simvastatin":  CYPProfile("simvastatin",  ("CYP3A4",)),
    "atorvastatin": CYPProfile("atorvastatin", ("CYP3A4",)),
    "lovastatin":   CYPProfile("lovastatin",   ("CYP3A4",)),
    "fluvastatin":  CYPProfile("fluvastatin",  ("CYP2C9",)),
    "rosuvastatin": CYPProfile("rosuvastatin", ("CYP2C9",)),
    "omeprazole":   CYPProfile("omeprazole",   ("CYP2C19",), inhibits=("CYP2C19",)),
    "esomeprazole": CYPProfile("esomeprazole", ("CYP2C19",), inhibits=("CYP2C19",)),
    "diazepam":     CYPProfile("diazepam",     ("CYP2C19", "CYP3A4")),
    "phenytoin":    CYPProfile("phenytoin",    ("CYP2C9",), induces=("CYP3A4",)),
    "tamoxifen":    CYPProfile("tamoxifen",    ("CYP2D6", "CYP3A4")),
    "codeine":      CYPProfile("codeine",      ("CYP2D6",)),
    "tramadol":     CYPProfile("tramadol",     ("CYP2D6",)),
    "metoprolol":   CYPProfile("metoprolol",   ("CYP2D6",)),
    "propranolol":  CYPProfile("propranolol",  ("CYP2D6", "CYP1A2")),
    "fluoxetine":   CYPProfile("fluoxetine",   ("CYP2D6",), inhibits=("CYP2D6",)),
    "paroxetine":   CYPProfile("paroxetine",   ("CYP2D6",), inhibits=("CYP2D6",)),
    "sertraline":   CYPProfile("sertraline",   ("CYP2C19", "CYP3A4")),
    "caffeine":     CYPProfile("caffeine",     ("CYP1A2",)),
    "theophylline": CYPProfile("theophylline", ("CYP1A2",)),
    "clozapine":    CYPProfile("clozapine",    ("CYP1A2",)),
    "midazolam":    CYPProfile("midazolam",    ("CYP3A4",)),
    "cyclosporine": CYPProfile("cyclosporine", ("CYP3A4",), inhibits=("CYP3A4",)),
    "erythromycin": CYPProfile("erythromycin", ("CYP3A4",), inhibits=("CYP3A4",)),
    "clarithromycin": CYPProfile("clarithromycin", ("CYP3A4",), inhibits=("CYP3A4",)),
    "ketoconazole": CYPProfile("ketoconazole", ("CYP3A4",), inhibits=("CYP3A4",)),
    "itraconazole": CYPProfile("itraconazole", ("CYP3A4",), inhibits=("CYP3A4",)),
    "rifampin":     CYPProfile("rifampin",     ("CYP3A4",), induces=("CYP3A4", "CYP2C9")),
    "carbamazepine": CYPProfile("carbamazepine", ("CYP3A4",), induces=("CYP3A4",)),
    "ibuprofen":    CYPProfile("ibuprofen",    ("CYP2C9",)),
    "celecoxib":    CYPProfile("celecoxib",    ("CYP2C9",)),
    "sildenafil":   CYPProfile("sildenafil",   ("CYP3A4",)),
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_CYP_RE = re.compile(
    r"\bCYP\s*([1-4][A-E]\d?)\b",
    re.IGNORECASE,
)


def _extract_drug(text: str) -> str | None:
    lowered = text.lower()
    for drug in _PROFILES:
        if re.search(rf"\b{re.escape(drug)}\b", lowered):
            return drug
    return None


def _extract_cyps(text: str) -> list[str]:
    return [f"CYP{m.group(1).upper()}" for m in _CYP_RE.finditer(text)]


_METABOLISED_BY_RE = re.compile(
    r"\b(metabolised|metabolized|broken down|cleared) by\b", re.IGNORECASE
)
_INHIBITS_RE = re.compile(r"\binhibit(?:s|or|ion)?\b", re.IGNORECASE)
_INDUCES_RE = re.compile(r"\binduc(?:es|er|tion|ed)\b", re.IGNORECASE)


@register_verifier
class MetabolismVerifier(BaseVerifier):
    """Verify CYP-metabolism and inhibitor-classification claims."""

    name = "metabolism"
    supported_claim_types = [
        ClaimType.pharmacokinetic,
        ClaimType.mechanism_of_action,
    ]

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("MetabolismVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during metabolism verification: {exc}",
                source_db="metabolism",
            )

    def _verify_impl(self, claim_text: str) -> VerificationOutput:
        drug = _extract_drug(claim_text)
        cyps = _extract_cyps(claim_text)
        evidence: dict[str, Any] = {"drug": drug, "claimed_cyps": cyps}
        if not drug or not cyps:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No (drug, CYP) pair extracted from claim.",
                evidence=evidence,
                source_db="metabolism",
            )

        profile = _PROFILES[drug]
        evidence["known"] = {
            "primary_cyps": list(profile.primary_cyps),
            "inhibits": list(profile.inhibits),
            "induces": list(profile.induces),
        }

        says_metabolised_by = bool(_METABOLISED_BY_RE.search(claim_text))
        says_inhibits = bool(_INHIBITS_RE.search(claim_text))
        says_induces = bool(_INDUCES_RE.search(claim_text))

        problems: list[str] = []
        confirms: list[str] = []

        for cyp in cyps:
            if says_metabolised_by:
                if cyp in profile.primary_cyps:
                    confirms.append(f"{drug} metabolised by {cyp}")
                else:
                    problems.append(
                        f"Claim says {drug} is metabolised by {cyp}, but "
                        f"curated data lists {profile.primary_cyps}."
                    )
            if says_inhibits:
                if cyp in profile.inhibits:
                    confirms.append(f"{drug} inhibits {cyp}")
                else:
                    problems.append(
                        f"Claim calls {drug} a {cyp} inhibitor, but curated "
                        f"data lists inhibitors: {profile.inhibits or 'none'}."
                    )
            if says_induces:
                if cyp in profile.induces:
                    confirms.append(f"{drug} induces {cyp}")
                else:
                    problems.append(
                        f"Claim calls {drug} a {cyp} inducer, but curated data "
                        f"lists inducers: {profile.induces or 'none'}."
                    )

        if problems and not confirms:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=min(0.92, 0.75 + 0.05 * len(problems)),
                reasoning="; ".join(problems),
                evidence=evidence,
                source_db="metabolism",
            )
        if confirms:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.88,
                reasoning="; ".join(confirms),
                evidence=evidence,
                source_db="metabolism",
            )
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.4,
            reasoning=(
                f"(drug={drug}, cyps={cyps}) extracted but claim did not use "
                "a metabolism/inhibition/induction keyword."
            ),
            evidence=evidence,
            source_db="metabolism",
        )

    async def health_check(self) -> bool:
        return len(_PROFILES) > 0
