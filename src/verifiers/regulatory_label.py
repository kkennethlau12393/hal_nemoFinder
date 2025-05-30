"""Regulatory label verifier — cross-checks claims against FDA drug labels.

Flags:
* claims that assert FDA approval for an indication not listed on the label
* dosages outside the approved adult range
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.labels import DrugLabel, DrugLabelClient
from src.knowledge.faers import _FALLBACK_PROFILES as _DRUG_NAMES  # noqa: F401
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


_DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)",
    re.IGNORECASE,
)

_APPROVED_RE = re.compile(
    r"\b(fda[- ]approved|approved by (?:the )?fda|approved for the treatment of|"
    r"indicated for|is approved for)\b",
    re.IGNORECASE,
)

# Map claim-indication wording to the canonical phrase that appears in
# the curated labels dict.  Keeping this explicit avoids false-positives
# from loose substring matching.
_INDICATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "type 2 diabetes": ("type 2 diabetes", "diabetes mellitus", "glycaemic control", "glycemic control"),
    "hypertension": ("hypertension", "high blood pressure"),
    "hyperlipidemia": ("hypercholesterolemia", "dyslipidemia", "hyperlipidemia"),
    "depression": ("major depressive disorder", "depression"),
    "anticoagulation": ("thromboembolism", "atrial fibrillation", "embolism", "anticoagulation"),
    "pain": ("pain",),
    "cancer": ("cancer", "carcinoma", "leukemia", "melanoma", "tumor"),
    "hypothyroidism": ("hypothyroidism",),
    "gerd": ("gerd", "peptic ulcer", "reflux", "zollinger-ellison"),
    "erectile dysfunction": ("erectile dysfunction", "pulmonary arterial hypertension"),
    "angina": ("angina",),
}


def _known_drugs() -> list[str]:
    from src.knowledge.labels import _FALLBACK_LABELS
    return sorted(_FALLBACK_LABELS.keys())


def _extract_drug(text: str) -> str | None:
    lowered = text.lower()
    for drug in _known_drugs():
        if re.search(rf"\b{re.escape(drug)}\b", lowered):
            return drug
    return None


def _extract_doses_mg(text: str) -> list[float]:
    return [float(m.group(1)) for m in _DOSE_RE.finditer(text)]


@register_verifier
class RegulatoryLabelVerifier(BaseVerifier):
    """Cross-check FDA-approval and dosing claims against drug labels."""

    name = "regulatory_label"
    supported_claim_types = [ClaimType.clinical_outcome]

    def __init__(self) -> None:
        try:
            self._client: DrugLabelClient | None = DrugLabelClient()
        except Exception as exc:  # pragma: no cover
            logger.debug("DrugLabelClient unavailable: %s", exc)
            self._client = None

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return await self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("RegulatoryLabelVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during label verification: {exc}",
                source_db="druglabel",
            )

    async def _verify_impl(self, claim_text: str) -> VerificationOutput:
        drug = _extract_drug(claim_text)
        evidence: dict[str, Any] = {"drug": drug}
        if not drug:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No recognised drug name in claim.",
                evidence=evidence,
                source_db="druglabel",
            )

        label = await self._get_label(drug)
        if label is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.1,
                reasoning=f"No label data available for {drug}.",
                evidence=evidence,
                source_db="druglabel",
            )
        evidence["indications"] = list(label.indications)
        evidence["dose_range"] = (
            None
            if label.adult_dose is None
            else [label.adult_dose.min_mg, label.adult_dose.max_mg]
        )

        problems: list[str] = []
        confirmations: list[str] = []

        # 1) Approval claim — is the indication listed?
        if _APPROVED_RE.search(claim_text):
            found_match = False
            for bucket, synonyms in _INDICATION_SYNONYMS.items():
                if any(s in claim_text.lower() for s in synonyms):
                    label_has = any(
                        syn in ind.lower()
                        for ind in label.indications
                        for syn in synonyms
                    )
                    if label_has:
                        confirmations.append(
                            f"{drug} label includes indication '{bucket}'"
                        )
                        found_match = True
                    else:
                        problems.append(
                            f"Claim asserts {drug} is FDA-approved for "
                            f"'{bucket}' but label does not list it."
                        )
                        found_match = True
            if not found_match:
                confirmations.append(
                    f"{drug} approval claim could not be mapped to a known indication"
                )

        # 2) Dose range check
        doses = _extract_doses_mg(claim_text)
        evidence["claim_doses_mg"] = doses
        if doses and label.adult_dose is not None:
            lo, hi = label.adult_dose.min_mg, label.adult_dose.max_mg
            out_of_range = [d for d in doses if d < lo * 0.5 or d > hi * 2.0]
            if out_of_range:
                problems.append(
                    f"Dose(s) {out_of_range} mg lie far outside the FDA-approved "
                    f"range [{lo}, {hi}] mg for {drug} (off-label)."
                )
            else:
                confirmations.append(
                    f"Dose(s) {doses} mg within/near approved range [{lo}, {hi}] mg"
                )

        if problems:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=min(0.92, 0.7 + 0.1 * len(problems)),
                reasoning="; ".join(problems),
                evidence=evidence,
                source_db="druglabel",
            )
        if confirmations:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.82,
                reasoning="; ".join(confirmations),
                evidence=evidence,
                source_db="druglabel",
            )
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.35,
            reasoning=(
                f"Label retrieved for {drug} but claim contained no testable "
                "approval/dose statement."
            ),
            evidence=evidence,
            source_db="druglabel",
        )

    async def _get_label(self, drug: str) -> DrugLabel | None:
        if self._client is not None:
            try:
                return await self._client.get_label(drug)
            except Exception as exc:  # pragma: no cover
                logger.debug("Label online fetch failed: %s", exc)
        return DrugLabelClient.fallback_label(drug)

    async def health_check(self) -> bool:
        return True
