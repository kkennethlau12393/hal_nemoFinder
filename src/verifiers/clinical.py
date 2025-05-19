"""Clinical trial verifier — validates NCT IDs and trial attributes."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_NCT_RE = re.compile(r"\b(NCT\d{8})\b")

_PHASE_RE = re.compile(
    r"phase\s*(?:(\d)|([IViv]{1,3}))\b",
    re.IGNORECASE,
)

_STATUS_RE = re.compile(
    r"\b(completed|recruiting|active|terminated|withdrawn|suspended|"
    r"not\s+yet\s+recruiting|enrolling\s+by\s+invitation|"
    r"unknown\s+status)\b",
    re.IGNORECASE,
)

# Roman numeral mapping
_ROMAN_TO_INT: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4,
    "I": 1, "II": 2, "III": 3, "IV": 4,
}


def _normalise_phase(raw: str) -> int | None:
    """Convert a phase string to an integer (1-4) or None."""
    raw = raw.strip().lower()
    if raw.isdigit():
        return int(raw)
    return _ROMAN_TO_INT.get(raw)


def _extract_phase_from_text(text: str) -> int | None:
    """Extract the first phase mention from free text."""
    match = _PHASE_RE.search(text)
    if not match:
        return None
    digit, roman = match.group(1), match.group(2)
    if digit:
        return int(digit)
    if roman:
        return _ROMAN_TO_INT.get(roman.lower())
    return None


def _extract_status_from_text(text: str) -> str | None:
    """Extract the first status mention from free text."""
    match = _STATUS_RE.search(text)
    return match.group(0).lower().strip() if match else None


def _normalise_status(status: str) -> str:
    """Normalise a trial status string for comparison."""
    return re.sub(r"\s+", " ", status.strip().lower())


def _phase_from_api(api_phase: str) -> int | None:
    """Parse the phase value returned by the ClinicalTrials API.

    Common values: "Phase 1", "Phase 2", "Phase 1/Phase 2", "Early Phase 1",
    "Phase 3", "Phase 4", "Not Applicable".
    """
    nums = re.findall(r"\d", api_phase)
    if nums:
        return int(nums[-1])  # take the highest phase mentioned
    return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@register_verifier
class ClinicalTrialVerifier(BaseVerifier):
    """Verify clinical-trial claims by looking up NCT IDs and checking attributes."""

    name = "clinical"
    supported_claim_types = [ClaimType.clinical_outcome]

    def __init__(self) -> None:
        self._ct_client: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from src.knowledge.clinicaltrials import ClinicalTrialsClient  # type: ignore[import-untyped]
            self._ct_client = ClinicalTrialsClient()
        except Exception:
            logger.debug("ClinicalTrialsClient unavailable.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return await self._verify_impl(claim_text, claim_type, context)
        except Exception as exc:
            logger.exception("ClinicalTrialVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during clinical trial verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        nct_ids = _NCT_RE.findall(claim_text)
        stated_phase = _extract_phase_from_text(claim_text)
        stated_status = _extract_status_from_text(claim_text)

        evidence: dict[str, Any] = {
            "extracted_nct_ids": nct_ids,
            "stated_phase": stated_phase,
            "stated_status": stated_status,
        }

        if not nct_ids:
            # Try to verify by search if we have enough context
            return await self._verify_without_nct(claim_text, evidence)

        if self._ct_client is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=(
                    f"Found NCT ID(s) {', '.join(nct_ids)} but ClinicalTrials "
                    "client is unavailable for lookup."
                ),
                evidence=evidence,
            )

        # Look up each NCT ID
        all_matches: list[str] = []
        all_mismatches: list[str] = []
        found_ids: list[str] = []
        missing_ids: list[str] = []

        for nct_id in nct_ids:
            study = await self._lookup_study(nct_id)
            if study is None:
                missing_ids.append(nct_id)
                continue

            found_ids.append(nct_id)
            evidence[f"study_{nct_id}"] = study
            matches, mismatches = self._compare_attributes(
                study, stated_phase, stated_status, claim_text
            )
            all_matches.extend(matches)
            all_mismatches.extend(mismatches)

        evidence["found_ids"] = found_ids
        evidence["missing_ids"] = missing_ids
        evidence["attribute_matches"] = all_matches
        evidence["attribute_mismatches"] = all_mismatches

        return self._resolve_verdict(
            found_ids=found_ids,
            missing_ids=missing_ids,
            matches=all_matches,
            mismatches=all_mismatches,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _lookup_study(self, nct_id: str) -> dict[str, Any] | None:
        """Retrieve study data from ClinicalTrials.gov."""
        try:
            result = await self._ct_client.get_study(nct_id)
            if result:
                return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            logger.debug("ClinicalTrials lookup failed for %s: %s", nct_id, exc)
        return None

    @staticmethod
    def _compare_attributes(
        study: dict[str, Any],
        stated_phase: int | None,
        stated_status: str | None,
        claim_text: str,
    ) -> tuple[list[str], list[str]]:
        """Compare stated trial attributes against actual API data."""
        matches: list[str] = []
        mismatches: list[str] = []

        # Phase comparison
        api_phase_raw = study.get("phase", study.get("Phase", ""))
        if stated_phase is not None and api_phase_raw:
            api_phase = _phase_from_api(str(api_phase_raw))
            if api_phase is not None:
                if api_phase == stated_phase:
                    matches.append(f"Phase matches: stated {stated_phase}, actual {api_phase}")
                else:
                    mismatches.append(
                        f"Phase mismatch: stated phase {stated_phase}, "
                        f"actual phase {api_phase} ('{api_phase_raw}')"
                    )

        # Status comparison
        api_status = study.get("status", study.get("overall_status", study.get("Status", "")))
        if stated_status and api_status:
            norm_stated = _normalise_status(stated_status)
            norm_actual = _normalise_status(str(api_status))
            if norm_stated in norm_actual or norm_actual in norm_stated:
                matches.append(f"Status matches: stated '{stated_status}', actual '{api_status}'")
            else:
                mismatches.append(
                    f"Status mismatch: stated '{stated_status}', actual '{api_status}'"
                )

        # Condition / intervention keyword overlap (best effort)
        conditions = study.get("conditions", study.get("Conditions", []))
        if isinstance(conditions, list):
            cond_text = " ".join(str(c) for c in conditions).lower()
            # Check if any condition keyword appears in claim
            for condition in conditions:
                cond_str = str(condition).lower()
                if len(cond_str) > 3 and cond_str in claim_text.lower():
                    matches.append(f"Condition '{condition}' mentioned in claim")
                    break

        interventions = study.get("interventions", study.get("Interventions", []))
        if isinstance(interventions, list):
            for intervention in interventions:
                intv_str = str(intervention).lower()
                if len(intv_str) > 3 and intv_str in claim_text.lower():
                    matches.append(f"Intervention '{intervention}' mentioned in claim")
                    break

        return matches, mismatches

    async def _verify_without_nct(
        self,
        claim_text: str,
        evidence: dict[str, Any],
    ) -> VerificationOutput:
        """Attempt to verify a clinical-outcome claim that lacks an NCT ID."""
        if self._ct_client is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=(
                    "No NCT ID found in claim and ClinicalTrials client "
                    "is unavailable for keyword search."
                ),
                evidence=evidence,
            )

        # Try keyword search
        try:
            search_results = await self._ct_client.search_studies(claim_text[:200])
            if search_results:
                evidence["keyword_search_results"] = (
                    search_results[:5]
                    if isinstance(search_results, list)
                    else search_results
                )
                return VerificationOutput(
                    verdict=Verdict.partially_supported,
                    confidence=0.3,
                    reasoning=(
                        "No NCT ID found in claim, but keyword search returned "
                        "potentially related trials. Cannot confirm specific "
                        "clinical outcomes without an explicit trial identifier."
                    ),
                    evidence=evidence,
                    source_db="clinicaltrials.gov",
                )
        except Exception as exc:
            logger.debug("ClinicalTrials keyword search failed: %s", exc)

        return VerificationOutput(
            verdict=Verdict.unverifiable,
            confidence=0.0,
            reasoning="No NCT ID found and keyword search yielded no results.",
            evidence=evidence,
        )

    @staticmethod
    def _resolve_verdict(
        *,
        found_ids: list[str],
        missing_ids: list[str],
        matches: list[str],
        mismatches: list[str],
        evidence: dict[str, Any],
    ) -> VerificationOutput:
        """Combine lookup and comparison results into a verdict."""
        reasons: list[str] = []

        if found_ids:
            reasons.append(f"Trial(s) found on ClinicalTrials.gov: {', '.join(found_ids)}.")
        if missing_ids:
            reasons.append(f"Trial(s) NOT found: {', '.join(missing_ids)}.")
        if matches:
            reasons.append(f"Attribute matches: {'; '.join(matches)}.")
        if mismatches:
            reasons.append(f"Attribute mismatches: {'; '.join(mismatches)}.")

        if missing_ids and not found_ids:
            verdict = Verdict.refuted
            confidence = 0.85
        elif mismatches and not matches:
            verdict = Verdict.refuted
            confidence = 0.75
        elif found_ids and matches and not mismatches:
            verdict = Verdict.verified
            confidence = min(0.95, 0.7 + 0.1 * len(matches))
        elif found_ids:
            verdict = Verdict.partially_supported
            confidence = 0.5
        else:
            verdict = Verdict.unverifiable
            confidence = 0.2

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=" ".join(reasons),
            evidence=evidence,
            source_db="clinicaltrials.gov",
        )

    async def health_check(self) -> bool:
        return self._ct_client is not None
