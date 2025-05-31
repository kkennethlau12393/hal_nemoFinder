"""Acme Corp's proprietary verifiers.

These classes demonstrate how a pharma company plugs its internal
knowledge into hal_nemoFinder. In production, swap the hardcoded
dictionaries for real database / REST calls against your systems.

Two verifiers are defined:

* :class:`AcmeInternalLibraryVerifier` — looks up molecular property
  claims against Acme's proprietary compound library.
* :class:`AcmeAssayDataVerifier` — checks target interaction claims
  against Acme's internal high-throughput-screening assay data.

Both register themselves with the global verifier registry at import
time via the ``@register_verifier`` decorator, so simply importing the
module (via ``PLUGIN_MODULES`` or an entry point) is enough to wire
them into the routing layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake Acme internal data (stand-in for Oracle / Snowflake / REST APIs)
# ---------------------------------------------------------------------------


#: Proprietary compound library keyed by internal Acme code.
_ACME_COMPOUND_LIBRARY: dict[str, dict[str, Any]] = {
    "ACME-123": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "molecular_weight": 151.16,
        "logp": 0.46,
        "project": "pain-a1",
    },
    "ACME-555": {
        "smiles": "CCN(CC)CCNC(=O)c1ccc(N)cc1",
        "molecular_weight": 235.33,
        "logp": 1.12,
        "project": "onco-b2",
    },
    "BX-7219": {
        # This compound code appears in hallucinated LLM output but does
        # not actually exist in the Acme library.
        "smiles": None,
        "molecular_weight": None,
        "logp": None,
        "project": None,
    },
}


#: Proprietary assay results (IC50 in nM).
_ACME_ASSAY_DATA: dict[tuple[str, str], dict[str, Any]] = {
    ("ACME-123", "EGFR"): {"ic50_nm": 420.0, "n_replicates": 6, "assay": "LanthaScreen"},
    ("ACME-123", "COX2"): {"ic50_nm": 1.7, "n_replicates": 8, "assay": "Fluor-COX"},
    ("ACME-555", "BRAF"): {"ic50_nm": 12.3, "n_replicates": 4, "assay": "Alphascreen"},
}


_COMPOUND_CODE_RE = re.compile(r"\b([A-Z]{2,5}-\d{1,6})\b")
_NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)")
_TARGET_ALIASES = {
    "EGFR": {"egfr", "erbb1", "her1"},
    "COX2": {"cox-2", "cox2", "ptgs2"},
    "BRAF": {"braf", "b-raf"},
}


# ---------------------------------------------------------------------------
# Compound library verifier
# ---------------------------------------------------------------------------


@register_verifier
class AcmeInternalLibraryVerifier(BaseVerifier):
    """Verify molecular-property claims against Acme's internal library.

    Flags claims that mention an Acme compound code (e.g. ``ACME-123``)
    together with a molecular weight or logP number, and compares the
    stated value against the library's record of truth.
    """

    name = "acme_internal_library"
    supported_claim_types = [ClaimType.molecular_property]

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        match = _COMPOUND_CODE_RE.search(claim_text)
        if not match:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No Acme compound code detected in claim.",
                evidence={},
                source_db=self.name,
            )

        code = match.group(1)
        record = _ACME_COMPOUND_LIBRARY.get(code)
        if record is None or record.get("smiles") is None:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.92,
                reasoning=(
                    f"Compound code {code} not found in Acme internal library; "
                    f"likely fabricated."
                ),
                evidence={"compound_code": code, "found": False},
                source_db=self.name,
            )

        lower = claim_text.lower()
        # Validate molecular weight when present.
        if "molecular weight" in lower or "g/mol" in lower:
            numbers = [float(n) for n in _NUMERIC_RE.findall(claim_text)]
            if numbers:
                stated = numbers[0]
                expected = float(record["molecular_weight"])
                delta = abs(stated - expected)
                if delta > max(1.0, 0.05 * expected):
                    return VerificationOutput(
                        verdict=Verdict.refuted,
                        confidence=0.95,
                        reasoning=(
                            f"{code}: stated MW {stated:.2f} g/mol differs from "
                            f"internal value {expected:.2f} g/mol (|Δ|={delta:.2f})."
                        ),
                        evidence={
                            "compound_code": code,
                            "stated_mw": stated,
                            "internal_mw": expected,
                        },
                        source_db=self.name,
                    )
                return VerificationOutput(
                    verdict=Verdict.verified,
                    confidence=0.95,
                    reasoning=(
                        f"{code}: stated MW {stated:.2f} g/mol matches internal "
                        f"library value {expected:.2f} g/mol."
                    ),
                    evidence={"compound_code": code, "internal_mw": expected},
                    source_db=self.name,
                )

        # Fall back to verifying that the compound exists.
        return VerificationOutput(
            verdict=Verdict.verified,
            confidence=0.6,
            reasoning=(
                f"{code} exists in Acme internal library (project {record['project']})."
            ),
            evidence={"compound_code": code, "project": record["project"]},
            source_db=self.name,
        )


# ---------------------------------------------------------------------------
# Proprietary assay verifier
# ---------------------------------------------------------------------------


@register_verifier
class AcmeAssayDataVerifier(BaseVerifier):
    """Verify target-interaction claims against Acme's HTS assay data.

    Parses claims of the form "compound X has IC50 of Y nM against
    target Z" and cross-checks them against Acme's assay database.
    """

    name = "acme_assay_data"
    supported_claim_types = [ClaimType.target_interaction]

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        lower = claim_text.lower()
        code_match = _COMPOUND_CODE_RE.search(claim_text)
        if not code_match:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No Acme compound code detected in claim.",
                evidence={},
                source_db=self.name,
            )

        code = code_match.group(1)
        target: str | None = None
        for canonical, aliases in _TARGET_ALIASES.items():
            if any(a in lower for a in aliases):
                target = canonical
                break
        if target is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"No known Acme assay target recognised for {code}.",
                evidence={"compound_code": code},
                source_db=self.name,
            )

        record = _ACME_ASSAY_DATA.get((code, target))
        if record is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.3,
                reasoning=(
                    f"No Acme assay data on record for {code} against {target}."
                ),
                evidence={"compound_code": code, "target": target},
                source_db=self.name,
            )

        # Parse the stated IC50 in nM.
        ic50_match = re.search(
            r"ic50\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*nm",
            lower,
        )
        if not ic50_match:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.5,
                reasoning=(
                    f"{code} has assay data against {target}, but no numeric "
                    f"IC50 detected in the claim."
                ),
                evidence={"compound_code": code, "target": target, "record": record},
                source_db=self.name,
            )

        stated = float(ic50_match.group(1))
        expected = float(record["ic50_nm"])
        # "Within 3x" is the customary Acme tolerance for HTS data.
        if stated <= 0.0 or not (expected / 3.0 <= stated <= expected * 3.0):
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.9,
                reasoning=(
                    f"Stated IC50 of {stated} nM for {code} vs {target} is "
                    f"inconsistent with Acme assay value {expected} nM "
                    f"(n={record['n_replicates']}, assay={record['assay']})."
                ),
                evidence={
                    "compound_code": code,
                    "target": target,
                    "stated_ic50_nm": stated,
                    "internal_ic50_nm": expected,
                },
                source_db=self.name,
            )

        return VerificationOutput(
            verdict=Verdict.verified,
            confidence=0.9,
            reasoning=(
                f"Stated IC50 of {stated} nM for {code} vs {target} matches "
                f"Acme assay (expected {expected} nM)."
            ),
            evidence={
                "compound_code": code,
                "target": target,
                "stated_ic50_nm": stated,
                "internal_ic50_nm": expected,
            },
            source_db=self.name,
        )
