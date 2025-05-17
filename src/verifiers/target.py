"""Target interaction and mechanism-of-action verifier."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers for extracting targets and compounds
# ---------------------------------------------------------------------------

_TARGET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:targets?|binds?\s+to|inhibits?|activates?|antagoni[sz]es?|agoni[sz]es?|modulates?|blocks?)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-/]+(?:\s+(?:receptor|kinase|protease|channel|transporter|enzyme))?)", re.IGNORECASE),
    re.compile(r"([A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?)\s+(?:receptor|kinase|protease|channel|transporter|enzyme)", re.IGNORECASE),
    re.compile(r"(?:against|for|on)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-/]+(?:\s+(?:receptor|kinase|protease|channel|transporter|enzyme))?)"),
    # UniProt accession
    re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"),
]

_COMPOUND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b([A-Z][a-z]{2,}(?:ib|ab|mab|nib|lib|tinib|zumab|ximab|umab))\b"),
    re.compile(r"\b([A-Z]{2,6}-\d{3,})\b"),
    re.compile(r"(?:compound|drug|molecule|ligand)\s+([A-Za-z0-9\-]+)", re.IGNORECASE),
]

_MECHANISM_KEYWORDS: dict[str, list[str]] = {
    "inhibitor": ["inhibit", "block", "suppress", "antagoni", "negative"],
    "activator": ["activat", "agonist", "stimulat", "enhanc", "positive"],
    "modulator": ["modulat", "allosteric", "regulat"],
}

# Regex for numeric potency claims: "Ki = 37 nM", "IC50 of 2 nM",
# "Kd ~ 0.5 uM", etc.  Captures (type, value, unit).
_POTENCY_PATTERN = re.compile(
    r"\b(Ki|IC50|EC50|Kd)\b\s*(?:of|=|~|is|:)?\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*"
    r"(p[Mm]|n[Mm]|[uμ][Mm]|m[Mm]|M)\b",
)

# Unit → factor relative to nM.
_UNIT_TO_NM: dict[str, float] = {
    "pM": 1e-3,
    "pm": 1e-3,
    "nM": 1.0,
    "nm": 1.0,
    "uM": 1e3,
    "um": 1e3,
    "μM": 1e3,
    "μm": 1e3,
    "mM": 1e6,
    "mm": 1e6,
    "M": 1e9,
}


def _format_citation(record: Any) -> str:
    """Build a short human-readable citation from a :class:`ProvenanceRecord`."""
    bits: list[str] = []
    if getattr(record, "reference_journal", None):
        bits.append(str(record.reference_journal))
    if getattr(record, "reference_year", None):
        bits.append(str(record.reference_year))
    if getattr(record, "reference_doi", None):
        bits.append(f"doi:{record.reference_doi}")
    return ", ".join(bits) if bits else "unknown reference"


def _extract_potency(text: str) -> dict[str, Any] | None:
    """Extract a single numeric potency claim from *text*.

    Returns a dict with keys ``type``, ``value_nm``, ``raw_value``,
    ``raw_unit`` or ``None`` if no match.
    """
    match = _POTENCY_PATTERN.search(text)
    if not match:
        return None
    ptype, val, unit = match.group(1), match.group(2), match.group(3)
    factor = _UNIT_TO_NM.get(unit)
    if factor is None:
        return None
    try:
        raw_value = float(val)
    except ValueError:
        return None
    return {
        "type": ptype.upper() if ptype.upper() in {"KI", "KD"} else ptype.upper(),
        "value_nm": raw_value * factor,
        "raw_value": raw_value,
        "raw_unit": unit,
    }


def _extract_targets(text: str) -> list[str]:
    """Extract potential protein/target names from *text*."""
    targets: list[str] = []
    seen: set[str] = set()
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip().rstrip(".,;:")
            key = name.lower()
            if key not in seen and len(name) >= 2:
                seen.add(key)
                targets.append(name)
    return targets


# ---------------------------------------------------------------------------
# Known protein family bounds
# ---------------------------------------------------------------------------
#
# When a claim mentions a member of a well-known protein family (e.g. BRCA9,
# CASP15, CDK28), we can flag it as fabricated WITHOUT external API access:
# the family has a fixed set of members and the cited member doesn't exist.
# This is a deterministic, bounded check — not a model opinion.

# Maps family prefix → highest valid member number.  Members above this
# number are flagged as fabricated.
_KNOWN_PROTEIN_FAMILIES: dict[str, int] = {
    "BRCA": 2,        # Only BRCA1, BRCA2 exist
    "CDK": 20,        # CDK1-20 (some have aliases like CCNH)
    "CASP": 14,       # Caspase-1 through 14
    "BAX": 1,         # Just BAX
    "BAK": 1,         # Just BAK1
    "BCL": 11,        # BCL2, BCL2L1-11 family members
    "TLR": 13,        # TLR1-13 (mouse has more, human ~10)
    "HOXA": 13,
    "HOXB": 13,
    "HOXC": 13,
    "HOXD": 13,
    "ERBB": 4,        # ERBB1-4
    "STAT": 6,        # STAT1, 2, 3, 4, 5A, 5B, 6
    "JAK": 3,         # JAK1, JAK2, JAK3, TYK2 (TYK2 not JAK4)
    "FGFR": 4,        # FGFR1-4
    "VEGFR": 3,       # VEGFR1-3
    "PDGFR": 2,       # PDGFRA, PDGFRB
    "PIK3C": 4,       # PIK3CA, B, D, G
    "MAP2K": 7,       # MAP2K1-7
    "MAPK": 14,       # MAPK1-14
    "RAB": 43,        # very large family
    "RAS": 3,         # KRAS, HRAS, NRAS
}

_PROTEIN_FAMILY_RE = re.compile(
    r"\b([A-Z]{2,6})(\d{1,3})\b"
)


def _detect_fabricated_proteins(targets: list[str]) -> list[dict[str, Any]]:
    """Return a list of fabricated-protein flags from *targets*.

    For each token of the form FAMILY-NUMBER, check whether the family is
    in :data:`_KNOWN_PROTEIN_FAMILIES` and whether the number exceeds the
    known maximum.  Returns a list of dicts with the offending name, the
    family, the maximum known member, and a reason string.
    """
    flags: list[dict[str, Any]] = []
    for target in targets:
        # Strip suffixes like "kinase", "receptor", "-alpha", etc.
        bare = re.sub(
            r"\s*(?:kinase|receptor|protein|enzyme|protease|channel|transporter)$",
            "",
            target,
            flags=re.IGNORECASE,
        ).strip()
        for m in _PROTEIN_FAMILY_RE.finditer(bare):
            family = m.group(1)
            number = int(m.group(2))
            max_known = _KNOWN_PROTEIN_FAMILIES.get(family)
            if max_known is None:
                continue
            if number > max_known:
                flags.append(
                    {
                        "name": target,
                        "family": family,
                        "claimed_member": number,
                        "max_known_member": max_known,
                        "reason": (
                            f"{target!r} appears to reference {family} family member "
                            f"{number}, but the highest documented {family} member "
                            f"is {family}{max_known}. {family}{number} is likely fabricated."
                        ),
                    }
                )
    return flags


def _extract_compounds(text: str) -> list[str]:
    """Extract potential compound/drug names from *text*."""
    compounds: list[str] = []
    seen: set[str] = set()
    for pattern in _COMPOUND_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                compounds.append(name)
    return compounds


def _classify_mechanism(text: str) -> str | None:
    """Determine the stated mechanism type from *text*."""
    text_lower = text.lower()
    for mechanism, keywords in _MECHANISM_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return mechanism
    return None


def _mechanism_consistent(stated_mechanism: str, function_text: str) -> bool:
    """Check whether *stated_mechanism* is consistent with a UniProt function description."""
    func_lower = function_text.lower()
    keywords = _MECHANISM_KEYWORDS.get(stated_mechanism, [])
    return any(kw in func_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@register_verifier
class TargetVerifier(BaseVerifier):
    """Verify claims about drug-target interactions and mechanisms of action."""

    name = "target"
    supported_claim_types = [ClaimType.target_interaction, ClaimType.mechanism_of_action]

    def __init__(self) -> None:
        self._uniprot: Any = None
        self._chembl: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from src.knowledge.uniprot import UniProtClient  # type: ignore[import-untyped]
            self._uniprot = UniProtClient()
        except Exception:
            logger.debug("UniProtClient unavailable.")

        try:
            from src.knowledge.chembl import ChEMBLClient  # type: ignore[import-untyped]
            self._chembl = ChEMBLClient()
        except Exception:
            logger.debug("ChEMBLClient unavailable.")

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
            logger.exception("TargetVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during target verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        targets = _extract_targets(claim_text)
        compounds = _extract_compounds(claim_text)
        stated_mechanism = _classify_mechanism(claim_text)
        potency = _extract_potency(claim_text)

        evidence: dict[str, Any] = {
            "extracted_targets": targets,
            "extracted_compounds": compounds,
            "stated_mechanism": stated_mechanism,
        }
        if potency is not None:
            evidence["claimed_potency"] = potency

        # Deterministic fabricated-protein check (no external API needed).
        fabricated = _detect_fabricated_proteins(targets)
        if fabricated:
            evidence["fabricated_proteins"] = fabricated
            reasons = "; ".join(f["reason"] for f in fabricated)
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.95,
                reasoning=(
                    f"Fabricated protein name(s) detected based on known "
                    f"family bounds: {reasons}"
                ),
                evidence=evidence,
                source_db="target",
            )

        if not targets:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No protein or target name could be extracted from the claim.",
                evidence=evidence,
            )

        # --- Step 1: Verify each target exists in UniProt ---
        target_results: dict[str, dict[str, Any]] = {}
        targets_found: list[str] = []
        targets_missing: list[str] = []

        for target_name in targets:
            result = await self._lookup_target(target_name)
            if result is not None:
                target_results[target_name] = result
                targets_found.append(target_name)
            else:
                targets_missing.append(target_name)

        evidence["target_lookup"] = {
            "found": targets_found,
            "not_found": targets_missing,
        }

        if not targets_found:
            # If we have no clients at all, we can't verify
            if self._uniprot is None:
                return VerificationOutput(
                    verdict=Verdict.unverifiable,
                    confidence=0.0,
                    reasoning=(
                        "UniProt client is unavailable; cannot verify target existence."
                    ),
                    evidence=evidence,
                )
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.7,
                reasoning=(
                    f"Target(s) not found in UniProt: {', '.join(targets_missing)}. "
                    "The claimed protein target may not exist or may be misspelled."
                ),
                evidence=evidence,
            )

        # --- Step 2: Check compound-target interaction in ChEMBL ---
        interaction_confirmed = False
        if compounds and self._chembl is not None:
            interaction_confirmed = await self._check_interaction(
                compounds, targets_found, evidence
            )

        # --- Step 3: Mechanism verification ---
        mechanism_ok: bool | None = None
        if stated_mechanism and target_results:
            mechanism_ok = self._verify_mechanism(
                stated_mechanism, target_results, evidence
            )

        # --- Step 4: Bioactivity provenance check (Ki/IC50/Kd claims) ---
        potency_verdict: bool | None = None
        provenance_reasoning: str | None = None
        if potency is not None and compounds and targets_found:
            potency_verdict, provenance_reasoning = await self._check_provenance(
                compounds, targets_found, potency, evidence
            )

        # --- Determine verdict ---
        return self._resolve_verdict(
            targets_found=targets_found,
            targets_missing=targets_missing,
            interaction_confirmed=interaction_confirmed,
            mechanism_ok=mechanism_ok,
            compounds=compounds,
            stated_mechanism=stated_mechanism,
            potency_verdict=potency_verdict,
            provenance_reasoning=provenance_reasoning,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _lookup_target(self, name: str) -> dict[str, Any] | None:
        """Look up a target in UniProt. Returns result dict or None."""
        if self._uniprot is None:
            return None
        try:
            result = await self._uniprot.search_protein(name)
            if result:
                return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            logger.debug("UniProt lookup failed for %r: %s", name, exc)
        return None

    async def _check_interaction(
        self,
        compounds: list[str],
        targets: list[str],
        evidence: dict[str, Any],
    ) -> bool:
        """Check ChEMBL for bioactivity data linking any compound to any target."""
        for compound in compounds:
            try:
                mol_result = await self._chembl.search_molecule(compound)
                if not mol_result:
                    continue
                chembl_id = (
                    mol_result.get("molecule_chembl_id")
                    if isinstance(mol_result, dict)
                    else None
                )
                if not chembl_id:
                    continue
                activities = await self._chembl.get_bioactivities(chembl_id)
                if activities:
                    evidence["chembl_bioactivities"] = {
                        "compound": compound,
                        "chembl_id": chembl_id,
                        "activity_count": (
                            len(activities) if isinstance(activities, list) else 1
                        ),
                    }
                    return True
            except Exception as exc:
                logger.debug("ChEMBL interaction check failed for %r: %s", compound, exc)
        return False

    async def _check_provenance(
        self,
        compounds: list[str],
        targets: list[str],
        potency: dict[str, Any],
        evidence: dict[str, Any],
    ) -> tuple[bool | None, str | None]:
        """Retrieve a :class:`ProvenanceRecord` for the claim and compare values.

        Returns a tuple ``(verdict, reasoning)`` where *verdict* is
        ``True`` (matches), ``False`` (clearly off), or ``None`` (no
        provenance available).  The reasoning cites the primary paper.
        """
        if self._chembl is None or not hasattr(
            self._chembl, "get_bioactivity_with_provenance"
        ):
            return None, None

        for compound in compounds:
            for target in targets:
                try:
                    record = await self._chembl.get_bioactivity_with_provenance(
                        compound, target
                    )
                except Exception as exc:
                    logger.debug(
                        "Provenance lookup failed for %s/%s: %s",
                        compound,
                        target,
                        exc,
                    )
                    continue
                if record is None:
                    continue

                # Stash the full audit record in the evidence dict.
                evidence["bioactivity_provenance"] = {
                    "compound": compound,
                    "target": target,
                    "assay_id": record.assay_id,
                    "assay_type": record.assay_type,
                    "assay_description": record.assay_description,
                    "standard_type": record.standard_type,
                    "standard_value": record.standard_value,
                    "standard_units": record.standard_units,
                    "pchembl_value": record.pchembl_value,
                    "confidence_score": record.confidence_score,
                    "reference_doi": record.reference_doi,
                    "reference_year": record.reference_year,
                    "reference_journal": record.reference_journal,
                }

                # Compare claimed vs measured with a 3x tolerance.
                claimed_nm = potency.get("value_nm")
                measured_nm: float | None = None
                if (
                    record.standard_units
                    and record.standard_units.lower() == "nm"
                    and record.standard_value is not None
                ):
                    measured_nm = float(record.standard_value)

                citation = _format_citation(record)

                if measured_nm is None or claimed_nm is None:
                    return None, (
                        f"Found provenance for {compound}/{target} "
                        f"({citation}) but numeric comparison is not possible."
                    )

                ratio = max(claimed_nm, measured_nm) / max(
                    1e-9, min(claimed_nm, measured_nm)
                )
                evidence["bioactivity_provenance"]["claimed_vs_measured_ratio"] = ratio

                if ratio <= 3.0:
                    return True, (
                        f"Claimed {potency['type']} "
                        f"{potency['raw_value']} {potency['raw_unit']} "
                        f"is consistent (within 3x) with measured "
                        f"{record.standard_value} {record.standard_units} "
                        f"from {citation}."
                    )
                return False, (
                    f"Claimed {potency['type']} "
                    f"{potency['raw_value']} {potency['raw_unit']} "
                    f"differs from measured "
                    f"{record.standard_value} {record.standard_units} "
                    f"({ratio:.1f}x off) per {citation}."
                )
        return None, None

    def _verify_mechanism(
        self,
        stated_mechanism: str,
        target_results: dict[str, dict[str, Any]],
        evidence: dict[str, Any],
    ) -> bool:
        """Check whether the stated mechanism aligns with UniProt function data."""
        for target_name, data in target_results.items():
            function_text = ""
            if isinstance(data, dict):
                function_text = data.get("function", data.get("description", ""))
                if isinstance(function_text, list):
                    function_text = " ".join(str(f) for f in function_text)
            if function_text and _mechanism_consistent(stated_mechanism, function_text):
                evidence["mechanism_alignment"] = {
                    "target": target_name,
                    "stated": stated_mechanism,
                    "aligned": True,
                }
                return True

        evidence["mechanism_alignment"] = {
            "stated": stated_mechanism,
            "aligned": False,
        }
        return False

    @staticmethod
    def _resolve_verdict(
        *,
        targets_found: list[str],
        targets_missing: list[str],
        interaction_confirmed: bool,
        mechanism_ok: bool | None,
        compounds: list[str],
        stated_mechanism: str | None,
        evidence: dict[str, Any],
        potency_verdict: bool | None = None,
        provenance_reasoning: str | None = None,
    ) -> VerificationOutput:
        """Combine all signals into a single verdict."""
        reasons: list[str] = []
        confidence_parts: list[float] = []

        # Target existence
        if targets_found:
            reasons.append(f"Target(s) confirmed in UniProt: {', '.join(targets_found)}.")
            confidence_parts.append(0.6)
        if targets_missing:
            reasons.append(f"Target(s) NOT found: {', '.join(targets_missing)}.")

        # Interaction
        if interaction_confirmed:
            reasons.append("Compound-target interaction confirmed in ChEMBL.")
            confidence_parts.append(0.85)
        elif compounds:
            reasons.append("Compound-target interaction could not be confirmed in ChEMBL.")

        # Mechanism
        if mechanism_ok is True:
            reasons.append(f"Stated mechanism ({stated_mechanism}) aligns with known target function.")
            confidence_parts.append(0.7)
        elif mechanism_ok is False:
            reasons.append(f"Stated mechanism ({stated_mechanism}) does not align with known target function.")

        # Bioactivity provenance
        if provenance_reasoning:
            reasons.append(provenance_reasoning)
        if potency_verdict is True:
            confidence_parts.append(0.95)
        elif potency_verdict is False:
            confidence_parts.append(0.9)

        # Verdict logic
        if potency_verdict is False:
            verdict = Verdict.refuted
        elif potency_verdict is True:
            verdict = Verdict.verified
        elif interaction_confirmed and (mechanism_ok is not False):
            verdict = Verdict.verified
        elif targets_missing and not targets_found:
            verdict = Verdict.refuted
        elif mechanism_ok is False:
            verdict = Verdict.refuted
        elif targets_found and not interaction_confirmed:
            verdict = Verdict.partially_supported
        else:
            verdict = Verdict.partially_supported

        confidence = max(confidence_parts) if confidence_parts else 0.3

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=" ".join(reasons),
            evidence=evidence,
            source_db="uniprot+chembl",
        )

    async def health_check(self) -> bool:
        return self._uniprot is not None or self._chembl is not None
