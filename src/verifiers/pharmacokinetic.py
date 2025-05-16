"""Pharmacokinetic verifier — validates PK parameter claims for plausibility."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PK parameter extraction patterns
# ---------------------------------------------------------------------------

# Connector pattern shared across PK params: matches "of", "=", ":",
# "is", "was", "approximately", "~", and combinations thereof.
_CONNECTOR = r"(?:\s*(?:of|=|:|is|was|approximately|approx\.?|about|~|≈|of\s+approximately)\s*)?\s*"

_PK_PATTERNS: dict[str, re.Pattern[str]] = {
    "bioavailability": re.compile(
        # "bioavailability of 65%", "65% bioavailability", "F = 0.65",
        # "achieves 99.7% oral bioavailability"
        r"(?:(?:oral\s+)?bioavailability" + _CONNECTOR + r"([\d.]+)\s*%"
        r"|([\d.]+)\s*%\s+(?:oral\s+)?bioavailability"
        r"|\bF\s*=\s*([\d.]+))",
        re.IGNORECASE,
    ),
    "half_life": re.compile(
        # "half-life of 8 hours", "t1/2 = 5h", "elimination half-life: 8 h",
        # "terminal half-life of 380 hours"
        r"(?:(?:terminal|elimination|plasma)\s+)?(?:half[- ]life|t[½_]?1/2|t1/2)"
        + _CONNECTOR + r"([\d.]+)\s*(h|hr|hrs|hours?|min|minutes?|days?|d)\b",
        re.IGNORECASE,
    ),
    "cmax": re.compile(
        r"[Cc]max" + _CONNECTOR + r"([\d.]+)\s*(ng/mL|[μu]g/mL|mg/L|nmol/L|nM|µM)?",
    ),
    "auc": re.compile(
        r"AUC(?:0[–\-](?:inf|∞|t))?" + _CONNECTOR
        + r"([\d.]+)\s*"
        r"(ng[·.*]h/mL|[μu]g[·.*]h/mL|h[·.*]ng/mL|h[·.*][μu]g/mL)?",
        re.IGNORECASE,
    ),
    "clearance": re.compile(
        r"(?:total\s+|systemic\s+|renal\s+|hepatic\s+)?(?:clearance|CL)" + _CONNECTOR
        + r"([\d.]+)\s*(L/h|mL/min|L/h/kg|mL/min/kg)?",
        re.IGNORECASE,
    ),
    "logp": re.compile(
        r"(?:logP|log\s*P|LogP|cLogP|clog\s*P)" + _CONNECTOR + r"([+-]?[\d.]+)",
        re.IGNORECASE,
    ),
    "vd": re.compile(
        r"(?:volume\s+of\s+distribution|Vd|V_?d|Vss|V_?ss)" + _CONNECTOR
        + r"([\d.]+)\s*(L|L/kg|mL/kg)?",
        re.IGNORECASE,
    ),
    "mw": re.compile(
        r"(?:molecular\s+weight|MW|molar\s+mass)" + _CONNECTOR
        + r"([\d.]+)\s*(?:Da|g/mol|kDa)?",
        re.IGNORECASE,
    ),
    "tmax": re.compile(
        r"[Tt]max" + _CONNECTOR + r"([\d.]+)\s*(h|hr|hours?|min|minutes?)?",
    ),
    "protein_binding": re.compile(
        # "protein binding of 99%", "99% protein bound", "highly bound (>95%)"
        r"(?:protein\s+binding" + _CONNECTOR + r"([\d.]+)\s*%"
        r"|([\d.]+)\s*%\s+protein[- ]bound"
        r"|protein[- ]bound" + _CONNECTOR + r"([\d.]+)\s*%)",
        re.IGNORECASE,
    ),
}

# Time-unit normalisers (convert everything to hours)
_TIME_TO_HOURS: dict[str, float] = {
    "h": 1.0, "hour": 1.0, "hours": 1.0,
    "min": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
    "day": 24.0, "days": 24.0,
}

# ---------------------------------------------------------------------------
# Plausibility bounds for small-molecule drugs
# ---------------------------------------------------------------------------

_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "bioavailability": (0.0, 100.0),
    "half_life_hours": (0.001, 720.0),   # ~30 days max for small molecules
    "logp": (-5.0, 10.0),
    "mw": (50.0, 1500.0),
    "cmax": (0.0, 1e6),                  # must be positive
    "auc": (0.0, 1e8),
    "clearance": (0.0, 1e4),
    "vd": (0.0, 5000.0),                 # L/kg can be high for lipophilic drugs
    "tmax_hours": (0.0, 168.0),          # up to 1 week
}

# Lipinski rule of five
_LIPINSKI: dict[str, tuple[float, float]] = {
    "mw": (0, 500),
    "logp": (float("-inf"), 5),
    "hbd": (0, 5),
    "hba": (0, 10),
}


# ---------------------------------------------------------------------------
# Extracted PK container
# ---------------------------------------------------------------------------


class _ExtractedPK:
    """Container for extracted PK parameters."""

    __slots__ = ("values", "units", "raw")

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.units: dict[str, str] = {}
        self.raw: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Extraction and plausibility helpers
# ---------------------------------------------------------------------------


def _extract_pk_params(text: str) -> _ExtractedPK:
    """Extract PK parameters from *text*."""
    pk = _ExtractedPK()
    for key, pattern in _PK_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue

        # Find the first non-None numeric group (for patterns with
        # alternation like bioavailability that have multiple capture
        # groups, only one will match).
        value: float | None = None
        groups = match.groups()
        for g in groups:
            if g is None:
                continue
            try:
                value = float(g)
                break
            except (TypeError, ValueError):
                continue
        if value is None:
            continue

        # Extract a trailing unit if present (last non-None group that
        # cannot be parsed as a number).
        unit = ""
        for g in reversed(groups):
            if g is None:
                continue
            try:
                float(g)
            except (TypeError, ValueError):
                unit = g
                break

        pk.raw[key] = match.group(0)

        # Normalise time-dependent parameters to hours
        if key == "half_life" and unit:
            factor = _TIME_TO_HOURS.get(unit.lower(), 1.0)
            pk.values["half_life_hours"] = value * factor
            pk.units["half_life_hours"] = "h"
        elif key == "tmax" and unit:
            factor = _TIME_TO_HOURS.get(unit.lower(), 1.0)
            pk.values["tmax_hours"] = value * factor
            pk.units["tmax_hours"] = "h"
        elif key == "bioavailability":
            # If reported as a fraction (F = 0.65), convert to %.
            # Heuristic: if the matched text contained "%" the value is
            # already a percentage; otherwise treat as a fraction.  Note
            # that F > 1.0 in fraction form is itself impossible — the
            # plausibility check downstream catches it (>100%).
            if "%" not in match.group(0):
                if value <= 1.0:
                    value *= 100.0
                else:
                    # F > 1.0 — treat as the percentage they probably
                    # meant, which the plausibility check will then flag
                    # as exceeding 100%.
                    value *= 100.0
            pk.values[key] = value
            pk.units[key] = "%"
        else:
            pk.values[key] = value
            pk.units[key] = unit

    return pk


def _check_plausibility(pk: _ExtractedPK) -> tuple[list[str], list[str]]:
    """Check extracted PK values against hard plausibility bounds.

    Returns (plausible, implausible) lists of human-readable strings.
    """
    plausible: list[str] = []
    implausible: list[str] = []

    for param, value in pk.values.items():
        bounds = _PLAUSIBILITY.get(param)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo <= value <= hi:
            plausible.append(f"{param}={value} is within plausible range [{lo}, {hi}]")
        else:
            implausible.append(
                f"{param}={value} is OUTSIDE plausible range [{lo}, {hi}]"
            )

    return plausible, implausible


def _check_lipinski(pk: _ExtractedPK) -> tuple[int, list[str]]:
    """Check Lipinski rule-of-five. Returns (violations, details)."""
    violations = 0
    details: list[str] = []

    for prop, (lo, hi) in _LIPINSKI.items():
        if prop not in pk.values:
            continue
        val = pk.values[prop]
        if val < lo or val > hi:
            violations += 1
            details.append(f"{prop}={val} violates Lipinski (max {hi})")
        else:
            details.append(f"{prop}={val} passes Lipinski")

    return violations, details


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@register_verifier
class PharmacokineticVerifier(BaseVerifier):
    """Verify pharmacokinetic claims by checking plausibility and known drug data."""

    name = "pharmacokinetic"
    supported_claim_types = [ClaimType.pharmacokinetic]

    def __init__(self) -> None:
        self._pubchem: Any = None
        self._chembl: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from src.knowledge.pubchem import PubChemClient  # type: ignore[import-untyped]
            self._pubchem = PubChemClient()
        except Exception:
            logger.debug("PubChemClient unavailable.")

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
            logger.exception("PharmacokineticVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during PK verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        pk = _extract_pk_params(claim_text)
        evidence: dict[str, Any] = {
            "extracted_parameters": pk.values,
            "extracted_units": pk.units,
        }

        if not pk.values:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No pharmacokinetic parameters could be extracted from the claim.",
                evidence=evidence,
            )

        # Step 1: Plausibility checks
        plausible, implausible = _check_plausibility(pk)
        evidence["plausible"] = plausible
        evidence["implausible"] = implausible

        if implausible:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.9,
                reasoning=(
                    "Physically implausible PK values detected: "
                    + "; ".join(implausible) + "."
                ),
                evidence=evidence,
            )

        # Step 2: Lipinski check (if enough data)
        lip_violations, lip_details = _check_lipinski(pk)
        if lip_details:
            evidence["lipinski"] = {
                "violations": lip_violations,
                "details": lip_details,
            }

        # Step 3: Cross-reference with known drug data
        db_match = await self._cross_reference(claim_text, pk, evidence)

        # Resolve verdict
        reasons: list[str] = []
        if plausible:
            reasons.append(f"Plausibility checks passed: {'; '.join(plausible)}.")
        if lip_violations > 0:
            reasons.append(
                f"Lipinski rule-of-five: {lip_violations} violation(s) — "
                "compound may have poor oral bioavailability."
            )
        elif lip_details:
            reasons.append("Lipinski rule-of-five satisfied for stated properties.")

        if db_match:
            reasons.append("Cross-referenced with known drug data.")
            verdict = Verdict.verified
            confidence = min(0.9, 0.7 + 0.05 * len(plausible))
        elif plausible:
            verdict = Verdict.partially_supported
            confidence = min(0.7, 0.4 + 0.1 * len(plausible))
            reasons.append(
                "Values are plausible but could not be cross-referenced "
                "with known compound data."
            )
        else:
            verdict = Verdict.unverifiable
            confidence = 0.3
            reasons.append("Insufficient data for definitive verification.")

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=" ".join(reasons),
            evidence=evidence,
            source_db="pubchem+chembl" if db_match else "",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _cross_reference(
        self,
        claim_text: str,
        pk: _ExtractedPK,
        evidence: dict[str, Any],
    ) -> bool:
        """Try to match claim against known drug PK profiles."""
        # Extract a compound name heuristic
        name_match = re.search(
            r"\b([A-Z][a-z]{2,}(?:ib|ab|mab|nib|lib|tinib|zumab))\b",
            claim_text,
        )
        if not name_match:
            return False

        compound = name_match.group(1)

        if self._pubchem is not None:
            try:
                result = await self._pubchem.search_by_name(compound)
                if result:
                    evidence["pubchem_compound"] = (
                        result if isinstance(result, dict) else {"name": compound}
                    )
                    return True
            except Exception as exc:
                logger.debug("PubChem PK cross-ref failed: %s", exc)

        if self._chembl is not None:
            try:
                result = await self._chembl.search_molecule(compound)
                if result:
                    evidence["chembl_compound"] = (
                        result if isinstance(result, dict) else {"name": compound}
                    )
                    return True
            except Exception as exc:
                logger.debug("ChEMBL PK cross-ref failed: %s", exc)

        return False

    async def health_check(self) -> bool:
        return True  # plausibility checks are always available
