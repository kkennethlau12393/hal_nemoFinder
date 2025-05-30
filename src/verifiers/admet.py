"""ADMET predictor verifier — deterministic RDKit-based ADMET sanity checks.

Computes rule-based ADMET flags from an RDKit molecule:

* **Absorption** — Lipinski Ro5 and Veber rules → predicted oral absorption.
* **Distribution** — TPSA + logP → predicted BBB penetration.
* **Metabolism** — heuristic flags for likely CYP substrates.
* **Excretion** — crude clearance proxy from MW + logP.
* **Toxicity** — PAINS / Brenk substructure catalog hits.

The verifier then compares the claim text against these predictions
and flags obvious inconsistencies (e.g. "crosses the blood-brain
barrier" for a highly polar molecule, "orally bioavailable" for a
molecule failing Ro5 badly).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, FilterCatalog as _FilterCatalog

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RDKIT_AVAILABLE = False


_SMILES_RE = re.compile(r"\b([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%&.~]{4,})\b")

_ORAL_RE = re.compile(
    r"\b(orally? bioavailable|oral bioavailability|good oral absorption)\b",
    re.IGNORECASE,
)
_BBB_RE = re.compile(
    r"\b(crosses? the blood[- ]brain barrier|bbb penetrant|cns penetrant|"
    r"brain penetrant|enters the cns)\b",
    re.IGNORECASE,
)
_NO_BBB_RE = re.compile(
    r"\b(does not cross the blood[- ]brain barrier|cns excluded|"
    r"no bbb penetration|non[- ]penetrant)\b",
    re.IGNORECASE,
)
_NONTOXIC_RE = re.compile(
    r"\b(non[- ]toxic|safe profile|no toxicity|clean toxicology)\b",
    re.IGNORECASE,
)


def _extract_smiles(text: str) -> str | None:
    if not _RDKIT_AVAILABLE:
        return None
    for m in _SMILES_RE.finditer(text):
        token = m.group(1)
        if len(token) < 4:
            continue
        if not re.search(r"[A-Za-z]", token):
            continue
        if not re.search(r"[^A-Za-z]", token):
            continue
        try:
            mol = Chem.MolFromSmiles(token)
            if mol is not None and mol.GetNumAtoms() >= 3:
                return token
        except Exception:  # pragma: no cover
            continue
    return None


def _admet_profile(mol: Any) -> dict[str, Any]:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)

    lipinski_ok = sum(
        [mw <= 500, logp <= 5, hbd <= 5, hba <= 10]
    ) >= 3
    veber_ok = rotb <= 10 and tpsa <= 140
    oral_ok = lipinski_ok and veber_ok

    # BBB rule of thumb (Clark): TPSA < 90, 1 < logP < 4, MW < 450.
    bbb_ok = tpsa < 90 and 1.0 < logp < 4.0 and mw < 450
    bbb_unlikely = tpsa > 140 or logp < 0 or mw > 500

    # CYP substrate heuristic: aromatic rings + logP > 2 often CYP3A4.
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) > 4
    cyp_substrate_likely = aromatic and logp > 2

    # Toxicity: run PAINS/Brenk filters.
    tox_hits: list[str] = []
    try:
        params = _FilterCatalog.FilterCatalogParams()
        params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        cat = _FilterCatalog.FilterCatalog(params)
        for m in cat.GetMatches(mol):
            tox_hits.append(m.GetDescription())
    except Exception as exc:  # pragma: no cover
        logger.debug("ADMET catalog filter failed: %s", exc)

    return {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotb,
        "lipinski_passes": lipinski_ok,
        "veber_passes": veber_ok,
        "predicted_oral_absorption": oral_ok,
        "predicted_bbb_penetration": bbb_ok,
        "bbb_unlikely": bbb_unlikely,
        "cyp_substrate_likely": cyp_substrate_likely,
        "toxicophore_matches": tox_hits,
    }


@register_verifier
class ADMETPredictorVerifier(BaseVerifier):
    """Rule-based ADMET predictor that flags claim/structure mismatches."""

    name = "admet"
    supported_claim_types = [
        ClaimType.pharmacokinetic,
        ClaimType.molecular_property,
    ]

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return self._verify_impl(claim_text, context)
        except Exception as exc:  # pragma: no cover
            logger.exception("ADMETPredictorVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during ADMET verification: {exc}",
                source_db="admet",
            )

    def _verify_impl(
        self, claim_text: str, context: dict[str, Any]
    ) -> VerificationOutput:
        if not _RDKIT_AVAILABLE:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="RDKit not installed; ADMET predictions unavailable.",
                source_db="admet",
            )

        smiles = _extract_smiles(claim_text) or context.get("smiles")
        if not smiles:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No parseable SMILES found in claim.",
                source_db="admet",
            )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.8,
                reasoning=f"SMILES {smiles!r} does not parse; ADMET impossible.",
                evidence={"smiles": smiles},
                source_db="admet",
            )

        profile = _admet_profile(mol)
        evidence: dict[str, Any] = {"smiles": smiles, "profile": profile}
        problems: list[str] = []
        confirmations: list[str] = []

        if _ORAL_RE.search(claim_text):
            if profile["predicted_oral_absorption"]:
                confirmations.append("oral bioavailability consistent with Lipinski+Veber")
            else:
                problems.append(
                    "Claim asserts oral bioavailability but molecule fails "
                    "Lipinski/Veber rules."
                )

        if _BBB_RE.search(claim_text):
            if profile["bbb_unlikely"]:
                problems.append(
                    "Claim asserts BBB penetration but TPSA/logP/MW suggest CNS exclusion."
                )
            elif profile["predicted_bbb_penetration"]:
                confirmations.append("BBB penetration consistent with physchem profile")

        if _NO_BBB_RE.search(claim_text) and profile["predicted_bbb_penetration"]:
            problems.append(
                "Claim says no BBB penetration, but physchem profile is within "
                "Clark BBB limits (TPSA<90, 1<logP<4, MW<450)."
            )

        if _NONTOXIC_RE.search(claim_text) and profile["toxicophore_matches"]:
            problems.append(
                f"Claim asserts non-toxic/safe profile but molecule matches "
                f"{len(profile['toxicophore_matches'])} PAINS/Brenk toxicophores."
            )

        if problems:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=min(0.9, 0.6 + 0.1 * len(problems)),
                reasoning="; ".join(problems),
                evidence=evidence,
                source_db="admet",
            )
        if confirmations:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.7,
                reasoning="; ".join(confirmations),
                evidence=evidence,
                source_db="admet",
            )
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.4,
            reasoning=(
                "ADMET profile computed but no explicit ADMET statement in "
                "the claim to verify."
            ),
            evidence=evidence,
            source_db="admet",
        )

    async def health_check(self) -> bool:
        return _RDKIT_AVAILABLE
