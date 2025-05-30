"""Toxicity-predictor verifier — deterministic toxicophore detection.

Runs a fixed set of SMARTS patterns over an RDKit-parsed molecule to
flag well-known structural alerts (aromatic amines, aliphatic halides,
epoxides, quinones, nitroaromatics, Michael acceptors).  When the
claim asserts "no toxicity" / "safe profile" and one or more alerts
fire, the verifier refutes the claim.
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

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RDKIT_AVAILABLE = False


_TOXICOPHORES: list[tuple[str, str, str]] = [
    ("aromatic_amine", "[NX3;H2,H1;!$(NC=O)][c]",
     "aromatic amine — potential DNA adduct formation"),
    ("aliphatic_halide", "[CX4][F,Cl,Br,I]",
     "aliphatic halide — potential alkylating agent"),
    ("epoxide", "C1OC1",
     "epoxide — electrophilic, potential alkylating agent"),
    ("quinone", "O=C1C=CC(=O)C=C1",
     "quinone — oxidative stress / redox cycling"),
    ("nitroaromatic", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
     "nitroaromatic — nitroreduction to reactive species"),
    ("azo", "[NX2]=[NX2]",
     "azo group — potential carcinogen via reduction"),
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]",
     "Michael acceptor — electrophilic, potential thiol reactivity"),
    ("thiocarbonyl", "[#6]=[SX1]",
     "thiocarbonyl — reactive; hepatotoxicity alerts"),
    ("hydrazine", "[NX3][NX3]",
     "hydrazine — genotoxic"),
    ("aldehyde", "[CX3H1](=O)[#6]",
     "reactive aldehyde — protein adduct formation"),
]


_NONTOX_RE = re.compile(
    r"\b(non[- ]toxic|no toxicity|safe profile|clean tox(?:icology)?|"
    r"no toxicophores?|no structural alerts?|excellent safety profile)\b",
    re.IGNORECASE,
)

_SMILES_RE = re.compile(r"\b([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%&.~]{4,})\b")


def _extract_smiles(text: str) -> str | None:
    if not _RDKIT_AVAILABLE:
        return None
    for m in _SMILES_RE.finditer(text):
        token = m.group(1)
        if len(token) < 4 or not re.search(r"[A-Za-z]", token) or not re.search(r"[^A-Za-z]", token):
            continue
        try:
            mol = Chem.MolFromSmiles(token)
        except Exception:  # pragma: no cover
            continue
        if mol is not None and mol.GetNumAtoms() >= 3:
            return token
    return None


def _match_toxicophores(mol: Any) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for name, smarts, description in _TOXICOPHORES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            hits.append({"name": name, "description": description, "smarts": smarts})
    return hits


@register_verifier
class ToxicityPredictorVerifier(BaseVerifier):
    """Flag structural-alert toxicophores against "no toxicity" claims."""

    name = "toxicity"
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
            logger.exception("ToxicityPredictorVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during toxicity verification: {exc}",
                source_db="toxicity",
            )

    def _verify_impl(
        self, claim_text: str, context: dict[str, Any]
    ) -> VerificationOutput:
        if not _RDKIT_AVAILABLE:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="RDKit not installed; toxicophore scan unavailable.",
                source_db="toxicity",
            )

        smiles = _extract_smiles(claim_text) or context.get("smiles")
        if not smiles:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No parseable SMILES found in claim.",
                source_db="toxicity",
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.8,
                reasoning=f"SMILES {smiles!r} does not parse; cannot scan toxicophores.",
                evidence={"smiles": smiles},
                source_db="toxicity",
            )

        hits = _match_toxicophores(mol)
        evidence: dict[str, Any] = {"smiles": smiles, "toxicophores": hits}
        says_nontoxic = bool(_NONTOX_RE.search(claim_text))

        if says_nontoxic and hits:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=min(0.95, 0.75 + 0.05 * len(hits)),
                reasoning=(
                    f"Claim asserts non-toxic/safe profile but molecule contains "
                    f"{len(hits)} toxicophore(s): {[h['name'] for h in hits]}."
                ),
                evidence=evidence,
                source_db="toxicity",
            )
        if says_nontoxic and not hits:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.75,
                reasoning=(
                    "Claim of non-toxic profile consistent with absence of "
                    "structural alerts in the SMARTS catalog."
                ),
                evidence=evidence,
                source_db="toxicity",
            )
        if hits:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.5,
                reasoning=(
                    f"Toxicophore(s) detected ({[h['name'] for h in hits]}) "
                    "but claim makes no toxicity statement to verify."
                ),
                evidence=evidence,
                source_db="toxicity",
            )
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.35,
            reasoning=(
                "No toxicophores found and claim contains no toxicity language; "
                "partial support only."
            ),
            evidence=evidence,
            source_db="toxicity",
        )

    async def health_check(self) -> bool:
        return _RDKIT_AVAILABLE
