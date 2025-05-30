"""Retrosynthesis complexity verifier.

Uses an RDKit-native synthetic-accessibility proxy (spiro + bridgehead +
FractionCSP3 + ring complexity + size) to predict whether a molecule is
easy or hard to synthesize, then flags claim/structure mismatches such
as "easily synthesized" for score > 6 or "challenging synthesis" for
score < 3.  Also rejects molecules whose SMILES contains impossible
valences.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RDKIT_AVAILABLE = False


_SMILES_RE = re.compile(r"\b([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%&.~]{4,})\b")


_EASY_RE = re.compile(
    r"\b(easily synthesi[sz]ed|simple synthesis|trivial to make|"
    r"readily synthesi[sz]ed|straightforward synthesis|one[- ]step synthesis|"
    r"easy to (?:make|prepare|synthesi[sz]e))\b",
    re.IGNORECASE,
)
_HARD_RE = re.compile(
    r"\b(challenging synthesis|complex route|difficult to synthesi[sz]e|"
    r"total synthesis|multi[- ]step synthesis|complex synthesis)\b",
    re.IGNORECASE,
)


def _sa_score(mol: Any) -> dict[str, Any]:
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    macro = sum(1 for r in ring_info.AtomRings() if len(r) >= 8)
    n_heavy = mol.GetNumHeavyAtoms()
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    size_penalty = math.log10(max(n_heavy, 1)) * 0.5
    raw = (
        1.0
        + 1.2 * n_spiro
        + 1.0 * n_bridge
        + 2.0 * fsp3
        + 0.3 * n_rings
        + 1.5 * macro
        + 0.25 * n_stereo
        + size_penalty
    )
    score = max(1.0, min(10.0, raw))
    return {
        "score": round(score, 2),
        "num_spiro": n_spiro,
        "num_bridgehead": n_bridge,
        "fraction_csp3": round(fsp3, 3),
        "num_rings": n_rings,
        "num_macrocycles": macro,
        "num_stereocenters": n_stereo,
        "num_heavy_atoms": n_heavy,
    }


def _extract_smiles(text: str) -> str | None:
    if not _RDKIT_AVAILABLE:
        return None
    best: tuple[int, str] | None = None
    for m in _SMILES_RE.finditer(text):
        token = m.group(1)
        if len(token) < 4 or not re.search(r"[A-Za-z]", token) or not re.search(r"[^A-Za-z]", token):
            continue
        try:
            mol = Chem.MolFromSmiles(token)
        except Exception:  # pragma: no cover
            continue
        if mol is None:
            continue
        natoms = mol.GetNumAtoms()
        if natoms < 3:
            continue
        if best is None or natoms > best[0]:
            best = (natoms, token)
    return best[1] if best else None


@register_verifier
class RetrosynthesisVerifier(BaseVerifier):
    """Verify synthesis-complexity claims against an SA-proxy score."""

    name = "retrosynthesis"
    supported_claim_types = [
        ClaimType.molecular_property,
        ClaimType.general_biomedical,
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
            logger.exception("RetrosynthesisVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during retrosynthesis verification: {exc}",
                source_db="retrosynthesis",
            )

    def _verify_impl(
        self, claim_text: str, context: dict[str, Any]
    ) -> VerificationOutput:
        if not _RDKIT_AVAILABLE:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="RDKit not installed; retrosynthesis prediction unavailable.",
                source_db="retrosynthesis",
            )

        smiles = _extract_smiles(claim_text) or context.get("smiles")
        if not smiles:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No parseable SMILES found in claim.",
                source_db="retrosynthesis",
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.85,
                reasoning=(
                    f"SMILES {smiles!r} does not parse — impossible valences or "
                    "invalid structure."
                ),
                evidence={"smiles": smiles},
                source_db="retrosynthesis",
            )

        sa = _sa_score(mol)
        evidence: dict[str, Any] = {"smiles": smiles, "sa": sa}
        score = sa["score"]

        says_easy = bool(_EASY_RE.search(claim_text))
        says_hard = bool(_HARD_RE.search(claim_text))

        if says_easy and score > 6.0:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.82,
                reasoning=(
                    f"Claim calls the molecule easily synthesized but SA-proxy "
                    f"score is {score} (>6 = difficult)."
                ),
                evidence=evidence,
                source_db="retrosynthesis",
            )
        if says_hard and score < 3.0:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.55,
                reasoning=(
                    f"Claim calls the molecule challenging to synthesize but "
                    f"SA-proxy score is only {score} (<3 = simple)."
                ),
                evidence=evidence,
                source_db="retrosynthesis",
            )
        if says_easy and score <= 6.0:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.7,
                reasoning=(
                    f"Claim of easy synthesis consistent with SA-proxy score {score}."
                ),
                evidence=evidence,
                source_db="retrosynthesis",
            )
        if says_hard and score >= 3.0:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.7,
                reasoning=(
                    f"Claim of challenging synthesis consistent with SA-proxy score {score}."
                ),
                evidence=evidence,
                source_db="retrosynthesis",
            )
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.4,
            reasoning=(
                f"SA-proxy score computed ({score}) but claim contains no "
                "synthesis-difficulty language to verify."
            ),
            evidence=evidence,
            source_db="retrosynthesis",
        )

    async def health_check(self) -> bool:
        return _RDKIT_AVAILABLE
