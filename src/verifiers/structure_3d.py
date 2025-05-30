"""3D structure (PDB) verifier — validates claims that cite a PDB entry.

Extracts PDB ids from the claim and cross-checks them against the RCSB
PDB (via :class:`~src.knowledge.pdb.PDBClient`) or the curated offline
fallback.  Flags:

* non-existent PDB ids → refuted
* claims that cite "ligand Y binds at X in PDB Z" when Z has no ligands
* claims that reference a protein not present in the entry
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.pdb import PDBClient, PDBEntry
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


_PDB_ID_RE = re.compile(r"\b([1-9][A-Za-z0-9]{3})\b")

_STRUCTURAL_CONTEXT_RE = re.compile(
    r"\b(pdb|crystal structure|structure of|x-ray|bound to|in complex with|binds at)\b",
    re.IGNORECASE,
)


def _extract_pdb_ids(text: str) -> list[str]:
    if not _STRUCTURAL_CONTEXT_RE.search(text):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _PDB_ID_RE.finditer(text):
        pid = m.group(1).upper()
        # Must contain at least one digit and one letter.
        if pid.isdigit() or pid.isalpha():
            continue
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


@register_verifier
class Structure3DVerifier(BaseVerifier):
    """Verify PDB-citation and ligand-binding claims."""

    name = "structure_3d"
    supported_claim_types = [
        ClaimType.target_interaction,
        ClaimType.mechanism_of_action,
    ]

    def __init__(self) -> None:
        try:
            self._client: PDBClient | None = PDBClient()
        except Exception as exc:  # pragma: no cover
            logger.debug("PDBClient unavailable: %s", exc)
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
            logger.exception("Structure3DVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during PDB verification: {exc}",
                source_db="pdb",
            )

    async def _verify_impl(self, claim_text: str) -> VerificationOutput:
        ids = _extract_pdb_ids(claim_text)
        evidence: dict[str, Any] = {"pdb_ids": ids}
        if not ids:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No PDB identifier extracted from claim.",
                evidence=evidence,
                source_db="pdb",
            )

        entries: list[tuple[str, PDBEntry | None]] = []
        for pid in ids:
            entry = await self._fetch(pid)
            entries.append((pid, entry))
        evidence["entries"] = [
            {
                "pdb_id": pid,
                "found": e is not None,
                "title": e.title if e else None,
                "ligands": list(e.ligands) if e else [],
            }
            for pid, e in entries
        ]

        missing = [pid for pid, e in entries if e is None]
        if missing:
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.9,
                reasoning=(
                    f"PDB id(s) {missing} could not be resolved in RCSB or the "
                    "curated fallback; likely fabricated."
                ),
                evidence=evidence,
                source_db="pdb",
            )

        lowered = claim_text.lower()
        mentions_ligand = any(
            kw in lowered
            for kw in (
                "bound to", "in complex with", "binds at", "binds to",
                "ligand", "inhibitor",
            )
        )

        for pid, entry in entries:
            assert entry is not None
            if mentions_ligand and not entry.ligands:
                return VerificationOutput(
                    verdict=Verdict.refuted,
                    confidence=0.85,
                    reasoning=(
                        f"Claim references a ligand in PDB {pid}, but the entry "
                        f"'{entry.title}' has no deposited ligand records."
                    ),
                    evidence=evidence,
                    source_db="pdb",
                )
            # Check for protein-name mismatch with simple keyword test.
            for name in entry.protein_names:
                if name.split()[0].lower() in lowered:
                    return VerificationOutput(
                        verdict=Verdict.verified,
                        confidence=0.85,
                        reasoning=(
                            f"PDB {pid} ({entry.title}) matches the protein "
                            "mentioned in the claim."
                        ),
                        evidence=evidence,
                        source_db="pdb",
                    )

        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.55,
            reasoning=(
                f"PDB id(s) {ids} exist, but no protein/ligand mention in the "
                "claim could be explicitly matched to the entry metadata."
            ),
            evidence=evidence,
            source_db="pdb",
        )

    async def _fetch(self, pdb_id: str) -> PDBEntry | None:
        if self._client is not None:
            try:
                return await self._client.get_structure(pdb_id)
            except Exception as exc:  # pragma: no cover
                logger.debug("PDB online fetch failed: %s", exc)
        return PDBClient.fallback_entry(pdb_id)

    async def health_check(self) -> bool:
        return True
