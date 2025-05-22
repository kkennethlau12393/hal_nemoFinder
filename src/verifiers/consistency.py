"""Consistency verifier — detects contradictions across claims in the same job."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

_MW_RE = re.compile(
    r"(?:molecular\s+weight|MW)\s*(?:of|=|:|\s)\s*([\d.]+)",
    re.IGNORECASE,
)
_PHASE_RE = re.compile(r"phase\s*(\d|[IViv]{1,3})\b", re.IGNORECASE)
_NCT_RE = re.compile(r"\b(NCT\d{8})\b")
_SMILES_RE = re.compile(r"SMILES[:\s]*([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%&.~]+)")

_TARGET_ACTION_RE = re.compile(
    r"(agonist|antagonist|inhibitor|activator|blocker|modulator)\s+(?:of\s+|for\s+)?(\S+)",
    re.IGNORECASE,
)

_MOLECULE_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{2,}(?:ib|ab|mab|nib|lib|tinib|zumab|ximab|umab))\b"
)

_ROMAN_TO_INT: dict[str, str] = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4",
}


def _extract_molecule_names(text: str) -> set[str]:
    """Extract likely drug/molecule names from *text*."""
    names: set[str] = set()
    for m in _MOLECULE_NAME_RE.finditer(text):
        names.add(m.group(1).lower())
    # Also look for quoted or capitalised multi-word names
    for m in re.finditer(r'"([^"]+)"', text):
        names.add(m.group(1).lower())
    return names


def _extract_mw(text: str) -> float | None:
    m = _MW_RE.search(text)
    return float(m.group(1)) if m else None


def _extract_phase(text: str) -> str | None:
    m = _PHASE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    return _ROMAN_TO_INT.get(raw, raw)


def _extract_nct(text: str) -> str | None:
    m = _NCT_RE.search(text)
    return m.group(1) if m else None


def _extract_target_actions(text: str) -> list[tuple[str, str]]:
    """Return list of (action, target) pairs."""
    return [
        (m.group(1).lower(), m.group(2).lower())
        for m in _TARGET_ACTION_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def _find_contradictions(
    claim_text: str,
    other_claims: list[str],
) -> list[dict[str, Any]]:
    """Find contradictions between *claim_text* and *other_claims*.

    Returns a list of contradiction dicts.
    """
    contradictions: list[dict[str, Any]] = []
    my_molecules = _extract_molecule_names(claim_text)
    my_mw = _extract_mw(claim_text)
    my_phase = _extract_phase(claim_text)
    my_nct = _extract_nct(claim_text)
    my_actions = _extract_target_actions(claim_text)

    for other in other_claims:
        if other == claim_text:
            continue

        other_molecules = _extract_molecule_names(other)
        shared_molecules = my_molecules & other_molecules

        if not shared_molecules:
            continue

        molecule_label = ", ".join(sorted(shared_molecules))

        # MW contradiction
        if my_mw is not None:
            other_mw = _extract_mw(other)
            if other_mw is not None and abs(my_mw - other_mw) > max(my_mw * 0.01, 1.0):
                contradictions.append({
                    "type": "mw_conflict",
                    "molecule": molecule_label,
                    "this_claim_mw": my_mw,
                    "other_claim_mw": other_mw,
                    "other_claim": other[:200],
                })

        # Phase contradiction (same trial, different phase)
        if my_nct is not None and my_phase is not None:
            other_nct = _extract_nct(other)
            other_phase = _extract_phase(other)
            if other_nct == my_nct and other_phase is not None and other_phase != my_phase:
                contradictions.append({
                    "type": "phase_conflict",
                    "nct_id": my_nct,
                    "this_claim_phase": my_phase,
                    "other_claim_phase": other_phase,
                    "other_claim": other[:200],
                })

        # Action contradiction (same target, contradictory actions)
        for my_action, my_target in my_actions:
            for other_action, other_target in _extract_target_actions(other):
                if my_target != other_target:
                    continue
                if _actions_contradictory(my_action, other_action):
                    contradictions.append({
                        "type": "action_conflict",
                        "target": my_target,
                        "this_claim_action": my_action,
                        "other_claim_action": other_action,
                        "other_claim": other[:200],
                    })

    return contradictions


_CONTRADICTORY_PAIRS: set[frozenset[str]] = {
    frozenset({"agonist", "antagonist"}),
    frozenset({"activator", "inhibitor"}),
    frozenset({"activator", "blocker"}),
    frozenset({"agonist", "blocker"}),
    frozenset({"agonist", "inhibitor"}),
}


def _actions_contradictory(a: str, b: str) -> bool:
    if a == b:
        return False
    return frozenset({a, b}) in _CONTRADICTORY_PAIRS


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@register_verifier
class ConsistencyVerifier(BaseVerifier):
    """Cross-claim consistency verifier — detects contradictions within a job."""

    name = "consistency"
    # Accepts ALL claim types
    supported_claim_types = list(ClaimType)

    def __init__(self) -> None:
        self._embedding: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from src.knowledge.embeddings import EmbeddingService  # type: ignore[import-untyped]
            self._embedding = EmbeddingService()
        except Exception:
            logger.debug("EmbeddingService unavailable for consistency checks.")

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
            logger.exception("ConsistencyVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during consistency verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        all_claims: list[str] = context.get("all_claims", [])
        evidence: dict[str, Any] = {
            "total_claims_in_job": len(all_claims),
        }

        if len(all_claims) < 2:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="Only one claim in job; no cross-claim consistency check possible.",
                evidence=evidence,
                source_db="consistency",
            )

        # Step 1: Heuristic contradiction detection
        contradictions = _find_contradictions(claim_text, all_claims)
        evidence["heuristic_contradictions"] = contradictions

        # Step 2: Semantic similarity-based detection (find highly similar
        # claims that might be making different assertions)
        semantic_flags = await self._semantic_check(claim_text, all_claims)
        if semantic_flags:
            evidence["semantic_flags"] = semantic_flags

        # Combine
        total_issues = len(contradictions) + len(semantic_flags)
        clear_contradictions = len(contradictions)

        if clear_contradictions > 0:
            verdict = Verdict.refuted
            confidence = min(0.95, 0.7 + 0.1 * clear_contradictions)
            reasons = [
                f"Found {clear_contradictions} contradiction(s) with other claims in this job."
            ]
            for c in contradictions[:3]:  # summarise top 3
                ctype = c.get("type", "unknown")
                if ctype == "mw_conflict":
                    reasons.append(
                        f"MW conflict for {c['molecule']}: "
                        f"{c['this_claim_mw']} vs {c['other_claim_mw']}."
                    )
                elif ctype == "phase_conflict":
                    reasons.append(
                        f"Phase conflict for {c['nct_id']}: "
                        f"phase {c['this_claim_phase']} vs {c['other_claim_phase']}."
                    )
                elif ctype == "action_conflict":
                    reasons.append(
                        f"Action conflict on {c['target']}: "
                        f"{c['this_claim_action']} vs {c['other_claim_action']}."
                    )
        elif semantic_flags:
            verdict = Verdict.partially_supported
            confidence = 0.5
            reasons = [
                f"No clear contradictions found, but {len(semantic_flags)} "
                "semantically similar claim(s) may warrant manual review."
            ]
        else:
            verdict = Verdict.verified
            confidence = min(0.8, 0.5 + 0.05 * len(all_claims))
            reasons = [
                f"No contradictions detected across {len(all_claims)} claims in this job."
            ]

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=" ".join(reasons),
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _semantic_check(
        self,
        claim_text: str,
        all_claims: list[str],
    ) -> list[dict[str, Any]]:
        """Use embedding similarity to flag potentially related claims that
        might contain subtle contradictions.
        """
        if self._embedding is None:
            return []

        flags: list[dict[str, Any]] = []

        for other in all_claims:
            if other == claim_text:
                continue
            try:
                sim = await self._embedding.similarity(claim_text, other)
                if sim is None:
                    continue
                sim = float(sim)
                # High similarity (> 0.8) between distinct claims suggests
                # they're about the same thing — worth checking for conflicts.
                if sim > 0.8:
                    flags.append({
                        "other_claim": other[:200],
                        "similarity": round(sim, 4),
                        "note": "Very similar claim — verify consistency manually.",
                    })
            except Exception as exc:
                logger.debug("Embedding similarity check failed: %s", exc)

        return flags

    async def health_check(self) -> bool:
        return True  # heuristic checks always work
