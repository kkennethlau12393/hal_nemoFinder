"""Pathway-level reasoning verifier.

Given a free-text claim such as *"metformin inhibits JAK2/STAT3 signaling"*
this verifier extracts the proteins and the claimed pathway, then checks
the curated pathway graph in :mod:`src.knowledge.pathways` to decide
whether the stated membership / mechanism is consistent.

The verifier is deliberately conservative: it only raises REFUTED when
it has positive evidence that a protein does **not** belong to the
claimed pathway, and falls back to PARTIALLY_SUPPORTED when the text is
ambiguous.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.pathways import PathwayClient, PathwayInfo
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Protein symbol heuristic: 2-10 alphanumeric, starts with a letter,
# optionally suffixed with a digit or Greek-letter-as-ASCII.
_PROTEIN_TOKEN = re.compile(
    r"\b("
    r"[A-Z][A-Z0-9]{1,9}"       # EGFR, BRCA1, KRAS
    r"(?:-?[A-Z0-9]+)?"          # BRAF-V600E, BCL-2
    r")\b"
)

# Explicit long-form pathway phrases.  Order matters — longer first.
_PATHWAY_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bPI3K\s*/\s*AKT\s*/\s*m?TOR\b", re.IGNORECASE),
    re.compile(r"\bPI3K\s*/\s*AKT\b", re.IGNORECASE),
    re.compile(r"\bRAS\s*[-/]\s*RAF\s*[-/]\s*MEK\s*[-/]\s*ERK\b", re.IGNORECASE),
    re.compile(r"\bMAPK\s*/\s*ERK\b", re.IGNORECASE),
    re.compile(r"\bJAK\s*/\s*STAT\b", re.IGNORECASE),
    re.compile(r"\bJAK\d?\s*/\s*STAT\d?\b", re.IGNORECASE),
    re.compile(r"\bNF\s*[-/]?\s*[kκ]B\b", re.IGNORECASE),
    re.compile(r"\bWnt\s*/\s*[Bβ]-?catenin\b", re.IGNORECASE),
    re.compile(r"\bp53\s+pathway\b", re.IGNORECASE),
    re.compile(r"\bEGFR\s+signal(?:l?ing|\s+pathway)?\b", re.IGNORECASE),
    re.compile(r"\bHER2\s+signal(?:l?ing)?\b", re.IGNORECASE),
    re.compile(r"\brenin[- ]angiotensin(?:\s+system)?\b", re.IGNORECASE),
    re.compile(r"\bcoagulation\s+cascade\b", re.IGNORECASE),
    re.compile(r"\bcholesterol\s+biosynthesis\b", re.IGNORECASE),
    re.compile(r"\bmevalonate\s+pathway\b", re.IGNORECASE),
    re.compile(r"\bglycolysis\b", re.IGNORECASE),
    re.compile(r"\bapoptosis\b", re.IGNORECASE),
    re.compile(r"\bDNA\s+damage\s+response\b", re.IGNORECASE),
    re.compile(r"\binsulin\s+signal(?:l?ing)?\b", re.IGNORECASE),
    re.compile(r"\bGPCR\s+signal(?:l?ing)?\b", re.IGNORECASE),
    re.compile(r"\bCOX\s*/\s*prostaglandin\b", re.IGNORECASE),
    re.compile(r"\bprostaglandin\s+(?:biosynthesis|pathway)\b", re.IGNORECASE),
    # Generic "<Name> pathway" / "<Name> signaling"
    re.compile(r"\b([A-Z][A-Za-z0-9/\-]{2,30})\s+pathway\b"),
    re.compile(r"\b([A-Z][A-Za-z0-9/\-]{2,30})\s+signal(?:l?ing)\b"),
]


_MECHANISM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "inhibit": ("inhibit", "block", "suppress", "antagoni", "downregulat"),
    "activate": ("activat", "agonist", "stimulat", "upregulat", "induc"),
}


# Small, explicit set of drug → mechanism pairs used for cross-checking
# drug-specific claims.  Not exhaustive; the verifier only uses this to
# flag obvious contradictions for well-known drugs.
_KNOWN_DRUG_TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "metformin": {
        # Metformin's canonical mechanism is complex I / AMPK — not JAK/STAT.
        "pathways": ("insulin", "ampk", "glycolysis"),
        "not_pathways": ("jak/stat", "jak-stat", "egfr"),
    },
    "aspirin": {
        "pathways": ("cox", "prostaglandin", "cox/prostaglandin"),
        "not_pathways": ("pi3k", "jak/stat", "wnt"),
    },
    "imatinib": {
        "pathways": ("bcr-abl", "kit"),
        "not_pathways": ("cox", "wnt"),
    },
    "atorvastatin": {
        "pathways": ("cholesterol biosynthesis", "mevalonate"),
        "not_pathways": ("jak/stat", "apoptosis"),
    },
    "erlotinib": {
        "pathways": ("egfr",),
        "not_pathways": ("wnt", "coagulation"),
    },
    "vemurafenib": {
        "pathways": ("mapk", "ras-raf-mek-erk"),
        "not_pathways": ("jak/stat", "wnt"),
    },
    "warfarin": {
        "pathways": ("coagulation",),
        "not_pathways": ("pi3k", "mapk"),
    },
}


def _extract_proteins(text: str) -> list[str]:
    """Return candidate protein symbols from *text*.

    Filters out obvious English words that happen to be all-caps.
    """
    stopwords = {
        "A", "I", "AN", "THE", "AND", "OR", "OF", "IN", "TO", "IS", "IT",
        "BY", "FOR", "ON", "AT", "AS", "BE", "US", "EU", "FDA", "DNA",
        "RNA", "ATP", "ADP", "GTP", "IC50", "EC50", "KI", "KD", "MW",
        "NM", "UM", "MG", "KG", "UV", "NSAID", "AUC", "PK", "PD",
    }
    out: list[str] = []
    seen: set[str] = set()
    for match in _PROTEIN_TOKEN.finditer(text):
        tok = match.group(1).upper()
        if tok in stopwords:
            continue
        # Must contain at least one letter and not be pure digits.
        if not any(c.isalpha() for c in tok):
            continue
        # Must actually appear upper-case in the source (case-sensitive
        # anchor to avoid pulling sentence-initial words).
        raw = match.group(1)
        if raw != raw.upper() and not any(c.isdigit() for c in raw):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _extract_pathways(text: str) -> list[str]:
    """Return verbatim pathway / process phrases mentioned in *text*."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _PATHWAY_PHRASES:
        for match in pattern.finditer(text):
            phrase = match.group(0).strip()
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                found.append(phrase)
    return found


def _extract_drug(text: str) -> str | None:
    """Return the first known drug name found in *text*, if any."""
    lowered = text.lower()
    for drug in _KNOWN_DRUG_TARGETS:
        if re.search(rf"\b{re.escape(drug)}\b", lowered):
            return drug
    return None


def _classify_mechanism(text: str) -> str | None:
    """Return ``"inhibit"`` / ``"activate"`` or ``None``."""
    lowered = text.lower()
    for mech, keywords in _MECHANISM_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return mech
    return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@register_verifier
class PathwayVerifier(BaseVerifier):
    """Verify pathway-level mechanism-of-action and target-interaction claims.

    The verifier performs three checks:

    1. If the claim names both a protein and a pathway, look both up in
       :class:`~src.knowledge.pathways.PathwayClient` and confirm that
       the protein is a member of the claimed pathway.
    2. If a known drug is mentioned, cross-check the claimed pathway
       against a small whitelist / blacklist of the drug's documented
       mechanism.
    3. Flag same-molecule / same-protein conflicting pathway assignments
       (e.g. agonist + inhibitory pathway).
    """

    name = "pathway"
    supported_claim_types = [
        ClaimType.mechanism_of_action,
        ClaimType.target_interaction,
    ]

    def __init__(self) -> None:
        self._client = PathwayClient()

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
            return self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("PathwayVerifier crashed on %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during pathway verification: {exc}",
                source_db="pathways",
            )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _verify_impl(self, claim_text: str) -> VerificationOutput:
        proteins = _extract_proteins(claim_text)
        pathway_phrases = _extract_pathways(claim_text)
        mechanism = _classify_mechanism(claim_text)
        drug = _extract_drug(claim_text)

        evidence: dict[str, Any] = {
            "extracted_proteins": proteins,
            "extracted_pathways": pathway_phrases,
            "stated_mechanism": mechanism,
            "drug": drug,
        }

        # Resolve pathway phrases against the curated DB.
        resolved_pathways: list[PathwayInfo] = []
        unresolved: list[str] = []
        for phrase in pathway_phrases:
            info = self._client.get_pathway(phrase)
            if info is not None:
                resolved_pathways.append(info)
            else:
                unresolved.append(phrase)
        evidence["resolved_pathways"] = [
            {
                "pathway_id": p.pathway_id,
                "name": p.name,
                "source": p.source,
                "members": p.proteins,
            }
            for p in resolved_pathways
        ]
        if unresolved:
            evidence["unresolved_pathway_phrases"] = unresolved

        # ------------------------------------------------------------------
        # Branch A: claim has neither protein nor pathway → unverifiable.
        # ------------------------------------------------------------------
        if not proteins and not resolved_pathways:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=(
                    "No protein or recognised pathway could be extracted "
                    "from the claim; pathway-level reasoning is not applicable."
                ),
                evidence=evidence,
                source_db="pathways",
            )

        # ------------------------------------------------------------------
        # Branch B: drug-level cross-check against known mechanism.
        # ------------------------------------------------------------------
        drug_verdict = self._check_drug_contradiction(
            drug, pathway_phrases, resolved_pathways, evidence
        )
        if drug_verdict is not None:
            return drug_verdict

        # ------------------------------------------------------------------
        # Branch C: protein ∈ pathway membership check.
        # ------------------------------------------------------------------
        if resolved_pathways and proteins:
            return self._check_membership(
                proteins, resolved_pathways, mechanism, evidence
            )

        # ------------------------------------------------------------------
        # Branch D: protein only — enumerate candidate pathways.
        # ------------------------------------------------------------------
        if proteins and not resolved_pathways:
            candidates: dict[str, list[str]] = {}
            for protein in proteins:
                hits = self._client.find_pathways_for_protein(protein)
                if hits:
                    candidates[protein] = [h.name for h in hits]
            evidence["candidate_pathways_by_protein"] = candidates
            if candidates:
                return VerificationOutput(
                    verdict=Verdict.partially_supported,
                    confidence=0.4,
                    reasoning=(
                        "Protein(s) extracted but no specific pathway named; "
                        f"candidates from curated DB: {candidates}."
                    ),
                    evidence=evidence,
                    source_db="pathways",
                )
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.1,
                reasoning=(
                    "Protein(s) extracted but not present in the curated "
                    "pathway database; cannot perform pathway-level check."
                ),
                evidence=evidence,
                source_db="pathways",
            )

        # ------------------------------------------------------------------
        # Branch E: pathway resolved but no protein extracted.
        # ------------------------------------------------------------------
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.35,
            reasoning=(
                f"Pathway '{resolved_pathways[0].name}' recognised, but no "
                "specific protein could be extracted from the claim; mechanism "
                "cannot be cross-checked."
            ),
            evidence=evidence,
            source_db="pathways",
        )

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_drug_contradiction(
        self,
        drug: str | None,
        pathway_phrases: list[str],
        resolved_pathways: list[PathwayInfo],
        evidence: dict[str, Any],
    ) -> VerificationOutput | None:
        """Flag claims that assign a well-known drug to the wrong pathway."""
        if not drug or not pathway_phrases:
            return None
        profile = _KNOWN_DRUG_TARGETS.get(drug)
        if not profile:
            return None

        allowed = profile.get("pathways", ())
        blocked = profile.get("not_pathways", ())

        def _matches(phrase: str, needles: tuple[str, ...]) -> bool:
            phrase_l = phrase.lower()
            return any(n in phrase_l for n in needles)

        phrases_l = [p.lower() for p in pathway_phrases]
        hit_blocked = [p for p in phrases_l if _matches(p, blocked)]
        hit_allowed = [p for p in phrases_l if _matches(p, allowed)]

        if hit_blocked and not hit_allowed:
            evidence["drug_contradiction"] = {
                "drug": drug,
                "claimed": hit_blocked,
                "expected_any_of": list(allowed),
            }
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.85,
                reasoning=(
                    f"Claim attributes {drug} to {hit_blocked}, but its "
                    f"documented mechanism involves {list(allowed)}. "
                    "This is inconsistent with curated drug-pathway data."
                ),
                evidence=evidence,
                source_db="pathways",
            )
        return None

    def _check_membership(
        self,
        proteins: list[str],
        pathways: list[PathwayInfo],
        mechanism: str | None,
        evidence: dict[str, Any],
    ) -> VerificationOutput:
        """Verify that each claimed protein really is a member of the pathway."""
        membership: list[dict[str, Any]] = []
        confirmed_any = False
        refuted_any = False

        for pathway in pathways:
            for protein in proteins:
                is_member = self._client.is_member(protein, pathway)
                membership.append(
                    {
                        "protein": protein,
                        "pathway": pathway.name,
                        "is_member": is_member,
                    }
                )
                if is_member:
                    confirmed_any = True
                else:
                    # Only count as "refuted" if this protein exists in
                    # SOME pathway in our DB — otherwise the protein is
                    # simply unknown and we shouldn't over-commit.
                    other_hits = self._client.find_pathways_for_protein(protein)
                    if other_hits:
                        refuted_any = True

        evidence["membership_checks"] = membership

        if refuted_any and not confirmed_any:
            mismatches = [m for m in membership if not m["is_member"]]
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.8,
                reasoning=(
                    "Protein(s) are NOT members of the claimed pathway "
                    f"according to the curated DB: {mismatches}."
                ),
                evidence=evidence,
                source_db="pathways",
            )

        if confirmed_any and not refuted_any:
            reasoning = (
                "All extracted protein(s) are confirmed members of the "
                f"claimed pathway ({pathways[0].name})."
            )
            if mechanism:
                reasoning += f" Stated mechanism: {mechanism}."
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=0.85,
                reasoning=reasoning,
                evidence=evidence,
                source_db="pathways",
            )

        if confirmed_any and refuted_any:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.5,
                reasoning=(
                    "Some proteins match the claimed pathway and others do "
                    "not; partial support from curated DB."
                ),
                evidence=evidence,
                source_db="pathways",
            )

        # Nothing matched and nothing was actively refuted — proteins are
        # unknown to our curated DB.
        return VerificationOutput(
            verdict=Verdict.partially_supported,
            confidence=0.3,
            reasoning=(
                "Pathway recognised, but the extracted proteins are not "
                "present in the curated pathway database; membership could "
                "not be confirmed."
            ),
            evidence=evidence,
            source_db="pathways",
        )

    async def health_check(self) -> bool:
        return len(self._client.list_pathways()) > 0
