"""Literature-retrieval verifier — checks free-text biomedical claims against PubMed.

The verifier extracts "unanchored" claims (phrases like "was shown", "has
been reported", "evidence suggests") and searches PubMed (or its curated
offline fallback) for papers mentioning the same compound / target /
mechanism.  The top abstracts are then compared to the claim using
:class:`~src.knowledge.embeddings.EmbeddingService` and a deterministic
token-overlap fallback, producing a verdict driven by similarity.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.knowledge.embeddings import EmbeddingService
from src.knowledge.pubmed import PubMedClient, PubMedRecord
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


_UNANCHORED_RE = re.compile(
    r"\b(was shown|has been reported|evidence suggests|it is known|"
    r"recent studies|studies (?:have|demonstrate)|demonstrated|reported)\b",
    re.IGNORECASE,
)

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "that", "is",
    "was", "it", "with", "by", "on", "at", "as", "be", "for", "this",
    "has", "been", "shown", "evidence", "suggests", "reported",
    "studies", "study", "recent", "known",
})


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower()) if t not in _STOPWORDS]


def _token_overlap(a: str, b: str) -> float:
    toks_a = set(_tokens(a))
    toks_b = set(_tokens(b))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


@register_verifier
class LiteratureRetrievalVerifier(BaseVerifier):
    """Verify biomedical claims by retrieving supporting PubMed abstracts."""

    name = "literature"
    supported_claim_types = [
        ClaimType.general_biomedical,
        ClaimType.mechanism_of_action,
    ]

    def __init__(self) -> None:
        self._pubmed: PubMedClient | None
        try:
            self._pubmed = PubMedClient()
        except Exception as exc:  # pragma: no cover
            logger.debug("PubMedClient unavailable: %s", exc)
            self._pubmed = None
        try:
            self._embeddings = EmbeddingService()
        except Exception:  # pragma: no cover
            self._embeddings = EmbeddingService(enabled=False)

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return await self._verify_impl(claim_text)
        except Exception as exc:  # pragma: no cover
            logger.exception("LiteratureRetrievalVerifier crashed: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during literature verification: {exc}",
                source_db="literature",
            )

    async def _verify_impl(self, claim_text: str) -> VerificationOutput:
        is_unanchored = bool(_UNANCHORED_RE.search(claim_text))
        evidence: dict[str, Any] = {"is_unanchored": is_unanchored}

        query_tokens = _tokens(claim_text)
        if not query_tokens:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No searchable tokens in claim.",
                evidence=evidence,
                source_db="literature",
            )
        query = " ".join(query_tokens[:12])
        evidence["query"] = query

        records: list[PubMedRecord] = []
        if self._pubmed is not None:
            try:
                records = await self._pubmed.search_and_fetch(query, max_results=5)
            except Exception as exc:
                logger.warning("PubMed online search failed: %s", exc)
                records = PubMedClient._fallback_records(query, 5)
        else:
            records = PubMedClient._fallback_records(query, 5)

        evidence["matched_papers"] = [
            {"pmid": r.pmid, "title": r.title, "year": r.year} for r in records
        ]

        if not records:
            return VerificationOutput(
                verdict=Verdict.partially_supported if is_unanchored else Verdict.unverifiable,
                confidence=0.3 if is_unanchored else 0.1,
                reasoning=(
                    "No PubMed evidence (live or curated) matched the claim; "
                    "claim remains unsupported."
                ),
                evidence=evidence,
                source_db="literature",
            )

        # Compute similarity of each retrieved abstract to the claim.
        scored: list[tuple[float, PubMedRecord]] = []
        for rec in records:
            text = f"{rec.title}. {rec.abstract}"
            sim = self._similarity(claim_text, text)
            scored.append((sim, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        top_sim = scored[0][0]
        evidence["top_similarity"] = round(top_sim, 4)
        evidence["top_pmid"] = scored[0][1].pmid

        if top_sim >= 0.45:
            return VerificationOutput(
                verdict=Verdict.verified,
                confidence=min(0.95, 0.55 + top_sim / 2),
                reasoning=(
                    f"Top PubMed match PMID {scored[0][1].pmid} "
                    f"('{scored[0][1].title}') has similarity {top_sim:.2f} with the claim."
                ),
                evidence=evidence,
                source_db="literature",
            )
        if top_sim >= 0.2:
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=0.45,
                reasoning=(
                    f"PubMed matches found but top similarity is only {top_sim:.2f}; "
                    "claim partially supported."
                ),
                evidence=evidence,
                source_db="literature",
            )
        return VerificationOutput(
            verdict=Verdict.refuted if is_unanchored else Verdict.partially_supported,
            confidence=0.55 if is_unanchored else 0.35,
            reasoning=(
                f"No high-similarity PubMed match (best={top_sim:.2f}); "
                "unanchored claim appears unsupported by the literature."
            ),
            evidence=evidence,
            source_db="literature",
        )

    def _similarity(self, a: str, b: str) -> float:
        try:
            sim = float(self._embeddings.similarity(a, b))
            if sim != 0.5:  # 0.5 is the "disabled" sentinel
                return max(0.0, min(1.0, sim))
        except Exception as exc:  # pragma: no cover
            logger.debug("Embedding similarity failed: %s", exc)
        return _token_overlap(a, b)

    async def health_check(self) -> bool:
        return True
