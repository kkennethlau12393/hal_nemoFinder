"""Citation verifier — validates DOIs, PMIDs, and claim-paper relevance."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b(10\.\d{4,}/\S+)")
_PMID_RE = re.compile(r"PMID[:\s]*(\d+)", re.IGNORECASE)

# Reasonable PMID range (PubMed started in 1966; IDs are sequential)
_PMID_MAX = 100_000_000  # generous upper bound

# Similarity thresholds
_SIM_HIGH = 0.6
_SIM_LOW = 0.3


def _clean_doi(raw: str) -> str:
    """Strip trailing punctuation that is not part of the DOI."""
    return raw.rstrip(".,;:)>]}")


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@register_verifier
class CitationVerifier(BaseVerifier):
    """Verify citation claims by validating DOIs/PMIDs and checking content relevance."""

    name = "citation"
    supported_claim_types = [ClaimType.citation]

    def __init__(self) -> None:
        self._crossref: Any = None
        self._embedding: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from src.knowledge.crossref import CrossRefClient  # type: ignore[import-untyped]
            self._crossref = CrossRefClient()
        except Exception:
            logger.debug("CrossRefClient unavailable.")

        try:
            from src.knowledge.embeddings import EmbeddingService  # type: ignore[import-untyped]
            self._embedding = EmbeddingService()
        except Exception:
            logger.debug("EmbeddingService unavailable.")

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
            logger.exception("CitationVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during citation verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        dois = [_clean_doi(m.group(1)) for m in _DOI_RE.finditer(claim_text)]
        pmids = [m.group(1) for m in _PMID_RE.finditer(claim_text)]

        evidence: dict[str, Any] = {
            "extracted_dois": dois,
            "extracted_pmids": pmids,
        }

        if not dois and not pmids:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No DOI or PMID found in the claim text.",
                evidence=evidence,
            )

        # --- Validate PMIDs (format only) ---
        valid_pmids: list[str] = []
        invalid_pmids: list[str] = []
        for pmid in pmids:
            if pmid.isdigit() and 0 < int(pmid) < _PMID_MAX:
                valid_pmids.append(pmid)
            else:
                invalid_pmids.append(pmid)

        evidence["valid_pmids"] = valid_pmids
        evidence["invalid_pmids"] = invalid_pmids

        # --- Validate DOIs via CrossRef ---
        doi_results: list[dict[str, Any]] = []
        invalid_dois: list[str] = []

        for doi in dois:
            result = await self._validate_doi(doi)
            if result is not None:
                doi_results.append(result)
            else:
                invalid_dois.append(doi)

        evidence["doi_results"] = doi_results
        evidence["invalid_dois"] = invalid_dois

        # --- Semantic similarity check ---
        similarity_scores: list[dict[str, Any]] = []
        for result in doi_results:
            score = await self._check_relevance(claim_text, result)
            if score is not None:
                similarity_scores.append({
                    "doi": result.get("doi", ""),
                    "title": result.get("title", ""),
                    "similarity": round(score, 4),
                })

        evidence["similarity_scores"] = similarity_scores

        # --- Determine verdict ---
        return self._resolve_verdict(
            dois=dois,
            doi_results=doi_results,
            invalid_dois=invalid_dois,
            valid_pmids=valid_pmids,
            invalid_pmids=invalid_pmids,
            similarity_scores=similarity_scores,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _validate_doi(self, doi: str) -> dict[str, Any] | None:
        """Validate a DOI via CrossRef.

        Returns metadata dict if found, empty dict ``{"doi": doi, "unchecked": True}``
        if CrossRef is unavailable, or ``None`` if definitively not found.
        """
        if self._crossref is None:
            # Cannot reach CrossRef — DOI is unchecked, not invalid.
            return {"doi": doi, "unchecked": True}
        try:
            result = await self._crossref.validate_doi(doi)
            if result:
                return result if isinstance(result, dict) else {"doi": doi, "valid": True}
        except Exception as exc:
            logger.debug("CrossRef validation failed for DOI %r: %s", doi, exc)
            return {"doi": doi, "unchecked": True}
        return None

    async def _check_relevance(
        self,
        claim_text: str,
        paper_metadata: dict[str, Any],
    ) -> float | None:
        """Compute semantic similarity between claim and paper title/abstract."""
        if self._embedding is None:
            return None

        # Build comparison text from paper metadata
        parts: list[str] = []
        title = paper_metadata.get("title", "")
        if isinstance(title, list):
            title = " ".join(str(t) for t in title)
        if title:
            parts.append(str(title))

        abstract = paper_metadata.get("abstract", "")
        if abstract:
            parts.append(str(abstract))

        if not parts:
            return None

        paper_text = " ".join(parts)
        try:
            score = await self._embedding.similarity(claim_text, paper_text)
            return float(score) if score is not None else None
        except Exception as exc:
            logger.debug("Embedding similarity failed: %s", exc)
            return None

    @staticmethod
    def _resolve_verdict(
        *,
        dois: list[str],
        doi_results: list[dict[str, Any]],
        invalid_dois: list[str],
        valid_pmids: list[str],
        invalid_pmids: list[str],
        similarity_scores: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> VerificationOutput:
        """Combine validation signals into a final verdict."""
        reasons: list[str] = []
        total_ids = len(dois) + len(valid_pmids) + len(invalid_pmids)
        valid_count = len(doi_results) + len(valid_pmids)
        invalid_count = len(invalid_dois) + len(invalid_pmids)

        if doi_results:
            titles = [r.get("title", "?") for r in doi_results]
            reasons.append(f"DOI(s) validated via CrossRef: {', '.join(str(t) for t in titles)}.")
        if invalid_dois:
            reasons.append(f"DOI(s) could not be validated: {', '.join(invalid_dois)}.")
        if valid_pmids:
            reasons.append(f"PMID(s) have valid format: {', '.join(valid_pmids)}.")
        if invalid_pmids:
            reasons.append(f"PMID(s) have invalid format: {', '.join(invalid_pmids)}.")

        # Similarity verdicts
        high_sim = [s for s in similarity_scores if s["similarity"] >= _SIM_HIGH]
        low_sim = [s for s in similarity_scores if s["similarity"] < _SIM_LOW]

        if high_sim:
            reasons.append(
                "Claim content is semantically relevant to cited paper(s) "
                f"(similarity >= {_SIM_HIGH})."
            )
        if low_sim:
            reasons.append(
                "Claim content appears unrelated to some cited paper(s) "
                f"(similarity < {_SIM_LOW}) — possible hallucinated attribution."
            )

        # Determine verdict
        if invalid_count > 0 and valid_count == 0:
            verdict = Verdict.refuted
            confidence = 0.85
        elif valid_count > 0 and high_sim:
            verdict = Verdict.verified
            confidence = min(0.95, 0.7 + 0.1 * len(high_sim))
        elif valid_count > 0 and low_sim and not high_sim:
            # Exists but content doesn't align
            verdict = Verdict.partially_supported
            confidence = 0.6
        elif valid_count > 0:
            verdict = Verdict.partially_supported
            confidence = 0.5
        else:
            verdict = Verdict.unverifiable
            confidence = 0.2

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=" ".join(reasons),
            evidence=evidence,
            source_db="crossref",
        )

    async def health_check(self) -> bool:
        return self._crossref is not None
