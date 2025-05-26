"""LLM-backed claim extraction (Tier 4 — pluggable, currently a stub).

This module exposes :class:`LLMClaimExtractor`, a drop-in replacement for the
deterministic :class:`~src.core.claim_extractor.ClaimExtractor`.  It is designed
so that a real LLM (Claude, GPT-4, a local model, etc.) can be plugged in via
the :class:`LLMBackend` protocol — but the framework ships a deterministic
"stub" backend that delegates to the rule-based extractor.

Why a stub?

The whole point of hal_nemoFinder is that **verdicts** are grounded in
deterministic computation or authoritative databases, not in another LLM's
opinion.  However, the *initial decomposition* of free-text input into
atomic claims is a legitimate use case for an LLM, since the input is
unstructured and an LLM can pick up on implicit assertions that pure
regex misses.  We therefore expose this as an optional, pluggable layer
that defaults to the rule-based extractor — so the system has zero LLM
dependency out of the box, but users who *want* better extraction
quality can wire in their preferred backend.

Example
-------
>>> # Default: deterministic stub (no LLM, no network)
>>> extractor = LLMClaimExtractor()
>>> claims = extractor.extract("Aspirin has MW 180.16 Da.")

>>> # With a custom backend (pseudocode)
>>> class ClaudeBackend(LLMBackend):
...     async def extract_claims(self, text: str) -> list[ExtractedClaim]:
...         response = await anthropic.messages.create(...)
...         return parse_claims(response.content)
>>> extractor = LLMClaimExtractor(backend=ClaudeBackend())
"""

from __future__ import annotations

import logging
from typing import Protocol

from src.core.claim_extractor import ClaimExtractor, ExtractedClaim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Protocol for LLM-backed claim extraction.

    Implementations should accept raw text and return a list of
    :class:`ExtractedClaim` objects.  The backend is free to call any
    underlying LLM service (Anthropic, OpenAI, local model, etc.) as long
    as it returns the structured claim list.

    Important constraints:

    * The backend must NOT make verification judgments.  Its job is purely
      to *decompose* the input into atomic factual claims.
    * The backend must return claims with valid source spans pointing
      back into the original text.
    * The backend must be deterministic enough to be tested (use
      temperature=0 if available).
    """

    def extract_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract atomic factual claims from *text*."""
        ...


# ---------------------------------------------------------------------------
# Stub backend (default)
# ---------------------------------------------------------------------------


class StubLLMBackend:
    """Default backend that delegates to the deterministic ``ClaimExtractor``.

    This is the safe, no-network, no-LLM-dependency default.  It exists so
    that :class:`LLMClaimExtractor` can be instantiated and used without
    any external service, while still presenting the same interface that
    a real LLM backend would expose.

    Replace this with a real backend (e.g. ``ClaudeBackend``,
    ``OpenAIBackend``) by passing an alternate implementation to
    :class:`LLMClaimExtractor`.
    """

    def __init__(self) -> None:
        self._fallback = ClaimExtractor()
        logger.debug(
            "StubLLMBackend initialised — falling back to deterministic "
            "ClaimExtractor.  Plug in a real LLM backend to upgrade."
        )

    def extract_claims(self, text: str) -> list[ExtractedClaim]:
        """Delegate to the deterministic extractor."""
        return self._fallback.extract(text)


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------


class LLMClaimExtractor:
    """Claim extractor that can be backed by an LLM (or a stub).

    Parameters
    ----------
    backend : LLMBackend | None
        The backend to use.  If ``None`` (default), uses
        :class:`StubLLMBackend` which delegates to the deterministic
        :class:`~src.core.claim_extractor.ClaimExtractor`.

    Notes
    -----
    Even when a real LLM backend is plugged in, the *verification* of
    extracted claims is still done entirely by the deterministic verifiers
    (chemical, citation, clinical, etc.).  The LLM only sees the input
    text, never the verdicts.  This preserves the framework's epistemic
    grounding.
    """

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend: LLMBackend = backend or StubLLMBackend()

    @property
    def backend_name(self) -> str:
        """Name of the active backend (for logging / health checks)."""
        return type(self._backend).__name__

    def extract(self, text: str) -> list[ExtractedClaim]:
        """Extract atomic factual claims from *text*.

        Mirrors the signature of :meth:`ClaimExtractor.extract` so that
        :class:`LLMClaimExtractor` is a drop-in replacement.

        Parameters
        ----------
        text : str
            Free-text input (typically the raw output of a drug-discovery
            LLM).

        Returns
        -------
        list[ExtractedClaim]
            One :class:`ExtractedClaim` per atomic claim found.
        """
        try:
            claims = self._backend.extract_claims(text)
            logger.debug(
                "LLMClaimExtractor(%s) returned %d claims",
                self.backend_name,
                len(claims),
            )
            return claims
        except Exception:
            logger.exception(
                "LLM backend %s crashed during extraction; "
                "falling back to deterministic extractor",
                self.backend_name,
            )
            return ClaimExtractor().extract(text)


# ---------------------------------------------------------------------------
# Convenience: example sketch of a real backend (NOT wired up by default)
# ---------------------------------------------------------------------------


class ExampleClaudeBackend:
    """Sketch of how a real Claude-backed extractor would look.

    This class is **not** instantiated automatically.  It exists as
    documentation / scaffolding for users who want to plug in a real LLM.
    To use it: install ``anthropic``, set ``ANTHROPIC_API_KEY``, and pass
    an instance to :class:`LLMClaimExtractor`.

    The implementation deliberately raises ``NotImplementedError`` to
    make it clear this is a placeholder — replace the body with a real
    Anthropic SDK call to use it.
    """

    EXTRACTION_PROMPT = """\
You are a claim extractor for a hallucination-detection system.  Given a
block of AI-generated drug-discovery text, decompose it into atomic
factual claims that can be independently verified.

Return STRICT JSON in the following shape — no commentary, no markdown:

{
  "claims": [
    {
      "text": "<the exact claim sentence>",
      "start": <character offset in input>,
      "end": <character offset>,
      "confidence": <float 0.0-1.0 — how confident you are that this is
                     a verifiable factual claim, not opinion>
    },
    ...
  ]
}

Rules:
* Each claim must be a single atomic assertion (one fact per claim).
* Do NOT make verification judgments — just extract.
* Do NOT add claims that are not in the input text.
* Preserve the original wording verbatim in the "text" field.
"""

    def __init__(self, model: str = "claude-opus-4-6", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def extract_claims(self, text: str) -> list[ExtractedClaim]:
        raise NotImplementedError(
            "ExampleClaudeBackend is a scaffolding placeholder. "
            "Implement extract_claims() with a real Anthropic SDK call "
            "to enable LLM-backed extraction."
        )
