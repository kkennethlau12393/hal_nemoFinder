"""Claim extraction from free-text LLM output.

Identifies factual claims in text that can be verified against
external databases (ChEMBL, PubChem, PubMed, ClinicalTrials.gov, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Character-level location of a claim inside the original text."""

    start: int
    end: int
    context: str  # surrounding sentence / paragraph


@dataclass(slots=True)
class ExtractedClaim:
    """A single factual claim extracted from text."""

    claim_text: str
    source_span: SourceSpan
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Extraction-pattern registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionPattern:
    """A named regex pattern that flags a sentence as containing a claim.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"smiles_string"``).
    pattern : re.Pattern
        Compiled regex applied to each candidate sentence.
    confidence_boost : float
        Added to the base confidence when this pattern matches.
        Concrete identifiers (DOIs, SMILES) boost more than vague keywords.
    """

    name: str
    pattern: re.Pattern[str]
    confidence_boost: float


# Built-in patterns --------------------------------------------------------

_BUILTIN_PATTERNS: list[ExtractionPattern] = [
    # Concrete identifiers — high confidence
    ExtractionPattern(
        name="doi",
        pattern=re.compile(r"10\.\d{4,}/\S+"),
        confidence_boost=0.35,
    ),
    ExtractionPattern(
        name="pubmed_id",
        pattern=re.compile(r"PMID:\s*\d+", re.IGNORECASE),
        confidence_boost=0.35,
    ),
    ExtractionPattern(
        name="nct_id",
        pattern=re.compile(r"NCT\d{8}"),
        confidence_boost=0.35,
    ),
    ExtractionPattern(
        name="smiles_string",
        # SMILES: at least 6 characters from the SMILES alphabet with at least
        # one ring digit or branch and NOT purely alphabetical.
        pattern=re.compile(
            r"(?<![A-Za-z])"
            r"[A-Za-z0-9@+\-\[\]\\/()=#%]{6,}"
            r"(?![A-Za-z])"
        ),
        confidence_boost=0.30,
    ),
    ExtractionPattern(
        name="numerical_with_units",
        pattern=re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:nM|[uμµ]M|mM|mg|[uμµ]g|ng|kg|mL|[uμµ]L|"
            r"mol/L|g/mol|Da|kDa|%|°C|K|hr|min|sec|kcal|kJ|Å|nm)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.25,
    ),
    # Molecular property keywords
    ExtractionPattern(
        name="molecular_property_keyword",
        pattern=re.compile(
            r"\b(?:logP|log\s?D|IC50|IC_?50|Ki|EC50|EC_?50|"
            r"MW|molecular\s+weight|pKa|clogP|TPSA|HBA|HBD|"
            r"solubility|permeability|Lipinski)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.20,
    ),
    # Protein / target mentions
    ExtractionPattern(
        name="target_mention",
        pattern=re.compile(
            r"\b(?:receptor|kinase|inhibitor|agonist|antagonist|"
            r"ligand|substrate|enzyme|transporter|channel|protease|"
            r"phosphatase|transferase|synthase|reductase)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.15,
    ),
    # Clinical terms
    ExtractionPattern(
        name="clinical_term",
        pattern=re.compile(
            r"\b(?:phase\s+(?:I{1,3}|[1-3])|clinical\s+trial|"
            r"efficacy|adverse\s+(?:event|effect)|placebo|"
            r"endpoint|randomized|double[- ]blind|FDA|EMA|"
            r"approval|indication|dose[- ]response)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.15,
    ),
    # PK terms
    ExtractionPattern(
        name="pharmacokinetic_term",
        pattern=re.compile(
            r"\b(?:bioavailability|Cmax|C_?max|AUC|half[- ]life|t1/2|"
            r"clearance|Vd|volume\s+of\s+distribution|"
            r"first[- ]pass|absorption|Tmax|T_?max)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.18,
    ),
    # Mechanism of action terms
    ExtractionPattern(
        name="mechanism_term",
        pattern=re.compile(
            r"\b(?:pathway|signaling|phosphorylat|inhibit|activat|"
            r"upregulat|downregulat|modulat|cascade|apoptosis|"
            r"proliferation|differentiation|transcription)\b",
            re.IGNORECASE,
        ),
        confidence_boost=0.12,
    ),
]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that should NOT be treated as sentence boundaries.
_ABBREVIATION_SET = frozenset(
    "al Dr Mr Mrs Ms Prof Fig Figs Eq Eqs Ref Refs Vol "
    "Inc Ltd Jr Sr vs approx ca viz No no Nos".split()
)

# Simple sentence-boundary regex: punctuation followed by whitespace and an
# uppercase letter or digit.  We post-filter to reject abbreviation splits.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])"   # lookbehind for punctuation (fixed-width)
    r"\s+"           # whitespace between sentences
    r"(?=[A-Z0-9\[])"  # lookahead for sentence start
)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(sentence, start, end)`` tuples.

    Handles common scientific abbreviations so "et al." does not cause
    a spurious split.
    """
    # First pass: split at every candidate boundary.
    raw_parts: list[tuple[str, int, int]] = []
    prev = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        sentence = text[prev : m.start()].strip()
        if sentence:
            raw_parts.append((sentence, prev, m.start()))
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        raw_parts.append((tail, prev, len(text)))

    # Second pass: merge back splits that land right after an abbreviation.
    merged: list[tuple[str, int, int]] = []
    for sent, start, end in raw_parts:
        # Check if the previous segment ended with an abbreviation.
        if merged:
            prev_sent = merged[-1][0]
            last_word = prev_sent.rstrip(".!?").rsplit(None, 1)[-1] if prev_sent else ""
            if last_word in _ABBREVIATION_SET or last_word.rstrip(".") == "e.g" or last_word.rstrip(".") == "i.e":
                # Merge with previous segment.
                merged[-1] = (
                    text[merged[-1][1] : end].strip(),
                    merged[-1][1],
                    end,
                )
                continue
        merged.append((sent, start, end))

    return merged


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------

# Confidence floor for any matched sentence.
_BASE_CONFIDENCE = 0.30


class ClaimExtractor:
    """Extract verifiable factual claims from free-text LLM output.

    The extractor maintains a mutable registry of :class:`ExtractionPattern`
    instances.  Extend it at runtime via :meth:`register_pattern` or at
    subclass level by overriding :meth:`_default_patterns`.

    Parameters
    ----------
    patterns : Sequence[ExtractionPattern] | None
        Override the built-in patterns entirely.  When ``None`` the
        default scientific-claim patterns are used.
    base_confidence : float
        Minimum confidence assigned to any matched claim.
    """

    def __init__(
        self,
        patterns: Sequence[ExtractionPattern] | None = None,
        base_confidence: float = _BASE_CONFIDENCE,
    ) -> None:
        self._patterns: list[ExtractionPattern] = list(
            patterns if patterns is not None else self._default_patterns()
        )
        self._base_confidence = base_confidence

    # -- Pattern registry ---------------------------------------------------

    @staticmethod
    def _default_patterns() -> list[ExtractionPattern]:
        """Return a copy of the built-in extraction patterns."""
        return list(_BUILTIN_PATTERNS)

    @property
    def patterns(self) -> list[ExtractionPattern]:
        """Currently registered extraction patterns (read-only view)."""
        return list(self._patterns)

    def register_pattern(self, pattern: ExtractionPattern) -> None:
        """Add a custom extraction pattern at runtime.

        Parameters
        ----------
        pattern : ExtractionPattern
            Pattern to append to the registry.
        """
        self._patterns.append(pattern)

    # -- Extraction ---------------------------------------------------------

    def extract(self, text: str) -> list[ExtractedClaim]:
        """Extract verifiable claims from *text*.

        Parameters
        ----------
        text : str
            Raw text (typically LLM output) to scan for claims.

        Returns
        -------
        list[ExtractedClaim]
            Claims ordered by their position in the original text.
        """
        claims: list[ExtractedClaim] = []

        for sentence, start, end in _split_sentences(text):
            matched_patterns = self._match_patterns(sentence)
            if not matched_patterns:
                continue

            confidence = self._compute_confidence(matched_patterns)
            span = SourceSpan(start=start, end=end, context=sentence)
            claims.append(
                ExtractedClaim(
                    claim_text=sentence,
                    source_span=span,
                    confidence=confidence,
                )
            )

        return claims

    # -- Internals ----------------------------------------------------------

    def _match_patterns(self, sentence: str) -> list[ExtractionPattern]:
        """Return all patterns that match *sentence*."""
        return [p for p in self._patterns if p.pattern.search(sentence)]

    def _compute_confidence(self, matched: list[ExtractionPattern]) -> float:
        """Derive a confidence score from the set of matched patterns.

        Confidence starts at :pyattr:`_base_confidence` and increases with
        each matching pattern (diminishing returns via harmonic sum so the
        value never exceeds 1.0).
        """
        score = self._base_confidence
        for p in sorted(matched, key=lambda p: p.confidence_boost, reverse=True):
            # Each additional boost has a diminishing contribution.
            score += (1.0 - score) * p.confidence_boost
        return round(min(score, 1.0), 4)
