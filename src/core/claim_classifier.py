"""Rule-based claim classification.

Maps each :class:`~src.core.claim_extractor.ExtractedClaim` to a
:class:`~src.models.enums.ClaimType` using a prioritised registry of
regex-based classification rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.core.claim_extractor import ExtractedClaim
from src.models.enums import ClaimType

# ---------------------------------------------------------------------------
# Classification rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassificationRule:
    """A single classification rule applied in priority order.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"smiles_or_molecular_property"``).
    patterns : list[re.Pattern]
        One or more compiled regexes.  If **any** pattern matches the
        claim text the rule fires.
    claim_type : ClaimType
        The type to assign when the rule fires.
    priority : int
        Lower numbers are evaluated first.  When multiple rules could
        match, the highest-priority (lowest number) wins.
    """

    name: str
    patterns: list[re.Pattern[str]]
    claim_type: ClaimType
    priority: int


# ---------------------------------------------------------------------------
# Built-in rules  (sorted by priority — lower = higher precedence)
# ---------------------------------------------------------------------------


def _builtin_rules() -> list[ClassificationRule]:
    """Return the default classification rules."""
    return [
        # Priority 10 — citation (DOI / PMID / explicit citation markers)
        # Checked first because DOI patterns can false-match SMILES regex.
        ClassificationRule(
            name="citation",
            patterns=[
                re.compile(r"10\.\d{4,}/\S+"),
                re.compile(r"PMID:\s*\d+", re.IGNORECASE),
                re.compile(r"\bet\s+al\."),
                re.compile(r"\[\d+\]"),  # numbered references
            ],
            claim_type=ClaimType.citation,
            priority=10,
        ),
        # Priority 15 — clinical outcome (NCT IDs, trial language)
        # Checked before molecular_property because NCT IDs match SMILES regex.
        ClassificationRule(
            name="clinical_outcome",
            patterns=[
                re.compile(r"NCT\d{8}"),
                re.compile(
                    r"\b(?:phase\s+(?:I{1,3}|[1-3])|clinical\s+trial|"
                    r"efficacy|adverse\s+(?:event|effect)|placebo|"
                    r"endpoint|randomized|double[- ]blind|"
                    r"FDA\s+approv|EMA\s+approv|indication)\b",
                    re.IGNORECASE,
                ),
            ],
            claim_type=ClaimType.clinical_outcome,
            priority=15,
        ),
        # Priority 20 — pharmacokinetic (before molecular_property to avoid
        # false matches from the broad SMILES regex)
        ClassificationRule(
            name="pharmacokinetic",
            patterns=[
                re.compile(
                    r"\b(?:bioavailability|Cmax|C_?max|AUC|half[- ]life|t1/2|"
                    r"clearance|Vd|volume\s+of\s+distribution|"
                    r"first[- ]pass|absorption|Tmax|T_?max)\b",
                    re.IGNORECASE,
                ),
            ],
            claim_type=ClaimType.pharmacokinetic,
            priority=20,
        ),
        # Priority 25 — target interaction
        ClassificationRule(
            name="target_interaction",
            patterns=[
                re.compile(
                    r"\b(?:binding|receptor|kinase|inhibitor|agonist|"
                    r"antagonist|ligand|substrate|enzyme|IC50|Ki|Kd|"
                    r"selectivity|affinity)\b",
                    re.IGNORECASE,
                ),
            ],
            claim_type=ClaimType.target_interaction,
            priority=25,
        ),
        # Priority 28 — mechanism of action
        ClassificationRule(
            name="mechanism_of_action",
            patterns=[
                re.compile(
                    r"\b(?:pathway|signaling|phosphorylat|inhibit|activat|"
                    r"upregulat|downregulat|modulat|cascade|apoptosis|"
                    r"proliferation|differentiation|transcription|"
                    r"signal\s+transduction)\b",
                    re.IGNORECASE,
                ),
            ],
            claim_type=ClaimType.mechanism_of_action,
            priority=28,
        ),
        # Priority 30 — molecular property (SMILES / InChI / property terms)
        # Broadest patterns, checked last among specific types.
        ClassificationRule(
            name="molecular_property",
            patterns=[
                # SMILES must contain at least one non-letter char to
                # distinguish from English words (ring digits, bonds,
                # brackets, etc.)
                re.compile(
                    r"(?<![A-Za-z])"
                    r"(?=[A-Za-z0-9@+\-\[\]\\/()=#%]{6,}(?![A-Za-z]))"
                    r"(?=\S*[0-9@+\[\]\\\/()=#])"
                    r"[A-Za-z0-9@+\-\[\]\\/()=#%]+"
                ),
                re.compile(r"\bInChI=\S+", re.IGNORECASE),
                re.compile(
                    r"\b(?:logP|log\s?D|IC50|IC_?50|Ki|EC50|EC_?50|MW|"
                    r"molecular\s+weight|pKa|clogP|TPSA|HBA|HBD|"
                    r"solubility|permeability|Lipinski)\b",
                    re.IGNORECASE,
                ),
            ],
            claim_type=ClaimType.molecular_property,
            priority=30,
        ),
    ]


# ---------------------------------------------------------------------------
# ClaimClassifier
# ---------------------------------------------------------------------------


class ClaimClassifier:
    """Classify extracted claims into :class:`ClaimType` categories.

    Classification uses a priority-ordered list of regex rules.  The first
    matching rule (lowest ``priority`` value) determines the claim type.
    If no rule matches the claim is classified as
    :attr:`ClaimType.general_biomedical`.

    Parameters
    ----------
    rules : Sequence[ClassificationRule] | None
        Override the built-in rule set.  When ``None`` the default
        drug-discovery rules are loaded.
    fallback : ClaimType
        Type assigned when no rule matches.
    """

    def __init__(
        self,
        rules: Sequence[ClassificationRule] | None = None,
        fallback: ClaimType = ClaimType.general_biomedical,
    ) -> None:
        self._rules: list[ClassificationRule] = sorted(
            rules if rules is not None else _builtin_rules(),
            key=lambda r: r.priority,
        )
        self._fallback = fallback

    # -- Rule registry ------------------------------------------------------

    @property
    def rules(self) -> list[ClassificationRule]:
        """Currently registered rules, sorted by priority."""
        return list(self._rules)

    def register_rule(self, rule: ClassificationRule) -> None:
        """Add a custom classification rule at runtime.

        The internal rule list is re-sorted after insertion so priority
        ordering is maintained.

        Parameters
        ----------
        rule : ClassificationRule
            Rule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    # -- Classification -----------------------------------------------------

    def classify(self, claim: ExtractedClaim) -> ClaimType:
        """Assign a :class:`ClaimType` to *claim*.

        Parameters
        ----------
        claim : ExtractedClaim
            The claim to classify.

        Returns
        -------
        ClaimType
            Best-matching claim type, or :attr:`_fallback` when no rule
            matches.
        """
        text = claim.claim_text
        for rule in self._rules:
            if any(p.search(text) for p in rule.patterns):
                return rule.claim_type
        return self._fallback
