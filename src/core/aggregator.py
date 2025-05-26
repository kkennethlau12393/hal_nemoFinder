"""Result aggregation -- combine per-verifier outputs into final verdicts.

Two levels of aggregation:

1. **Per-claim** -- :meth:`ResultAggregator.aggregate_claim` merges
   multiple :class:`VerificationOutput` objects into a single
   :class:`AggregatedVerdict`.
2. **Per-job** -- :meth:`ResultAggregator.aggregate_job` rolls up all
   claims into a :class:`ReportData` with an overall hallucination score,
   severity label, and per-section breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.models.enums import ClaimType, Severity, Verdict
from src.verifiers.base import VerificationOutput

if TYPE_CHECKING:
    from src.core.confidence_intervals import CIResult

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClaimInfo:
    """Lightweight claim descriptor used during aggregation.

    Attributes
    ----------
    claim_text : str
        The original claim sentence.
    claim_type : ClaimType
        Classification result.
    extraction_confidence : float
        Confidence from the extraction stage (0.0 -- 1.0).
    """

    claim_text: str
    claim_type: ClaimType
    extraction_confidence: float


@dataclass(slots=True)
class AggregatedVerdict:
    """Merged verdict for a single claim across all verifiers.

    Attributes
    ----------
    verdict : Verdict
        Final consensus verdict.
    confidence : float
        Weighted confidence (0.0 -- 1.0).
    reasoning : list[str]
        Combined reasoning strings from contributing verifiers.
    verifier_results : list[VerificationOutput]
        The raw outputs that were aggregated.
    """

    verdict: Verdict
    confidence: float
    reasoning: list[str] = field(default_factory=list)
    verifier_results: list[VerificationOutput] = field(default_factory=list)


@dataclass(slots=True)
class SectionStats:
    """Per-claim-type statistics within a report.

    Attributes
    ----------
    claim_type : ClaimType
        The section's claim type.
    count : int
        Total claims of this type.
    verified : int
        Claims whose aggregated verdict is ``verified``.
    refuted : int
        Claims whose aggregated verdict is ``refuted``.
    score : float
        Section-level hallucination score (0.0 -- 1.0).
    """

    claim_type: ClaimType
    count: int = 0
    verified: int = 0
    refuted: int = 0
    score: float = 0.0


@dataclass(slots=True)
class ReportData:
    """Top-level report for an entire analysis job.

    Attributes
    ----------
    hallucination_score : float
        Overall score from 0.0 (clean) to 1.0 (fully hallucinated).
    severity : Severity
        Categorical label derived from :attr:`hallucination_score`.
    total_claims : int
        Number of claims analysed.
    sections : list[SectionStats]
        Per-claim-type breakdown.
    metadata : dict
        Extra information (timing, verifier versions, etc.).
    """

    hallucination_score: float
    severity: Severity
    total_claims: int
    sections: list[SectionStats] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Default per-verifier weights (keyed by verifier name).
# Unknown verifiers default to 1.0.
_DEFAULT_VERIFIER_WEIGHTS: dict[str, float] = {}

# Claim-type importance weights used in the job-level score.
_DEFAULT_TYPE_WEIGHTS: dict[ClaimType, float] = {
    ClaimType.citation: 1.5,
    ClaimType.molecular_property: 1.3,
    ClaimType.clinical_outcome: 1.3,
    ClaimType.target_interaction: 1.2,
    ClaimType.pharmacokinetic: 1.1,
    ClaimType.mechanism_of_action: 1.0,
    ClaimType.general_biomedical: 0.8,
}

# Severity thresholds.
_SEVERITY_THRESHOLDS: list[tuple[float, Severity]] = [
    (0.1, Severity.clean),
    (0.3, Severity.minor),
    (0.6, Severity.major),
]


class ResultAggregator:
    """Aggregate verification results at the claim and job levels.

    Parameters
    ----------
    verifier_weights : dict[str, float] | None
        Per-verifier confidence weights (keyed by verifier ``name``).
        Verifiers not listed default to ``1.0``.
    type_weights : dict[ClaimType, float] | None
        Override the default claim-type importance weights used when
        computing the job-level hallucination score.
    refuted_confidence_threshold : float
        Minimum confidence required on a ``refuted`` result for it to
        dominate the aggregated verdict.
    """

    def __init__(
        self,
        verifier_weights: dict[str, float] | None = None,
        type_weights: dict[ClaimType, float] | None = None,
        refuted_confidence_threshold: float = 0.7,
    ) -> None:
        self._verifier_weights = verifier_weights or dict(_DEFAULT_VERIFIER_WEIGHTS)
        self._type_weights = type_weights or dict(_DEFAULT_TYPE_WEIGHTS)
        self._refuted_threshold = refuted_confidence_threshold

    # -- Per-claim aggregation ----------------------------------------------

    def aggregate_claim(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Merge multiple verifier outputs into a single verdict.

        Aggregation logic
        -----------------
        1. If **any** result is ``refuted`` with confidence above
           :attr:`_refuted_threshold` the claim is ``refuted``.
        2. If all results are ``unverifiable`` the claim is
           ``unverifiable``.
        3. If all results are ``verified`` the claim is ``verified``.
        4. Otherwise the claim is ``partially_supported``.

        Parameters
        ----------
        results : list[VerificationOutput]
            Outputs from one or more verifiers for the same claim.

        Returns
        -------
        AggregatedVerdict
        """
        if not results:
            return AggregatedVerdict(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=["No verifier results available."],
                verifier_results=[],
            )

        verdicts = [r.verdict for r in results]
        reasoning = [r.reasoning for r in results if r.reasoning]

        # Rule 1: strong refutation dominates.
        if any(
            r.verdict == Verdict.refuted and r.confidence >= self._refuted_threshold
            for r in results
        ):
            verdict = Verdict.refuted
        # Rule 2: all unverifiable.
        elif all(v == Verdict.unverifiable for v in verdicts):
            verdict = Verdict.unverifiable
        # Rule 3: all verified.
        elif all(v == Verdict.verified for v in verdicts):
            verdict = Verdict.verified
        # Rule 4: mixed.
        else:
            verdict = Verdict.partially_supported

        confidence = self._weighted_confidence(results)

        return AggregatedVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )

    # -- Job-level aggregation ----------------------------------------------

    def aggregate_job(
        self,
        claims_with_verdicts: list[tuple[ClaimInfo, AggregatedVerdict]],
    ) -> ReportData:
        """Produce a full report from all claims in an analysis job.

        Parameters
        ----------
        claims_with_verdicts : list[tuple[ClaimInfo, AggregatedVerdict]]
            Each entry pairs a :class:`ClaimInfo` with its aggregated
            verdict.

        Returns
        -------
        ReportData
            Overall score, severity, and per-section breakdowns.
        """
        if not claims_with_verdicts:
            return ReportData(
                hallucination_score=0.0,
                severity=Severity.clean,
                total_claims=0,
            )

        # Build per-section stats.
        section_map: dict[ClaimType, SectionStats] = {}
        weighted_bad = 0.0
        weighted_total = 0.0

        for claim_info, agg in claims_with_verdicts:
            ct = claim_info.claim_type
            stats = section_map.setdefault(ct, SectionStats(claim_type=ct))
            stats.count += 1

            if agg.verdict == Verdict.verified:
                stats.verified += 1
            elif agg.verdict == Verdict.refuted:
                stats.refuted += 1

            type_weight = self._type_weights.get(ct, 1.0)
            weighted_total += type_weight
            if agg.verdict in (Verdict.refuted, Verdict.partially_supported):
                # Refuted contributes fully; partially_supported at half weight.
                factor = 1.0 if agg.verdict == Verdict.refuted else 0.5
                weighted_bad += type_weight * factor * agg.confidence

        # Per-section scores.
        for stats in section_map.values():
            if stats.count > 0:
                stats.score = round(stats.refuted / stats.count, 4)

        hallucination_score = round(
            weighted_bad / weighted_total if weighted_total > 0 else 0.0, 4
        )
        severity = self._severity_from_score(hallucination_score)

        return ReportData(
            hallucination_score=hallucination_score,
            severity=severity,
            total_claims=len(claims_with_verdicts),
            sections=sorted(
                section_map.values(), key=lambda s: s.score, reverse=True
            ),
        )

    # -- Helpers ------------------------------------------------------------

    def _weighted_confidence(self, results: list[VerificationOutput]) -> float:
        """Compute a weighted-average confidence across verifier results."""
        total_w = 0.0
        total_c = 0.0
        for r in results:
            w = self._verifier_weights.get(r.source_db, 1.0)
            total_w += w
            total_c += w * r.confidence
        return round(total_c / total_w if total_w > 0 else 0.0, 4)

    @staticmethod
    def _severity_from_score(score: float) -> Severity:
        """Map a hallucination score to a :class:`Severity` label."""
        for threshold, sev in _SEVERITY_THRESHOLDS:
            if score < threshold:
                return sev
        return Severity.critical


# ---------------------------------------------------------------------------
# Bayesian aggregator
# ---------------------------------------------------------------------------

#: Per-verifier reliability constants.  Sensitivity is P(verifier says
#: "refuted" | claim is truly a hallucination); specificity is
#: P(verifier says "verified" | claim is truly not a hallucination).
#: These seed values would be re-estimated from labelled data over time.
_VERIFIER_RELIABILITY: dict[str, dict[str, float]] = {
    "chemical": {"sensitivity": 0.95, "specificity": 0.92},
    "citation": {"sensitivity": 0.85, "specificity": 0.95},
    "clinical": {"sensitivity": 0.80, "specificity": 0.93},
    "pharmacokinetic": {"sensitivity": 0.90, "specificity": 0.88},
    "target": {"sensitivity": 0.80, "specificity": 0.85},
    "consistency": {"sensitivity": 0.70, "specificity": 0.95},
    "statistical": {"sensitivity": 0.93, "specificity": 0.90},
    "pathway": {"sensitivity": 0.78, "specificity": 0.92},
    # Common source_db aliases used by individual verifier implementations
    "rdkit": {"sensitivity": 0.95, "specificity": 0.92},   # chemical
    "crossref": {"sensitivity": 0.85, "specificity": 0.95},  # citation
    "uniprot": {"sensitivity": 0.80, "specificity": 0.85},   # target
    "chembl": {"sensitivity": 0.85, "specificity": 0.90},    # target/chemical
    "pubchem": {"sensitivity": 0.90, "specificity": 0.92},   # chemical
    "clinicaltrials": {"sensitivity": 0.80, "specificity": 0.93},
    "pathways": {"sensitivity": 0.78, "specificity": 0.92},  # pathway plural
    # New deterministic verifiers
    "literature": {"sensitivity": 0.72, "specificity": 0.85},
    "adverse_events": {"sensitivity": 0.82, "specificity": 0.90},
    "faers": {"sensitivity": 0.82, "specificity": 0.90},
    "drug_interaction": {"sensitivity": 0.88, "specificity": 0.92},
    "ddi": {"sensitivity": 0.88, "specificity": 0.92},
    "structure_3d": {"sensitivity": 0.80, "specificity": 0.95},
    "pdb": {"sensitivity": 0.80, "specificity": 0.95},
    "admet": {"sensitivity": 0.70, "specificity": 0.80},
    "retrosynthesis": {"sensitivity": 0.65, "specificity": 0.88},
    "patent": {"sensitivity": 0.85, "specificity": 0.95},
    "regulatory_label": {"sensitivity": 0.90, "specificity": 0.93},
    "druglabel": {"sensitivity": 0.90, "specificity": 0.93},
    "toxicity": {"sensitivity": 0.75, "specificity": 0.85},
    "metabolism": {"sensitivity": 0.80, "specificity": 0.92},
}

#: Fallback reliability for verifiers not listed above.
_DEFAULT_RELIABILITY: dict[str, float] = {"sensitivity": 0.70, "specificity": 0.70}

#: Base-rate prior for P(claim is a hallucination) in unverified LLM
#: drug-discovery output.  Literature estimates cluster around 0.25--0.35.
_DEFAULT_PRIOR: float = 0.30

#: Numerical guard for likelihood clipping to keep odds finite.
_EPSILON: float = 1e-6


class BayesianAggregator:
    """Aggregate verifier outputs via Bayesian updating.

    Each verifier is modelled as a noisy binary observer of the latent
    event ``H`` = "the claim is actually a hallucination", with a known
    sensitivity (true-positive rate) ``s`` and specificity (true-negative
    rate) ``t``.  Given a verdict from verifier *i*, we compute a
    likelihood ratio and update the prior odds.

    Math
    ----
    Let ``p0 = P(H)`` be the prior.  The prior odds are
    ``O0 = p0 / (1 - p0)``.  For each verifier *i* with sensitivity
    ``s_i`` and specificity ``t_i`` we compute a likelihood ratio
    depending on its verdict:

    * ``REFUTED``                -> ``LR_i = s_i / (1 - t_i)``
    * ``VERIFIED``               -> ``LR_i = (1 - s_i) / t_i``
    * ``PARTIALLY_SUPPORTED``    -> ``LR_i = 0.5 / 0.5 = 1`` (weak; see
      ``_weak_likelihood_ratio`` -- slightly above/below 1 is allowed)
    * ``UNVERIFIABLE``           -> ignored (no update)

    Assuming conditional independence of verifiers given ``H``, the
    posterior odds are::

        O_post = O0 * prod_i LR_i ^ w_i

    where ``w_i`` is the verifier's self-reported confidence (0..1).
    Raising the LR to a confidence exponent implements a soft
    "fractional observation": a verifier with confidence 0 contributes
    nothing; at confidence 1 it contributes its full LR.

    The posterior probability is recovered via
    ``p_post = O_post / (1 + O_post)``.

    Parameters
    ----------
    reliability : dict[str, dict[str, float]] | None
        Override per-verifier sensitivity/specificity.  Keys are matched
        against :attr:`VerificationOutput.source_db`.
    prior : float
        Prior probability of hallucination.  Defaults to
        :data:`_DEFAULT_PRIOR`.
    """

    def __init__(
        self,
        reliability: dict[str, dict[str, float]] | None = None,
        prior: float = _DEFAULT_PRIOR,
    ) -> None:
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self._reliability = reliability or dict(_VERIFIER_RELIABILITY)
        self._prior = prior

    # -- Public API ---------------------------------------------------------

    def aggregate_claim_bayesian(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Aggregate verifier results with a Bayesian posterior.

        Parameters
        ----------
        results : list[VerificationOutput]
            Per-verifier outputs for a single claim.

        Returns
        -------
        AggregatedVerdict
            Contains the mapped verdict, a decisiveness-based confidence
            score, and reasoning that reports the posterior probability.
        """
        if not results:
            return AggregatedVerdict(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=["No verifier results available."],
                verifier_results=[],
            )

        posterior = self._posterior_hallucination(results)
        verdict = self._verdict_from_posterior(posterior)

        # Decisiveness: distance from 0.5, scaled to 0..1.
        confidence = round(min(1.0, abs(posterior - 0.5) * 2.0), 4)

        # Count verifiers that actually contributed (i.e. were not UNVERIFIABLE).
        n_informative = sum(
            1 for r in results if r.verdict != Verdict.unverifiable
        )

        reasoning = [
            f"Bayesian posterior P(hallucination) = {posterior:.2f} "
            f"from {n_informative} verifiers"
        ]
        # Carry through individual verifier explanations for transparency.
        reasoning.extend(r.reasoning for r in results if r.reasoning)

        return AggregatedVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )

    def posterior_probability(
        self, results: list[VerificationOutput]
    ) -> float:
        """Return the posterior P(hallucination) without mapping to a verdict.

        Useful for calibration tracking where the raw probability is
        needed alongside a discrete prediction.
        """
        if not results:
            return self._prior
        return self._posterior_hallucination(results)

    def aggregate_claim_with_ci(
        self,
        results: list[VerificationOutput],
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> tuple[AggregatedVerdict, "CIResult"]:
        """Aggregate *results* and return both the verdict and a bootstrap CI.

        This convenience method wires the aggregator to
        :class:`~src.core.confidence_intervals.BootstrapConfidenceInterval`
        so callers can get uncertainty estimates in a single call.

        Parameters
        ----------
        results : list[VerificationOutput]
            Per-verifier outputs for a single claim.
        n_bootstrap : int
            Number of bootstrap replicates (default 1000).
        seed : int | None
            Optional RNG seed for reproducible intervals.

        Returns
        -------
        tuple[AggregatedVerdict, CIResult]
            The standard aggregated verdict plus a CI computed via
            non-parametric bootstrap over *results*.
        """
        # Local import to avoid a circular dependency at module load.
        from src.core.confidence_intervals import BootstrapConfidenceInterval

        verdict = self.aggregate_claim_bayesian(results)
        ci_tool = BootstrapConfidenceInterval(n_bootstrap=n_bootstrap, seed=seed)
        ci = ci_tool.compute(results, self)
        return verdict, ci

    # -- Core math ----------------------------------------------------------

    def _posterior_hallucination(
        self, results: list[VerificationOutput]
    ) -> float:
        """Compute the posterior probability of hallucination.

        Iteratively multiplies likelihood ratios into the prior odds,
        raising each LR to the verifier's self-reported confidence.
        """
        log_odds = self._log(self._odds_from_prob(self._prior))

        for r in results:
            if r.verdict == Verdict.unverifiable:
                continue

            # Resolve verifier reliability — try source_db first, then fall
            # back to a few common synonyms found in evidence dicts.
            source_key = (r.source_db or "").strip().lower()
            rel = self._reliability.get(source_key)
            if rel is None and isinstance(r.evidence, dict):
                # Some verifiers tag the verifier name in evidence['verifier']
                alt = str(r.evidence.get("verifier", "")).strip().lower()
                if alt:
                    rel = self._reliability.get(alt)
            if rel is None:
                rel = _DEFAULT_RELIABILITY
            sensitivity = float(rel.get("sensitivity", _DEFAULT_RELIABILITY["sensitivity"]))
            specificity = float(rel.get("specificity", _DEFAULT_RELIABILITY["specificity"]))

            lr = self._likelihood_ratio(r.verdict, sensitivity, specificity)
            if lr <= 0.0:
                continue

            # Confidence-weighted log-evidence: w * log(LR).
            weight = max(0.0, min(1.0, float(r.confidence)))
            log_odds += weight * self._log(lr)

        posterior_odds = self._exp(log_odds)
        return self._prob_from_odds(posterior_odds)

    @staticmethod
    def _likelihood_ratio(
        verdict: Verdict, sensitivity: float, specificity: float
    ) -> float:
        """Likelihood ratio P(verdict|H) / P(verdict|not H).

        ``REFUTED`` evidence pushes the posterior toward "hallucination";
        ``VERIFIED`` pushes toward "not hallucination".
        ``PARTIALLY_SUPPORTED`` is treated as weak/neutral evidence: we
        use a symmetric 0.5/0.5 likelihood which yields ``LR = 1`` and
        therefore does not move the odds.
        """
        # Clip to avoid division by zero / infinite log-odds.
        s = min(max(sensitivity, _EPSILON), 1.0 - _EPSILON)
        t = min(max(specificity, _EPSILON), 1.0 - _EPSILON)

        if verdict == Verdict.refuted:
            return s / (1.0 - t)
        if verdict == Verdict.verified:
            return (1.0 - s) / t
        if verdict == Verdict.partially_supported:
            # P(partial | H) = P(partial | not H) = 0.5 -> LR = 1.
            return 1.0
        # Unverifiable (or anything unknown) contributes nothing.
        return 1.0

    @staticmethod
    def _verdict_from_posterior(posterior: float) -> Verdict:
        """Map a posterior hallucination probability to a :class:`Verdict`."""
        if posterior > 0.7:
            return Verdict.refuted
        if posterior > 0.4:
            return Verdict.partially_supported
        # Both the "weakly verified" (0.1 < p <= 0.4) and "strongly
        # verified" (p <= 0.1) buckets map to VERIFIED at the enum level;
        # the returned confidence captures the strength of the call.
        return Verdict.verified

    # -- Numerical helpers --------------------------------------------------

    @staticmethod
    def _odds_from_prob(p: float) -> float:
        """Convert a probability in (0, 1) to odds ``p / (1 - p)``."""
        p = min(max(p, _EPSILON), 1.0 - _EPSILON)
        return p / (1.0 - p)

    @staticmethod
    def _prob_from_odds(odds: float) -> float:
        """Convert odds in (0, inf) back to a probability."""
        if odds <= 0.0:
            return 0.0
        return odds / (1.0 + odds)

    @staticmethod
    def _log(x: float) -> float:
        """Natural log with an epsilon floor."""
        import math

        return math.log(max(x, _EPSILON))

    @staticmethod
    def _exp(x: float) -> float:
        """Exponential that saturates on overflow."""
        import math

        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")
