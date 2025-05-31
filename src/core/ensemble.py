"""Ensemble / alternative aggregators for verifier outputs.

Each class below implements the same ``aggregate_claim`` interface as
:meth:`src.core.aggregator.BayesianAggregator.aggregate_claim_bayesian`
so they can be dropped in wherever the Bayesian aggregator is used.

When to choose which
--------------------
* :class:`MajorityVoteAggregator` -- transparent and predictable.  Use
  when every verifier has comparable reliability (e.g. after heavy
  recalibration) and you want stakeholders to be able to reason about
  the output without understanding Bayes.
* :class:`WeightedVoteAggregator` -- preferred over majority when
  verifier reliability varies.  Weights a refuted vote by sensitivity *
  confidence, balancing the strong-but-rare signals against the
  weak-but-frequent ones.
* :class:`BayesianAggregator` -- most principled; produces a posterior
  probability and is the default across the framework.
* :class:`EnsembleAggregator` -- combines Bayesian + majority +
  weighted.  A ``refuted`` verdict only sticks when at least two of the
  three sub-aggregators agree.  Use in regulated contexts where false
  positives carry a high cost.
* :class:`StackingAggregator` -- a thin meta-layer over any list of
  base aggregators.  Returns the verdict mode; ties broken by
  confidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

from src.core.aggregator import (
    AggregatedVerdict,
    BayesianAggregator,
    _DEFAULT_RELIABILITY,
    _VERIFIER_RELIABILITY,
)
from src.models.enums import Verdict
from src.verifiers.base import VerificationOutput

logger = logging.getLogger(__name__)

__all__ = [
    "MajorityVoteAggregator",
    "WeightedVoteAggregator",
    "StackingAggregator",
    "EnsembleAggregator",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_verdict(reason: str) -> AggregatedVerdict:
    """Return the canonical "no results" verdict used by every aggregator."""
    return AggregatedVerdict(
        verdict=Verdict.unverifiable,
        confidence=0.0,
        reasoning=[reason],
        verifier_results=[],
    )


def _reliability_for(source_key: str) -> dict[str, float]:
    """Resolve reliability for *source_key* with the same fallback logic
    used by :class:`BayesianAggregator`.
    """
    key = (source_key or "").strip().lower()
    return _VERIFIER_RELIABILITY.get(key, _DEFAULT_RELIABILITY)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


class MajorityVoteAggregator:
    """Simple majority vote across verifier verdicts.

    Ignores ``unverifiable`` results and resolves ties in favour of the
    vote with highest mean confidence.  Appropriate when every verifier
    has comparable reliability.
    """

    def aggregate_claim(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Return a :class:`AggregatedVerdict` computed by plain majority vote."""
        if not results:
            return _empty_verdict("No verifier results available.")

        informative = [
            r for r in results if r.verdict != Verdict.unverifiable
        ]
        if not informative:
            return _empty_verdict(
                "All verifiers returned 'unverifiable'; no majority possible."
            )

        counts: Counter[Verdict] = Counter(r.verdict for r in informative)
        top_count = max(counts.values())
        top_verdicts = [v for v, c in counts.items() if c == top_count]

        # Tie-break: highest mean confidence wins.
        if len(top_verdicts) > 1:
            def mean_conf(v: Verdict) -> float:
                conf = [r.confidence for r in informative if r.verdict == v]
                return sum(conf) / len(conf) if conf else 0.0

            top_verdicts.sort(key=mean_conf, reverse=True)
        chosen = top_verdicts[0]

        chosen_confs = [
            r.confidence for r in informative if r.verdict == chosen
        ]
        confidence = (
            round(sum(chosen_confs) / len(chosen_confs), 4)
            if chosen_confs
            else 0.0
        )

        reasoning = [
            f"Majority vote: {dict(counts)} "
            f"-> {chosen.value} ({top_count}/{len(informative)})"
        ]
        reasoning.extend(r.reasoning for r in results if r.reasoning)

        return AggregatedVerdict(
            verdict=chosen,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )


# ---------------------------------------------------------------------------
# Weighted vote
# ---------------------------------------------------------------------------


class WeightedVoteAggregator:
    """Weighted vote where weight = verifier sensitivity * self-confidence.

    Reuses :data:`_VERIFIER_RELIABILITY` from :mod:`aggregator`, matched
    against :attr:`VerificationOutput.source_db`.  Verifiers without a
    reliability entry fall back to :data:`_DEFAULT_RELIABILITY`.

    Preferred over majority voting when verifier reliability varies.
    """

    def __init__(
        self,
        reliability: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._reliability = reliability or dict(_VERIFIER_RELIABILITY)

    def aggregate_claim(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Return a verdict weighted by ``sensitivity * confidence``."""
        if not results:
            return _empty_verdict("No verifier results available.")

        informative = [r for r in results if r.verdict != Verdict.unverifiable]
        if not informative:
            return _empty_verdict(
                "All verifiers returned 'unverifiable'; no weighted vote possible."
            )

        verdict_weights: dict[Verdict, float] = {}
        total_weight = 0.0
        for r in informative:
            rel = self._reliability.get(
                (r.source_db or "").strip().lower(), _DEFAULT_RELIABILITY
            )
            w = float(rel.get("sensitivity", 0.5)) * max(
                0.0, min(1.0, float(r.confidence))
            )
            verdict_weights[r.verdict] = verdict_weights.get(r.verdict, 0.0) + w
            total_weight += w

        if total_weight <= 0.0:
            return _empty_verdict(
                "Total vote weight was zero (no confident verifiers)."
            )

        chosen, chosen_w = max(
            verdict_weights.items(), key=lambda kv: kv[1]
        )
        confidence = round(chosen_w / total_weight, 4)

        reasoning = [
            "Weighted vote: "
            + ", ".join(
                f"{v.value}={w:.3f}" for v, w in verdict_weights.items()
            )
            + f" -> {chosen.value}"
        ]
        reasoning.extend(r.reasoning for r in results if r.reasoning)

        return AggregatedVerdict(
            verdict=chosen,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


class StackingAggregator:
    """Meta-aggregator that takes the mode of several base aggregators.

    Each base is expected to expose either ``aggregate_claim`` or
    ``aggregate_claim_bayesian``.  The stacker calls them all, takes
    the most common verdict, and returns it.  Ties are broken by the
    base aggregator with the highest reported confidence on its own
    verdict.

    Use this when you want to layer together multiple aggregators
    (e.g. Bayesian + weighted) without hand-coding a voting rule.
    """

    def __init__(self, bases: list[object]) -> None:
        if not bases:
            raise ValueError("StackingAggregator requires at least one base")
        self._bases = list(bases)

    def aggregate_claim(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Return the mode verdict across all base aggregators."""
        if not results:
            return _empty_verdict("No verifier results available.")

        sub_verdicts: list[AggregatedVerdict] = []
        for base in self._bases:
            out = _run_base_aggregator(base, results)
            if out is not None:
                sub_verdicts.append(out)

        if not sub_verdicts:
            return _empty_verdict("No base aggregator produced a verdict.")

        counts: Counter[Verdict] = Counter(v.verdict for v in sub_verdicts)
        top_count = max(counts.values())
        top = [v for v, c in counts.items() if c == top_count]

        if len(top) > 1:
            # Tie -> pick the one with the highest reported confidence.
            best = max(
                sub_verdicts,
                key=lambda v: (v.confidence if v.verdict in top else -1.0),
            )
            chosen = best.verdict
        else:
            chosen = top[0]

        matching = [v for v in sub_verdicts if v.verdict == chosen]
        confidence = round(
            sum(v.confidence for v in matching) / len(matching), 4
        )

        reasoning = [
            "Stacking over "
            + ", ".join(type(b).__name__ for b in self._bases)
            + f": {dict(counts)} -> {chosen.value}"
        ]
        for v in sub_verdicts:
            reasoning.extend(v.reasoning)

        return AggregatedVerdict(
            verdict=chosen,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )


def _run_base_aggregator(
    base: object, results: list[VerificationOutput]
) -> AggregatedVerdict | None:
    """Invoke ``aggregate_claim`` or ``aggregate_claim_bayesian`` on *base*."""
    fn = getattr(base, "aggregate_claim", None)
    if fn is None:
        fn = getattr(base, "aggregate_claim_bayesian", None)
    if fn is None:
        logger.warning(
            "Stacking base %r has no aggregate_claim method", type(base).__name__
        )
        return None
    try:
        return fn(results)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Stacking base %r raised during aggregate_claim",
            type(base).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Ensemble (Bayesian + majority + weighted)
# ---------------------------------------------------------------------------


class EnsembleAggregator:
    """Combine Bayesian, majority, and weighted votes into one decision.

    Each sub-aggregator casts one verdict; the ensemble then applies the
    configured *strictness* rule:

    * ``"majority"`` (default) -- a ``refuted`` call only sticks if at
      least two of three sub-aggregators agree.  Otherwise the verdict
      falls back to the "softest" vote across the three.
    * ``"unanimous"`` -- all three sub-aggregators must agree for any
      verdict to stand; otherwise the claim becomes
      ``partially_supported``.
    * ``"any"`` -- the presence of any ``refuted`` vote is enough to
      mark the claim refuted.

    Recommended when false positives are more costly than false
    negatives (regulatory / safety-critical contexts).
    """

    _VALID_STRICTNESS = frozenset({"majority", "unanimous", "any"})

    def __init__(
        self,
        bayesian: BayesianAggregator | None = None,
        majority: MajorityVoteAggregator | None = None,
        weighted: WeightedVoteAggregator | None = None,
        strictness: str = "majority",
    ) -> None:
        if strictness not in self._VALID_STRICTNESS:
            raise ValueError(
                f"strictness must be one of {sorted(self._VALID_STRICTNESS)}"
            )
        self._bayesian = bayesian or BayesianAggregator()
        self._majority = majority or MajorityVoteAggregator()
        self._weighted = weighted or WeightedVoteAggregator()
        self._strictness = strictness

    def aggregate_claim(
        self, results: list[VerificationOutput]
    ) -> AggregatedVerdict:
        """Run all three sub-aggregators and apply the strictness rule."""
        if not results:
            return _empty_verdict("No verifier results available.")

        sub: list[AggregatedVerdict] = [
            self._bayesian.aggregate_claim_bayesian(results),
            self._majority.aggregate_claim(results),
            self._weighted.aggregate_claim(results),
        ]
        counts: Counter[Verdict] = Counter(v.verdict for v in sub)
        refuted_votes = counts.get(Verdict.refuted, 0)

        chosen: Verdict
        if self._strictness == "any":
            chosen = Verdict.refuted if refuted_votes > 0 else _softest(sub)
        elif self._strictness == "unanimous":
            top_verdict, top_count = counts.most_common(1)[0]
            chosen = (
                top_verdict
                if top_count == len(sub)
                else Verdict.partially_supported
            )
        else:  # majority (default)
            if refuted_votes >= 2:
                chosen = Verdict.refuted
            else:
                # Take the mode across the three; fall back to softer.
                top_verdict, top_count = counts.most_common(1)[0]
                if top_count >= 2 and top_verdict != Verdict.refuted:
                    chosen = top_verdict
                else:
                    chosen = _softest(sub)

        matching = [v for v in sub if v.verdict == chosen]
        confidence = round(
            sum(v.confidence for v in (matching or sub))
            / len(matching or sub),
            4,
        )
        reasoning = [
            f"Ensemble({self._strictness}) "
            f"bayesian={sub[0].verdict.value}, majority={sub[1].verdict.value}, "
            f"weighted={sub[2].verdict.value} -> {chosen.value}"
        ]
        for v in sub:
            reasoning.extend(v.reasoning)

        return AggregatedVerdict(
            verdict=chosen,
            confidence=confidence,
            reasoning=reasoning,
            verifier_results=results,
        )


#: Ordering from "strongest refutation" to "weakest/safest" for
#: ensemble fallback purposes.
_SOFTNESS_ORDER: tuple[Verdict, ...] = (
    Verdict.refuted,
    Verdict.partially_supported,
    Verdict.verified,
    Verdict.unverifiable,
)


def _softest(verdicts: Iterable[AggregatedVerdict]) -> Verdict:
    """Return the "softest" (least accusatory) verdict in *verdicts*.

    Used by :class:`EnsembleAggregator` to fall back when no majority
    exists -- avoiding false-positive refutations.
    """
    seen = {v.verdict for v in verdicts}
    for candidate in reversed(_SOFTNESS_ORDER):
        if candidate in seen:
            return candidate
    return Verdict.unverifiable
