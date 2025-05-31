"""Active-learning / uncertainty-sampling queue.

The :class:`UncertaintySampler` maintains a bounded in-memory queue of
claims whose aggregated posterior is informative enough to be worth
human review.  Reviewers then label the claims; the labels feed back
into the calibration loop as :class:`LabeledClaim` entries.

Three uncertainty strategies are implemented from scratch:

* ``least_confident`` -- ``1 - max(p, 1 - p)``.  Peaks at ``p = 0.5``.
* ``margin`` -- distance to the nearest Bayesian decision threshold
  (0.4 or 0.7).  A posterior exactly on a threshold has margin 0.
* ``entropy`` -- binary Shannon entropy ``-p log p - (1-p) log (1-p)``.

All three produce a higher score for more-uncertain claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.calibration import LabeledClaim
from src.models.enums import ClaimType, Verdict

__all__ = ["UncertaintySampler", "ReviewCandidate"]

#: Decision thresholds used by :class:`BayesianAggregator` for the
#: ``verified`` / ``partially_supported`` / ``refuted`` boundaries.  The
#: ``margin`` strategy measures how close a posterior sits to the
#: nearest of these.
_BAYES_THRESHOLDS: tuple[float, ...] = (0.4, 0.7)

_VALID_STRATEGIES: frozenset[str] = frozenset(
    {"least_confident", "margin", "entropy"}
)


@dataclass(slots=True)
class ReviewCandidate:
    """Claim awaiting human review.

    Attributes
    ----------
    claim_id : str
        Stable identifier (e.g. hash or database row id).
    claim_text : str
        The original claim sentence.
    predicted_verdict : Verdict
        Verdict the aggregator produced.
    posterior : float
        Bayesian posterior P(hallucination).
    uncertainty_score : float
        Strategy-dependent uncertainty score; higher = more uncertain.
    added_at : datetime
        When the candidate was queued.
    reviewed : bool
        ``True`` once :meth:`UncertaintySampler.mark_reviewed` is called.
    label : bool | None
        Reviewer-provided label (``True`` = hallucination).
    reviewer : str
        Identity of the reviewer who labelled the candidate.
    claim_type : ClaimType
        Claim type used when exporting to :class:`LabeledClaim`.
    """

    claim_id: str
    claim_text: str
    predicted_verdict: Verdict
    posterior: float
    uncertainty_score: float
    added_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reviewed: bool = False
    label: bool | None = None
    reviewer: str = ""
    claim_type: ClaimType = ClaimType.general_biomedical

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict representation."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "predicted_verdict": self.predicted_verdict.value,
            "posterior": self.posterior,
            "uncertainty_score": self.uncertainty_score,
            "added_at": self.added_at.isoformat(),
            "reviewed": self.reviewed,
            "label": self.label,
            "reviewer": self.reviewer,
            "claim_type": self.claim_type.value,
        }


class UncertaintySampler:
    """Bounded review queue selecting the most-uncertain claims.

    Parameters
    ----------
    review_queue_size : int
        Maximum number of *unreviewed* candidates retained.  When full,
        the least-uncertain candidate is evicted to make room for a new
        entry (only if the new entry is more uncertain).
    uncertainty_strategy : str
        Default strategy used by :meth:`score`.  One of
        ``least_confident``, ``margin``, ``entropy``.
    """

    def __init__(
        self,
        review_queue_size: int = 100,
        uncertainty_strategy: str = "least_confident",
    ) -> None:
        if review_queue_size <= 0:
            raise ValueError("review_queue_size must be positive")
        if uncertainty_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"uncertainty_strategy must be one of {sorted(_VALID_STRATEGIES)}"
            )
        self._capacity = int(review_queue_size)
        self._strategy = uncertainty_strategy
        self._candidates: dict[str, ReviewCandidate] = {}

    # -- Scoring ------------------------------------------------------------

    def score(self, posterior: float, strategy: str | None = None) -> float:
        """Return the uncertainty score for *posterior* (higher = more uncertain)."""
        s = strategy or self._strategy
        p = max(0.0, min(1.0, float(posterior)))
        if s == "least_confident":
            return 1.0 - max(p, 1.0 - p)
        if s == "margin":
            # Smaller distance to threshold = more uncertain, so invert.
            nearest = min(abs(p - t) for t in _BAYES_THRESHOLDS)
            return 1.0 - nearest
        if s == "entropy":
            if p <= 0.0 or p >= 1.0:
                return 0.0
            return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
        raise ValueError(f"unknown strategy: {s!r}")

    # -- Candidate management -----------------------------------------------

    def add_candidate(
        self,
        claim_id: str,
        claim_text: str,
        posterior: float,
        verdict: Verdict,
        claim_type: ClaimType = ClaimType.general_biomedical,
    ) -> ReviewCandidate | None:
        """Submit a claim for possible inclusion in the review queue.

        If the queue is at capacity and the new claim is *less*
        uncertain than the currently least-uncertain unreviewed
        candidate, the new claim is ignored.  Otherwise the weakest
        unreviewed candidate is evicted.

        Returns
        -------
        ReviewCandidate | None
            The stored candidate, or ``None`` if the claim was rejected.
        """
        uncertainty = self.score(posterior)
        candidate = ReviewCandidate(
            claim_id=claim_id,
            claim_text=claim_text,
            predicted_verdict=verdict,
            posterior=float(posterior),
            uncertainty_score=uncertainty,
            claim_type=claim_type,
        )

        # Allow re-submission of an existing claim id (refreshes score).
        if claim_id in self._candidates:
            self._candidates[claim_id] = candidate
            return candidate

        unreviewed_count = sum(
            1 for c in self._candidates.values() if not c.reviewed
        )
        if unreviewed_count >= self._capacity:
            # Evict the least-uncertain unreviewed candidate.
            weakest_id: str | None = None
            weakest_score = float("inf")
            for cid, c in self._candidates.items():
                if c.reviewed:
                    continue
                if c.uncertainty_score < weakest_score:
                    weakest_score = c.uncertainty_score
                    weakest_id = cid
            if weakest_id is None or uncertainty <= weakest_score:
                return None
            self._candidates.pop(weakest_id, None)

        self._candidates[claim_id] = candidate
        return candidate

    def get_review_queue(
        self,
        limit: int = 20,
        strategy: str | None = None,
    ) -> list[ReviewCandidate]:
        """Return the top-*limit* most-uncertain unreviewed candidates.

        Parameters
        ----------
        limit : int
            Maximum number of candidates to return.
        strategy : str | None
            If given, candidates are re-scored with this strategy before
            sorting.  Otherwise the stored (default-strategy) score is
            used as-is.
        """
        unreviewed = [c for c in self._candidates.values() if not c.reviewed]
        if strategy is not None and strategy != self._strategy:
            keyed = [
                (self.score(c.posterior, strategy=strategy), c) for c in unreviewed
            ]
            keyed.sort(key=lambda kv: kv[0], reverse=True)
            return [c for _, c in keyed[: max(0, limit)]]
        unreviewed.sort(key=lambda c: c.uncertainty_score, reverse=True)
        return unreviewed[: max(0, limit)]

    def mark_reviewed(self, claim_id: str, label: bool, reviewer: str) -> bool:
        """Mark *claim_id* as reviewed with *label* and *reviewer*.

        Returns ``True`` on success, ``False`` if the id is not queued.
        """
        c = self._candidates.get(claim_id)
        if c is None:
            return False
        c.reviewed = True
        c.label = bool(label)
        c.reviewer = str(reviewer)
        return True

    def export_labeled(self) -> list[LabeledClaim]:
        """Return all reviewed entries as :class:`LabeledClaim` objects.

        The exported claims are tagged with ``source='active_learning'``
        and carry the reviewer's identity in ``notes``.
        """
        out: list[LabeledClaim] = []
        for c in self._candidates.values():
            if not c.reviewed or c.label is None:
                continue
            out.append(
                LabeledClaim(
                    claim_text=c.claim_text,
                    claim_type=c.claim_type,
                    ground_truth_label=bool(c.label),
                    source="active_learning",
                    notes=f"reviewer={c.reviewer}",
                )
            )
        return out

    # -- Introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._candidates)

    def pending_count(self) -> int:
        """Number of unreviewed candidates currently queued."""
        return sum(1 for c in self._candidates.values() if not c.reviewed)

    def reviewed_count(self) -> int:
        """Number of reviewed candidates retained."""
        return sum(1 for c in self._candidates.values() if c.reviewed)

    def get(self, claim_id: str) -> ReviewCandidate | None:
        """Return the candidate with *claim_id*, or ``None`` if absent."""
        return self._candidates.get(claim_id)
