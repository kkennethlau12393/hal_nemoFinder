"""Tests for :mod:`src.core.active_learning`."""

from __future__ import annotations

import math

import pytest

from src.core.active_learning import ReviewCandidate, UncertaintySampler
from src.models.enums import ClaimType, Verdict


def test_least_confident_score_peaks_at_half() -> None:
    sampler = UncertaintySampler(uncertainty_strategy="least_confident")
    assert sampler.score(0.5) == pytest.approx(0.5, abs=1e-9)
    assert sampler.score(0.5) > sampler.score(0.1)
    assert sampler.score(0.5) > sampler.score(0.9)


def test_entropy_strategy_monotonic() -> None:
    sampler = UncertaintySampler(uncertainty_strategy="entropy")
    s_mid = sampler.score(0.5)
    s_low = sampler.score(0.1)
    s_high = sampler.score(0.9)
    assert s_mid > s_low
    assert s_mid > s_high
    assert math.isclose(s_mid, 1.0, abs_tol=1e-6)


def test_margin_strategy_near_thresholds() -> None:
    sampler = UncertaintySampler(uncertainty_strategy="margin")
    # Posterior exactly at a Bayes threshold -> max uncertainty (1.0).
    assert sampler.score(0.4) == pytest.approx(1.0)
    assert sampler.score(0.7) == pytest.approx(1.0)
    # Far from either threshold -> lower score.
    assert sampler.score(0.0) < sampler.score(0.35)


def test_queue_ordering_by_uncertainty() -> None:
    sampler = UncertaintySampler(
        review_queue_size=10, uncertainty_strategy="least_confident"
    )
    # Three claims with increasing uncertainty.
    sampler.add_candidate("a", "low uncertainty", 0.05, Verdict.verified)
    sampler.add_candidate("b", "mid uncertainty", 0.30, Verdict.partially_supported)
    sampler.add_candidate("c", "max uncertainty", 0.50, Verdict.partially_supported)

    queue = sampler.get_review_queue(limit=10)
    ids = [c.claim_id for c in queue]
    # Most-uncertain (c) must come first; least-uncertain (a) last.
    assert ids[0] == "c"
    assert ids[-1] == "a"
    for c in queue:
        assert isinstance(c, ReviewCandidate)


def test_queue_eviction_when_full() -> None:
    sampler = UncertaintySampler(
        review_queue_size=2, uncertainty_strategy="least_confident"
    )
    sampler.add_candidate("a", "x", 0.05, Verdict.verified)  # low uncertainty
    sampler.add_candidate("b", "x", 0.50, Verdict.partially_supported)
    # Adding a higher-uncertainty candidate should evict 'a'.
    sampler.add_candidate("c", "x", 0.45, Verdict.partially_supported)
    ids = {c.claim_id for c in sampler.get_review_queue(limit=10)}
    assert "a" not in ids
    assert {"b", "c"}.issubset(ids)


def test_mark_reviewed_and_export() -> None:
    sampler = UncertaintySampler(review_queue_size=10)
    sampler.add_candidate(
        "a",
        "Aspirin binds COX-3.",
        0.5,
        Verdict.partially_supported,
        claim_type=ClaimType.target_interaction,
    )
    assert sampler.mark_reviewed("a", True, "alice") is True
    assert sampler.mark_reviewed("missing", True, "alice") is False

    # Reviewed entries must be excluded from the unreviewed queue.
    assert sampler.pending_count() == 0
    assert sampler.reviewed_count() == 1

    exported = sampler.export_labeled()
    assert len(exported) == 1
    lc = exported[0]
    assert lc.claim_text.startswith("Aspirin")
    assert lc.ground_truth_label is True
    assert lc.claim_type == ClaimType.target_interaction
    assert "alice" in lc.notes
