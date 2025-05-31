"""Tests for :mod:`src.core.drift`."""

from __future__ import annotations

from src.core.drift import DriftDetector, DriftReport
from src.models.enums import Verdict


def _record_stream(
    detector: DriftDetector,
    verifier: str,
    correct_count: int,
    wrong_count: int,
    positive: bool = True,
) -> None:
    """Feed the detector a batch of labelled predictions.

    Each "correct" prediction has the verdict matching the label; each
    "wrong" one has the opposite verdict so the detector sees a specific
    accuracy rate.
    """
    for _ in range(correct_count):
        label = positive
        predicted = Verdict.refuted if label else Verdict.verified
        detector.record(verifier, predicted, label, 0.9)
    for _ in range(wrong_count):
        label = positive
        predicted = Verdict.verified if label else Verdict.refuted
        detector.record(verifier, predicted, label, 0.9)


def test_no_drift_when_distribution_stable() -> None:
    detector = DriftDetector(
        window_size=50, baseline_size=200, significance_threshold=0.05
    )
    # Baseline: 90% correct.
    _record_stream(detector, "chemical", 180, 20)
    # Window: also 90% correct.
    _record_stream(detector, "chemical", 45, 5)

    report = detector.check("chemical")
    assert report is None


def test_drift_fires_on_shifted_distribution() -> None:
    detector = DriftDetector(
        window_size=100, baseline_size=400, significance_threshold=0.05
    )
    # Baseline: 95% correct.
    _record_stream(detector, "chemical", 380, 20)
    # Window: 40% correct — big regression.
    _record_stream(detector, "chemical", 40, 60)

    report = detector.check("chemical")
    assert isinstance(report, DriftReport)
    assert report.verifier_name == "chemical"
    assert report.window_accuracy < report.baseline_accuracy
    assert report.delta < -0.1
    assert report.alert_level in {"warning", "critical"}
    assert report.p_value < 0.05


def test_insufficient_data_returns_none() -> None:
    detector = DriftDetector(window_size=50, baseline_size=200)
    _record_stream(detector, "pubchem", 10, 0)
    assert detector.check("pubchem") is None


def test_check_all_collects_all_verifiers() -> None:
    detector = DriftDetector(
        window_size=50, baseline_size=200, significance_threshold=0.05
    )
    # Drifting verifier.
    _record_stream(detector, "chemical", 190, 10)
    _record_stream(detector, "chemical", 10, 40)
    # Stable verifier.
    _record_stream(detector, "crossref", 180, 20)
    _record_stream(detector, "crossref", 45, 5)

    reports = detector.check_all()
    assert "chemical" in reports
    assert "crossref" not in reports
