"""Shared pytest fixtures for hal_nemoFinder unit tests."""

from __future__ import annotations

import pytest

from src.core.aggregator import ResultAggregator
from src.core.claim_classifier import ClaimClassifier
from src.core.claim_extractor import ClaimExtractor


@pytest.fixture()
def claim_extractor() -> ClaimExtractor:
    """Return a default ClaimExtractor instance."""
    return ClaimExtractor()


@pytest.fixture()
def claim_classifier() -> ClaimClassifier:
    """Return a default ClaimClassifier instance."""
    return ClaimClassifier()


@pytest.fixture()
def aggregator() -> ResultAggregator:
    """Return a default ResultAggregator instance."""
    return ResultAggregator()
