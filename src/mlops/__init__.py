"""ML-ops capabilities for hal-nemoFinder.

This package bundles production-grade operational tooling on top of the
core verification pipeline:

* :mod:`shadow_mode` -- side-by-side champion/challenger A/B testing.
* :mod:`cost_tracker` -- per-job and system-wide cost accounting.

Everything under ``src/mlops`` is optional: the core pipeline works
without any of it, but turning these on is how a production deployment
continuously measures and improves itself.
"""

from __future__ import annotations

from src.mlops.cost_tracker import CostReport, CostTracker
from src.mlops.shadow_mode import (
    ShadowComparisonReport,
    ShadowModeRouter,
    ShadowRecord,
    ShadowRecorder,
)

__all__ = [
    "CostReport",
    "CostTracker",
    "ShadowComparisonReport",
    "ShadowModeRouter",
    "ShadowRecord",
    "ShadowRecorder",
]
