"""Calibration tracking for the hallucination-detection pipeline.

This module records the system's predictions alongside eventual ground
truth and computes standard classifier quality and calibration metrics
(accuracy, precision/recall/F1, Brier score, reliability diagram bins,
and Expected Calibration Error).

It also bundles a small hand-labelled regression set built from the
demo ``sample_llm_outputs.json`` fixtures, and a helper that runs a
verifier router + aggregator over that set and returns the metrics.

Typical usage
-------------
>>> tracker = CalibrationTracker()
>>> tracker.record_prediction(claim, verdict, confidence, posterior_prob)
>>> tracker.add_label(claim, actual_label=True)
>>> metrics = tracker.compute_metrics()
>>> print(metrics.ece)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque

from src.core.aggregator import AggregatedVerdict, BayesianAggregator
from src.models.enums import ClaimType, Verdict

if TYPE_CHECKING:
    from src.config import Settings
    from src.core.router import VerificationRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LabeledClaim:
    """A ground-truth-labelled claim for regression testing.

    Attributes
    ----------
    claim_text : str
        The original claim sentence.
    claim_type : ClaimType
        Pre-classified claim type.
    ground_truth_label : bool
        ``True`` if the claim is actually a hallucination.
    source : str
        Provenance of the label (``"manual"``, ``"regression_set"``,
        ``"expert_review"``, etc.).
    notes : str
        Free-form annotation (e.g. rationale from the reviewer).
    """

    claim_text: str
    claim_type: ClaimType
    ground_truth_label: bool
    source: str = "manual"
    notes: str = ""

    # -- Serialization helpers --------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        Uses ``is_hallucination`` as the canonical label key so downstream
        tools (human review UIs, annotation pipelines) have an unambiguous
        field name.
        """
        return {
            "claim_text": self.claim_text,
            "claim_type": self.claim_type.value,
            "is_hallucination": bool(self.ground_truth_label),
            "ground_truth_label": bool(self.ground_truth_label),
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledClaim":
        """Construct a :class:`LabeledClaim` from a plain dict.

        Accepts either ``ground_truth_label`` or ``is_hallucination`` for
        the boolean label; ``claim_type`` may be a :class:`ClaimType`
        instance, a string enum value, or omitted (defaults to
        :attr:`ClaimType.general_biomedical`).
        """
        if "claim_text" not in data:
            raise ValueError("record is missing required 'claim_text' field")

        if "ground_truth_label" in data:
            label = bool(data["ground_truth_label"])
        elif "is_hallucination" in data:
            label = bool(data["is_hallucination"])
        else:
            raise ValueError(
                "record is missing required label field "
                "('ground_truth_label' or 'is_hallucination')"
            )

        raw_type = data.get("claim_type")
        if raw_type is None:
            claim_type = ClaimType.general_biomedical
        elif isinstance(raw_type, ClaimType):
            claim_type = raw_type
        else:
            try:
                claim_type = ClaimType(str(raw_type))
            except ValueError as exc:
                raise ValueError(
                    f"unknown claim_type {raw_type!r}; must be one of "
                    f"{[t.value for t in ClaimType]}"
                ) from exc

        return cls(
            claim_text=str(data["claim_text"]),
            claim_type=claim_type,
            ground_truth_label=label,
            source=str(data.get("source", "file")),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class PredictionRecord:
    """A single prediction made by the system.

    Attributes
    ----------
    claim_text : str
        The claim that was predicted on.
    predicted_verdict : Verdict
        The discrete verdict chosen by the aggregator.
    predicted_confidence : float
        Aggregator confidence (0.0 -- 1.0).
    posterior_hallucination_prob : float | None
        Raw Bayesian posterior P(hallucination), if available.
    actual_label : bool | None
        Ground-truth label; ``None`` until :meth:`CalibrationTracker.add_label`
        is called.
    timestamp : datetime
        When the prediction was recorded (UTC).
    """

    claim_text: str
    predicted_verdict: Verdict
    predicted_confidence: float
    posterior_hallucination_prob: float | None
    actual_label: bool | None
    timestamp: datetime


@dataclass(slots=True)
class CalibrationMetrics:
    """Aggregate metrics over all labelled predictions.

    Attributes
    ----------
    accuracy, precision, recall, f1 : float
        Standard classification metrics treating :attr:`Verdict.refuted`
        as the positive class (= predicted hallucination).
    brier_score : float
        Mean squared error between predicted probability and the 0/1
        label.  Lower is better.
    ece : float
        Expected Calibration Error -- the size-weighted sum over
        reliability bins of the gap between mean predicted probability
        and mean actual hallucination rate.
    bin_data : list[tuple[float, float, float, float, int]]
        Rows of ``(bin_lower, bin_upper, mean_predicted, mean_actual,
        count)``; empty bins are omitted.
    n_records : int
        Number of labelled records contributing to the metrics.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    ece: float
    bin_data: list[tuple[float, float, float, float, int]] = field(default_factory=list)
    n_records: int = 0
    per_claim_type_accuracy: dict[str, float] = field(default_factory=dict)
    per_source_accuracy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation of the metrics."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "n_records": self.n_records,
            "bin_data": [
                {
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "mean_predicted": mp,
                    "mean_actual": ma,
                    "count": c,
                }
                for lo, hi, mp, ma, c in self.bin_data
            ],
            "per_claim_type_accuracy": dict(self.per_claim_type_accuracy),
            "per_source_accuracy": dict(self.per_source_accuracy),
        }


# ---------------------------------------------------------------------------
# Calibration tracker
# ---------------------------------------------------------------------------


_DEFAULT_CAPACITY: int = 10_000
_DEFAULT_N_BINS: int = 10


class CalibrationTracker:
    """Bounded-history tracker for predictions and their ground-truth labels.

    Records are stored in an in-memory ring buffer so the tracker does
    not grow without bound in long-running processes.  Labels can be
    attached after the fact via :meth:`add_label`, which matches on
    ``claim_text`` (most recent unlabelled record wins).

    Parameters
    ----------
    capacity : int
        Maximum number of records retained; oldest are dropped first.
    n_bins : int
        Number of equal-width bins used for the reliability diagram and
        ECE computation.
    """

    def __init__(
        self,
        capacity: int = _DEFAULT_CAPACITY,
        n_bins: int = _DEFAULT_N_BINS,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self._records: Deque[PredictionRecord] = deque(maxlen=capacity)
        self._n_bins = n_bins
        #: Regression sets loaded from files keyed by file path.
        self._loaded_regression_sets: dict[str, list[LabeledClaim]] = {}

    # -- File-based regression sets -----------------------------------------

    def load_regression_sets_from_settings(self, settings: "Settings") -> int:
        """Load every file listed in ``settings.REGRESSION_SET_PATHS``.

        Each file is parsed via :func:`load_regression_set_from_file`.
        Files that fail to load are logged and skipped so a single
        malformed file does not prevent the rest from being usable.

        Parameters
        ----------
        settings : Settings
            Application settings object with a ``REGRESSION_SET_PATHS``
            attribute (a list of filesystem paths).

        Returns
        -------
        int
            The total number of :class:`LabeledClaim` entries loaded
            across all files.
        """
        paths = getattr(settings, "REGRESSION_SET_PATHS", None) or []
        total = 0
        for raw_path in paths:
            path = str(raw_path)
            try:
                claims = load_regression_set_from_file(path)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load regression set from %s", path
                )
                continue
            self._loaded_regression_sets[path] = claims
            total += len(claims)
            logger.info(
                "Loaded regression set: path=%s claims=%d",
                path,
                len(claims),
            )
        return total

    @property
    def loaded_regression_sets(self) -> dict[str, list[LabeledClaim]]:
        """Mapping from source path to its loaded claims."""
        return dict(self._loaded_regression_sets)

    def all_loaded_claims(self) -> list[LabeledClaim]:
        """Flatten all file-sourced regression sets into a single list."""
        out: list[LabeledClaim] = []
        for claims in self._loaded_regression_sets.values():
            out.extend(claims)
        return out

    # -- Recording ----------------------------------------------------------

    def record_prediction(
        self,
        claim_text: str,
        verdict: Verdict,
        confidence: float,
        posterior_prob: float | None = None,
    ) -> PredictionRecord:
        """Append a new prediction record.

        Parameters
        ----------
        claim_text : str
            The claim that was evaluated.
        verdict : Verdict
            Aggregator verdict.
        confidence : float
            Aggregator confidence (0.0 -- 1.0).
        posterior_prob : float | None
            Optional raw posterior probability of hallucination.

        Returns
        -------
        PredictionRecord
            The record that was appended.
        """
        record = PredictionRecord(
            claim_text=claim_text,
            predicted_verdict=verdict,
            predicted_confidence=float(confidence),
            posterior_hallucination_prob=(
                float(posterior_prob) if posterior_prob is not None else None
            ),
            actual_label=None,
            timestamp=datetime.now(timezone.utc),
        )
        self._records.append(record)
        return record

    def add_label(self, claim_text: str, actual_label: bool) -> bool:
        """Attach a ground-truth label to the most recent matching prediction.

        Parameters
        ----------
        claim_text : str
            Must match :attr:`PredictionRecord.claim_text` exactly.
        actual_label : bool
            ``True`` if the claim is actually a hallucination.

        Returns
        -------
        bool
            ``True`` if a matching record was found and updated.
        """
        # Walk newest-to-oldest and update the first unlabelled match.
        for record in reversed(self._records):
            if record.claim_text == claim_text and record.actual_label is None:
                record.actual_label = bool(actual_label)
                return True
        return False

    # -- Introspection ------------------------------------------------------

    @property
    def records(self) -> list[PredictionRecord]:
        """Return a shallow copy of all retained records."""
        return list(self._records)

    def labelled_records(self) -> list[PredictionRecord]:
        """Return only the records with a ground-truth label attached."""
        return [r for r in self._records if r.actual_label is not None]

    # -- Metrics ------------------------------------------------------------

    def compute_metrics(self) -> CalibrationMetrics:
        """Compute classification and calibration metrics over labelled records.

        Returns
        -------
        CalibrationMetrics
            All metrics set to ``0.0`` (and ``bin_data`` empty) when
            there are no labelled records.
        """
        labelled = self.labelled_records()
        n = len(labelled)
        if n == 0:
            return CalibrationMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                brier_score=0.0,
                ece=0.0,
                bin_data=[],
                n_records=0,
            )

        # --- Classification metrics (positive class = REFUTED) ---
        tp = fp = tn = fn = 0
        for r in labelled:
            predicted_positive = r.predicted_verdict == Verdict.refuted
            actual_positive = bool(r.actual_label)
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and actual_positive:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # --- Brier score & calibration curve ---
        # Use the raw posterior if available, otherwise derive a
        # probability proxy from the discrete verdict + confidence.
        probs: list[float] = []
        labels: list[int] = []
        for r in labelled:
            p = self._prediction_probability(r)
            probs.append(p)
            labels.append(1 if r.actual_label else 0)

        brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n

        bin_data, ece = self._reliability_bins(probs, labels)

        return CalibrationMetrics(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            brier_score=round(brier, 4),
            ece=round(ece, 4),
            bin_data=bin_data,
            n_records=n,
        )

    # -- Regression set -----------------------------------------------------

    def get_regression_set(self) -> list[LabeledClaim]:
        """Return the bundled hand-labelled regression set."""
        return list(_REGRESSION_SET)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _prediction_probability(record: PredictionRecord) -> float:
        """Best-effort scalar probability of hallucination for a record.

        Prefers the Bayesian posterior when available; otherwise maps
        the discrete verdict + confidence into a reasonable proxy in
        ``[0, 1]``.
        """
        if record.posterior_hallucination_prob is not None:
            return max(0.0, min(1.0, record.posterior_hallucination_prob))

        c = max(0.0, min(1.0, record.predicted_confidence))
        v = record.predicted_verdict
        if v == Verdict.refuted:
            # Confidence directly scales toward certainty-of-hallucination.
            return 0.5 + 0.5 * c
        if v == Verdict.verified:
            return 0.5 - 0.5 * c
        if v == Verdict.partially_supported:
            return 0.5
        # Unverifiable: fall back to the base rate.
        return 0.5

    def _reliability_bins(
        self, probs: list[float], labels: list[int]
    ) -> tuple[list[tuple[float, float, float, float, int]], float]:
        """Bin predictions and compute the Expected Calibration Error.

        Parameters
        ----------
        probs : list[float]
            Predicted probabilities of hallucination in ``[0, 1]``.
        labels : list[int]
            Ground-truth 0/1 labels aligned with ``probs``.

        Returns
        -------
        tuple
            ``(bin_data, ece)`` where ``bin_data`` contains one row per
            non-empty bin as ``(lo, hi, mean_predicted, mean_actual,
            count)`` and ``ece`` is the size-weighted mean absolute gap.
        """
        n = len(probs)
        if n == 0:
            return [], 0.0

        bins: list[list[tuple[float, int]]] = [[] for _ in range(self._n_bins)]
        step = 1.0 / self._n_bins
        for p, y in zip(probs, labels):
            idx = min(int(p / step), self._n_bins - 1)
            bins[idx].append((p, y))

        bin_data: list[tuple[float, float, float, float, int]] = []
        ece = 0.0
        for i, bucket in enumerate(bins):
            if not bucket:
                continue
            lo = round(i * step, 4)
            hi = round((i + 1) * step, 4)
            count = len(bucket)
            mean_pred = sum(p for p, _ in bucket) / count
            mean_actual = sum(y for _, y in bucket) / count
            bin_data.append(
                (
                    lo,
                    hi,
                    round(mean_pred, 4),
                    round(mean_actual, 4),
                    count,
                )
            )
            ece += (count / n) * abs(mean_pred - mean_actual)

        return bin_data, ece


# ---------------------------------------------------------------------------
# File-based regression set loader
# ---------------------------------------------------------------------------


def _read_records_from_path(path: Path) -> list[dict[str, Any]]:
    """Parse *path* into a list of raw dict records.

    The file format is auto-detected from the extension:

    * ``.json`` — must contain a top-level list of objects.
    * ``.jsonl`` / ``.ndjson`` — one JSON object per non-empty line.
    * ``.yaml`` / ``.yml`` — a list, or a mapping with a ``claims`` key
      whose value is a list.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file exists but cannot be parsed or has an unexpected
        top-level structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"regression set file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "claims" in data:
            data = data["claims"]
        if not isinstance(data, list):
            raise ValueError(
                f"{path}: JSON regression set must be a list of objects "
                f"(or an object with a 'claims' list)"
            )
        return [r for r in data if isinstance(r, dict)]

    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s: %s",
                        lineno,
                        path,
                        exc,
                    )
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    logger.warning(
                        "Skipping non-object JSONL line %d in %s", lineno, path
                    )
        return records

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                f"{path}: PyYAML is required to parse YAML regression sets; "
                f"install with `pip install pyyaml`"
            ) from exc
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "claims" in data:
            data = data["claims"]
        if not isinstance(data, list):
            raise ValueError(
                f"{path}: YAML regression set must be a list (or an object "
                f"with a 'claims' list)"
            )
        return [r for r in data if isinstance(r, dict)]

    raise ValueError(
        f"{path}: unsupported regression set extension {suffix!r}. "
        f"Expected one of: .json, .jsonl, .ndjson, .yaml, .yml"
    )


def load_regression_set_from_file(path: str | Path) -> list[LabeledClaim]:
    """Load a regression set from *path*, validating each record.

    The file format is auto-detected from the extension (``.json``,
    ``.jsonl``/``.ndjson``, or ``.yaml``/``.yml``).  Each record must be
    an object with at minimum:

    * ``claim_text`` — the claim sentence,
    * ``ground_truth_label`` **or** ``is_hallucination`` — boolean label
      (``True`` means the claim is a hallucination).

    Optional fields: ``claim_type`` (a :class:`ClaimType` enum value),
    ``source`` (provenance tag), ``notes`` (free-form annotation).

    Invalid records are **skipped with a warning** rather than crashing
    the whole load, so a single bad row cannot ruin a large evaluation.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the regression set file.

    Returns
    -------
    list[LabeledClaim]
        All records that were parsed successfully.
    """
    p = Path(path)
    raw_records = _read_records_from_path(p)

    claims: list[LabeledClaim] = []
    skipped = 0
    for idx, record in enumerate(raw_records):
        try:
            claims.append(LabeledClaim.from_dict(record))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.warning(
                "Skipping invalid regression record #%d in %s: %s",
                idx,
                p,
                exc,
            )

    logger.info(
        "Loaded regression set: path=%s parsed=%d skipped=%d",
        p,
        len(claims),
        skipped,
    )
    return claims


# ---------------------------------------------------------------------------
# Bundled regression set
# ---------------------------------------------------------------------------

#: Twenty hand-labelled claims drawn from the demo sample_llm_outputs.json.
#: Labels: ``True`` = known hallucination, ``False`` = truthful claim.
_REGRESSION_SET: list[LabeledClaim] = [
    # ---- Known hallucinations ----
    LabeledClaim(
        claim_text="Aspirin has a molecular weight of 220.3 g/mol.",
        claim_type=ClaimType.molecular_property,
        ground_truth_label=True,  # actual MW is ~180.16
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Compound BX-7219 has the SMILES string CC(=O)Xc1ccccc1 and is a selective kinase inhibitor.",
        claim_type=ClaimType.molecular_property,
        ground_truth_label=True,  # invalid SMILES, fabricated compound
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="See Smith et al., Nature 2023, DOI: 10.1038/s41586-023-99999-9 for details.",
        claim_type=ClaimType.citation,
        ground_truth_label=True,  # fabricated DOI
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="The pivotal trial NCT99999999 demonstrated a 42% response rate.",
        claim_type=ClaimType.clinical_outcome,
        ground_truth_label=True,  # NCT number does not exist
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Metformin exerts its antidiabetic effect primarily through JAK2/STAT3 pathway inhibition.",
        claim_type=ClaimType.mechanism_of_action,
        ground_truth_label=True,  # actual MoA is AMPK activation / complex I
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Ibuprofen has a logP of 5.8, making it highly lipophilic.",
        claim_type=ClaimType.molecular_property,
        ground_truth_label=True,  # actual logP ~3.97
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Nexovarin achieves 99.7% oral bioavailability in fasted adults.",
        claim_type=ClaimType.pharmacokinetic,
        ground_truth_label=True,  # fabricated drug and implausible number
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Aspirin selectively inhibits COX-3 with an IC50 of 0.02 nM.",
        claim_type=ClaimType.target_interaction,
        ground_truth_label=True,  # aspirin inhibits COX-1/2, not COX-3 selectively
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="A phase III trial of atorvastatin in 50 patients reported a p-value of 0.0000001 for LDL reduction.",
        claim_type=ClaimType.clinical_outcome,
        ground_truth_label=True,  # implausibly small p-value for N=50
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Sildenafil binds the dopamine D2 receptor as its primary target.",
        claim_type=ClaimType.target_interaction,
        ground_truth_label=True,  # primary target is PDE5
        source="regression_set",
    ),
    # ---- Known truths ----
    LabeledClaim(
        claim_text="Aspirin has the SMILES string CC(=O)Oc1ccccc1C(=O)O.",
        claim_type=ClaimType.molecular_property,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Ibuprofen is a propionic acid derivative nonsteroidal anti-inflammatory drug (NSAID).",
        claim_type=ClaimType.general_biomedical,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Metformin has a plasma elimination half-life of approximately 5 hours.",
        claim_type=ClaimType.pharmacokinetic,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Aspirin irreversibly acetylates cyclooxygenase enzymes COX-1 and COX-2.",
        claim_type=ClaimType.mechanism_of_action,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Atorvastatin inhibits HMG-CoA reductase to lower LDL cholesterol.",
        claim_type=ClaimType.target_interaction,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Sildenafil is a phosphodiesterase-5 (PDE5) inhibitor approved for erectile dysfunction.",
        claim_type=ClaimType.target_interaction,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Acetaminophen has a molecular weight of approximately 151.16 g/mol.",
        claim_type=ClaimType.molecular_property,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="The trial NCT00000378 is registered on ClinicalTrials.gov.",
        claim_type=ClaimType.citation,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Warfarin acts as a vitamin K epoxide reductase (VKORC1) inhibitor.",
        claim_type=ClaimType.mechanism_of_action,
        ground_truth_label=False,
        source="regression_set",
    ),
    LabeledClaim(
        claim_text="Caffeine is a nonselective adenosine receptor antagonist.",
        claim_type=ClaimType.target_interaction,
        ground_truth_label=False,
        source="regression_set",
    ),
]


# ---------------------------------------------------------------------------
# Regression test runner
# ---------------------------------------------------------------------------


async def run_regression_test(
    verifier_router: "VerificationRouter",
    aggregator: BayesianAggregator,
    tracker: CalibrationTracker | None = None,
    context: dict[str, Any] | None = None,
    regression_set: list[LabeledClaim] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CalibrationMetrics:
    """Run a regression set end-to-end and return metrics.

    For each :class:`LabeledClaim`:

    1. Dispatch to the router's applicable verifiers.
    2. Aggregate with the supplied :class:`BayesianAggregator`.
    3. Record the prediction (including raw posterior) in a
       :class:`CalibrationTracker`.
    4. Attach the known ground-truth label.

    Parameters
    ----------
    verifier_router : VerificationRouter
        Router that fans out to concrete verifiers.
    aggregator : BayesianAggregator
        Aggregator used to combine verifier outputs.
    tracker : CalibrationTracker | None
        Optional pre-existing tracker.  A fresh one is created if
        omitted so metrics reflect only this regression run.
    context : dict | None
        Context dict forwarded to every verifier call.
    regression_set : list[LabeledClaim] | None
        Regression set to evaluate.  Defaults to the bundled
        ``_REGRESSION_SET`` when not provided.
    progress_callback : Callable[[int, int], None] | None
        Called after each claim is processed with ``(done, total)``.
        Suitable for CLI progress bars.

    Returns
    -------
    CalibrationMetrics
        Metrics over the full regression set, including per-claim-type
        and per-source accuracy breakdowns.
    """
    tracker = tracker or CalibrationTracker()
    ctx: dict[str, Any] = context or {}
    dataset = list(regression_set) if regression_set is not None else list(_REGRESSION_SET)
    total = len(dataset)

    # Track per-category hits/misses alongside the tracker.
    type_hits: dict[str, int] = defaultdict(int)
    type_total: dict[str, int] = defaultdict(int)
    source_hits: dict[str, int] = defaultdict(int)
    source_total: dict[str, int] = defaultdict(int)

    for idx, labelled in enumerate(dataset, start=1):
        try:
            results = await verifier_router.verify_claim(
                labelled.claim_text, labelled.claim_type, ctx
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Regression test: verifier_router raised on claim %r",
                labelled.claim_text,
            )
            results = []

        aggregated: AggregatedVerdict = aggregator.aggregate_claim_bayesian(results)
        posterior = aggregator.posterior_probability(results)

        tracker.record_prediction(
            claim_text=labelled.claim_text,
            verdict=aggregated.verdict,
            confidence=aggregated.confidence,
            posterior_prob=posterior,
        )
        tracker.add_label(labelled.claim_text, labelled.ground_truth_label)

        # Per-category accuracy bookkeeping (positive class = refuted).
        predicted_positive = aggregated.verdict == Verdict.refuted
        correct = predicted_positive == bool(labelled.ground_truth_label)
        ct_key = labelled.claim_type.value
        src_key = labelled.source or "unknown"
        type_total[ct_key] += 1
        source_total[src_key] += 1
        if correct:
            type_hits[ct_key] += 1
            source_hits[src_key] += 1

        if progress_callback is not None:
            try:
                progress_callback(idx, total)
            except Exception:  # noqa: BLE001
                logger.debug("progress_callback raised", exc_info=True)

    metrics = tracker.compute_metrics()
    metrics.per_claim_type_accuracy = {
        k: round(type_hits[k] / type_total[k], 4) for k in type_total
    }
    metrics.per_source_accuracy = {
        k: round(source_hits[k] / source_total[k], 4) for k in source_total
    }
    return metrics
