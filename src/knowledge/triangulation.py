"""Cross-database triangulation service.

Implements the "two-clocks principle": when verifying a compound, query
multiple independent authoritative sources (PubChem, ChEMBL, DrugBank) and
flag disagreements. If two independent clocks of truth disagree, something
is wrong upstream — either in our claim, in one of the sources, or in the
normalization layer between them.

Used by the ChemicalVerifier to enrich the evidence chain and by downstream
audit pipelines to quantify source agreement as a confidence signal.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from .chembl import ChEMBLClient, MoleculeInfo
from .drugbank import DrugBankClient, DrugInfo
from .pubchem import CompoundInfo, PubChemClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_MW_TOLERANCE_DA = 0.5
_LOGP_TOLERANCE = 0.3

_SEVERITY_MINOR = "minor"
_SEVERITY_MODERATE = "moderate"
_SEVERITY_MAJOR = "major"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Disagreement:
    """A structured record of a single cross-source disagreement.

    Attributes
    ----------
    field:
        The property name on which sources disagree (e.g. ``"mw"``).
    values:
        Mapping of source name to the value that source reported.
    magnitude:
        Numerical spread of the disagreement (``max - min``) for numeric
        fields. Zero for non-numeric fields.
    severity:
        One of ``"minor"``, ``"moderate"``, ``"major"``.
    outlier:
        Name of the source that appears to be the outlier, if identifiable
        (only meaningful when 3+ sources have data).
    """

    field: str
    values: dict[str, Any]
    magnitude: float
    severity: str
    outlier: str | None = None


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    """Structured result of a cross-database triangulation query.

    Attributes
    ----------
    query:
        The original query string (name or SMILES).
    sources_consulted:
        All databases that were queried.
    sources_with_data:
        Subset that actually returned a hit.
    consensus_mw:
        Median molecular weight across sources with data.
    consensus_logp:
        Median logP across sources with data.
    disagreements:
        List of :class:`Disagreement` records for any fields with
        out-of-tolerance variance.
    confidence:
        Agreement-based confidence score in ``[0, 1]``. Higher means more
        sources agreed within tolerance.
    raw:
        Dict of source name to the raw record returned by that source
        (kept for audit trail).
    """

    query: str
    sources_consulted: list[str]
    sources_with_data: list[str]
    consensus_mw: float | None
    consensus_logp: float | None
    disagreements: list[Disagreement]
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_agree(self) -> bool:
        """True if every source with data agrees within tolerance."""
        return not self.disagreements and len(self.sources_with_data) >= 2

    @property
    def has_disagreement(self) -> bool:
        """True if any disagreement was recorded."""
        return bool(self.disagreements)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TriangulationService:
    """Coordinates cross-database queries and detects source disagreements.

    The service is designed for graceful degradation: any subset of the
    three underlying clients may be missing or fail at runtime. The result
    will reflect only the sources that successfully returned data.
    """

    def __init__(
        self,
        pubchem: PubChemClient | None,
        chembl: ChEMBLClient | None,
        drugbank: DrugBankClient | None,
    ) -> None:
        self._pubchem = pubchem
        self._chembl = chembl
        self._drugbank = drugbank

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def triangulate_compound(self, name_or_smiles: str) -> TriangulationResult:
        """Query PubChem, ChEMBL and DrugBank for the same compound.

        Parameters
        ----------
        name_or_smiles:
            A common drug name or a SMILES string. Heuristics decide
            which lookup mode to use on a per-client basis.

        Returns
        -------
        TriangulationResult
            Consolidated cross-source record with consensus values,
            disagreements, and a confidence score.
        """
        query = name_or_smiles.strip()
        is_smiles = self._looks_like_smiles(query)

        sources_consulted: list[str] = []
        raw: dict[str, Any] = {}
        values_mw: dict[str, float] = {}
        values_logp: dict[str, float] = {}

        # --- PubChem ------------------------------------------------------
        if self._pubchem is not None:
            sources_consulted.append("pubchem")
            try:
                if is_smiles:
                    pc: CompoundInfo | None = await self._pubchem.search_by_smiles(query)
                else:
                    pc = await self._pubchem.search_by_name(query)
                if pc is not None:
                    raw["pubchem"] = pc
                    if pc.molecular_weight:
                        values_mw["pubchem"] = float(pc.molecular_weight)
                    # PubChem CompoundInfo does not currently expose logP.
            except Exception as exc:
                logger.warning("PubChem triangulation failed for %r: %s", query, exc)

        # --- ChEMBL -------------------------------------------------------
        if self._chembl is not None:
            sources_consulted.append("chembl")
            try:
                results: list[MoleculeInfo] = await self._chembl.search_molecule(query)
                if results:
                    mol = results[0]
                    raw["chembl"] = mol
                    if mol.molecular_weight is not None:
                        values_mw["chembl"] = float(mol.molecular_weight)
            except Exception as exc:
                logger.warning("ChEMBL triangulation failed for %r: %s", query, exc)

        # --- DrugBank -----------------------------------------------------
        if self._drugbank is not None:
            sources_consulted.append("drugbank")
            try:
                if is_smiles:
                    db: DrugInfo | None = await self._drugbank.get_by_smiles(query)
                else:
                    db = await self._drugbank.get_drug(query)
                if db is not None:
                    raw["drugbank"] = db
                    if db.molecular_weight is not None:
                        values_mw["drugbank"] = float(db.molecular_weight)
                    if db.logP is not None:
                        values_logp["drugbank"] = float(db.logP)
            except Exception as exc:
                logger.warning("DrugBank triangulation failed for %r: %s", query, exc)

        sources_with_data = sorted(raw.keys())

        disagreements: list[Disagreement] = []
        d = self._analyze_field("mw", values_mw, _MW_TOLERANCE_DA)
        if d is not None:
            disagreements.append(d)
        d = self._analyze_field("logp", values_logp, _LOGP_TOLERANCE)
        if d is not None:
            disagreements.append(d)

        consensus_mw = (
            statistics.median(values_mw.values()) if values_mw else None
        )
        consensus_logp = (
            statistics.median(values_logp.values()) if values_logp else None
        )

        confidence = self._compute_confidence(
            n_sources_with_data=len(sources_with_data),
            n_disagreements=len(disagreements),
            n_fields_checked=sum(1 for v in (values_mw, values_logp) if v),
        )

        return TriangulationResult(
            query=query,
            sources_consulted=sources_consulted,
            sources_with_data=sources_with_data,
            consensus_mw=consensus_mw,
            consensus_logp=consensus_logp,
            disagreements=disagreements,
            confidence=confidence,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_smiles(s: str) -> bool:
        """Heuristic: does this string look like SMILES rather than a name?"""
        if not s:
            return False
        # SMILES very commonly contains these structural chars; plain names don't.
        structural_chars = set("()[]=#/\\@+-")
        if any(c in structural_chars for c in s):
            return True
        # Names are usually alphabetic with maybe spaces/hyphens.
        if s.replace("-", "").replace(" ", "").isalpha():
            return False
        return False

    @staticmethod
    def _analyze_field(
        field_name: str,
        values: dict[str, float],
        tolerance: float,
    ) -> Disagreement | None:
        """Check if the per-source values for a numeric field disagree.

        Returns ``None`` if there is no disagreement (fewer than two
        sources, or all within ``tolerance`` of each other). Otherwise
        returns a :class:`Disagreement` with severity and likely outlier.
        """
        if len(values) < 2:
            return None

        lo = min(values.values())
        hi = max(values.values())
        magnitude = hi - lo

        if magnitude <= tolerance:
            return None

        # Severity buckets: magnitude relative to tolerance
        if magnitude <= tolerance * 3:
            severity = _SEVERITY_MINOR
        elif magnitude <= tolerance * 10:
            severity = _SEVERITY_MODERATE
        else:
            severity = _SEVERITY_MAJOR

        outlier: str | None = None
        if len(values) >= 3:
            median = statistics.median(values.values())
            # Outlier = source whose deviation from median is largest AND
            # whose removal brings remaining values within tolerance.
            ranked = sorted(
                values.items(),
                key=lambda kv: abs(kv[1] - median),
                reverse=True,
            )
            candidate, _ = ranked[0]
            remaining = [v for k, v in values.items() if k != candidate]
            if remaining and (max(remaining) - min(remaining)) <= tolerance:
                outlier = candidate

        return Disagreement(
            field=field_name,
            values=dict(values),
            magnitude=round(magnitude, 4),
            severity=severity,
            outlier=outlier,
        )

    @staticmethod
    def _compute_confidence(
        n_sources_with_data: int,
        n_disagreements: int,
        n_fields_checked: int,
    ) -> float:
        """Compute a 0..1 confidence score from agreement statistics.

        * 0 sources   -> 0.0
        * 1 source    -> 0.4  (no cross-check possible)
        * 2 sources   -> 0.75 baseline
        * 3+ sources  -> 0.9 baseline
        Each disagreement subtracts 0.2, floored at 0.1.
        """
        if n_sources_with_data == 0:
            return 0.0
        if n_sources_with_data == 1:
            return 0.4
        base = 0.75 if n_sources_with_data == 2 else 0.9
        penalty = 0.2 * n_disagreements
        return max(0.1, round(base - penalty, 3))
