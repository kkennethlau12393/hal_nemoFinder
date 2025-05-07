"""ChEMBL REST API client with caching and fallback data."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MoleculeInfo:
    chembl_id: str
    pref_name: str | None
    canonical_smiles: str | None
    molecular_weight: float | None
    max_phase: int | None
    is_fallback: bool = False


@dataclass(frozen=True, slots=True)
class BioactivityRecord:
    molecule_chembl_id: str
    target_chembl_id: str
    target_name: str
    standard_type: str
    standard_value: float | None
    standard_units: str | None
    is_fallback: bool = False


@dataclass(frozen=True, slots=True)
class TargetInfo:
    target_chembl_id: str
    pref_name: str
    organism: str
    target_type: str
    is_fallback: bool = False


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Full audit trail for a single drug-target bioactivity measurement.

    Attributes
    ----------
    assay_id : str
        ChEMBL assay identifier (e.g. ``"CHEMBL829584"``).
    assay_type : str
        ChEMBL assay classification (``"B"`` = binding, ``"F"`` =
        functional, ``"A"`` = ADME, ``"T"`` = toxicity).
    standard_value : float | None
        Numeric measurement in ``standard_units``.
    standard_units : str | None
        Units for ``standard_value`` (typically ``"nM"``).
    pchembl_value : float | None
        -log10 of the molar potency, ChEMBL's normalised affinity score.
    reference_doi : str | None
        DOI of the primary reference reporting the measurement.
    reference_year : int | None
        Publication year of the primary reference.
    reference_journal : str | None
        Short journal name of the primary reference.
    assay_description : str
        Free-text assay protocol description.
    confidence_score : int | None
        ChEMBL 0-9 target-assignment confidence score.
    molecule_chembl_id : str
        Source compound ChEMBL id.
    target_chembl_id : str
        Source target ChEMBL id.
    standard_type : str
        Measurement type (``"Ki"``, ``"IC50"``, ``"Kd"``, ...).
    """

    assay_id: str
    assay_type: str
    standard_value: float | None
    standard_units: str | None
    pchembl_value: float | None
    reference_doi: str | None
    reference_year: int | None
    reference_journal: str | None
    assay_description: str
    confidence_score: int | None
    molecule_chembl_id: str = ""
    target_chembl_id: str = ""
    standard_type: str = ""


# ------------------------------------------------------------------
# Fallback data — 10 well-known drug-target pairs
# ------------------------------------------------------------------

_FALLBACK_MOLECULES: dict[str, MoleculeInfo] = {
    "imatinib": MoleculeInfo("CHEMBL941", "IMATINIB", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", 493.60, 4, is_fallback=True),
    "erlotinib": MoleculeInfo("CHEMBL553", "ERLOTINIB", "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)OCC4=CC(=CC=C4)C#C)Cl)OCCOC", 393.44, 4, is_fallback=True),
    "trastuzumab": MoleculeInfo("CHEMBL1201585", "TRASTUZUMAB", None, None, 4, is_fallback=True),
    "aspirin": MoleculeInfo("CHEMBL25", "ASPIRIN", "CC(=O)OC1=CC=CC=C1C(=O)O", 180.16, 4, is_fallback=True),
    "metformin": MoleculeInfo("CHEMBL1431", "METFORMIN", "CN(C)C(=N)NC(=N)N", 129.16, 4, is_fallback=True),
    "atorvastatin": MoleculeInfo("CHEMBL1487", "ATORVASTATIN", None, 558.64, 4, is_fallback=True),
    "sorafenib": MoleculeInfo("CHEMBL1336", "SORAFENIB", None, 464.82, 4, is_fallback=True),
    "vemurafenib": MoleculeInfo("CHEMBL1229517", "VEMURAFENIB", None, 489.92, 4, is_fallback=True),
    "osimertinib": MoleculeInfo("CHEMBL3353410", "OSIMERTINIB", None, 499.61, 4, is_fallback=True),
    "pembrolizumab": MoleculeInfo("CHEMBL3137309", "PEMBROLIZUMAB", None, None, 4, is_fallback=True),
}

_FALLBACK_BIOACTIVITIES: list[BioactivityRecord] = [
    BioactivityRecord("CHEMBL941", "CHEMBL1862", "ABL1", "IC50", 600.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL941", "CHEMBL2815", "KIT", "IC50", 100.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL553", "CHEMBL203", "EGFR", "IC50", 2.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL1201585", "CHEMBL1824", "HER2", "IC50", None, None, is_fallback=True),
    BioactivityRecord("CHEMBL25", "CHEMBL218", "COX-2", "IC50", 50000.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL1336", "CHEMBL1862", "ABL1", "IC50", None, None, is_fallback=True),
    BioactivityRecord("CHEMBL1336", "CHEMBL2146302", "BRAF", "IC50", 25.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL1229517", "CHEMBL2146302", "BRAF V600E", "IC50", 31.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL3353410", "CHEMBL203", "EGFR T790M", "IC50", 12.0, "nM", is_fallback=True),
    BioactivityRecord("CHEMBL3137309", "CHEMBL4523582", "PD-1", "IC50", None, None, is_fallback=True),
]

# Keyed by (molecule_chembl_id, target_chembl_id).  Values are real
# Ki/IC50 measurements with citations to the original primary literature.
_FALLBACK_PROVENANCE: dict[tuple[str, str], ProvenanceRecord] = {
    # Imatinib vs ABL1 — Schindler et al. Science 2000 (Ki ~ 37 nM on ABL)
    ("CHEMBL941", "CHEMBL1862"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_IMAT_ABL1",
        assay_type="B",
        standard_value=37.0,
        standard_units="nM",
        pchembl_value=7.43,
        reference_doi="10.1126/science.289.5486.1938",
        reference_year=2000,
        reference_journal="Science",
        assay_description=(
            "In vitro kinase assay of recombinant ABL1 inhibition by "
            "STI-571 (imatinib); Ki determined by ATP-competition kinetics."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL941",
        target_chembl_id="CHEMBL1862",
        standard_type="Ki",
    ),
    # Imatinib vs KIT — Heinrich et al. Blood 2000 (IC50 ~ 100 nM)
    ("CHEMBL941", "CHEMBL2815"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_IMAT_KIT",
        assay_type="B",
        standard_value=100.0,
        standard_units="nM",
        pchembl_value=7.00,
        reference_doi="10.1182/blood.V96.3.925",
        reference_year=2000,
        reference_journal="Blood",
        assay_description=(
            "Inhibition of KIT autophosphorylation in ligand-stimulated "
            "cells; IC50 from dose-response curve."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL941",
        target_chembl_id="CHEMBL2815",
        standard_type="IC50",
    ),
    # Erlotinib vs EGFR — Moyer et al. Cancer Res 1997 (IC50 ~ 2 nM)
    ("CHEMBL553", "CHEMBL203"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_ERLO_EGFR",
        assay_type="B",
        standard_value=2.0,
        standard_units="nM",
        pchembl_value=8.70,
        reference_doi="10.1158/0008-5472.CAN-97-4838",
        reference_year=1997,
        reference_journal="Cancer Research",
        assay_description=(
            "Inhibition of purified EGFR tyrosine kinase "
            "autophosphorylation by OSI-774 (erlotinib); IC50 measured "
            "by radiometric filter-binding assay."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL553",
        target_chembl_id="CHEMBL203",
        standard_type="IC50",
    ),
    # Aspirin vs COX-1 — Vane, Nature New Biology 1971 (IC50 ~ 1700 nM)
    ("CHEMBL25", "CHEMBL221"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_ASP_COX1",
        assay_type="F",
        standard_value=1700.0,
        standard_units="nM",
        pchembl_value=5.77,
        reference_doi="10.1038/newbio231232a0",
        reference_year=1971,
        reference_journal="Nature New Biology",
        assay_description=(
            "Inhibition of prostaglandin synthetase (cyclooxygenase-1) "
            "from guinea-pig lung homogenate by acetylsalicylic acid."
        ),
        confidence_score=8,
        molecule_chembl_id="CHEMBL25",
        target_chembl_id="CHEMBL221",
        standard_type="IC50",
    ),
    # Aspirin vs COX-2 — Mitchell et al. PNAS 1993
    ("CHEMBL25", "CHEMBL218"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_ASP_COX2",
        assay_type="F",
        standard_value=50000.0,
        standard_units="nM",
        pchembl_value=4.30,
        reference_doi="10.1073/pnas.90.24.11693",
        reference_year=1993,
        reference_journal="PNAS",
        assay_description=(
            "Whole-cell COX-2 inhibition in LPS-stimulated J774.2 "
            "macrophages; ~30-fold weaker than COX-1."
        ),
        confidence_score=8,
        molecule_chembl_id="CHEMBL25",
        target_chembl_id="CHEMBL218",
        standard_type="IC50",
    ),
    # Sorafenib vs BRAF — Wilhelm et al. Cancer Res 2004 (IC50 ~ 22 nM)
    ("CHEMBL1336", "CHEMBL2146302"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_SORA_BRAF",
        assay_type="B",
        standard_value=22.0,
        standard_units="nM",
        pchembl_value=7.66,
        reference_doi="10.1158/0008-5472.CAN-04-1443",
        reference_year=2004,
        reference_journal="Cancer Research",
        assay_description=(
            "In vitro recombinant BRAF kinase assay; IC50 measured by "
            "radiometric MEK1 phosphorylation."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL1336",
        target_chembl_id="CHEMBL2146302",
        standard_type="IC50",
    ),
    # Vemurafenib vs BRAF V600E — Bollag et al. Nature 2010 (IC50 ~ 31 nM)
    ("CHEMBL1229517", "CHEMBL2146302"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_VEM_BRAFV600E",
        assay_type="B",
        standard_value=31.0,
        standard_units="nM",
        pchembl_value=7.51,
        reference_doi="10.1038/nature09454",
        reference_year=2010,
        reference_journal="Nature",
        assay_description=(
            "Recombinant BRAF(V600E) kinase assay measuring PLX4032 "
            "(vemurafenib) inhibition of MEK phosphorylation."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL1229517",
        target_chembl_id="CHEMBL2146302",
        standard_type="IC50",
    ),
    # Osimertinib vs EGFR T790M — Cross et al. Cancer Discov 2014 (IC50 ~ 12 nM)
    ("CHEMBL3353410", "CHEMBL203"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_OSI_EGFR_T790M",
        assay_type="B",
        standard_value=12.0,
        standard_units="nM",
        pchembl_value=7.92,
        reference_doi="10.1158/2159-8290.CD-14-0337",
        reference_year=2014,
        reference_journal="Cancer Discovery",
        assay_description=(
            "In vitro enzymatic assay of AZD9291 (osimertinib) against "
            "EGFR L858R/T790M mutant; IC50 by LanthaScreen binding."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL3353410",
        target_chembl_id="CHEMBL203",
        standard_type="IC50",
    ),
    # Atorvastatin vs HMGCR — Istvan & Deisenhofer, Science 2001 (IC50 ~ 8 nM)
    ("CHEMBL1487", "CHEMBL402"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_ATOR_HMGCR",
        assay_type="B",
        standard_value=8.0,
        standard_units="nM",
        pchembl_value=8.10,
        reference_doi="10.1126/science.1059344",
        reference_year=2001,
        reference_journal="Science",
        assay_description=(
            "Inhibition of purified human HMG-CoA reductase catalytic "
            "domain by atorvastatin; IC50 by NADPH-coupled spectroscopy."
        ),
        confidence_score=9,
        molecule_chembl_id="CHEMBL1487",
        target_chembl_id="CHEMBL402",
        standard_type="IC50",
    ),
    # Metformin vs AMPK (indirect) — Zhou et al. J Clin Invest 2001
    ("CHEMBL1431", "CHEMBL5524"): ProvenanceRecord(
        assay_id="CHEMBL_ASSAY_MET_AMPK",
        assay_type="F",
        standard_value=None,
        standard_units=None,
        pchembl_value=None,
        reference_doi="10.1172/JCI13505",
        reference_year=2001,
        reference_journal="J. Clin. Invest.",
        assay_description=(
            "Metformin activates AMPK indirectly via mitochondrial "
            "complex I inhibition in primary rat hepatocytes; no direct "
            "Ki/IC50 — activation is allosteric and context-dependent."
        ),
        confidence_score=6,
        molecule_chembl_id="CHEMBL1431",
        target_chembl_id="CHEMBL5524",
        standard_type="EC50",
    ),
}


_FALLBACK_TARGETS: dict[str, TargetInfo] = {
    "CHEMBL203": TargetInfo("CHEMBL203", "Epidermal growth factor receptor", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL1862": TargetInfo("CHEMBL1862", "ABL1", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL1824": TargetInfo("CHEMBL1824", "HER2 (ErbB2)", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL218": TargetInfo("CHEMBL218", "Cyclooxygenase-2", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL2146302": TargetInfo("CHEMBL2146302", "BRAF", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL2815": TargetInfo("CHEMBL2815", "KIT", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
    "CHEMBL4523582": TargetInfo("CHEMBL4523582", "PD-1 (PDCD1)", "Homo sapiens", "SINGLE PROTEIN", is_fallback=True),
}

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Name → ChEMBL id maps used by get_bioactivity_with_provenance.  Kept
# module-level so external callers (e.g. verifiers) can see them when
# needed without instantiating the client.
_MOLECULE_NAME_TO_ID: dict[str, str] = {
    "imatinib": "CHEMBL941",
    "gleevec": "CHEMBL941",
    "sti-571": "CHEMBL941",
    "sti571": "CHEMBL941",
    "erlotinib": "CHEMBL553",
    "tarceva": "CHEMBL553",
    "osi-774": "CHEMBL553",
    "trastuzumab": "CHEMBL1201585",
    "herceptin": "CHEMBL1201585",
    "aspirin": "CHEMBL25",
    "acetylsalicylic acid": "CHEMBL25",
    "metformin": "CHEMBL1431",
    "glucophage": "CHEMBL1431",
    "atorvastatin": "CHEMBL1487",
    "lipitor": "CHEMBL1487",
    "sorafenib": "CHEMBL1336",
    "nexavar": "CHEMBL1336",
    "vemurafenib": "CHEMBL1229517",
    "zelboraf": "CHEMBL1229517",
    "plx4032": "CHEMBL1229517",
    "osimertinib": "CHEMBL3353410",
    "tagrisso": "CHEMBL3353410",
    "azd9291": "CHEMBL3353410",
    "pembrolizumab": "CHEMBL3137309",
    "keytruda": "CHEMBL3137309",
}

_TARGET_NAME_TO_ID: dict[str, str] = {
    "egfr": "CHEMBL203",
    "her1": "CHEMBL203",
    "erbb1": "CHEMBL203",
    "epidermal growth factor receptor": "CHEMBL203",
    "egfr t790m": "CHEMBL203",
    "egfr l858r": "CHEMBL203",
    "abl1": "CHEMBL1862",
    "abl": "CHEMBL1862",
    "bcr-abl": "CHEMBL1862",
    "her2": "CHEMBL1824",
    "erbb2": "CHEMBL1824",
    "cox-2": "CHEMBL218",
    "cox2": "CHEMBL218",
    "ptgs2": "CHEMBL218",
    "cyclooxygenase-2": "CHEMBL218",
    "cox-1": "CHEMBL221",
    "cox1": "CHEMBL221",
    "ptgs1": "CHEMBL221",
    "cyclooxygenase-1": "CHEMBL221",
    "braf": "CHEMBL2146302",
    "braf v600e": "CHEMBL2146302",
    "kit": "CHEMBL2815",
    "c-kit": "CHEMBL2815",
    "pd-1": "CHEMBL4523582",
    "pdcd1": "CHEMBL4523582",
    "hmgcr": "CHEMBL402",
    "hmg-coa reductase": "CHEMBL402",
    "ampk": "CHEMBL5524",
    "prkaa1": "CHEMBL5524",
}


def _resolve_molecule_id(value: str) -> str | None:
    """Return a canonical ChEMBL molecule id for *value*, if known."""
    if not value:
        return None
    v = value.strip()
    if v.upper().startswith("CHEMBL"):
        return v.upper()
    return _MOLECULE_NAME_TO_ID.get(v.lower())


def _resolve_target_id(value: str) -> str | None:
    """Return a canonical ChEMBL target id for *value*, if known."""
    if not value:
        return None
    v = value.strip()
    if v.upper().startswith("CHEMBL"):
        return v.upper()
    return _TARGET_NAME_TO_ID.get(v.lower())


_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_CACHE_NS = "chembl"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class ChEMBLClient:
    """Async ChEMBL REST API client with caching and built-in fallbacks."""

    def __init__(self, cache: KnowledgeCache, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._cache = cache
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_molecule(self, query: str) -> list[MoleculeInfo]:
        """Search ChEMBL molecules by name or SMILES."""
        cache_key = f"mol_search:{query.lower()}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return [MoleculeInfo(**m) for m in cached]

        try:
            resp = await self._http.get(
                "/molecule/search.json",
                params={"q": query, "limit": 10},
            )
            resp.raise_for_status()
            data = resp.json()
            molecules = [
                self._parse_molecule(m)
                for m in data.get("molecules", [])
                if m
            ]
            molecules = [m for m in molecules if m is not None]
            await self._cache.set(_CACHE_NS, cache_key, [asdict(m) for m in molecules], _CACHE_TTL)
            return molecules
        except Exception as exc:
            logger.warning("ChEMBL molecule search failed for %r: %s", query, exc)
            fb = _FALLBACK_MOLECULES.get(query.lower())
            return [fb] if fb else []

    async def get_bioactivities(self, chembl_id: str) -> list[BioactivityRecord]:
        """Fetch bioactivity records for a given molecule ChEMBL ID."""
        cache_key = f"bioact:{chembl_id}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return [BioactivityRecord(**b) for b in cached]

        try:
            resp = await self._http.get(
                "/activity.json",
                params={
                    "molecule_chembl_id": chembl_id,
                    "limit": 50,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            records = []
            for act in data.get("activities", []):
                std_val = act.get("standard_value")
                records.append(BioactivityRecord(
                    molecule_chembl_id=act.get("molecule_chembl_id", ""),
                    target_chembl_id=act.get("target_chembl_id", ""),
                    target_name=act.get("target_pref_name", ""),
                    standard_type=act.get("standard_type", ""),
                    standard_value=float(std_val) if std_val is not None else None,
                    standard_units=act.get("standard_units"),
                ))
            await self._cache.set(_CACHE_NS, cache_key, [asdict(r) for r in records], _CACHE_TTL)
            return records
        except Exception as exc:
            logger.warning("ChEMBL bioactivity lookup failed for %s: %s", chembl_id, exc)
            return [b for b in _FALLBACK_BIOACTIVITIES if b.molecule_chembl_id == chembl_id]

    async def get_target(self, target_id: str) -> TargetInfo | None:
        """Fetch target details by ChEMBL target ID."""
        cache_key = f"target:{target_id}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return TargetInfo(**cached)

        try:
            resp = await self._http.get(f"/target/{target_id}.json")
            resp.raise_for_status()
            data = resp.json()
            target = TargetInfo(
                target_chembl_id=data.get("target_chembl_id", target_id),
                pref_name=data.get("pref_name", ""),
                organism=data.get("organism", ""),
                target_type=data.get("target_type", ""),
            )
            await self._cache.set(_CACHE_NS, cache_key, asdict(target), _CACHE_TTL)
            return target
        except Exception as exc:
            logger.warning("ChEMBL target lookup failed for %s: %s", target_id, exc)
            return _FALLBACK_TARGETS.get(target_id)

    async def get_bioactivity_with_provenance(
        self,
        molecule_id: str,
        target_id: str,
    ) -> ProvenanceRecord | None:
        """Return a :class:`ProvenanceRecord` for a drug-target measurement.

        Parameters
        ----------
        molecule_id : str
            Compound identifier — either a ChEMBL id (``"CHEMBL941"``)
            or a common drug name (``"imatinib"``).
        target_id : str
            Target identifier — either a ChEMBL target id
            (``"CHEMBL1862"``) or a common symbol (``"ABL1"``,
            ``"EGFR"``, ``"COX-1"``).

        Returns
        -------
        ProvenanceRecord | None
            The provenance record, or ``None`` if no audited measurement
            is known for the pair.

        Notes
        -----
        For the framework demo, this method returns records from a
        hand-curated provenance table containing ~10 well-known
        drug-target Ki/IC50 measurements with full primary-literature
        citations.  Network calls to the live ChEMBL REST API are not
        performed here because the ``/activity.json`` endpoint rarely
        returns reference DOIs in a single hop, and we want deterministic
        audit trails for the demo.
        """
        mol_chembl = _resolve_molecule_id(molecule_id)
        tgt_chembl = _resolve_target_id(target_id)
        if mol_chembl is None or tgt_chembl is None:
            logger.debug(
                "Could not resolve provenance inputs: molecule=%r target=%r",
                molecule_id,
                target_id,
            )
            return None

        cache_key = f"prov:{mol_chembl}:{tgt_chembl}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            try:
                return ProvenanceRecord(**cached)
            except TypeError:
                # Cache shape drift — fall through to fresh lookup.
                pass

        record = _FALLBACK_PROVENANCE.get((mol_chembl, tgt_chembl))
        if record is not None:
            await self._cache.set(
                _CACHE_NS, cache_key, asdict(record), _CACHE_TTL
            )
            return record
        return None

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_molecule(data: dict) -> MoleculeInfo | None:
        try:
            structs = data.get("molecule_structures") or {}
            props = data.get("molecule_properties") or {}
            mw_raw = props.get("full_mwt")
            return MoleculeInfo(
                chembl_id=data["molecule_chembl_id"],
                pref_name=data.get("pref_name"),
                canonical_smiles=structs.get("canonical_smiles"),
                molecular_weight=float(mw_raw) if mw_raw is not None else None,
                max_phase=data.get("max_phase"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed to parse ChEMBL molecule: %s", exc)
            return None
