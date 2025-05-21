"""Pathway knowledge client with a hand-curated graph of well-known pathways.

This module ships a small, deterministic pathway database for the
hal-nemoFinder framework demo.  The data is embedded as a Python dict
so the client has **no external dependencies** and can be used in
offline / air-gapped environments.

The curated set covers ~20 canonical signaling, metabolic, and
regulatory pathways with their member proteins.  Sources are noted per
entry (Reactome / KEGG style).

Typical usage::

    from src.knowledge.pathways import PathwayClient

    client = PathwayClient()
    info = client.get_pathway("PI3K/AKT/mTOR")
    pathways = client.find_pathways_for_protein("AKT1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PathwayInfo:
    """A single biological pathway record.

    Attributes
    ----------
    pathway_id : str
        Stable identifier (Reactome- or KEGG-style) for the pathway.
    name : str
        Human-readable pathway name.
    source : str
        Provenance of the curation (e.g. ``"reactome"`` or ``"kegg"``).
    proteins : list[str]
        Canonical HGNC gene symbols (and common aliases) of the member
        proteins.
    description : str
        Short free-text summary of the pathway's biology.
    parent_pathway : str | None
        Optional parent pathway id for hierarchical grouping.
    aliases : tuple[str, ...]
        Alternative colloquial names for name-based lookup.
    """

    pathway_id: str
    name: str
    source: str
    proteins: list[str]
    description: str
    parent_pathway: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Hand-curated pathway graph
# ---------------------------------------------------------------------------

# Protein aliases → canonical symbol.  Used to normalise user input when
# checking membership.
_PROTEIN_ALIASES: dict[str, str] = {
    "pi3k": "PI3K",
    "pik3ca": "PI3K",
    "akt": "AKT1",
    "akt1": "AKT1",
    "pkb": "AKT1",
    "mtor": "MTOR",
    "pten": "PTEN",
    "tsc1": "TSC1",
    "tsc2": "TSC2",
    "kras": "KRAS",
    "braf": "BRAF",
    "mek1": "MAP2K1",
    "map2k1": "MAP2K1",
    "erk1": "MAPK3",
    "mapk3": "MAPK3",
    "erk2": "MAPK1",
    "mapk1": "MAPK1",
    "jak1": "JAK1",
    "jak2": "JAK2",
    "jak3": "JAK3",
    "stat1": "STAT1",
    "stat3": "STAT3",
    "stat5": "STAT5",
    "egfr": "EGFR",
    "her1": "EGFR",
    "erbb1": "EGFR",
    "erbb2": "ERBB2",
    "her2": "ERBB2",
    "erbb3": "ERBB3",
    "her3": "ERBB3",
    "grb2": "GRB2",
    "sos1": "SOS1",
    "bax": "BAX",
    "bcl2": "BCL2",
    "bak": "BAK1",
    "bak1": "BAK1",
    "casp3": "CASP3",
    "caspase-3": "CASP3",
    "casp9": "CASP9",
    "caspase-9": "CASP9",
    "cytc": "CYCS",
    "cycs": "CYCS",
    "wnt": "WNT",
    "fzd": "FZD",
    "dvl": "DVL",
    "gsk3b": "GSK3B",
    "ctnnb1": "CTNNB1",
    "beta-catenin": "CTNNB1",
    "tp53": "TP53",
    "p53": "TP53",
    "mdm2": "MDM2",
    "cdkn1a": "CDKN1A",
    "p21": "CDKN1A",
    "ikk": "IKK",
    "ikb": "NFKBIA",
    "nfkbia": "NFKBIA",
    "rela": "RELA",
    "p65": "RELA",
    "nfkb1": "NFKB1",
    "p50": "NFKB1",
    "ptgs1": "PTGS1",
    "cox1": "PTGS1",
    "cox-1": "PTGS1",
    "ptgs2": "PTGS2",
    "cox2": "PTGS2",
    "cox-2": "PTGS2",
    "pla2": "PLA2G4A",
    "pla2g4a": "PLA2G4A",
    "insr": "INSR",
    "irs1": "IRS1",
    "irs2": "IRS2",
    "gnas": "GNAS",
    "gnai": "GNAI1",
    "gnai1": "GNAI1",
    "adcy": "ADCY1",
    "adcy1": "ADCY1",
    "pka": "PRKACA",
    "prkaca": "PRKACA",
    "ren": "REN",
    "agt": "AGT",
    "ace": "ACE",
    "agtr1": "AGTR1",
    "f2": "F2",
    "thrombin": "F2",
    "f7": "F7",
    "f10": "F10",
    "vkorc1": "VKORC1",
    "hmgcr": "HMGCR",
    "hmgcs": "HMGCS1",
    "hmgcs1": "HMGCS1",
    "sqle": "SQLE",
    "hk1": "HK1",
    "pfk1": "PFKM",
    "pfkm": "PFKM",
    "gapdh": "GAPDH",
    "pkm": "PKM",
    "atm": "ATM",
    "atr": "ATR",
    "brca1": "BRCA1",
    "brca2": "BRCA2",
}


# Pathway registry: pathway_id → PathwayInfo
_PATHWAY_DB: dict[str, PathwayInfo] = {
    "R-HSA-PI3K-AKT-MTOR": PathwayInfo(
        pathway_id="R-HSA-PI3K-AKT-MTOR",
        name="PI3K/AKT/mTOR signaling pathway",
        source="reactome",
        proteins=["PI3K", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2"],
        description=(
            "Growth-factor-driven pro-survival and cell-growth cascade; "
            "frequently dysregulated in cancer."
        ),
        aliases=("pi3k/akt", "pi3k-akt", "pi3k/akt/mtor", "akt pathway", "mtor pathway"),
    ),
    "R-HSA-MAPK-ERK": PathwayInfo(
        pathway_id="R-HSA-MAPK-ERK",
        name="MAPK/ERK (RAS-RAF-MEK-ERK) pathway",
        source="reactome",
        proteins=[
            # Upstream receptor tyrosine kinases that signal through MAPK
            "EGFR", "ERBB2", "ERBB3", "FGFR1", "FGFR2", "MET", "KIT", "PDGFRA",
            # Adapter / GEF
            "GRB2", "SOS1",
            # Core cascade
            "KRAS", "HRAS", "NRAS", "BRAF", "ARAF", "RAF1", "MAP2K1", "MAP2K2",
            "MAPK3", "MAPK1",
        ],
        description=(
            "Canonical RAS/RAF/MEK/ERK mitogen-activated protein kinase "
            "cascade regulating proliferation and differentiation. Receives "
            "input from upstream receptor tyrosine kinases (EGFR, FGFR, etc)."
        ),
        aliases=("mapk", "erk", "ras/raf/mek/erk", "mapk/erk", "ras-raf-mek-erk"),
    ),
    "R-HSA-JAK-STAT": PathwayInfo(
        pathway_id="R-HSA-JAK-STAT",
        name="JAK/STAT signaling pathway",
        source="reactome",
        proteins=["JAK1", "JAK2", "JAK3", "STAT1", "STAT3", "STAT5"],
        description=(
            "Cytokine-activated signaling cascade relaying extracellular "
            "signals to transcriptional responses."
        ),
        aliases=("jak/stat", "jak-stat", "jak stat"),
    ),
    "R-HSA-EGFR": PathwayInfo(
        pathway_id="R-HSA-EGFR",
        name="EGFR signaling",
        source="reactome",
        proteins=["EGFR", "ERBB2", "ERBB3", "GRB2", "SOS1"],
        description="ErbB receptor tyrosine kinase signalling network.",
        aliases=("egfr", "erbb", "her"),
    ),
    "R-HSA-APOPTOSIS": PathwayInfo(
        pathway_id="R-HSA-APOPTOSIS",
        name="Apoptosis (intrinsic pathway)",
        source="reactome",
        proteins=["BAX", "BCL2", "BAK1", "CASP3", "CASP9", "CYCS"],
        description=(
            "Mitochondrial (intrinsic) programmed cell-death pathway "
            "balancing pro- and anti-apoptotic Bcl-2 family members."
        ),
        aliases=("apoptosis", "programmed cell death", "intrinsic apoptosis"),
    ),
    "R-HSA-WNT": PathwayInfo(
        pathway_id="R-HSA-WNT",
        name="Wnt signaling pathway",
        source="reactome",
        proteins=["WNT", "FZD", "DVL", "GSK3B", "CTNNB1"],
        description="Canonical Wnt/β-catenin developmental pathway.",
        aliases=("wnt", "wnt/beta-catenin", "canonical wnt"),
    ),
    "R-HSA-P53": PathwayInfo(
        pathway_id="R-HSA-P53",
        name="p53 tumour suppressor pathway",
        source="reactome",
        proteins=["TP53", "MDM2", "CDKN1A", "BAX"],
        description=(
            "DNA-damage-activated tumour suppressor network driving "
            "cell-cycle arrest and apoptosis."
        ),
        aliases=("p53", "tp53", "p53 pathway"),
    ),
    "R-HSA-NFKB": PathwayInfo(
        pathway_id="R-HSA-NFKB",
        name="NF-κB signaling pathway",
        source="reactome",
        proteins=["IKK", "NFKBIA", "RELA", "NFKB1"],
        description="Master inflammatory transcription-factor cascade.",
        aliases=("nf-kb", "nfkb", "nf-kappab", "nfkappab"),
    ),
    "R-HSA-COX-PROSTAGLANDIN": PathwayInfo(
        pathway_id="R-HSA-COX-PROSTAGLANDIN",
        name="Cyclooxygenase / prostaglandin biosynthesis",
        source="reactome",
        proteins=["PTGS1", "PTGS2", "PLA2G4A"],
        description=(
            "Arachidonic-acid to prostaglandin biosynthesis cascade; "
            "target of NSAIDs."
        ),
        aliases=("cox", "cox/prostaglandin", "prostaglandin", "cyclooxygenase"),
    ),
    "R-HSA-INSULIN": PathwayInfo(
        pathway_id="R-HSA-INSULIN",
        name="Insulin signaling pathway",
        source="reactome",
        proteins=["INSR", "IRS1", "IRS2", "PI3K", "AKT1"],
        description="Insulin-receptor-driven metabolic and mitogenic signalling.",
        aliases=("insulin", "insulin signaling"),
    ),
    "R-HSA-GPCR": PathwayInfo(
        pathway_id="R-HSA-GPCR",
        name="GPCR downstream signalling",
        source="reactome",
        proteins=["GNAS", "GNAI1", "ADCY1", "PRKACA"],
        description=(
            "Heterotrimeric G-protein cascades downstream of G-protein-"
            "coupled receptors."
        ),
        aliases=("gpcr", "g-protein", "g protein coupled", "camp/pka"),
    ),
    "K-HSA-RAAS": PathwayInfo(
        pathway_id="K-HSA-RAAS",
        name="Renin-angiotensin system",
        source="kegg",
        proteins=["REN", "AGT", "ACE", "AGTR1"],
        description=(
            "Blood-pressure and fluid-balance regulatory cascade; target "
            "of ACE-inhibitors and ARBs."
        ),
        aliases=("raas", "renin-angiotensin", "renin angiotensin"),
    ),
    "K-HSA-COAGULATION": PathwayInfo(
        pathway_id="K-HSA-COAGULATION",
        name="Coagulation cascade",
        source="kegg",
        proteins=["F2", "F7", "F10", "VKORC1"],
        description=(
            "Extrinsic / common coagulation cascade; target of warfarin "
            "and direct oral anticoagulants."
        ),
        aliases=("coagulation", "clotting cascade", "blood coagulation"),
    ),
    "K-HSA-CHOLESTEROL": PathwayInfo(
        pathway_id="K-HSA-CHOLESTEROL",
        name="Cholesterol biosynthesis (mevalonate pathway)",
        source="kegg",
        proteins=["HMGCR", "HMGCS1", "SQLE"],
        description=(
            "Mevalonate → cholesterol biosynthesis; rate-limited by "
            "HMG-CoA reductase, the statin target."
        ),
        aliases=("cholesterol biosynthesis", "mevalonate", "statin pathway"),
    ),
    "K-HSA-GLYCOLYSIS": PathwayInfo(
        pathway_id="K-HSA-GLYCOLYSIS",
        name="Glycolysis",
        source="kegg",
        proteins=["HK1", "PFKM", "GAPDH", "PKM"],
        description="Glucose → pyruvate catabolic energy pathway.",
        aliases=("glycolysis", "embden-meyerhof"),
    ),
    "R-HSA-DDR": PathwayInfo(
        pathway_id="R-HSA-DDR",
        name="DNA damage response",
        source="reactome",
        proteins=["ATM", "ATR", "BRCA1", "BRCA2", "TP53"],
        description=(
            "Sensor-transducer-effector network responding to DNA "
            "double-strand breaks and replication stress."
        ),
        aliases=("ddr", "dna damage response", "dna repair"),
    ),
}


# Keyword / alias → pathway_id for name-based lookup.  Built lazily from
# pathway names and their ``aliases`` field.
def _build_name_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for pid, info in _PATHWAY_DB.items():
        index[info.name.lower()] = pid
        for alias in info.aliases:
            index[alias.lower()] = pid
    return index


_NAME_INDEX: dict[str, str] = _build_name_index()


def _canonicalise_protein(name: str) -> str:
    """Return the canonical symbol for *name* (upper-case, aliases resolved)."""
    key = name.strip().lower().replace(" ", "")
    return _PROTEIN_ALIASES.get(key, name.strip().upper())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PathwayClient:
    """Read-only client over the curated pathway graph.

    The client has no network dependencies and is safe to instantiate
    eagerly.  All lookups are in-memory dict operations.
    """

    def __init__(self) -> None:
        self._db: dict[str, PathwayInfo] = dict(_PATHWAY_DB)
        self._name_index: dict[str, str] = dict(_NAME_INDEX)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_pathway(self, query: str) -> PathwayInfo | None:
        """Look up a pathway by id, name, or alias.

        Parameters
        ----------
        query : str
            Pathway identifier (e.g. ``"R-HSA-PI3K-AKT-MTOR"``) or a
            human-readable name / alias (e.g. ``"PI3K/AKT pathway"``).

        Returns
        -------
        PathwayInfo | None
            The matching pathway, or ``None`` if nothing is found.
        """
        if not query:
            return None

        # Direct id hit
        if query in self._db:
            return self._db[query]

        key = query.strip().lower()

        # Exact alias / name hit
        if key in self._name_index:
            return self._db[self._name_index[key]]

        # Substring / token match (e.g. "PI3K/AKT" inside "pi3k/akt/mtor").
        # We pick the best (longest) match to avoid spurious short hits.
        best_id: str | None = None
        best_len = 0
        for alias, pid in self._name_index.items():
            if alias in key or key in alias:
                if len(alias) > best_len:
                    best_id = pid
                    best_len = len(alias)
        if best_id is not None:
            return self._db[best_id]

        return None

    def find_pathways_for_protein(self, protein: str) -> list[PathwayInfo]:
        """Return all pathways that contain *protein* as a member.

        Matching is case-insensitive and alias-aware, so inputs like
        ``"p53"``, ``"TP53"``, and ``"tp53"`` all resolve to the same
        canonical symbol.
        """
        if not protein:
            return []
        canonical = _canonicalise_protein(protein)
        hits: list[PathwayInfo] = []
        for info in self._db.values():
            members = {_canonicalise_protein(p) for p in info.proteins}
            if canonical in members:
                hits.append(info)
        return hits

    def is_member(self, protein: str, pathway: str | PathwayInfo) -> bool:
        """Return ``True`` iff *protein* is a member of *pathway*.

        ``pathway`` may be a :class:`PathwayInfo` or any string accepted
        by :meth:`get_pathway`.
        """
        info = pathway if isinstance(pathway, PathwayInfo) else self.get_pathway(pathway)
        if info is None:
            return False
        canonical = _canonicalise_protein(protein)
        return canonical in {_canonicalise_protein(p) for p in info.proteins}

    def list_pathways(self) -> list[PathwayInfo]:
        """Return every pathway in the curated database."""
        return list(self._db.values())

    def list_members(self, pathway: str) -> list[str]:
        """Return the member proteins of *pathway* (canonical symbols)."""
        info = self.get_pathway(pathway)
        if info is None:
            return []
        return [_canonicalise_protein(p) for p in info.proteins]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Compatibility no-op so the client matches the other KB clients."""
        return None
