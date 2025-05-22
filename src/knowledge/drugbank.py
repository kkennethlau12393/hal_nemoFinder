"""DrugBank client with local fallback data.

DrugBank's full REST API requires a paid commercial license, so this client
is primarily a curated fallback store backed by the same Redis cache layer
used by the other knowledge-base clients. When the (optional) DrugBank API
endpoint is configured, the client will attempt a best-effort remote lookup
and fall back to the hardcoded dataset on any failure.

All hardcoded values below are taken from public sources (DrugBank public
pages, FDA labels, Wikipedia-cited primary literature). They are intended
for demonstration of the "two-clocks" cross-database triangulation concept
in the hal-nemoFinder framework.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrugInfo:
    """Structured DrugBank record for a single small-molecule drug."""

    name: str
    drugbank_id: str
    canonical_smiles: str | None
    molecular_weight: float | None
    logP: float | None
    indication: str | None
    mechanism: str | None
    half_life: str | None
    bioavailability: str | None
    protein_binding: str | None
    targets: tuple[str, ...] = field(default_factory=tuple)
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Fallback data — real values for ~30 well-known drugs
# Sources: DrugBank public pages, FDA labels, primary literature.
# ---------------------------------------------------------------------------

_FALLBACK_DRUGS: dict[str, DrugInfo] = {
    "aspirin": DrugInfo(
        name="Aspirin",
        drugbank_id="DB00945",
        canonical_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        molecular_weight=180.16,
        logP=1.19,
        indication="Pain, fever, inflammation; antiplatelet for cardiovascular prevention.",
        mechanism="Irreversible acetylation of COX-1 and COX-2 cyclooxygenases.",
        half_life="2-3 hours (low dose); up to 15-30 hours at high dose",
        bioavailability="50-75%",
        protein_binding="99.5%",
        targets=("PTGS1", "PTGS2"),
        is_fallback=True,
    ),
    "ibuprofen": DrugInfo(
        name="Ibuprofen",
        drugbank_id="DB01050",
        canonical_smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        molecular_weight=206.28,
        logP=3.97,
        indication="Mild-to-moderate pain, fever, inflammation.",
        mechanism="Non-selective reversible inhibition of COX-1 and COX-2.",
        half_life="1.8-2 hours",
        bioavailability="80-100%",
        protein_binding="99%",
        targets=("PTGS1", "PTGS2"),
        is_fallback=True,
    ),
    "metformin": DrugInfo(
        name="Metformin",
        drugbank_id="DB00331",
        canonical_smiles="CN(C)C(=N)NC(=N)N",
        molecular_weight=129.16,
        logP=-2.64,
        indication="Type 2 diabetes mellitus.",
        mechanism="Activates AMPK; decreases hepatic gluconeogenesis and increases peripheral insulin sensitivity.",
        half_life="4-8.7 hours",
        bioavailability="50-60%",
        protein_binding="Negligible",
        targets=("PRKAB1", "ETFDH", "GPD1"),
        is_fallback=True,
    ),
    "atorvastatin": DrugInfo(
        name="Atorvastatin",
        drugbank_id="DB01076",
        canonical_smiles="CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
        molecular_weight=558.64,
        logP=5.7,
        indication="Hypercholesterolemia and prevention of cardiovascular events.",
        mechanism="Competitive inhibition of HMG-CoA reductase.",
        half_life="14 hours (parent); 20-30 hours (active metabolites)",
        bioavailability="14%",
        protein_binding="98%",
        targets=("HMGCR", "DPP4", "NR1I3"),
        is_fallback=True,
    ),
    "omeprazole": DrugInfo(
        name="Omeprazole",
        drugbank_id="DB00338",
        canonical_smiles="CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
        molecular_weight=345.42,
        logP=2.23,
        indication="GERD, peptic ulcer disease, Zollinger-Ellison syndrome.",
        mechanism="Irreversible inhibition of gastric H+/K+-ATPase (proton pump).",
        half_life="0.5-1 hour",
        bioavailability="30-40%",
        protein_binding="95%",
        targets=("ATP4A", "ATP4B"),
        is_fallback=True,
    ),
    "losartan": DrugInfo(
        name="Losartan",
        drugbank_id="DB00678",
        canonical_smiles="CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
        molecular_weight=422.91,
        logP=4.01,
        indication="Hypertension, diabetic nephropathy.",
        mechanism="Selective angiotensin II type 1 (AT1) receptor antagonist.",
        half_life="2 hours (parent); 6-9 hours (active metabolite EXP3174)",
        bioavailability="33%",
        protein_binding="98.7%",
        targets=("AGTR1",),
        is_fallback=True,
    ),
    "amlodipine": DrugInfo(
        name="Amlodipine",
        drugbank_id="DB00381",
        canonical_smiles="CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
        molecular_weight=408.88,
        logP=3.0,
        indication="Hypertension, coronary artery disease, angina.",
        mechanism="L-type calcium channel blocker (dihydropyridine).",
        half_life="30-50 hours",
        bioavailability="64-90%",
        protein_binding="93%",
        targets=("CACNA1C", "CACNA1D", "CACNA1S"),
        is_fallback=True,
    ),
    "simvastatin": DrugInfo(
        name="Simvastatin",
        drugbank_id="DB00641",
        canonical_smiles="CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
        molecular_weight=418.57,
        logP=4.68,
        indication="Hypercholesterolemia; prevention of cardiovascular disease.",
        mechanism="Prodrug; active form competitively inhibits HMG-CoA reductase.",
        half_life="1.9 hours",
        bioavailability="<5%",
        protein_binding="95%",
        targets=("HMGCR",),
        is_fallback=True,
    ),
    "warfarin": DrugInfo(
        name="Warfarin",
        drugbank_id="DB00682",
        canonical_smiles="CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
        molecular_weight=308.33,
        logP=2.7,
        indication="Prevention and treatment of venous thromboembolism; atrial fibrillation stroke prophylaxis.",
        mechanism="Inhibits vitamin K epoxide reductase (VKORC1), impairing synthesis of clotting factors II, VII, IX, X.",
        half_life="20-60 hours (mean ~40)",
        bioavailability="100%",
        protein_binding="99%",
        targets=("VKORC1", "NQO1", "CALU"),
        is_fallback=True,
    ),
    "dexamethasone": DrugInfo(
        name="Dexamethasone",
        drugbank_id="DB01234",
        canonical_smiles="CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",
        molecular_weight=392.46,
        logP=1.83,
        indication="Inflammatory conditions, cerebral edema, COVID-19 severe/critical illness.",
        mechanism="Glucocorticoid receptor agonist; broad anti-inflammatory and immunosuppressive effects.",
        half_life="3-4 hours (plasma); 36-54 hours (biologic)",
        bioavailability="80-90%",
        protein_binding="77%",
        targets=("NR3C1", "ANXA1", "NOS2"),
        is_fallback=True,
    ),
    "tamoxifen": DrugInfo(
        name="Tamoxifen",
        drugbank_id="DB00675",
        canonical_smiles="CC/C(=C(/C1=CC=CC=C1)\\C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3",
        molecular_weight=371.52,
        logP=6.3,
        indication="Hormone-receptor-positive breast cancer (treatment and chemoprevention).",
        mechanism="Selective estrogen receptor modulator (SERM); competitive antagonist in breast tissue.",
        half_life="5-7 days",
        bioavailability="~100%",
        protein_binding=">99%",
        targets=("ESR1", "ESR2", "PGR", "NR1I2"),
        is_fallback=True,
    ),
    "erlotinib": DrugInfo(
        name="Erlotinib",
        drugbank_id="DB00530",
        canonical_smiles="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        molecular_weight=393.44,
        logP=2.7,
        indication="Non-small cell lung cancer, pancreatic cancer.",
        mechanism="Reversible inhibitor of EGFR (HER1/ERBB1) tyrosine kinase.",
        half_life="36 hours",
        bioavailability="~60%",
        protein_binding="93%",
        targets=("EGFR",),
        is_fallback=True,
    ),
    "imatinib": DrugInfo(
        name="Imatinib",
        drugbank_id="DB00619",
        canonical_smiles="CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        molecular_weight=493.60,
        logP=3.0,
        indication="Chronic myeloid leukemia (BCR-ABL+), GIST (KIT+), other tyrosine-kinase-driven malignancies.",
        mechanism="Competitive inhibitor of BCR-ABL, KIT, PDGFR tyrosine kinases.",
        half_life="18 hours (parent); 40 hours (N-desmethyl metabolite)",
        bioavailability="98%",
        protein_binding="95%",
        targets=("ABL1", "KIT", "PDGFRA", "PDGFRB"),
        is_fallback=True,
    ),
    "gefitinib": DrugInfo(
        name="Gefitinib",
        drugbank_id="DB00317",
        canonical_smiles="COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        molecular_weight=446.90,
        logP=3.75,
        indication="EGFR-mutation-positive non-small cell lung cancer.",
        mechanism="Reversible selective inhibitor of EGFR tyrosine kinase.",
        half_life="48 hours",
        bioavailability="~60%",
        protein_binding="90%",
        targets=("EGFR",),
        is_fallback=True,
    ),
    "sorafenib": DrugInfo(
        name="Sorafenib",
        drugbank_id="DB00398",
        canonical_smiles="CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
        molecular_weight=464.82,
        logP=4.1,
        indication="Advanced renal cell carcinoma, hepatocellular carcinoma, thyroid cancer.",
        mechanism="Multi-kinase inhibitor (RAF, VEGFR2/3, PDGFR, KIT, FLT3, RET).",
        half_life="25-48 hours",
        bioavailability="38-49%",
        protein_binding="99.5%",
        targets=("RAF1", "BRAF", "KDR", "FLT4", "PDGFRB", "KIT", "FLT3", "RET"),
        is_fallback=True,
    ),
    "sunitinib": DrugInfo(
        name="Sunitinib",
        drugbank_id="DB01268",
        canonical_smiles="CCN(CC)CCNC(=O)C1=C(NC(=C1C)/C=C\\2/C3=C(C=CC(=C3)F)NC2=O)C",
        molecular_weight=398.47,
        logP=2.54,
        indication="Gastrointestinal stromal tumor, renal cell carcinoma, pancreatic neuroendocrine tumors.",
        mechanism="Multi-targeted receptor tyrosine kinase inhibitor (VEGFR, PDGFR, KIT, FLT3, RET).",
        half_life="40-60 hours (parent); 80-110 hours (active metabolite)",
        bioavailability="~50%",
        protein_binding="95%",
        targets=("KDR", "FLT1", "FLT4", "PDGFRA", "PDGFRB", "KIT", "FLT3", "RET", "CSF1R"),
        is_fallback=True,
    ),
    "morphine": DrugInfo(
        name="Morphine",
        drugbank_id="DB00295",
        canonical_smiles="CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        molecular_weight=285.34,
        logP=0.89,
        indication="Moderate-to-severe acute and chronic pain.",
        mechanism="Mu-opioid receptor agonist.",
        half_life="1.5-7 hours",
        bioavailability="20-40% (oral)",
        protein_binding="30-40%",
        targets=("OPRM1", "OPRD1", "OPRK1"),
        is_fallback=True,
    ),
    "codeine": DrugInfo(
        name="Codeine",
        drugbank_id="DB00318",
        canonical_smiles="CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(C=C4)O",
        molecular_weight=299.36,
        logP=1.19,
        indication="Mild-to-moderate pain; cough suppression.",
        mechanism="Prodrug; CYP2D6 converts to morphine, a mu-opioid agonist.",
        half_life="2.5-3 hours",
        bioavailability="~90%",
        protein_binding="7-25%",
        targets=("OPRM1",),
        is_fallback=True,
    ),
    "diazepam": DrugInfo(
        name="Diazepam",
        drugbank_id="DB00829",
        canonical_smiles="CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(=CC=C31)Cl",
        molecular_weight=284.74,
        logP=2.82,
        indication="Anxiety, status epilepticus, alcohol withdrawal, muscle spasm.",
        mechanism="Positive allosteric modulator of GABA-A receptor (benzodiazepine site).",
        half_life="20-100 hours (parent); up to 200 hours (desmethyldiazepam)",
        bioavailability="~100% (oral)",
        protein_binding="98.5%",
        targets=("GABRA1", "GABRA2", "GABRA3", "GABRA5", "TSPO"),
        is_fallback=True,
    ),
    "caffeine": DrugInfo(
        name="Caffeine",
        drugbank_id="DB00201",
        canonical_smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        molecular_weight=194.19,
        logP=-0.07,
        indication="CNS stimulant; neonatal apnea; adjunct analgesic.",
        mechanism="Non-selective adenosine A1/A2A receptor antagonist.",
        half_life="3-5 hours (adults)",
        bioavailability="99%",
        protein_binding="30-36%",
        targets=("ADORA1", "ADORA2A", "ADORA2B", "PDE4B"),
        is_fallback=True,
    ),
    "acetaminophen": DrugInfo(
        name="Acetaminophen",
        drugbank_id="DB00316",
        canonical_smiles="CC(=O)NC1=CC=C(C=C1)O",
        molecular_weight=151.16,
        logP=0.46,
        indication="Mild-to-moderate pain and fever.",
        mechanism="Central COX inhibition and modulation of endocannabinoid system; exact mechanism debated.",
        half_life="1.25-3 hours",
        bioavailability="~88%",
        protein_binding="10-25%",
        targets=("PTGS1", "PTGS2", "PTGS3"),
        is_fallback=True,
    ),
    "penicillin": DrugInfo(
        name="Penicillin G",
        drugbank_id="DB01053",
        canonical_smiles="CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        molecular_weight=334.39,
        logP=1.83,
        indication="Gram-positive bacterial infections (streptococcal, syphilis, meningococcal).",
        mechanism="Inhibits bacterial cell wall synthesis by binding penicillin-binding proteins (PBPs).",
        half_life="0.5 hours",
        bioavailability="15-30% (oral)",
        protein_binding="60%",
        targets=("dacB", "ponA", "pbpA"),
        is_fallback=True,
    ),
    "amoxicillin": DrugInfo(
        name="Amoxicillin",
        drugbank_id="DB01060",
        canonical_smiles="CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        molecular_weight=365.40,
        logP=0.87,
        indication="Broad-spectrum beta-lactam antibiotic for respiratory, ear, urinary and dental infections.",
        mechanism="Inhibits bacterial cell wall synthesis by binding PBPs.",
        half_life="1-1.5 hours",
        bioavailability="~95%",
        protein_binding="17-20%",
        targets=("ponA", "pbpA", "dacA"),
        is_fallback=True,
    ),
    "clopidogrel": DrugInfo(
        name="Clopidogrel",
        drugbank_id="DB00758",
        canonical_smiles="COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3",
        molecular_weight=321.82,
        logP=3.81,
        indication="Prevention of atherothrombotic events in acute coronary syndrome and after stroke/PCI.",
        mechanism="Prodrug; active metabolite irreversibly inhibits P2Y12 ADP receptor on platelets.",
        half_life="6 hours (parent); 30 minutes (active metabolite)",
        bioavailability="~50%",
        protein_binding="94-98%",
        targets=("P2RY12",),
        is_fallback=True,
    ),
    "lisinopril": DrugInfo(
        name="Lisinopril",
        drugbank_id="DB00722",
        canonical_smiles="C(CC(=O)O)CC(C(=O)O)NC(CCCCN)C(=O)N1CCCC1C(=O)O",
        molecular_weight=405.49,
        logP=-1.22,
        indication="Hypertension, heart failure, post-MI, diabetic nephropathy.",
        mechanism="Competitive inhibitor of angiotensin-converting enzyme (ACE).",
        half_life="12 hours",
        bioavailability="~25%",
        protein_binding="Negligible",
        targets=("ACE",),
        is_fallback=True,
    ),
    "metoprolol": DrugInfo(
        name="Metoprolol",
        drugbank_id="DB00264",
        canonical_smiles="CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
        molecular_weight=267.36,
        logP=1.88,
        indication="Hypertension, angina, heart failure, post-MI, arrhythmias.",
        mechanism="Selective beta-1 adrenergic receptor antagonist.",
        half_life="3-7 hours",
        bioavailability="~50%",
        protein_binding="12%",
        targets=("ADRB1", "ADRB2"),
        is_fallback=True,
    ),
    "prednisone": DrugInfo(
        name="Prednisone",
        drugbank_id="DB00635",
        canonical_smiles="CC12CC(=O)C3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C",
        molecular_weight=358.43,
        logP=1.46,
        indication="Inflammatory and autoimmune disorders; immunosuppression.",
        mechanism="Prodrug of prednisolone; glucocorticoid receptor agonist.",
        half_life="2-3 hours (plasma); 18-36 hours (biologic)",
        bioavailability="~80%",
        protein_binding="70-90%",
        targets=("NR3C1", "ANXA1"),
        is_fallback=True,
    ),
}

# SMILES-indexed lookup for reverse search
_FALLBACK_BY_SMILES: dict[str, DrugInfo] = {
    d.canonical_smiles: d for d in _FALLBACK_DRUGS.values() if d.canonical_smiles
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://go.drugbank.com/public_api/v1"
_CACHE_NS = "drugbank"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DrugBankClient:
    """Async DrugBank client with caching and built-in fallback dataset.

    The DrugBank public API is paywalled; for open-source demo purposes this
    client relies primarily on a curated in-memory dataset of ~30 well-known
    drugs. If an ``api_key`` is provided, remote lookups are attempted first
    and the fallback is used only if the API call fails.
    """

    def __init__(
        self,
        cache: KnowledgeCache,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._cache = cache
        self._api_key = api_key
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = api_key
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_drug(self, name: str) -> DrugInfo | None:
        """Look up a drug by common name.

        Parameters
        ----------
        name:
            Common or generic drug name (case-insensitive).

        Returns
        -------
        DrugInfo | None
            The corresponding record, or ``None`` if not found.
        """
        key = name.strip().lower()
        cache_key = f"name:{key}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return DrugInfo(**cached)

        # Try remote API only if authenticated
        if self._api_key:
            try:
                resp = await self._http.get("/drugs", params={"q": name})
                resp.raise_for_status()
                data = resp.json()
                drug = self._parse_drug(data)
                if drug:
                    await self._cache.set(_CACHE_NS, cache_key, asdict(drug), _CACHE_TTL)
                    return drug
            except Exception as exc:
                logger.warning("DrugBank API lookup failed for %r: %s", name, exc)

        fallback = _FALLBACK_DRUGS.get(key)
        if fallback is not None:
            await self._cache.set(_CACHE_NS, cache_key, asdict(fallback), _CACHE_TTL)
        return fallback

    async def get_by_smiles(self, smiles: str) -> DrugInfo | None:
        """Look up a drug by canonical SMILES string.

        Parameters
        ----------
        smiles:
            Canonical SMILES string. Only exact matches are resolved from
            the fallback dataset — normalization is not performed here.

        Returns
        -------
        DrugInfo | None
            The matching record, or ``None`` if not found.
        """
        cache_key = f"smiles:{smiles}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return DrugInfo(**cached)

        if self._api_key:
            try:
                resp = await self._http.get(
                    "/structures",
                    params={"smiles": smiles, "search_type": "exact"},
                )
                resp.raise_for_status()
                data = resp.json()
                drug = self._parse_drug(data)
                if drug:
                    await self._cache.set(_CACHE_NS, cache_key, asdict(drug), _CACHE_TTL)
                    return drug
            except Exception as exc:
                logger.warning("DrugBank SMILES lookup failed for %r: %s", smiles, exc)

        fallback = _FALLBACK_BY_SMILES.get(smiles)
        if fallback is not None:
            await self._cache.set(_CACHE_NS, cache_key, asdict(fallback), _CACHE_TTL)
        return fallback

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_drug(data: dict[str, Any]) -> DrugInfo | None:
        """Parse a DrugBank API payload into a :class:`DrugInfo`.

        The real DrugBank schema is rich; we extract only the fields we
        surface on :class:`DrugInfo`. Missing fields are set to ``None``.
        """
        try:
            # DrugBank API may return a list or a single object.
            record = data[0] if isinstance(data, list) else data
            if not record:
                return None

            props = record.get("calculated_properties") or {}
            mw = props.get("molecular_weight")
            logp = props.get("logP") or props.get("logp")
            targets_raw = record.get("targets") or []
            targets = tuple(
                t.get("gene_name") or t.get("name", "")
                for t in targets_raw
                if isinstance(t, dict)
            )

            return DrugInfo(
                name=record.get("name", ""),
                drugbank_id=record.get("drugbank_id") or record.get("id", ""),
                canonical_smiles=record.get("canonical_smiles")
                or props.get("smiles"),
                molecular_weight=float(mw) if mw is not None else None,
                logP=float(logp) if logp is not None else None,
                indication=record.get("indication"),
                mechanism=record.get("mechanism_of_action"),
                half_life=record.get("half_life"),
                bioavailability=record.get("bioavailability"),
                protein_binding=record.get("protein_binding"),
                targets=targets,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed to parse DrugBank response: %s", exc)
            return None
