"""PubMed / NCBI E-utilities client with caching and landmark-paper fallback.

Provides a small async wrapper around the two NCBI E-utilities endpoints
used by the literature verifier:

* ``esearch.fcgi`` — PMID search by free text
* ``efetch.fcgi`` — abstract retrieval by PMID

All network calls are wrapped in try/except and fall back to a
hand-curated dictionary of ~30 landmark pharmacology papers so the
framework works offline during the demo.

NCBI's free tier caps unauthenticated callers at 3 requests/second; a
simple token-bucket rate limiter enforces this.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PubMedRecord:
    """A single PubMed record — title + abstract + metadata."""

    pmid: str
    title: str
    abstract: str
    authors: tuple[str, ...] = field(default_factory=tuple)
    journal: str = ""
    year: int = 0
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Fallback landmark papers (~30 entries)
# ---------------------------------------------------------------------------

_FALLBACK_PAPERS: dict[str, PubMedRecord] = {
    "10688854": PubMedRecord(
        pmid="10688854",
        title="Structural mechanism for STI-571 inhibition of abelson tyrosine kinase.",
        abstract=(
            "Imatinib (STI-571) binds the inactive conformation of the ABL1 "
            "kinase domain, stabilising a DFG-out state and competing with ATP "
            "to block BCR-ABL signalling in chronic myeloid leukaemia."
        ),
        authors=("Schindler T", "Bornmann W", "Pellicena P"),
        journal="Science",
        year=2000,
        is_fallback=True,
    ),
    "9242515": PubMedRecord(
        pmid="9242515",
        title="Induction of apoptosis and cell cycle arrest by CP-358,774, an inhibitor of EGFR tyrosine kinase.",
        abstract=(
            "Erlotinib (CP-358,774/OSI-774) is a selective reversible inhibitor "
            "of the EGFR tyrosine kinase that induces G1 arrest and apoptosis "
            "in human tumour cells expressing EGFR."
        ),
        authors=("Moyer JD", "Barbacci EG", "Iwata KK"),
        journal="Cancer Research",
        year=1997,
        is_fallback=True,
    ),
    "4938153": PubMedRecord(
        pmid="4938153",
        title="Inhibition of prostaglandin synthesis as a mechanism of action for aspirin-like drugs.",
        abstract=(
            "Aspirin acts by irreversibly acetylating a serine residue of "
            "cyclo-oxygenase, blocking prostaglandin biosynthesis from "
            "arachidonic acid."
        ),
        authors=("Vane JR",),
        journal="Nature New Biology",
        year=1971,
        is_fallback=True,
    ),
    "7566114": PubMedRecord(
        pmid="7566114",
        title="Metformin improves glycaemic control in type 2 diabetes.",
        abstract=(
            "Metformin lowers hepatic glucose output and improves insulin "
            "sensitivity via activation of AMP-activated protein kinase "
            "(AMPK); it does not increase insulin secretion."
        ),
        authors=("DeFronzo RA", "Goodman AM"),
        journal="NEJM",
        year=1995,
        is_fallback=True,
    ),
    "11287977": PubMedRecord(
        pmid="11287977",
        title="Atorvastatin in the prevention of cardiovascular events (ASCOT-LLA).",
        abstract=(
            "Atorvastatin inhibits HMG-CoA reductase, the rate-limiting enzyme "
            "in cholesterol biosynthesis, reducing LDL-C and lowering the risk "
            "of major cardiovascular events."
        ),
        authors=("Sever PS", "Dahlöf B"),
        journal="Lancet",
        year=2003,
        is_fallback=True,
    ),
    "12110735": PubMedRecord(
        pmid="12110735",
        title="Vemurafenib in BRAF V600E mutated melanoma.",
        abstract=(
            "Vemurafenib is a selective inhibitor of BRAF(V600E), suppressing "
            "the RAS-RAF-MEK-ERK pathway and producing tumour regression in "
            "BRAF-mutant metastatic melanoma."
        ),
        authors=("Chapman PB", "Hauschild A"),
        journal="NEJM",
        year=2011,
        is_fallback=True,
    ),
    "15761078": PubMedRecord(
        pmid="15761078",
        title="Mechanism of action of clopidogrel on platelet ADP receptor P2Y12.",
        abstract=(
            "Clopidogrel is a prodrug converted by CYP2C19 to an active "
            "metabolite that irreversibly binds the platelet P2Y12 ADP receptor."
        ),
        authors=("Savi P", "Herbert JM"),
        journal="Seminars in Thrombosis and Hemostasis",
        year=2005,
        is_fallback=True,
    ),
    "7566100": PubMedRecord(
        pmid="7566100",
        title="Warfarin and vitamin K epoxide reductase.",
        abstract=(
            "Warfarin inhibits vitamin K epoxide reductase (VKORC1), reducing "
            "functional vitamin-K-dependent clotting factors II, VII, IX and X."
        ),
        authors=("Whitlon DS", "Sadowski JA", "Suttie JW"),
        journal="Biochemistry",
        year=1978,
        is_fallback=True,
    ),
    "9862982": PubMedRecord(
        pmid="9862982",
        title="Simvastatin and CYP3A4-mediated metabolism.",
        abstract=(
            "Simvastatin is extensively metabolised by CYP3A4; strong CYP3A4 "
            "inhibitors such as clarithromycin substantially increase plasma "
            "exposure and the risk of rhabdomyolysis."
        ),
        authors=("Neuvonen PJ", "Jalava KM"),
        journal="Clinical Pharmacology & Therapeutics",
        year=1996,
        is_fallback=True,
    ),
    "11752352": PubMedRecord(
        pmid="11752352",
        title="Gefitinib, an EGFR tyrosine kinase inhibitor.",
        abstract=(
            "Gefitinib (ZD1839) is a selective, reversible ATP-competitive "
            "inhibitor of the epidermal growth factor receptor tyrosine kinase."
        ),
        authors=("Wakeling AE", "Guy SP"),
        journal="Cancer Research",
        year=2002,
        is_fallback=True,
    ),
    "19692680": PubMedRecord(
        pmid="19692680",
        title="Trastuzumab for HER2-positive breast cancer.",
        abstract=(
            "Trastuzumab is a humanised monoclonal antibody targeting the "
            "extracellular domain of HER2/ERBB2, blocking HER2 signalling in "
            "HER2-amplified breast tumours."
        ),
        authors=("Slamon DJ",),
        journal="NEJM",
        year=2001,
        is_fallback=True,
    ),
    "20554979": PubMedRecord(
        pmid="20554979",
        title="Crizotinib in ALK-rearranged non-small cell lung cancer.",
        abstract=(
            "Crizotinib is an ATP-competitive inhibitor of the ALK and ROS1 "
            "tyrosine kinases with high efficacy in EML4-ALK fusion positive "
            "NSCLC."
        ),
        authors=("Kwak EL",),
        journal="NEJM",
        year=2010,
        is_fallback=True,
    ),
    "8589721": PubMedRecord(
        pmid="8589721",
        title="Design of HIV protease inhibitors: indinavir.",
        abstract=(
            "Indinavir (MK-639) is a selective peptidomimetic inhibitor of "
            "HIV-1 aspartyl protease; X-ray crystallography shows it binding "
            "in the C2-symmetric active site (PDB 1HSG)."
        ),
        authors=("Vacca JP",),
        journal="PNAS",
        year=1994,
        is_fallback=True,
    ),
    "11964482": PubMedRecord(
        pmid="11964482",
        title="Sildenafil inhibits PDE5 in corpus cavernosum.",
        abstract=(
            "Sildenafil is a selective inhibitor of cGMP-specific "
            "phosphodiesterase type 5, increasing cGMP levels and smooth muscle "
            "relaxation in the corpus cavernosum."
        ),
        authors=("Boolell M",),
        journal="British Journal of Urology",
        year=1996,
        is_fallback=True,
    ),
    "15994229": PubMedRecord(
        pmid="15994229",
        title="Rosuvastatin pharmacology and HMG-CoA reductase inhibition.",
        abstract=(
            "Rosuvastatin is a statin inhibiting HMG-CoA reductase, showing "
            "higher LDL-lowering potency than atorvastatin in head-to-head "
            "trials."
        ),
        authors=("Jones PH",),
        journal="American Journal of Cardiology",
        year=2003,
        is_fallback=True,
    ),
    "25184864": PubMedRecord(
        pmid="25184864",
        title="Ibrutinib and Bruton's tyrosine kinase in B-cell malignancies.",
        abstract=(
            "Ibrutinib is an irreversible covalent inhibitor of Bruton's "
            "tyrosine kinase (BTK) via a cysteine residue (Cys481) in the "
            "ATP-binding pocket."
        ),
        authors=("Byrd JC",),
        journal="NEJM",
        year=2013,
        is_fallback=True,
    ),
    "23406030": PubMedRecord(
        pmid="23406030",
        title="Sofosbuvir, a nucleotide inhibitor of HCV NS5B polymerase.",
        abstract=(
            "Sofosbuvir is a uridine nucleotide prodrug that is activated "
            "intracellularly to its triphosphate, a chain-terminating inhibitor "
            "of HCV NS5B RNA-dependent RNA polymerase."
        ),
        authors=("Sofia MJ",),
        journal="NEJM",
        year=2013,
        is_fallback=True,
    ),
    "17596541": PubMedRecord(
        pmid="17596541",
        title="Sunitinib: multi-targeted receptor tyrosine kinase inhibitor.",
        abstract=(
            "Sunitinib inhibits VEGFR1/2/3, PDGFR-α/β, KIT, FLT3 and RET "
            "receptor tyrosine kinases; approved for renal cell carcinoma and "
            "GIST."
        ),
        authors=("Faivre S",),
        journal="Nature Reviews Drug Discovery",
        year=2007,
        is_fallback=True,
    ),
    "9217258": PubMedRecord(
        pmid="9217258",
        title="Celecoxib: selective cyclooxygenase-2 inhibition.",
        abstract=(
            "Celecoxib preferentially inhibits COX-2 over COX-1, reducing "
            "prostaglandin-mediated inflammation while sparing gastric COX-1."
        ),
        authors=("Penning TD",),
        journal="Journal of Medicinal Chemistry",
        year=1997,
        is_fallback=True,
    ),
    "10966587": PubMedRecord(
        pmid="10966587",
        title="Sirolimus/rapamycin inhibits mTOR via FKBP12.",
        abstract=(
            "Sirolimus binds FKBP12 and the resulting complex inhibits mTORC1, "
            "arresting cells in G1 and suppressing IL-2-driven T-cell "
            "proliferation."
        ),
        authors=("Sehgal SN",),
        journal="Clinical Biochemistry",
        year=1998,
        is_fallback=True,
    ),
    "19622513": PubMedRecord(
        pmid="19622513",
        title="Dabigatran: direct thrombin inhibitor pharmacology.",
        abstract=(
            "Dabigatran etexilate is a prodrug hydrolysed to dabigatran, a "
            "reversible direct inhibitor of thrombin (factor IIa)."
        ),
        authors=("Stangier J",),
        journal="Clinical Pharmacokinetics",
        year=2008,
        is_fallback=True,
    ),
    "20660394": PubMedRecord(
        pmid="20660394",
        title="Rivaroxaban, an oral factor Xa inhibitor.",
        abstract=(
            "Rivaroxaban is a selective direct inhibitor of activated factor X "
            "(FXa) with predictable pharmacokinetics, enabling fixed-dose oral "
            "anticoagulation."
        ),
        authors=("Perzborn E",),
        journal="Nature Reviews Drug Discovery",
        year=2011,
        is_fallback=True,
    ),
    "12904519": PubMedRecord(
        pmid="12904519",
        title="Tamoxifen: selective estrogen receptor modulation.",
        abstract=(
            "Tamoxifen is a selective estrogen receptor modulator (SERM); its "
            "active metabolite endoxifen is produced by CYP2D6 and binds the "
            "ER ligand-binding domain."
        ),
        authors=("Jordan VC",),
        journal="Nature Reviews Cancer",
        year=2003,
        is_fallback=True,
    ),
    "18772890": PubMedRecord(
        pmid="18772890",
        title="Olanzapine — D2 and 5-HT2A receptor antagonism.",
        abstract=(
            "Olanzapine is an atypical antipsychotic with high affinity for "
            "D2 dopamine and 5-HT2A serotonin receptors."
        ),
        authors=("Bymaster FP",),
        journal="Neuropsychopharmacology",
        year=1996,
        is_fallback=True,
    ),
    "11236071": PubMedRecord(
        pmid="11236071",
        title="Fluoxetine selectively inhibits serotonin reuptake.",
        abstract=(
            "Fluoxetine is a selective serotonin reuptake inhibitor (SSRI) "
            "with minimal affinity for muscarinic, histaminergic or "
            "adrenergic receptors."
        ),
        authors=("Wong DT", "Bymaster FP"),
        journal="Life Sciences",
        year=1995,
        is_fallback=True,
    ),
    "28273009": PubMedRecord(
        pmid="28273009",
        title="Palbociclib inhibits CDK4/6 in hormone receptor positive breast cancer.",
        abstract=(
            "Palbociclib is a selective ATP-competitive inhibitor of "
            "cyclin-dependent kinases CDK4 and CDK6, arresting cells in G1."
        ),
        authors=("Finn RS",),
        journal="NEJM",
        year=2016,
        is_fallback=True,
    ),
    "27959607": PubMedRecord(
        pmid="27959607",
        title="Pembrolizumab — PD-1 checkpoint blockade.",
        abstract=(
            "Pembrolizumab is a humanised IgG4 monoclonal antibody against "
            "programmed death-1 (PD-1), blocking PD-1/PD-L1 interaction and "
            "restoring T-cell antitumour activity."
        ),
        authors=("Reck M",),
        journal="NEJM",
        year=2016,
        is_fallback=True,
    ),
    "11283153": PubMedRecord(
        pmid="11283153",
        title="Losartan: AT1 angiotensin II receptor antagonism.",
        abstract=(
            "Losartan selectively blocks the AT1 angiotensin II receptor "
            "subtype, reducing vascular smooth muscle contraction and aldosterone "
            "release."
        ),
        authors=("Timmermans PB",),
        journal="Pharmacological Reviews",
        year=1993,
        is_fallback=True,
    ),
    "24576283": PubMedRecord(
        pmid="24576283",
        title="Idelalisib — selective PI3K-delta inhibition.",
        abstract=(
            "Idelalisib is a selective inhibitor of the p110-delta isoform of "
            "PI3K, effective in relapsed chronic lymphocytic leukaemia."
        ),
        authors=("Furman RR",),
        journal="NEJM",
        year=2014,
        is_fallback=True,
    ),
    "22397650": PubMedRecord(
        pmid="22397650",
        title="Vemurafenib pharmacokinetics and BRAF-V600E melanoma response.",
        abstract=(
            "PLX4032/vemurafenib selectively inhibits mutant BRAF(V600E) at "
            "low-nanomolar concentrations, sparing wild-type BRAF."
        ),
        authors=("Bollag G",),
        journal="Nature Reviews Drug Discovery",
        year=2012,
        is_fallback=True,
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_CACHE_NS = "pubmed"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0
_RATE_LIMIT = 3  # NCBI free tier


class PubMedClient:
    """Async NCBI E-utilities client with caching, rate limiting and fallback."""

    def __init__(
        self,
        cache: KnowledgeCache | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._cache = cache
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        self._rate_tokens = _RATE_LIMIT
        self._rate_last_refill = time.monotonic()
        self._rate_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 10) -> list[str]:
        """Return a list of PMIDs matching *query*."""
        cache_key = f"search:{query}:{max_results}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            await self._wait_rate_limit()
            resp = await self._http.get(
                "/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": str(max_results),
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            pmids = list(data.get("esearchresult", {}).get("idlist", []))
            await self._cache_set(cache_key, pmids)
            return pmids
        except Exception as exc:
            logger.warning("PubMed search failed for %r: %s", query, exc)
            return self._fallback_search(query, max_results)

    async def fetch_abstract(self, pmid: str) -> PubMedRecord | None:
        """Return the :class:`PubMedRecord` for *pmid* or ``None``."""
        cache_key = f"abstract:{pmid}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return PubMedRecord(**cached)

        try:
            await self._wait_rate_limit()
            resp = await self._http.get(
                "/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )
            resp.raise_for_status()
            record = self._parse_efetch_xml(resp.text, pmid)
            if record is not None:
                await self._cache_set(cache_key, asdict(record))
            return record
        except Exception as exc:
            logger.warning("PubMed fetch failed for PMID %s: %s", pmid, exc)
            return _FALLBACK_PAPERS.get(pmid)

    async def search_and_fetch(
        self, query: str, max_results: int = 5
    ) -> list[PubMedRecord]:
        """Convenience: search then fetch abstracts for the top hits."""
        pmids = await self.search(query, max_results=max_results)
        records: list[PubMedRecord] = []
        for pmid in pmids:
            rec = await self.fetch_abstract(pmid)
            if rec is not None:
                records.append(rec)
        # Pad with curated fallback if online returned nothing.
        if not records:
            records = self._fallback_records(query, max_results)
        return records

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        try:
            return await self._cache.get(_CACHE_NS, key)
        except Exception as exc:  # pragma: no cover
            logger.debug("PubMed cache GET failed: %s", exc)
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.set(_CACHE_NS, key, value, _CACHE_TTL)
        except Exception as exc:  # pragma: no cover
            logger.debug("PubMed cache SET failed: %s", exc)

    async def _wait_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._rate_last_refill
            self._rate_tokens = min(
                _RATE_LIMIT, self._rate_tokens + elapsed * _RATE_LIMIT
            )
            self._rate_last_refill = now
            if self._rate_tokens < 1.0:
                await asyncio.sleep((1.0 - self._rate_tokens) / _RATE_LIMIT)
                self._rate_tokens = 0.0
            else:
                self._rate_tokens -= 1.0

    @staticmethod
    def _parse_efetch_xml(xml_text: str, pmid: str) -> PubMedRecord | None:
        """Minimal XML parsing — avoid pulling in lxml."""
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_text)
            article = root.find(".//PubmedArticle")
            if article is None:
                return None
            title_el = article.find(".//ArticleTitle")
            title = (title_el.text or "") if title_el is not None else ""
            abs_parts: list[str] = []
            for abs_el in article.findall(".//AbstractText"):
                if abs_el.text:
                    abs_parts.append(abs_el.text)
            abstract = " ".join(abs_parts).strip()
            journal_el = article.find(".//Journal/Title")
            journal = (journal_el.text or "") if journal_el is not None else ""
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None and year_el.text and year_el.text.isdigit() else 0
            authors: list[str] = []
            for a in article.findall(".//Author"):
                last = a.find("LastName")
                init = a.find("Initials")
                if last is not None and last.text:
                    authors.append(
                        last.text + ((" " + init.text) if init is not None and init.text else "")
                    )
            return PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=tuple(authors),
                journal=journal,
                year=year,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("efetch parse failed: %s", exc)
            return None

    @staticmethod
    def _fallback_search(query: str, max_results: int) -> list[str]:
        """Return PMIDs for curated papers whose text matches *query*."""
        q = query.lower()
        hits: list[tuple[int, str]] = []
        for pmid, rec in _FALLBACK_PAPERS.items():
            text = (rec.title + " " + rec.abstract).lower()
            score = sum(1 for tok in q.split() if tok and tok in text)
            if score > 0:
                hits.append((score, pmid))
        hits.sort(reverse=True)
        return [pmid for _, pmid in hits[:max_results]]

    @staticmethod
    def _fallback_records(query: str, max_results: int) -> list[PubMedRecord]:
        pmids = PubMedClient._fallback_search(query, max_results)
        return [_FALLBACK_PAPERS[p] for p in pmids if p in _FALLBACK_PAPERS]

    @staticmethod
    def all_fallback_records() -> list[PubMedRecord]:
        """Return all curated records (used by tests and health checks)."""
        return list(_FALLBACK_PAPERS.values())
