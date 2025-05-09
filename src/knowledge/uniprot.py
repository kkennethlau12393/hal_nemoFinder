"""UniProt REST API client with caching and fallback data for common drug targets."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProteinInfo:
    accession: str
    protein_name: str
    gene_name: str | None
    organism: str
    function_description: str | None
    keywords: tuple[str, ...] = ()
    is_fallback: bool = False


# ------------------------------------------------------------------
# Fallback data — ~15 common drug targets
# ------------------------------------------------------------------

_FALLBACK_PROTEINS: dict[str, ProteinInfo] = {
    "P00533": ProteinInfo("P00533", "Epidermal growth factor receptor", "EGFR", "Homo sapiens", "Receptor tyrosine kinase involved in cell proliferation and signaling.", ("Kinase", "Receptor", "Tyrosine-protein kinase"), is_fallback=True),
    "P15056": ProteinInfo("P15056", "Serine/threonine-protein kinase B-raf", "BRAF", "Homo sapiens", "Serine/threonine-protein kinase that plays a role in regulating the MAP kinase/ERK signaling pathway.", ("Kinase", "Transferase", "Proto-oncogene"), is_fallback=True),
    "Q9BYF1": ProteinInfo("Q9BYF1", "Angiotensin-converting enzyme 2", "ACE2", "Homo sapiens", "Carboxypeptidase that converts angiotensin II to angiotensin 1-7. Also serves as receptor for SARS-CoV-2.", ("Hydrolase", "Metalloprotease", "Receptor"), is_fallback=True),
    "P35354": ProteinInfo("P35354", "Prostaglandin G/H synthase 2", "PTGS2", "Homo sapiens", "Converts arachidonate to prostaglandin H2. Target of NSAIDs (COX-2).", ("Oxidoreductase", "Peroxidase"), is_fallback=True),
    "P04626": ProteinInfo("P04626", "Receptor tyrosine-protein kinase erbB-2", "ERBB2", "Homo sapiens", "Receptor tyrosine kinase (HER2). Overexpression is associated with breast cancer.", ("Kinase", "Receptor", "Proto-oncogene"), is_fallback=True),
    "P00918": ProteinInfo("P00918", "Carbonic anhydrase 2", "CA2", "Homo sapiens", "Essential enzyme for CO2 hydration. Target of diuretics and antiglaucoma agents.", ("Lyase", "Zinc"), is_fallback=True),
    "P10275": ProteinInfo("P10275", "Androgen receptor", "AR", "Homo sapiens", "Steroid hormone receptor involved in male sexual differentiation.", ("Receptor", "Transcription regulation"), is_fallback=True),
    "P03372": ProteinInfo("P03372", "Estrogen receptor alpha", "ESR1", "Homo sapiens", "Nuclear hormone receptor. Target for breast cancer therapy.", ("Receptor", "Transcription regulation"), is_fallback=True),
    "P28223": ProteinInfo("P28223", "5-hydroxytryptamine receptor 2A", "HTR2A", "Homo sapiens", "G-protein coupled serotonin receptor. Target of atypical antipsychotics.", ("Receptor", "G-protein coupled receptor"), is_fallback=True),
    "P08172": ProteinInfo("P08172", "Muscarinic acetylcholine receptor M2", "CHRM2", "Homo sapiens", "G-protein coupled receptor for acetylcholine in the parasympathetic nervous system.", ("Receptor", "G-protein coupled receptor"), is_fallback=True),
    "P14416": ProteinInfo("P14416", "D(2) dopamine receptor", "DRD2", "Homo sapiens", "Dopamine receptor that mediates inhibitory G-protein signaling. Target of antipsychotics.", ("Receptor", "G-protein coupled receptor"), is_fallback=True),
    "P11362": ProteinInfo("P11362", "Fibroblast growth factor receptor 1", "FGFR1", "Homo sapiens", "Receptor tyrosine kinase involved in cell growth and differentiation.", ("Kinase", "Receptor", "Tyrosine-protein kinase"), is_fallback=True),
    "P12821": ProteinInfo("P12821", "Angiotensin-converting enzyme", "ACE", "Homo sapiens", "Zinc metallopeptidase that converts angiotensin I to angiotensin II. Target of ACE inhibitors.", ("Hydrolase", "Metalloprotease"), is_fallback=True),
    "P27487": ProteinInfo("P27487", "Dipeptidyl peptidase 4", "DPP4", "Homo sapiens", "Serine protease that cleaves incretin hormones. Target of gliptins for diabetes.", ("Hydrolase", "Protease"), is_fallback=True),
    "P07550": ProteinInfo("P07550", "Beta-2 adrenergic receptor", "ADRB2", "Homo sapiens", "G-protein coupled receptor for epinephrine. Target of bronchodilators.", ("Receptor", "G-protein coupled receptor"), is_fallback=True),
}

# Index by gene name (case-insensitive) for easier lookup
_FALLBACK_BY_GENE: dict[str, ProteinInfo] = {
    p.gene_name.upper(): p
    for p in _FALLBACK_PROTEINS.values()
    if p.gene_name
}

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_BASE_URL = "https://rest.uniprot.org/uniprotkb"
_CACHE_NS = "uniprot"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class UniProtClient:
    """Async UniProt REST client with caching and built-in fallback data."""

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

    async def search_protein(self, query: str) -> list[ProteinInfo]:
        """Search UniProt by protein name, gene name, or keyword."""
        cache_key = f"search:{query.lower()}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return [self._from_cache(p) for p in cached]

        try:
            resp = await self._http.get(
                "/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": 10,
                    "fields": "accession,protein_name,gene_names,organism_name,cc_function,keyword",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = [
                self._parse_entry(entry) for entry in data.get("results", [])
            ]
            results = [r for r in results if r is not None]
            await self._cache.set(_CACHE_NS, cache_key, [asdict(r) for r in results], _CACHE_TTL)
            return results
        except Exception as exc:
            logger.warning("UniProt search failed for %r: %s", query, exc)
            return self._fallback_search(query)

    async def get_protein(self, accession: str) -> ProteinInfo | None:
        """Fetch a single protein entry by UniProt accession."""
        cache_key = f"entry:{accession}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return self._from_cache(cached)

        try:
            resp = await self._http.get(f"/{accession}.json")
            resp.raise_for_status()
            data = resp.json()
            protein = self._parse_entry(data)
            if protein:
                await self._cache.set(_CACHE_NS, cache_key, asdict(protein), _CACHE_TTL)
            return protein
        except Exception as exc:
            logger.warning("UniProt entry lookup failed for %s: %s", accession, exc)
            return _FALLBACK_PROTEINS.get(accession)

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entry(data: dict) -> ProteinInfo | None:
        try:
            accession = data.get("primaryAccession", "")

            # Protein name
            pname_data = data.get("proteinDescription", {})
            rec_name = pname_data.get("recommendedName", {})
            protein_name = rec_name.get("fullName", {}).get("value", "")
            if not protein_name:
                sub_names = pname_data.get("submissionNames", [])
                if sub_names:
                    protein_name = sub_names[0].get("fullName", {}).get("value", "")

            # Gene name
            genes = data.get("genes", [])
            gene_name = None
            if genes:
                gene_name = genes[0].get("geneName", {}).get("value")

            # Organism
            organism = data.get("organism", {}).get("scientificName", "")

            # Function
            function_desc = None
            comments = data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_desc = texts[0].get("value")
                    break

            # Keywords
            keywords = tuple(
                kw.get("name", "") for kw in data.get("keywords", [])
            )

            return ProteinInfo(
                accession=accession,
                protein_name=protein_name,
                gene_name=gene_name,
                organism=organism,
                function_description=function_desc,
                keywords=keywords,
            )
        except (KeyError, TypeError) as exc:
            logger.debug("Failed to parse UniProt entry: %s", exc)
            return None

    @staticmethod
    def _from_cache(data: dict) -> ProteinInfo:
        data["keywords"] = tuple(data.get("keywords", []))
        return ProteinInfo(**data)

    @staticmethod
    def _fallback_search(query: str) -> list[ProteinInfo]:
        """Search fallback data by gene name or protein name substring."""
        q_upper = query.upper()
        # Try exact gene name match first
        if q_upper in _FALLBACK_BY_GENE:
            return [_FALLBACK_BY_GENE[q_upper]]
        # Substring match on protein name or gene name
        results = []
        for protein in _FALLBACK_PROTEINS.values():
            if (
                q_upper in (protein.gene_name or "").upper()
                or query.lower() in protein.protein_name.lower()
            ):
                results.append(protein)
        return results
