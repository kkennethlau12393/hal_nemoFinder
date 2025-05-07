"""PubChem REST (PUG) client with caching, rate limiting, and fallback data."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompoundInfo:
    cid: int
    name: str
    molecular_formula: str
    molecular_weight: float
    canonical_smiles: str
    iupac_name: str
    is_fallback: bool = False


# ------------------------------------------------------------------
# Fallback data — real properties for ~20 common drugs
# ------------------------------------------------------------------

_FALLBACK_COMPOUNDS: dict[str, CompoundInfo] = {
    "aspirin": CompoundInfo(2244, "Aspirin", "C9H8O4", 180.16, "CC(=O)OC1=CC=CC=C1C(=O)O", "2-acetoxybenzoic acid", is_fallback=True),
    "ibuprofen": CompoundInfo(3672, "Ibuprofen", "C13H18O2", 206.28, "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "2-[4-(2-methylpropyl)phenyl]propanoic acid", is_fallback=True),
    "metformin": CompoundInfo(4091, "Metformin", "C4H11N5", 129.16, "CN(C)C(=N)NC(=N)N", "3-(diaminomethylidene)-1,1-dimethylguanidine", is_fallback=True),
    "caffeine": CompoundInfo(2519, "Caffeine", "C8H10N4O2", 194.19, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", is_fallback=True),
    "acetaminophen": CompoundInfo(1983, "Acetaminophen", "C8H9NO2", 151.16, "CC(=O)NC1=CC=C(C=C1)O", "N-(4-hydroxyphenyl)acetamide", is_fallback=True),
    "amoxicillin": CompoundInfo(33613, "Amoxicillin", "C16H19N3O5S", 365.40, "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C", "(2S,5R,6R)-6-[(2R)-2-amino-2-(4-hydroxyphenyl)acetamido]-3,3-dimethyl-7-oxo-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid", is_fallback=True),
    "lisinopril": CompoundInfo(5362119, "Lisinopril", "C21H31N3O5", 405.49, "C(CC(=O)O)CC(C(=O)O)NC(CCCCN)C(=O)N1CCCC1C(=O)O", "(2S)-1-[(2S)-6-amino-2-{[(1S)-1-carboxy-3-phenylpropyl]amino}hexanoyl]pyrrolidine-2-carboxylic acid", is_fallback=True),
    "atorvastatin": CompoundInfo(60823, "Atorvastatin", "C33H35FN2O5", 558.64, "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "(3R,5R)-7-[2-(4-fluorophenyl)-3-phenyl-4-(phenylcarbamoyl)-5-propan-2-ylpyrrol-1-yl]-3,5-dihydroxyheptanoic acid", is_fallback=True),
    "omeprazole": CompoundInfo(4594, "Omeprazole", "C17H19N3O3S", 345.42, "CC1=CN=C(C2=CC=C(C=C2)OC)C3=CC=C(S3)S(=O)CC4=NC=C(N4)C", "5-methoxy-2-[(4-methoxy-3,5-dimethylpyridin-2-yl)methylsulfinyl]-1H-benzimidazole", is_fallback=True),
    "metoprolol": CompoundInfo(4171, "Metoprolol", "C15H25NO3", 267.36, "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O", "1-[4-(2-methoxyethyl)phenoxy]-3-(propan-2-ylamino)propan-2-ol", is_fallback=True),
    "amlodipine": CompoundInfo(2162, "Amlodipine", "C20H25ClN2O5", 408.88, "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN", "3-O-ethyl 5-O-methyl 2-(2-aminoethoxymethyl)-4-(2-chlorophenyl)-6-methyl-1,4-dihydropyridine-3,5-dicarboxylate", is_fallback=True),
    "warfarin": CompoundInfo(54678486, "Warfarin", "C19H16O4", 308.33, "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "4-hydroxy-3-(3-oxo-1-phenylbutyl)chromen-2-one", is_fallback=True),
    "ciprofloxacin": CompoundInfo(2764, "Ciprofloxacin", "C17H18FN3O3", 331.34, "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "1-cyclopropyl-6-fluoro-4-oxo-7-piperazin-1-ylquinoline-3-carboxylic acid", is_fallback=True),
    "dexamethasone": CompoundInfo(5743, "Dexamethasone", "C22H29FO5", 392.46, "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C", "(8S,9R,10S,11S,13S,14S,16R,17R)-9-fluoro-11,17-dihydroxy-17-(2-hydroxyacetyl)-10,13,16-trimethyl-6,7,8,11,12,14,15,16-octahydrocyclopenta[a]phenanthren-3-one", is_fallback=True),
    "penicillin": CompoundInfo(5904, "Penicillin G", "C16H18N2O4S", 334.39, "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", "(2S,5R,6R)-3,3-dimethyl-7-oxo-6-(2-phenylacetamido)-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid", is_fallback=True),
    "prednisone": CompoundInfo(5865, "Prednisone", "C21H26O5", 358.43, "CC12CC(=O)C3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C", "17,21-dihydroxypregna-1,4-diene-3,11,20-trione", is_fallback=True),
    "simvastatin": CompoundInfo(54454, "Simvastatin", "C25H38O5", 418.57, "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C", "[(1S,3R,7S,8S,8aR)-8-{2-[(2R,4R)-4-hydroxy-6-oxooxan-2-yl]ethyl}-3,7-dimethyl-1,2,3,7,8,8a-hexahydronaphthalen-1-yl] 2,2-dimethylbutanoate", is_fallback=True),
    "losartan": CompoundInfo(3961, "Losartan", "C22H23ClN6O", 422.91, "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl", "2-butyl-4-chloro-1-{[2'-(1H-tetrazol-5-yl)-[1,1'-biphenyl]-4-yl]methyl}-1H-imidazole-5-methanol", is_fallback=True),
    "gabapentin": CompoundInfo(3446, "Gabapentin", "C9H17NO2", 171.24, "C1CCC(CC1)(CC(=O)O)CN", "2-[1-(aminomethyl)cyclohexyl]acetic acid", is_fallback=True),
    "levothyroxine": CompoundInfo(5819, "Levothyroxine", "C15H11I4NO4", 776.87, "C1=CC(=C(C=C1CC(C(=O)O)N)I)OC2=CC(=C(C=C2)O)I", "(2S)-2-amino-3-[4-(4-hydroxy-3,5-diiodophenoxy)-3,5-diiodophenyl]propanoic acid", is_fallback=True),
}

# Build a CID-keyed index for fast lookup
_FALLBACK_BY_CID: dict[int, CompoundInfo] = {c.cid: c for c in _FALLBACK_COMPOUNDS.values()}

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CACHE_NS = "pubchem"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0  # seconds
_RATE_LIMIT = 5  # requests per second


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class PubChemClient:
    """Async PubChem PUG REST client with caching, rate limiting and fallback."""

    def __init__(self, cache: KnowledgeCache, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._cache = cache
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        # Simple token-bucket rate limiter state
        self._rate_tokens = _RATE_LIMIT
        self._rate_last_refill = time.monotonic()
        self._rate_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_by_smiles(self, smiles: str) -> CompoundInfo | None:
        """Look up a compound by SMILES string."""
        cache_key = f"smiles:{smiles}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return CompoundInfo(**cached)

        try:
            await self._wait_rate_limit()
            resp = await self._http.get(
                f"/compound/smiles/{httpx.QueryParams({'smiles': smiles}).get('smiles')}/JSON",
                params={"smiles": smiles},
            )
            resp.raise_for_status()
            data = resp.json()
            compound = self._parse_compound(data)
            if compound:
                await self._cache.set(_CACHE_NS, cache_key, asdict(compound), _CACHE_TTL)
            return compound
        except Exception as exc:
            logger.warning("PubChem SMILES lookup failed for %r: %s", smiles, exc)
            return self._fallback_by_smiles(smiles)

    async def search_by_name(self, name: str) -> CompoundInfo | None:
        """Look up a compound by common or IUPAC name."""
        cache_key = f"name:{name.lower()}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return CompoundInfo(**cached)

        try:
            await self._wait_rate_limit()
            resp = await self._http.get(f"/compound/name/{name}/JSON")
            resp.raise_for_status()
            data = resp.json()
            compound = self._parse_compound(data)
            if compound:
                await self._cache.set(_CACHE_NS, cache_key, asdict(compound), _CACHE_TTL)
            return compound
        except Exception as exc:
            logger.warning("PubChem name lookup failed for %r: %s", name, exc)
            return _FALLBACK_COMPOUNDS.get(name.lower())

    async def get_properties(self, cid: int, properties: list[str]) -> dict:
        """Fetch specific properties for a compound by CID."""
        props_str = ",".join(properties)
        cache_key = f"props:{cid}:{props_str}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return cached

        try:
            await self._wait_rate_limit()
            resp = await self._http.get(
                f"/compound/cid/{cid}/property/{props_str}/JSON",
            )
            resp.raise_for_status()
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            await self._cache.set(_CACHE_NS, cache_key, props, _CACHE_TTL)
            return props
        except Exception as exc:
            logger.warning("PubChem property lookup failed for CID %s: %s", cid, exc)
            fb = _FALLBACK_BY_CID.get(cid)
            if fb:
                return {p: getattr(fb, self._prop_map(p), None) for p in properties}
            return {}

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_compound(data: dict[str, Any]) -> CompoundInfo | None:
        """Extract a :class:`CompoundInfo` from PUG JSON response."""
        try:
            compounds = data["PC_Compounds"]
            if not compounds:
                return None
            c = compounds[0]
            cid = c["id"]["id"]["cid"]

            props: dict[str, Any] = {}
            for p in c.get("props", []):
                urn = p.get("urn", {})
                label = urn.get("label", "")
                name = urn.get("name", "")
                val = p.get("value", {})
                value = val.get("sval") or val.get("fval") or val.get("ival")

                if label == "IUPAC Name" and name == "Preferred":
                    props["iupac_name"] = value
                elif label == "Molecular Formula":
                    props["molecular_formula"] = value
                elif label == "Molecular Weight":
                    props["molecular_weight"] = float(value) if value else 0.0
                elif label == "SMILES" and name == "Canonical":
                    props["canonical_smiles"] = value

            # Try to get the compound name
            name = ""
            for p in c.get("props", []):
                urn = p.get("urn", {})
                if urn.get("label") == "IUPAC Name" and urn.get("name") == "Traditional":
                    name = p.get("value", {}).get("sval", "")
                    break
            if not name:
                name = props.get("iupac_name", "")

            return CompoundInfo(
                cid=cid,
                name=name,
                molecular_formula=props.get("molecular_formula", ""),
                molecular_weight=props.get("molecular_weight", 0.0),
                canonical_smiles=props.get("canonical_smiles", ""),
                iupac_name=props.get("iupac_name", ""),
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.debug("Failed to parse PubChem response: %s", exc)
            return None

    @staticmethod
    def _prop_map(api_name: str) -> str:
        """Map PubChem property API name to CompoundInfo attr."""
        mapping = {
            "MolecularFormula": "molecular_formula",
            "MolecularWeight": "molecular_weight",
            "CanonicalSMILES": "canonical_smiles",
            "IUPACName": "iupac_name",
        }
        return mapping.get(api_name, api_name.lower())

    def _fallback_by_smiles(self, smiles: str) -> CompoundInfo | None:
        """Try to find fallback data matching the SMILES string."""
        for compound in _FALLBACK_COMPOUNDS.values():
            if compound.canonical_smiles == smiles:
                return compound
        return None

    async def _wait_rate_limit(self) -> None:
        """Simple token-bucket rate limiter (5 req/s)."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._rate_last_refill
            self._rate_tokens = min(
                _RATE_LIMIT,
                self._rate_tokens + elapsed * _RATE_LIMIT,
            )
            self._rate_last_refill = now

            if self._rate_tokens < 1:
                wait = (1 - self._rate_tokens) / _RATE_LIMIT
                await asyncio.sleep(wait)
                self._rate_tokens = 0
            else:
                self._rate_tokens -= 1
