"""FDA Adverse Event Reporting System (FAERS) client via openFDA.

Provides a deterministic lookup of adverse event counts for a drug from
the openFDA ``/drug/event.json`` endpoint, with a hand-curated fallback
covering ~20 common drugs so the framework functions offline.

Data model
----------
Each drug resolves to a :class:`AdverseEventProfile` listing ranked
adverse reactions (reaction term, report count, serious flag).  The
verifier layer (``src.verifiers.adverse_events``) consumes this to
cross-check "no adverse events" / "the only side effect is X" claims.
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
class AdverseReaction:
    term: str
    count: int
    serious: bool = False


@dataclass(frozen=True, slots=True)
class AdverseEventProfile:
    drug: str
    total_reports: int
    reactions: tuple[AdverseReaction, ...] = field(default_factory=tuple)
    is_fallback: bool = False

    def top_terms(self, n: int = 20) -> list[str]:
        """Return the top-*n* reaction terms (lower-cased)."""
        return [r.term.lower() for r in list(self.reactions)[:n]]

    def count_for(self, term: str) -> int:
        term_l = term.lower()
        for r in self.reactions:
            if r.term.lower() == term_l:
                return r.count
        return 0


# ---------------------------------------------------------------------------
# Fallback profiles (~20 drugs)
# ---------------------------------------------------------------------------


def _profile(drug: str, total: int, *reactions: tuple[str, int, bool]) -> AdverseEventProfile:
    return AdverseEventProfile(
        drug=drug,
        total_reports=total,
        reactions=tuple(AdverseReaction(t, c, s) for t, c, s in reactions),
        is_fallback=True,
    )


_FALLBACK_PROFILES: dict[str, AdverseEventProfile] = {
    "aspirin": _profile(
        "aspirin", 48000,
        ("gastrointestinal haemorrhage", 6800, True),
        ("nausea", 4200, False),
        ("dyspepsia", 3100, False),
        ("tinnitus", 1900, False),
        ("bronchospasm", 1500, True),
        ("urticaria", 1200, False),
        ("reye syndrome", 210, True),
    ),
    "metformin": _profile(
        "metformin", 95000,
        ("diarrhoea", 14200, False),
        ("nausea", 11500, False),
        ("abdominal pain", 8800, False),
        ("lactic acidosis", 3400, True),
        ("vitamin b12 deficiency", 2100, False),
        ("decreased appetite", 1600, False),
    ),
    "simvastatin": _profile(
        "simvastatin", 62000,
        ("myalgia", 9800, False),
        ("rhabdomyolysis", 5200, True),
        ("myopathy", 4100, True),
        ("increased blood creatine phosphokinase", 3300, False),
        ("hepatitis", 1800, True),
    ),
    "atorvastatin": _profile(
        "atorvastatin", 78000,
        ("myalgia", 12100, False),
        ("rhabdomyolysis", 3800, True),
        ("myopathy", 3200, True),
        ("memory impairment", 2100, False),
        ("hepatic enzyme increased", 1900, False),
    ),
    "warfarin": _profile(
        "warfarin", 105000,
        ("haemorrhage", 22000, True),
        ("international normalised ratio increased", 14500, False),
        ("epistaxis", 6100, False),
        ("haematuria", 3800, False),
        ("intracranial haemorrhage", 3100, True),
    ),
    "ibuprofen": _profile(
        "ibuprofen", 54000,
        ("gastrointestinal haemorrhage", 5200, True),
        ("renal failure", 3800, True),
        ("dyspepsia", 3600, False),
        ("nausea", 2900, False),
        ("hypertension", 1500, False),
    ),
    "acetaminophen": _profile(
        "acetaminophen", 72000,
        ("hepatotoxicity", 11200, True),
        ("rash", 2300, False),
        ("nausea", 2100, False),
        ("drug-induced liver injury", 6800, True),
    ),
    "clopidogrel": _profile(
        "clopidogrel", 58000,
        ("haemorrhage", 8900, True),
        ("thrombotic thrombocytopenic purpura", 510, True),
        ("bruising", 3200, False),
        ("epistaxis", 2100, False),
    ),
    "omeprazole": _profile(
        "omeprazole", 68000,
        ("diarrhoea", 5100, False),
        ("headache", 4200, False),
        ("clostridium difficile colitis", 1900, True),
        ("hypomagnesaemia", 1800, True),
        ("fracture", 1200, False),
    ),
    "metoprolol": _profile(
        "metoprolol", 49000,
        ("bradycardia", 6100, True),
        ("hypotension", 5200, False),
        ("fatigue", 3100, False),
        ("dizziness", 2900, False),
    ),
    "amlodipine": _profile(
        "amlodipine", 42000,
        ("peripheral oedema", 7900, False),
        ("headache", 3200, False),
        ("dizziness", 2700, False),
        ("flushing", 1800, False),
    ),
    "lisinopril": _profile(
        "lisinopril", 52000,
        ("cough", 9200, False),
        ("angioedema", 3100, True),
        ("hyperkalaemia", 2600, True),
        ("hypotension", 2200, False),
    ),
    "levothyroxine": _profile(
        "levothyroxine", 38000,
        ("palpitations", 3100, False),
        ("weight decreased", 2200, False),
        ("anxiety", 1700, False),
        ("atrial fibrillation", 950, True),
    ),
    "fluoxetine": _profile(
        "fluoxetine", 64000,
        ("nausea", 6800, False),
        ("insomnia", 5100, False),
        ("suicidal ideation", 3400, True),
        ("serotonin syndrome", 1100, True),
        ("sexual dysfunction", 2800, False),
    ),
    "sertraline": _profile(
        "sertraline", 61000,
        ("nausea", 6400, False),
        ("diarrhoea", 4100, False),
        ("insomnia", 3900, False),
        ("suicidal ideation", 2900, True),
    ),
    "imatinib": _profile(
        "imatinib", 41000,
        ("oedema", 5200, False),
        ("nausea", 4100, False),
        ("myelosuppression", 3600, True),
        ("hepatotoxicity", 1800, True),
        ("cardiac failure", 950, True),
    ),
    "erlotinib": _profile(
        "erlotinib", 29000,
        ("rash", 9200, False),
        ("diarrhoea", 4800, False),
        ("interstitial lung disease", 1100, True),
        ("hepatotoxicity", 820, True),
    ),
    "sildenafil": _profile(
        "sildenafil", 22000,
        ("headache", 4100, False),
        ("flushing", 2900, False),
        ("visual disturbance", 1800, False),
        ("priapism", 310, True),
        ("myocardial infarction", 450, True),
    ),
    "warfarin_nsaid": _profile(
        "warfarin+nsaid", 8900,
        ("gastrointestinal haemorrhage", 3200, True),
        ("international normalised ratio increased", 2100, False),
    ),
    "pembrolizumab": _profile(
        "pembrolizumab", 36000,
        ("immune-mediated pneumonitis", 2300, True),
        ("colitis", 1900, True),
        ("hypothyroidism", 2100, False),
        ("hepatitis", 1100, True),
        ("rash", 2700, False),
    ),
    "dexamethasone": _profile(
        "dexamethasone", 45000,
        ("hyperglycaemia", 4100, False),
        ("insomnia", 2900, False),
        ("mood altered", 2200, False),
        ("adrenal suppression", 1800, True),
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.fda.gov/drug/event.json"
_CACHE_NS = "faers"
_CACHE_TTL = 86400
_DEFAULT_TIMEOUT = 15.0


class FAERSClient:
    """Async openFDA FAERS client with caching and offline fallback."""

    def __init__(
        self,
        cache: KnowledgeCache | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._cache = cache
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    async def search_adverse_events(
        self, drug_name: str, limit: int = 100
    ) -> AdverseEventProfile:
        """Return the aggregated adverse-event profile for *drug_name*."""
        cache_key = f"ae:{drug_name.lower()}:{limit}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return _profile_from_dict(cached)

        try:
            resp = await self._http.get(
                _BASE_URL,
                params={
                    "search": (
                        f'patient.drug.medicinalproduct:"{drug_name}"'
                        f' patient.drug.openfda.generic_name:"{drug_name}"'.replace(
                            "  ", " "
                        )
                    ),
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": str(limit),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            profile = self._parse_profile(drug_name, data)
            if profile.reactions:
                await self._cache_set(cache_key, asdict(profile))
                return profile
        except Exception as exc:
            logger.warning("FAERS query failed for %r: %s", drug_name, exc)
        return _FALLBACK_PROFILES.get(drug_name.lower(), AdverseEventProfile(drug_name, 0))

    async def get_reaction_counts(self, drug_name: str) -> dict[str, int]:
        """Return ``{reaction_term: count}`` mapping for *drug_name*."""
        profile = await self.search_adverse_events(drug_name)
        return {r.term: r.count for r in profile.reactions}

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_profile(drug_name: str, data: dict[str, Any]) -> AdverseEventProfile:
        results = data.get("results", [])
        reactions = tuple(
            AdverseReaction(term=str(r.get("term", "")), count=int(r.get("count", 0)))
            for r in results
        )
        total = sum(r.count for r in reactions)
        return AdverseEventProfile(drug=drug_name, total_reports=total, reactions=reactions)

    async def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        try:
            return await self._cache.get(_CACHE_NS, key)
        except Exception as exc:  # pragma: no cover
            logger.debug("FAERS cache GET failed: %s", exc)
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.set(_CACHE_NS, key, value, _CACHE_TTL)
        except Exception as exc:  # pragma: no cover
            logger.debug("FAERS cache SET failed: %s", exc)

    @staticmethod
    def fallback_profile(drug_name: str) -> AdverseEventProfile | None:
        return _FALLBACK_PROFILES.get(drug_name.lower())


def _profile_from_dict(data: dict[str, Any]) -> AdverseEventProfile:
    reactions = tuple(
        AdverseReaction(**r) if isinstance(r, dict) else r
        for r in data.get("reactions", ())
    )
    return AdverseEventProfile(
        drug=str(data.get("drug", "")),
        total_reports=int(data.get("total_reports", 0)),
        reactions=reactions,
        is_fallback=bool(data.get("is_fallback", False)),
    )
