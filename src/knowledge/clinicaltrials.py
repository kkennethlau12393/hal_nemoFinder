"""ClinicalTrials.gov v2 API client with caching and fallback validation."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StudyInfo:
    nct_id: str
    title: str | None
    status: str | None
    phase: str | None
    conditions: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    start_date: str | None = None
    completion_date: str | None = None
    format_valid: bool = True
    is_fallback: bool = False


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_CACHE_NS = "clinicaltrials"
_CACHE_TTL = 86400  # 24 h
_DEFAULT_TIMEOUT = 10.0
_NCT_RE = re.compile(r"^NCT\d{8}$")


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class ClinicalTrialsClient:
    """Async ClinicalTrials.gov v2 API client with caching and format fallback."""

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

    async def get_study(self, nct_id: str) -> StudyInfo | None:
        """Fetch a single clinical study by NCT ID."""
        cache_key = f"study:{nct_id}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return self._from_cache(cached)

        try:
            resp = await self._http.get(f"/{nct_id}")
            resp.raise_for_status()
            data = resp.json()
            study = self._parse_study(data)
            if study:
                await self._cache.set(_CACHE_NS, cache_key, asdict(study), _CACHE_TTL)
            return study
        except Exception as exc:
            logger.warning("ClinicalTrials.gov lookup failed for %s: %s", nct_id, exc)
            return self._format_fallback(nct_id)

    async def search_studies(self, query: str, max_results: int = 10) -> list[StudyInfo]:
        """Search ClinicalTrials.gov for studies matching *query*."""
        cache_key = f"search:{query}:{max_results}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            return [self._from_cache(s) for s in cached]

        try:
            resp = await self._http.get(
                "",
                params={
                    "query.term": query,
                    "pageSize": max_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            studies = []
            for item in data.get("studies", []):
                study = self._parse_study(item)
                if study:
                    studies.append(study)
            await self._cache.set(
                _CACHE_NS, cache_key, [asdict(s) for s in studies], _CACHE_TTL,
            )
            return studies
        except Exception as exc:
            logger.warning("ClinicalTrials.gov search failed for %r: %s", query, exc)
            return []

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_study(data: dict) -> StudyInfo | None:
        """Parse ClinicalTrials.gov v2 JSON into :class:`StudyInfo`."""
        try:
            proto = data.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conditions_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})

            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle") or ident.get("officialTitle")
            status = status_mod.get("overallStatus")

            # Phase
            phases = design.get("phases", [])
            phase = ", ".join(phases) if phases else None

            # Conditions
            conditions = tuple(conditions_mod.get("conditions", []))

            # Interventions
            interventions_raw = arms_mod.get("interventions", [])
            interventions = tuple(
                iv.get("name", "") for iv in interventions_raw if iv.get("name")
            )

            # Dates
            start_date = status_mod.get("startDateStruct", {}).get("date")
            completion_date = status_mod.get("completionDateStruct", {}).get("date")

            return StudyInfo(
                nct_id=nct_id,
                title=title,
                status=status,
                phase=phase,
                conditions=conditions,
                interventions=interventions,
                start_date=start_date,
                completion_date=completion_date,
            )
        except (KeyError, TypeError) as exc:
            logger.debug("Failed to parse ClinicalTrials.gov study: %s", exc)
            return None

    @staticmethod
    def _format_fallback(nct_id: str) -> StudyInfo:
        """Return a StudyInfo with only NCT ID format validation."""
        valid = bool(_NCT_RE.match(nct_id))
        return StudyInfo(
            nct_id=nct_id,
            title=None,
            status=None,
            phase=None,
            format_valid=valid,
            is_fallback=True,
        )

    @staticmethod
    def _from_cache(data: dict) -> StudyInfo:
        data["conditions"] = tuple(data.get("conditions", []))
        data["interventions"] = tuple(data.get("interventions", []))
        return StudyInfo(**data)
