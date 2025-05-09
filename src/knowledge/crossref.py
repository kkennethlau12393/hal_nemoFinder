"""CrossRef API client for DOI validation and bibliographic metadata."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DOIInfo:
    doi: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    journal: str | None = None
    year: int | None = None
    abstract: str | None = None
    format_valid: bool = True
    is_fallback: bool = False


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_BASE_URL = "https://api.crossref.org/works"
_CACHE_NS = "crossref"
_CACHE_TTL = 259200  # 72 h
_DEFAULT_TIMEOUT = 10.0
_DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s]+$")


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class CrossRefClient:
    """Async CrossRef client with caching and fallback DOI format validation."""

    def __init__(self, cache: KnowledgeCache, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._cache = cache
        headers: dict[str, str] = {"Accept": "application/json"}
        email = os.environ.get("CROSSREF_EMAIL") or os.environ.get("HAL_CROSSREF_EMAIL")
        if email:
            headers["User-Agent"] = f"hal_nemoFinder/0.1.0 (mailto:{email})"
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate_doi(self, doi: str) -> DOIInfo | None:
        """Check if a DOI exists and return its metadata.

        On failure, falls back to regex-based format validation only.
        """
        cache_key = f"doi:{doi}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            # Restore tuple for authors
            cached["authors"] = tuple(cached.get("authors", []))
            return DOIInfo(**cached)

        try:
            resp = await self._http.get(f"/{doi}")
            resp.raise_for_status()
            data = resp.json().get("message", {})
            info = self._parse_work(data)
            await self._cache.set(_CACHE_NS, cache_key, asdict(info), _CACHE_TTL)
            return info
        except Exception as exc:
            logger.warning("CrossRef DOI validation failed for %r: %s", doi, exc)
            return self._format_fallback(doi)

    async def search_works(self, query: str, rows: int = 5) -> list[DOIInfo]:
        """Search CrossRef for works matching *query*."""
        cache_key = f"search:{query}:{rows}"
        cached = await self._cache.get(_CACHE_NS, cache_key)
        if cached is not None:
            results = []
            for item in cached:
                item["authors"] = tuple(item.get("authors", []))
                results.append(DOIInfo(**item))
            return results

        try:
            resp = await self._http.get(
                "",
                params={"query": query, "rows": rows},
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
            results = [self._parse_work(item) for item in items]
            await self._cache.set(_CACHE_NS, cache_key, [asdict(r) for r in results], _CACHE_TTL)
            return results
        except Exception as exc:
            logger.warning("CrossRef search failed for %r: %s", query, exc)
            return []

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_work(data: dict) -> DOIInfo:
        titles = data.get("title", [])
        title = titles[0] if titles else None

        authors_raw = data.get("author", [])
        authors = tuple(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors_raw
        )

        journal_raw = data.get("container-title", [])
        journal = journal_raw[0] if journal_raw else None

        year = None
        date_parts = (
            data.get("published-print", {}).get("date-parts")
            or data.get("published-online", {}).get("date-parts")
            or data.get("created", {}).get("date-parts")
        )
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        return DOIInfo(
            doi=data.get("DOI", ""),
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=data.get("abstract"),
        )

    @staticmethod
    def _format_fallback(doi: str) -> DOIInfo:
        """Return a DOIInfo with only format validation (no metadata)."""
        valid = bool(_DOI_RE.match(doi))
        return DOIInfo(doi=doi, format_valid=valid, is_fallback=True)
