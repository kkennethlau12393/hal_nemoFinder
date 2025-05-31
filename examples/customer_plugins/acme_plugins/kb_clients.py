"""Acme Corp's internal knowledge-base client replacements.

These classes demonstrate how a customer swaps a built-in hal_nemoFinder
KB client (PubChem, ChEMBL, ...) for an internal mirror containing
proprietary data that must never leave the corporate network.

Each class exposes the **same method surface** as the built-in client
it replaces. hal_nemoFinder will instantiate them via the KB client
registry based on the ``kb_client_overrides`` section of
``hal_nemofinder.yaml``:

.. code-block:: yaml

    kb_client_overrides:
      pubchem: acme_plugins.kb_clients.AcmeCompoundDB
      chembl: acme_plugins.kb_clients.AcmeBioactivityDB

In production, replace the hardcoded dicts with real connection pools
to Oracle / Snowflake / internal REST APIs.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge.client_registry import register_kb_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AcmeCompoundDB — drop-in replacement for PubChemClient
# ---------------------------------------------------------------------------


_ACME_COMPOUND_MIRROR: dict[str, dict[str, Any]] = {
    "CC(=O)Oc1ccccc1C(=O)O": {
        "cid": "ACME-ASPIRIN",
        "molecular_weight": 180.16,
        "iupac_name": "2-acetoxybenzoic acid",
        "source": "acme_internal_mirror",
    },
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": {
        "cid": "ACME-IBU",
        "molecular_weight": 206.28,
        "iupac_name": "ibuprofen",
        "source": "acme_internal_mirror",
    },
}


class AcmeCompoundDB:
    """Drop-in replacement for :class:`PubChemClient`.

    Backed by Acme's internal compound mirror, which includes every
    PubChem entry plus the proprietary Acme series. Implements the
    same async surface as the built-in PubChem client so the rest of
    the framework can call it without any awareness of the swap.
    """

    def __init__(self, cache: Any | None = None) -> None:
        self._cache = cache
        logger.info("AcmeCompoundDB initialised (internal mirror)")

    async def close(self) -> None:
        """Release any open database connections."""
        logger.debug("AcmeCompoundDB closed")

    async def search_by_smiles(self, smiles: str) -> dict[str, Any] | None:
        """Look up a compound by SMILES against the internal mirror."""
        return _ACME_COMPOUND_MIRROR.get(smiles)

    async def search_by_name(self, name: str) -> dict[str, Any] | None:
        """Look up a compound by common name."""
        lower = name.lower()
        for record in _ACME_COMPOUND_MIRROR.values():
            if record["iupac_name"].lower() == lower:
                return record
        return None

    async def get_properties(self, cid: str) -> dict[str, Any] | None:
        """Fetch molecular properties by compound ID."""
        for record in _ACME_COMPOUND_MIRROR.values():
            if record["cid"] == cid:
                return record
        return None


# ---------------------------------------------------------------------------
# AcmeBioactivityDB — drop-in replacement for ChEMBLClient
# ---------------------------------------------------------------------------


_ACME_BIOACTIVITY_MIRROR: dict[tuple[str, str], dict[str, Any]] = {
    ("ASPIRIN", "PTGS1"): {
        "ic50_nm": 500.0,
        "source": "acme_mirror",
        "confidence_score": 8,
    },
    ("ASPIRIN", "PTGS2"): {
        "ic50_nm": 1700.0,
        "source": "acme_mirror",
        "confidence_score": 8,
    },
}


class AcmeBioactivityDB:
    """Drop-in replacement for :class:`ChEMBLClient`.

    Exposes Acme's curated bioactivity mirror alongside the public
    ChEMBL dataset. Proprietary assays are tagged with
    ``source="acme_internal"`` so downstream audits can distinguish
    them from public data.
    """

    def __init__(self, cache: Any | None = None) -> None:
        self._cache = cache
        logger.info("AcmeBioactivityDB initialised (internal mirror)")

    async def close(self) -> None:
        """Release connections."""

    async def search_bioactivity(
        self, compound: str, target: str
    ) -> dict[str, Any] | None:
        """Look up IC50 for a compound/target pair."""
        return _ACME_BIOACTIVITY_MIRROR.get(
            (compound.upper(), target.upper())
        )

    async def get_target_info(self, target: str) -> dict[str, Any] | None:
        """Fetch target metadata."""
        return {"target": target, "source": "acme_internal_mirror"}


# ---------------------------------------------------------------------------
# Optional self-registration
# ---------------------------------------------------------------------------
#
# Customers who prefer explicit registration (instead of listing these in
# kb_client_overrides in YAML) can uncomment the two calls below. The
# plugin loader will still honour KB_CLIENT_OVERRIDES if both are present.
#
# register_kb_client("pubchem", AcmeCompoundDB)
# register_kb_client("chembl", AcmeBioactivityDB)
