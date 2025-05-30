"""Knowledge base client layer for hal-nemoFinder.

Provides cached, fault-tolerant clients for external biomedical and
bibliographic data sources used during hallucination verification.

Quick start::

    from src.config import settings
    from src.knowledge import create_knowledge_base

    kb = create_knowledge_base(settings)
    compound = await kb.pubchem.search_by_name("aspirin")
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .cache import KnowledgeCache, create_cache
from .client_registry import (
    get_kb_client,
    list_kb_clients,
    override_kb_client,
    register_kb_client,
)
from .chembl import (
    BioactivityRecord,
    ChEMBLClient,
    MoleculeInfo,
    ProvenanceRecord,
    TargetInfo,
)
from .clinicaltrials import ClinicalTrialsClient, StudyInfo
from .crossref import CrossRefClient, DOIInfo
from .ddi import DrugInteraction, DrugInteractionClient
from .drugbank import DrugBankClient, DrugInfo
from .embeddings import EmbeddingService
from .faers import AdverseEventProfile, AdverseReaction, FAERSClient
from .labels import DoseRange, DrugLabel, DrugLabelClient
from .patents import PatentClient, PatentRecord
from .pathways import PathwayClient, PathwayInfo
from .pdb import PDBClient, PDBEntry
from .pubchem import CompoundInfo, PubChemClient
from .pubmed import PubMedClient, PubMedRecord
from .triangulation import Disagreement, TriangulationResult, TriangulationService
from .uniprot import ProteinInfo, UniProtClient

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    # Container
    "KnowledgeBase",
    "create_knowledge_base",
    # Registry
    "get_kb_client",
    "list_kb_clients",
    "override_kb_client",
    "register_kb_client",
    # Cache
    "KnowledgeCache",
    "create_cache",
    # Clients
    "PubChemClient",
    "ChEMBLClient",
    "CrossRefClient",
    "UniProtClient",
    "ClinicalTrialsClient",
    "DrugBankClient",
    "EmbeddingService",
    "PathwayClient",
    "TriangulationService",
    "PubMedClient",
    "FAERSClient",
    "DrugInteractionClient",
    "PDBClient",
    "PatentClient",
    "DrugLabelClient",
    # Data models
    "CompoundInfo",
    "MoleculeInfo",
    "BioactivityRecord",
    "TargetInfo",
    "ProvenanceRecord",
    "DOIInfo",
    "ProteinInfo",
    "StudyInfo",
    "DrugInfo",
    "PathwayInfo",
    "TriangulationResult",
    "Disagreement",
    "PubMedRecord",
    "AdverseEventProfile",
    "AdverseReaction",
    "DrugInteraction",
    "PDBEntry",
    "PatentRecord",
    "DrugLabel",
    "DoseRange",
]


@dataclass
class KnowledgeBase:
    """Container holding all initialized knowledge-base clients.

    Created via :func:`create_knowledge_base` which wires a shared
    :class:`KnowledgeCache` into every client.
    """

    cache: KnowledgeCache
    pubchem: PubChemClient
    chembl: ChEMBLClient
    crossref: CrossRefClient
    uniprot: UniProtClient
    clinicaltrials: ClinicalTrialsClient
    drugbank: DrugBankClient
    triangulation: TriangulationService
    embeddings: EmbeddingService
    pathways: PathwayClient
    pubmed: PubMedClient
    faers: FAERSClient
    ddi: DrugInteractionClient
    pdb: PDBClient
    patents: PatentClient
    labels: DrugLabelClient

    async def close(self) -> None:
        """Gracefully shut down all HTTP clients."""
        await self.pubchem.close()
        await self.chembl.close()
        await self.crossref.close()
        await self.uniprot.close()
        await self.clinicaltrials.close()
        await self.drugbank.close()
        await self.pubmed.close()
        await self.faers.close()
        await self.pdb.close()
        await self.labels.close()


def _apply_kb_overrides(settings: Settings) -> None:
    """Apply ``settings.KB_CLIENT_OVERRIDES`` to the client registry.

    For each ``name -> "module.ClassName"`` entry the dotted path is
    imported, validated, and registered via
    :func:`override_kb_client`.  Failures are logged but never raise —
    a broken override must not take the whole app down; the built-in
    client remains active instead.
    """
    overrides: dict[str, str] = getattr(settings, "KB_CLIENT_OVERRIDES", {}) or {}
    for name, path in overrides.items():
        if not isinstance(path, str) or "." not in path:
            logger.error(
                "Invalid KB_CLIENT_OVERRIDES entry %r=%r "
                "(expected 'module.ClassName')",
                name,
                path,
            )
            continue
        module_name, class_name = path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to import KB override %r for %r", path, name
            )
            continue
        if not isinstance(cls, type):
            logger.error(
                "KB override %r for %r is not a class", path, name
            )
            continue
        override_kb_client(name, cls)


def _build_client(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate the KB client registered under *name*.

    Looks up the class in the registry (honoring any overrides that
    have been applied) and instantiates it with the supplied
    constructor arguments.

    Raises
    ------
    KeyError
        If no client is registered under *name*.
    """
    cls = get_kb_client(name)
    if cls is None:
        raise KeyError(f"No knowledge-base client registered under {name!r}")
    return cls(*args, **kwargs)


def create_knowledge_base(settings: Settings) -> KnowledgeBase:
    """Factory that creates all clients with a shared Redis cache.

    Any entries in ``settings.KB_CLIENT_OVERRIDES`` are applied first,
    so each built-in client may be transparently replaced with a
    customer-provided subclass (e.g. a mirror of PubChem behind a
    corporate firewall).

    Parameters
    ----------
    settings:
        Application :class:`~src.config.Settings` instance.  The factory
        reads ``REDIS_URL``, ``ENABLE_EMBEDDINGS``, ``DRUGBANK_API_KEY``,
        and ``KB_CLIENT_OVERRIDES`` from it.

    Returns
    -------
    KnowledgeBase
        Fully initialized container — ready for ``await`` calls.
    """
    _apply_kb_overrides(settings)

    cache = create_cache(settings.REDIS_URL)

    pubchem = _build_client("pubchem", cache)
    chembl = _build_client("chembl", cache)
    # Resolve the DrugBank API key through the pluggable secret
    # provider.  Customers running Vault / AWS Secrets Manager can
    # override the provider via ``SECRET_PROVIDER`` without code
    # changes.  The lookup is synchronous-friendly: the default env
    # provider has zero I/O overhead and any async provider will have
    # the value cached by the event loop before this factory runs.
    drugbank_api_key: str | None = None
    try:
        from src.secrets import EnvSecretProvider, SecretNotFoundError, get_secret_provider

        provider = get_secret_provider(settings)
        if isinstance(provider, EnvSecretProvider):
            # Fast-path: env provider is sync-safe, avoid spinning up a
            # fresh event loop inside this factory.
            import os

            drugbank_api_key = os.environ.get("DRUGBANK_API_KEY") or os.environ.get(
                "HAL_DRUGBANK_API_KEY"
            )
            if drugbank_api_key is None:
                drugbank_api_key = getattr(settings, "DRUGBANK_API_KEY", None)
        else:
            import asyncio

            async def _fetch() -> str | None:
                try:
                    return await provider.get_secret("DRUGBANK_API_KEY")
                except (SecretNotFoundError, NotImplementedError):
                    return getattr(settings, "DRUGBANK_API_KEY", None)

            try:
                drugbank_api_key = asyncio.run(_fetch())
            except RuntimeError:
                # Already inside a running loop — best-effort fall back
                # to the settings value.
                drugbank_api_key = getattr(settings, "DRUGBANK_API_KEY", None)
    except Exception:  # noqa: BLE001
        logger.exception("secrets.drugbank.lookup_failed")
        drugbank_api_key = getattr(settings, "DRUGBANK_API_KEY", None)

    drugbank = _build_client(
        "drugbank",
        cache,
        api_key=drugbank_api_key,
    )
    triangulation = TriangulationService(
        pubchem=pubchem,
        chembl=chembl,
        drugbank=drugbank,
    )

    return KnowledgeBase(
        cache=cache,
        pubchem=pubchem,
        chembl=chembl,
        crossref=_build_client("crossref", cache),
        uniprot=_build_client("uniprot", cache),
        clinicaltrials=_build_client("clinicaltrials", cache),
        drugbank=drugbank,
        triangulation=triangulation,
        embeddings=EmbeddingService(enabled=settings.ENABLE_EMBEDDINGS),
        pathways=_build_client("pathways"),
        pubmed=PubMedClient(cache=cache),
        faers=FAERSClient(cache=cache),
        ddi=DrugInteractionClient(),
        pdb=PDBClient(cache=cache),
        patents=PatentClient(),
        labels=DrugLabelClient(cache=cache),
    )
