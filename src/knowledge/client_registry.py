"""Registry for knowledge-base clients.

This is a dead-simple name -> class mapping that lets customers:

* register brand-new clients (e.g. an in-house proprietary database), or
* **override** an existing built-in client (e.g. swap ``PubChemClient``
  for ``PubChemMirrorClient`` that talks to an internal mirror).

Built-in clients self-register at import time via
:func:`register_kb_client`, so they become available as soon as
``src.knowledge`` is imported.  Customer plugins call
:func:`register_kb_client` (for new clients) or
:func:`override_kb_client` (for replacements) from within their own
module's import-time code — typically triggered by the
:mod:`src.plugins` loader.

Example::

    from src.knowledge.client_registry import register_kb_client

    @register_kb_client("internal_db")
    class InternalDBClient:
        def __init__(self, cache): ...
        async def close(self): ...
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)


_T = TypeVar("_T", bound=type)

_lock = threading.Lock()
_clients: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_kb_client(name: str, cls: type | None = None) -> Callable[[_T], _T] | type:
    """Register a knowledge-base client class under *name*.

    Can be used either as a plain function call or as a class decorator::

        register_kb_client("pubchem", PubChemClient)

        @register_kb_client("chembl")
        class ChEMBLClient:
            ...

    If a client with the same name is already registered, a warning is
    logged and the new class is **ignored** — use
    :func:`override_kb_client` for explicit replacements.

    Parameters
    ----------
    name:
        Short name used to look up the client (e.g. ``"pubchem"``).
    cls:
        The client class.  If omitted, this function acts as a
        decorator factory.
    """
    if cls is None:
        def _decorator(klass: _T) -> _T:
            register_kb_client(name, klass)
            return klass

        return _decorator

    with _lock:
        if name in _clients:
            logger.warning(
                "KB client %r is already registered (%s); "
                "use override_kb_client to replace it. Ignoring new %s.",
                name,
                _clients[name].__name__,
                cls.__name__,
            )
            return cls
        _clients[name] = cls
    logger.debug("Registered KB client %r -> %s", name, cls.__name__)
    return cls


def override_kb_client(name: str, cls: type) -> None:
    """Explicitly replace the class registered under *name*.

    Unlike :func:`register_kb_client`, this function always wins —
    intended for swapping a built-in client for a customer-specific
    variant via :data:`Settings.KB_CLIENT_OVERRIDES`.
    """
    with _lock:
        previous = _clients.get(name)
        _clients[name] = cls
    if previous is not None:
        logger.info(
            "Overriding KB client %r: %s -> %s",
            name,
            previous.__name__,
            cls.__name__,
        )
    else:
        logger.info("Registered new KB client %r -> %s", name, cls.__name__)


def get_kb_client(name: str) -> type | None:
    """Return the class registered under *name*, or ``None``."""
    with _lock:
        return _clients.get(name)


def list_kb_clients() -> list[str]:
    """Return every registered KB client name, sorted."""
    with _lock:
        return sorted(_clients)


def clear_kb_clients() -> None:
    """Wipe the registry (intended for tests)."""
    with _lock:
        _clients.clear()


# ---------------------------------------------------------------------------
# Built-in client self-registration
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register the built-in clients shipped with hal-nemoFinder.

    This is a function (rather than module-level code) so that it can
    be called lazily and doesn't trigger circular imports between
    ``src.knowledge`` and ``src.knowledge.client_registry``.
    """
    try:
        from .chembl import ChEMBLClient
        from .clinicaltrials import ClinicalTrialsClient
        from .crossref import CrossRefClient
        from .drugbank import DrugBankClient
        from .pathways import PathwayClient
        from .pubchem import PubChemClient
        from .uniprot import UniProtClient
    except Exception:  # noqa: BLE001
        logger.exception("Failed to import built-in KB clients for registration")
        return

    for name, cls in {
        "pubchem": PubChemClient,
        "chembl": ChEMBLClient,
        "crossref": CrossRefClient,
        "uniprot": UniProtClient,
        "clinicaltrials": ClinicalTrialsClient,
        "drugbank": DrugBankClient,
        "pathways": PathwayClient,
    }.items():
        # Use direct assignment to avoid the "already registered" warning
        # on re-imports (e.g. Celery autodiscovery).
        with _lock:
            _clients.setdefault(name, cls)


_register_builtins()
