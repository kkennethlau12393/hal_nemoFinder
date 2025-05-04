"""Verifier plugin system.

Provides the abstract base class for all verification backends and a
singleton registry that maps :class:`~src.models.enums.ClaimType` values
to concrete verifier implementations.

Third-party verifiers only need to subclass :class:`BaseVerifier` and
decorate the class with :func:`register_verifier`::

    from src.verifiers.base import BaseVerifier, register_verifier
    from src.models.enums import ClaimType

    @register_verifier
    class PubChemVerifier(BaseVerifier):
        name = "pubchem"
        supported_claim_types = [ClaimType.molecular_property]

        async def verify(self, claim_text, claim_type, context):
            ...
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.models.enums import ClaimType, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verification output
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VerificationOutput:
    """Result produced by a single verifier for a single claim.

    Attributes
    ----------
    verdict : Verdict
        The verification outcome.
    confidence : float
        Verifier's self-assessed confidence in the verdict (0.0 -- 1.0).
    reasoning : str
        Human-readable explanation of how the verdict was reached.
    evidence : dict
        Structured evidence payload (varies per verifier).
    source_db : str
        Name of the authoritative database consulted (e.g. ``"ChEMBL"``).
    """

    verdict: Verdict
    confidence: float
    reasoning: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source_db: str = ""


# ---------------------------------------------------------------------------
# Abstract base verifier
# ---------------------------------------------------------------------------


class BaseVerifier(ABC):
    """Abstract base for all verification backends.

    Subclasses **must** set the class-level attributes :attr:`name` and
    :attr:`supported_claim_types` and implement :meth:`verify`.
    """

    #: Unique short name for this verifier (e.g. ``"chembl"``).
    name: str = ""

    #: Claim types this verifier can handle.
    supported_claim_types: list[ClaimType] = []

    @abstractmethod
    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        """Verify a single claim.

        Parameters
        ----------
        claim_text : str
            The raw claim sentence.
        claim_type : ClaimType
            Pre-classified claim type.
        context : dict
            Arbitrary metadata (e.g. surrounding text, job parameters).

        Returns
        -------
        VerificationOutput
        """

    async def health_check(self) -> bool:
        """Return ``True`` if the verifier's external dependencies are reachable.

        The default implementation always returns ``True``.  Override to
        add connectivity checks for external APIs or databases.
        """
        return True

    def supports(self, claim_type: ClaimType) -> bool:
        """Check whether this verifier handles *claim_type*."""
        return claim_type in self.supported_claim_types


# ---------------------------------------------------------------------------
# Verifier registry (thread-safe singleton)
# ---------------------------------------------------------------------------


class VerifierRegistry:
    """Singleton registry that maps claim types to verifier instances.

    Use :meth:`register` as a class decorator or call it manually.
    The registry is thread-safe and can be used from both sync and async
    code.

    Examples
    --------
    >>> registry = VerifierRegistry()
    >>> @registry.register
    ... class MyVerifier(BaseVerifier):
    ...     name = "my_verifier"
    ...     supported_claim_types = [ClaimType.citation]
    ...     async def verify(self, claim_text, claim_type, context):
    ...         ...
    """

    _instance: VerifierRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> VerifierRegistry:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._verifiers: dict[str, BaseVerifier] = {}
                cls._instance = inst
            return cls._instance

    # -- Registration -------------------------------------------------------

    def register(self, verifier_cls: type[BaseVerifier]) -> type[BaseVerifier]:
        """Register a verifier class (can be used as a decorator).

        An instance of *verifier_cls* is created immediately via its
        zero-argument constructor and stored in the registry.

        Parameters
        ----------
        verifier_cls : type[BaseVerifier]
            A concrete subclass of :class:`BaseVerifier`.

        Returns
        -------
        type[BaseVerifier]
            The same class, unchanged -- so it can be used as a decorator.

        Raises
        ------
        TypeError
            If *verifier_cls* is not a subclass of :class:`BaseVerifier`.
        ValueError
            If *verifier_cls* does not define a ``name``.
        """
        if not (isinstance(verifier_cls, type) and issubclass(verifier_cls, BaseVerifier)):
            raise TypeError(
                f"Expected a BaseVerifier subclass, got {verifier_cls!r}"
            )
        instance = verifier_cls()
        if not instance.name:
            raise ValueError(
                f"{verifier_cls.__name__} must define a non-empty 'name' attribute"
            )
        with self._lock:
            if instance.name in self._verifiers:
                logger.warning("Overwriting previously registered verifier %r", instance.name)
            self._verifiers[instance.name] = instance
        logger.debug("Registered verifier %r", instance.name)
        return verifier_cls

    # -- Lookup -------------------------------------------------------------

    def get_verifiers_for_type(self, claim_type: ClaimType) -> list[BaseVerifier]:
        """Return all registered verifiers that support *claim_type*."""
        with self._lock:
            return [v for v in self._verifiers.values() if v.supports(claim_type)]

    def get_all(self) -> list[BaseVerifier]:
        """Return all registered verifier instances."""
        with self._lock:
            return list(self._verifiers.values())

    def get_by_name(self, name: str) -> BaseVerifier:
        """Return the verifier registered under *name*.

        Raises
        ------
        KeyError
            If no verifier with that name is registered.
        """
        with self._lock:
            return self._verifiers[name]

    # -- Housekeeping -------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered verifiers.  Intended for testing."""
        with self._lock:
            self._verifiers.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._verifiers)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._verifiers


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

#: Global registry instance.
registry = VerifierRegistry()


def register_verifier(cls: type[BaseVerifier]) -> type[BaseVerifier]:
    """Decorator shortcut that registers a verifier class in the global registry."""
    return registry.register(cls)


def get_verifier_registry() -> VerifierRegistry:
    """Return the global :class:`VerifierRegistry` singleton."""
    return registry
