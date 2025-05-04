"""Verification routing -- dispatch claims to the appropriate verifiers.

The :class:`VerificationRouter` consults the :class:`VerifierRegistry` to
find all verifiers capable of handling a given :class:`ClaimType`, then
executes them concurrently with per-verifier timeout and error isolation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from src.models.enums import ClaimType
from src.verifiers.base import (
    BaseVerifier,
    Verdict,
    VerificationOutput,
    VerifierRegistry,
    registry as _global_registry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class VerificationRouter:
    """Route claims to matching verifiers and run them in parallel.

    Parameters
    ----------
    registry : VerifierRegistry | None
        Registry to query.  Defaults to the global singleton.
    exclude_verifiers : set[str] | None
        Verifier names to skip (useful for disabling slow / optional
        backends in certain environments).
    default_timeout : float
        Per-verifier timeout in seconds.  ``0`` means no timeout.
    per_verifier_timeouts : dict[str, float] | None
        Override timeout for specific verifier names.
    """

    def __init__(
        self,
        registry: VerifierRegistry | None = None,
        exclude_verifiers: set[str] | None = None,
        default_timeout: float = 30.0,
        per_verifier_timeouts: dict[str, float] | None = None,
    ) -> None:
        self._registry = registry or _global_registry
        self._exclude = exclude_verifiers or set()
        self._default_timeout = default_timeout
        self._per_verifier_timeouts = per_verifier_timeouts or {}

    # -- Public API ---------------------------------------------------------

    def route_claim(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> list[tuple[str, Coroutine[Any, Any, VerificationOutput]]]:
        """Select verifiers and return ``(name, coroutine)`` pairs.

        The caller is responsible for awaiting the coroutines.  This is
        useful when you need fine-grained control over scheduling.

        Parameters
        ----------
        claim_text : str
            Raw claim sentence.
        claim_type : ClaimType
            Pre-classified type.
        context : dict
            Arbitrary metadata forwarded to each verifier.

        Returns
        -------
        list[tuple[str, Coroutine]]
        """
        verifiers = self._applicable_verifiers(claim_type)
        return [
            (v.name, v.verify(claim_text, claim_type, context))
            for v in verifiers
        ]

    async def verify_claim(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> list[VerificationOutput]:
        """Run all applicable verifiers in parallel and collect results.

        Individual verifier failures (exceptions or timeouts) are caught
        and logged -- they do **not** prevent other verifiers from
        completing.

        Parameters
        ----------
        claim_text : str
            Raw claim sentence.
        claim_type : ClaimType
            Pre-classified type.
        context : dict
            Arbitrary metadata forwarded to each verifier.

        Returns
        -------
        list[VerificationOutput]
            One entry per verifier that completed successfully.
        """
        verifiers = self._applicable_verifiers(claim_type)
        if not verifiers:
            logger.warning(
                "No verifiers registered for claim type %s", claim_type.value
            )
            return []

        tasks = [
            self._run_with_timeout(v, claim_text, claim_type, context)
            for v in verifiers
        ]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[VerificationOutput] = []
        for verifier, outcome in zip(verifiers, settled):
            if isinstance(outcome, VerificationOutput):
                # Stamp the verifier name onto source_db if the verifier
                # didn't already set it — the BayesianAggregator needs this
                # to look up per-verifier sensitivity/specificity.
                if not outcome.source_db:
                    outcome.source_db = verifier.name
                results.append(outcome)
                continue
            if isinstance(outcome, BaseException):
                logger.error(
                    "Verifier %r failed for claim: %s",
                    verifier.name,
                    outcome,
                    exc_info=outcome,
                )
                continue
            results.append(outcome)
        return results

    # -- Internals ----------------------------------------------------------

    def _applicable_verifiers(self, claim_type: ClaimType) -> list[BaseVerifier]:
        """Return registry verifiers for *claim_type*, minus exclusions."""
        return [
            v
            for v in self._registry.get_verifiers_for_type(claim_type)
            if v.name not in self._exclude
        ]

    def _timeout_for(self, verifier: BaseVerifier) -> float:
        """Resolve the timeout for a specific verifier."""
        return self._per_verifier_timeouts.get(
            verifier.name, self._default_timeout
        )

    async def _run_with_timeout(
        self,
        verifier: BaseVerifier,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        """Execute a single verifier with an optional timeout."""
        timeout = self._timeout_for(verifier)
        coro = verifier.verify(claim_text, claim_type, context)
        if timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro
