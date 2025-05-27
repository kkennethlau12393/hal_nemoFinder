"""OpenID Connect / JWT validation.

When a customer configures an OIDC issuer (via ``OIDC_ISSUER``,
``OIDC_AUDIENCE`` etc. in :mod:`src.config`) the API will accept
``Authorization: Bearer <jwt>`` tokens signed by that issuer's JWKS.
The validator:

* Downloads the JWKS document from ``<issuer>/.well-known/openid-configuration``
  (or a direct ``jwks_url``) and caches it for one hour.
* Verifies the RS256/ES256 signature, the ``iss`` claim, the ``aud``
  claim (if configured), and the ``exp`` / ``nbf`` claims.
* Returns the decoded payload on success so callers can map the
  ``sub`` claim to a local :class:`src.models.tenant.User`.

If OIDC is not configured the returned validator is a
:class:`NoOpValidator` that rejects every bearer token with a clear
error — the caller can still fall back to API-key authentication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from src.observability import get_logger

logger = get_logger(__name__)

#: JWKS cache lifetime in seconds.  One hour matches the recommended
#: default for most commercial IdPs.
_JWKS_CACHE_TTL_SECONDS: int = 3600


class OIDCError(Exception):
    """Raised when a bearer token fails validation."""


@dataclass
class OIDCConfig:
    """Static configuration for the OIDC validator.

    Only ``issuer_url`` is strictly required; the JWKS URL is derived
    from it when not explicitly supplied.
    """

    issuer_url: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    audience: Optional[str] = None
    jwks_url: Optional[str] = None

    def resolved_jwks_url(self) -> str:
        """Return the JWKS URL, deriving from the issuer if needed."""
        if self.jwks_url:
            return self.jwks_url
        base = self.issuer_url.rstrip("/")
        return f"{base}/.well-known/jwks.json"


class OIDCValidator:
    """Validate JWTs against a remote OIDC issuer's JWKS.

    This class is safe to share across requests: the JWKS cache is
    in-memory and refreshed opportunistically when the first call after
    :data:`_JWKS_CACHE_TTL_SECONDS` observes a stale timestamp.
    """

    def __init__(self, config: OIDCConfig) -> None:
        self.config = config
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0

    # ------------------------------------------------------------------
    # JWKS
    # ------------------------------------------------------------------

    def _jwks_is_stale(self) -> bool:
        if self._jwks is None:
            return True
        return (time.time() - self._jwks_fetched_at) > _JWKS_CACHE_TTL_SECONDS

    def _fetch_jwks(self) -> dict[str, Any]:
        """Synchronously fetch (and cache) the issuer's JWKS document."""
        url = self.config.resolved_jwks_url()
        logger.info("oidc.jwks.fetch", url=url)
        try:
            response = httpx.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("oidc.jwks.fetch_failed", url=url, error=str(exc))
            raise OIDCError(f"Failed to fetch JWKS from {url}: {exc}") from exc

        if not isinstance(data, dict) or "keys" not in data:
            raise OIDCError(
                f"JWKS response from {url} is missing the 'keys' field"
            )
        self._jwks = data
        self._jwks_fetched_at = time.time()
        return data

    def _get_jwks(self) -> dict[str, Any]:
        if self._jwks_is_stale():
            return self._fetch_jwks()
        assert self._jwks is not None  # narrowed by _jwks_is_stale
        return self._jwks

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_bearer_token(self, token: str) -> dict[str, Any]:
        """Validate *token* and return the decoded JWT payload.

        Raises :class:`OIDCError` on any problem — expired, invalid
        signature, wrong audience, wrong issuer, missing key.  Callers
        should translate the error into an HTTP 401 response.
        """
        try:
            from jose import jwt  # type: ignore[import-not-found]
            from jose.exceptions import JWTError  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise OIDCError(
                "python-jose is not installed; OIDC validation disabled. "
                "Install with `pip install python-jose[cryptography]`."
            ) from exc

        jwks = self._get_jwks()
        try:
            unverified_header = jwt.get_unverified_header(token)
        except JWTError as exc:
            raise OIDCError(f"Malformed JWT header: {exc}") from exc

        kid = unverified_header.get("kid")
        key = None
        for candidate in jwks.get("keys", []):
            if candidate.get("kid") == kid:
                key = candidate
                break
        if key is None:
            # Possibly a key rotation — force a refresh and try once more.
            logger.info("oidc.jwks.cache_miss", kid=kid)
            jwks = self._fetch_jwks()
            for candidate in jwks.get("keys", []):
                if candidate.get("kid") == kid:
                    key = candidate
                    break
        if key is None:
            raise OIDCError(f"No JWKS key matches kid={kid!r}")

        try:
            payload = jwt.decode(
                token,
                key,
                algorithms=[unverified_header.get("alg", "RS256")],
                audience=self.config.audience,
                issuer=self.config.issuer_url,
                options={
                    "verify_aud": self.config.audience is not None,
                    "verify_iss": True,
                    "verify_exp": True,
                },
            )
        except JWTError as exc:
            raise OIDCError(f"JWT validation failed: {exc}") from exc

        return payload


class NoOpValidator:
    """Stand-in validator used when OIDC is not configured.

    It always rejects bearer tokens — the caller should fall back to
    API-key authentication before invoking this validator, and the
    error raised here is meant to produce a clean HTTP 401.
    """

    def validate_bearer_token(self, token: str) -> dict[str, Any]:
        raise OIDCError(
            "OIDC is not configured on this deployment; "
            "bearer tokens are not accepted."
        )


def build_oidc_validator(settings: Any) -> OIDCValidator | NoOpValidator:
    """Return an :class:`OIDCValidator` if configured, else a no-op."""
    issuer = getattr(settings, "OIDC_ISSUER", None)
    if not issuer:
        return NoOpValidator()
    config = OIDCConfig(
        issuer_url=issuer,
        client_id=getattr(settings, "OIDC_CLIENT_ID", None),
        client_secret=getattr(settings, "OIDC_CLIENT_SECRET", None),
        audience=getattr(settings, "OIDC_AUDIENCE", None),
    )
    return OIDCValidator(config)
