"""Authentication, authorization, and multi-tenancy for hal_nemoFinder.

This package groups everything related to *who* is calling the API and
*what* they are allowed to do once authenticated:

* :mod:`src.auth.api_keys` — opaque bearer-key generation and hashing.
* :mod:`src.auth.oidc` — JWT / JWKS validation for OpenID Connect.
* :mod:`src.auth.rbac` — roles, permissions, and the role → permission
  mapping.
* :mod:`src.auth.dependencies` — FastAPI dependency injection primitives
  that surface an :class:`AuthContext` for every request handler.

Downstream code should import from this package rather than from the
individual submodules so that internals can be refactored freely.
"""

from src.auth.api_keys import (
    API_KEY_PLAINTEXT_PREFIX,
    API_KEY_PREFIX_LENGTH,
    extract_prefix,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from src.auth.dependencies import (
    AuthContext,
    get_auth_context,
    get_current_tenant,
    require_auth,
    require_permission,
)
from src.auth.oidc import NoOpValidator, OIDCConfig, OIDCValidator, build_oidc_validator
from src.auth.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    has_permission,
)

__all__ = [
    # api_keys
    "API_KEY_PLAINTEXT_PREFIX",
    "API_KEY_PREFIX_LENGTH",
    "extract_prefix",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    # oidc
    "NoOpValidator",
    "OIDCConfig",
    "OIDCValidator",
    "build_oidc_validator",
    # rbac
    "Permission",
    "ROLE_PERMISSIONS",
    "has_permission",
    # dependencies
    "AuthContext",
    "get_auth_context",
    "get_current_tenant",
    "require_auth",
    "require_permission",
]
