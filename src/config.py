"""Application settings for hal-nemoFinder.

Settings are populated, in order of increasing precedence, from:

1. Hard-coded defaults below.
2. A YAML config file — either ``$HAL_CONFIG_FILE`` or, if unset,
   ``./hal_nemofinder.yaml`` or ``./config.yaml`` in the current working
   directory.  Values inside the YAML file may be nested; nested keys
   are flattened with underscores and uppercased (``database.url`` ->
   ``DATABASE_URL``).
3. Environment variables prefixed with ``HAL_`` (e.g. ``HAL_DATABASE_URL``)
   — env vars always override YAML.
4. ``.env`` file in the current working directory.

The precedence order follows the twelve-factor convention: configuration
baked into the image (YAML) is overridable at deploy time via env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_DEFAULT_YAML_CANDIDATES = ("hal_nemofinder.yaml", "config.yaml")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _flatten_yaml(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into an UPPER_SNAKE_CASE dict.

    ``{"database": {"url": "..."}}`` becomes ``{"DATABASE_URL": "..."}``.
    Non-dict values at any depth are returned as-is.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict) and value and all(
            isinstance(k, str) for k in value.keys()
        ):
            # Only recurse if every child key is a string and the dict
            # is non-empty — preserves dict-valued settings like
            # VERIFIER_RELIABILITY_OVERRIDES.
            if _is_structural_dict(value):
                flat.update(_flatten_yaml(value, new_key))
                continue
        flat[new_key.upper()] = value
    return flat


def _is_structural_dict(value: dict[str, Any]) -> bool:
    """Return True if *value* looks like nested configuration groups.

    Heuristic: all values are either dicts or scalars — i.e. the dict
    groups sub-settings rather than being itself a setting.  Dict-valued
    settings (like ``KB_CLIENT_OVERRIDES`` which maps names to import
    paths) will have scalar leaf values, so we only treat a dict as
    "structural" if at least one child is itself a dict.
    """
    return any(isinstance(v, dict) for v in value.values())


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load *path* as YAML and return a flattened settings dict.

    Returns ``{}`` if the file is missing or empty.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "PyYAML is not installed; skipping YAML config file at %s", p
        )
        return {}

    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse YAML config %s", p)
        return {}

    if not isinstance(raw, dict):
        logger.warning("YAML config %s must be a mapping at the top level", p)
        return {}
    return _flatten_yaml(raw)


def _resolve_yaml_path() -> Path | None:
    """Find the active YAML config file, if any."""
    env_path = os.environ.get("HAL_CONFIG_FILE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        logger.warning("HAL_CONFIG_FILE=%s does not exist", env_path)
        return None
    for candidate in _DEFAULT_YAML_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Central configuration for hal-nemoFinder.

    Every field can be overridden via an environment variable prefixed
    with ``HAL_`` (e.g. ``HAL_DATABASE_URL``) or declared in a YAML
    config file (see the module docstring).
    """

    # -- Database ---------------------------------------------------------------
    DATABASE_URL: str = (
        "postgresql+asyncpg://hal:hal@localhost:5432/hal_nemofinder"
    )
    SYNC_DATABASE_URL: str = (
        "postgresql+psycopg2://hal:hal@localhost:5432/hal_nemofinder"
    )

    # -- Redis / Celery ---------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # -- External API keys ------------------------------------------------------
    PUBCHEM_API_KEY: str | None = None
    CROSSREF_EMAIL: str | None = None  # used for Crossref polite pool
    DRUGBANK_API_KEY: str | None = None

    # -- Tuning -----------------------------------------------------------------
    CACHE_TTL_SECONDS: int = 3600
    VERIFICATION_TIMEOUT_SECONDS: int = 30
    CONFIDENCE_THRESHOLD: float = 0.7

    # -- Cost tracking ----------------------------------------------------------
    #: Dollar cost per CPU-hour used by :class:`src.mlops.cost_tracker.CostTracker`.
    #: Defaults to a typical commodity-cloud vCPU on-demand price.
    COST_CPU_PER_HOUR: float = 0.10

    # -- Feature flags ----------------------------------------------------------
    ENABLE_EMBEDDINGS: bool = True

    # -- Observability ----------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console"  # "json" or "console"
    ENVIRONMENT: str = "development"
    VERSION: str = "0.1.0"

    # -- Rate limiting ----------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_RPM: int = 60
    RATE_LIMIT_ANALYZE_PM: int = 10

    # -- Plugin discovery -------------------------------------------------------
    PLUGIN_ENTRY_POINT_GROUPS: list[str] = [
        "hal_nemofinder.verifiers",
        "hal_nemofinder.knowledge_clients",
        "hal_nemofinder.aggregators",
        "hal_nemofinder.extractors",
    ]
    #: Dotted module paths to import at startup (triggers decorators).
    PLUGIN_MODULES: list[str] = []
    #: Absolute filesystem paths to ``.py`` files to load at startup.
    PLUGIN_FILES: list[str] = []

    # -- Knowledge base overrides -----------------------------------------------
    #: Map of client_name -> Python import path (``module.ClassName``).
    #: Each entry replaces the built-in client of the same name.
    #: Example: ``{"pubchem": "mycompany.internal.PubChemMirror"}``.
    KB_CLIENT_OVERRIDES: dict[str, str] = {}

    # -- Regression sets --------------------------------------------------------
    #: List of file paths. Each is JSON (list) or JSONL (one per line).
    REGRESSION_SET_PATHS: list[str] = []

    # -- Bayesian reliability overrides -----------------------------------------
    #: Customers calibrate verifier reliability from their own data.
    #: Example: ``{"chemical": {"sensitivity": 0.97, "specificity": 0.94}}``.
    VERIFIER_RELIABILITY_OVERRIDES: dict[str, dict[str, float]] = {}

    # -- Auth -------------------------------------------------------------------
    #: When ``False`` (the default for backwards compatibility) the API is
    #: reachable without credentials and every request is associated with
    #: a synthetic "default" tenant.  Turn this on in production.
    AUTH_REQUIRED: bool = False
    #: A long-lived shared secret used exactly once to create the first
    #: tenant + admin user via ``hal-nemofinder auth bootstrap``.  Must
    #: only be set at install time; delete it after the first admin key
    #: has been issued.
    AUTH_BOOTSTRAP_KEY: str | None = None
    #: OIDC issuer URL, e.g. ``https://login.example.com/``.  When set,
    #: ``Authorization: Bearer <jwt>`` tokens are validated against this
    #: provider's JWKS document.
    OIDC_ISSUER: str | None = None
    OIDC_CLIENT_ID: str | None = None
    OIDC_CLIENT_SECRET: str | None = None
    OIDC_AUDIENCE: str | None = None

    # -- Multi-tenancy ----------------------------------------------------------
    #: Slug of the tenant used when auth is disabled or when bootstrapping.
    DEFAULT_TENANT_SLUG: str = "default"

    # -- Audit logging ----------------------------------------------------------
    #: Master switch for the HMAC-chained audit log.  When True the
    #: application refuses to start unless ``AUDIT_HMAC_KEY`` is set.
    AUDIT_ENABLED: bool = True
    #: HMAC secret used to sign every audit entry.  Generate with
    #: ``openssl rand -hex 32``.  Rotating this key breaks the chain
    #: from the point of rotation, so plan for it carefully.
    AUDIT_HMAC_KEY: str | None = None
    #: When True, startup runs :meth:`AuditRecorder.verify_integrity`
    #: and aborts if the chain is broken.
    AUDIT_REQUIRE_INTEGRITY: bool = True

    # -- Secrets provider -------------------------------------------------------
    #: Which SecretProvider implementation to use.  One of:
    #: ``env`` | ``file`` | ``vault`` | ``aws`` | ``gcp`` | ``azure``.
    SECRET_PROVIDER: str = "env"
    #: Path to a JSON/YAML file used by :class:`FileSecretProvider`.
    SECRETS_FILE: str | None = None
    #: HashiCorp Vault URL (e.g. ``https://vault.example.com:8200``).
    VAULT_URL: str | None = None
    #: Name of the secret (read via another provider — typically env)
    #: that holds the Vault auth token.
    VAULT_TOKEN_SECRET_NAME: str | None = None

    # -- Config file ------------------------------------------------------------
    #: Optional path to a YAML config file. If set, YAML values are
    #: loaded before env vars are applied.
    CONFIG_FILE: str | None = None

    model_config = {
        "env_prefix": "HAL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, **overrides: Any) -> None:
        """Merge YAML config (if present) with constructor kwargs.

        YAML values are injected as kwargs before pydantic evaluates
        environment variables, so the resulting precedence is:

        ``defaults < YAML < env < explicit kwargs``.
        """
        yaml_data: dict[str, Any] = {}
        yaml_path = Path(overrides.pop("_yaml_path", "")) if "_yaml_path" in overrides else None
        if yaml_path is None:
            yaml_path = _resolve_yaml_path()
        if yaml_path is not None:
            yaml_data = _load_yaml_file(yaml_path)

        # Drop YAML keys that are already provided as env vars — env
        # vars must win over YAML.
        env_prefix = "HAL_"
        for key in list(yaml_data.keys()):
            if f"{env_prefix}{key}" in os.environ or key in os.environ:
                yaml_data.pop(key, None)

        # Explicit overrides passed to the constructor win outright.
        merged = {**yaml_data, **overrides}

        super().__init__(**merged)
        if yaml_path is not None:
            # Record which file was applied, for observability.
            object.__setattr__(self, "CONFIG_FILE", str(yaml_path))

    # ------------------------------------------------------------------
    # YAML round-trip helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Build a :class:`Settings` instance from a specific YAML file.

        Unlike the default constructor (which auto-discovers a YAML
        file), this method loads *path* unconditionally.  Environment
        variables still take precedence over YAML values.
        """
        return cls(_yaml_path=str(path))

    def save_yaml(self, path: str | Path) -> None:
        """Write the current settings to *path* as a flat YAML mapping.

        Useful for generating a bootstrap config file or snapshotting
        the effective configuration of a running process.
        """
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required for Settings.save_yaml; "
                "install with `pip install pyyaml`."
            ) from exc

        data = self.model_dump()
        with Path(path).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=True, default_flow_style=False)


settings = Settings()
