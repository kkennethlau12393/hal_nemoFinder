"""Pluggable secrets abstraction.

Customers running hal-nemoFinder in regulated environments rarely want
their secrets sitting in environment variables.  This module defines a
:class:`SecretProvider` protocol plus a handful of concrete
implementations (env, file) and stubs for the major cloud offerings
(Vault, AWS, GCP, Azure).

Typical usage::

    from src.secrets import get_secret_provider
    from src.config import settings

    provider = get_secret_provider(settings)
    api_key = await provider.get_secret("DRUGBANK_API_KEY")

The framework ships with ``env`` as the default provider so existing
deployments continue to work unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SecretNotFoundError(KeyError):
    """Raised when a named secret cannot be found in the backend."""


@runtime_checkable
class SecretProvider(Protocol):
    """Minimal interface every secret backend must implement.

    The two optional methods (``set_secret`` and ``list_secrets``) are
    provided so hal-nemoFinder's CLI can offer consistent secret-
    management ergonomics across backends that support write access.
    Providers that do not implement them should raise
    :class:`NotImplementedError`.
    """

    async def get_secret(self, name: str) -> str:  # pragma: no cover - protocol
        ...

    async def set_secret(
        self, name: str, value: str
    ) -> None:  # pragma: no cover - protocol
        ...

    async def list_secrets(self) -> list[str]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Env provider (default)
# ---------------------------------------------------------------------------


class EnvSecretProvider:
    """Read secrets from environment variables.

    The provider normalises lookups by trying both the bare name and
    the ``HAL_``-prefixed form used by :class:`~src.config.Settings`,
    so either ``DRUGBANK_API_KEY`` or ``HAL_DRUGBANK_API_KEY`` works.
    """

    def __init__(self, *, prefix: str = "HAL_") -> None:
        self._prefix = prefix

    async def get_secret(self, name: str) -> str:
        if name in os.environ:
            return os.environ[name]
        prefixed = f"{self._prefix}{name}"
        if prefixed in os.environ:
            return os.environ[prefixed]
        raise SecretNotFoundError(
            f"Secret {name!r} not found in environment (also tried {prefixed!r})"
        )

    async def set_secret(self, name: str, value: str) -> None:
        os.environ[name] = value

    async def list_secrets(self) -> list[str]:
        return sorted(os.environ.keys())


# ---------------------------------------------------------------------------
# File provider (Docker secrets, Kubernetes files)
# ---------------------------------------------------------------------------


class FileSecretProvider:
    """Read secrets from a JSON or YAML file on disk.

    Well suited to Docker secrets, Kubernetes mounted files, or
    Vault-Agent-rendered templates.  The file is reloaded on every
    ``get_secret`` call so rotations are picked up without a process
    restart.  Use :class:`FileSecretProvider.cached` to disable this
    if filesystem reads become a bottleneck.
    """

    def __init__(self, path: str | Path, *, cached: bool = False) -> None:
        self._path = Path(path)
        self._cached = cached
        self._cache: Optional[dict[str, Any]] = None

    def _load(self) -> dict[str, Any]:
        if self._cached and self._cache is not None:
            return self._cache
        if not self._path.exists():
            raise SecretNotFoundError(
                f"Secrets file {self._path} does not exist"
            )
        text = self._path.read_text(encoding="utf-8")
        data: Any
        if self._path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML required for YAML secrets files"
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Secrets file {self._path} must contain a mapping at top level"
            )
        if self._cached:
            self._cache = data
        return data

    async def get_secret(self, name: str) -> str:
        data = self._load()
        if name not in data:
            raise SecretNotFoundError(
                f"Secret {name!r} not in {self._path}"
            )
        return str(data[name])

    async def set_secret(self, name: str, value: str) -> None:
        data = self._load() if self._path.exists() else {}
        data[name] = value
        self._path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        if self._cached:
            self._cache = data

    async def list_secrets(self) -> list[str]:
        try:
            return sorted(self._load().keys())
        except SecretNotFoundError:
            return []


# ---------------------------------------------------------------------------
# Cloud provider stubs
# ---------------------------------------------------------------------------


class HashiCorpVaultProvider:
    """Stub provider illustrating the Vault KV v2 integration shape.

    Example (for customers to implement in their fork)::

        # read from kv v2 path "secret/data/hal_nemofinder"
        client = hvac.Client(url=self._url, token=self._token)
        resp = client.secrets.kv.v2.read_secret_version(
            path=f"hal_nemofinder/{name}",
            mount_point="secret",
        )
        return resp["data"]["data"]["value"]
    """

    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._token = token

    async def get_secret(self, name: str) -> str:
        raise NotImplementedError(
            "HashiCorpVaultProvider is a stub. Install `hvac` and implement "
            "the KV v2 read path for your deployment."
        )

    async def set_secret(self, name: str, value: str) -> None:
        raise NotImplementedError

    async def list_secrets(self) -> list[str]:
        raise NotImplementedError


class AWSSecretsManagerProvider:
    """Stub — use boto3 ``secretsmanager`` client in a real deployment.

    Example::

        resp = await asyncio.to_thread(
            self._client.get_secret_value, SecretId=name
        )
        return resp["SecretString"]
    """

    def __init__(self, region: str, prefix: str = "hal_nemofinder/") -> None:
        self._region = region
        self._prefix = prefix

    async def get_secret(self, name: str) -> str:
        raise NotImplementedError(
            "AWSSecretsManagerProvider is a stub. Install `boto3` and "
            "implement get_secret_value for your deployment."
        )

    async def set_secret(self, name: str, value: str) -> None:
        raise NotImplementedError

    async def list_secrets(self) -> list[str]:
        raise NotImplementedError


class GCPSecretManagerProvider:
    """Stub — use ``google-cloud-secret-manager`` in a real deployment.

    Example::

        name = f"projects/{project}/secrets/{name}/versions/latest"
        resp = await asyncio.to_thread(self._client.access_secret_version, name=name)
        return resp.payload.data.decode("utf-8")
    """

    def __init__(self, project: str) -> None:
        self._project = project

    async def get_secret(self, name: str) -> str:
        raise NotImplementedError(
            "GCPSecretManagerProvider is a stub. Install "
            "`google-cloud-secret-manager` and implement "
            "access_secret_version for your deployment."
        )

    async def set_secret(self, name: str, value: str) -> None:
        raise NotImplementedError

    async def list_secrets(self) -> list[str]:
        raise NotImplementedError


class AzureKeyVaultProvider:
    """Stub — use ``azure-keyvault-secrets`` in a real deployment.

    Example::

        secret = await self._client.get_secret(name)
        return secret.value
    """

    def __init__(self, vault_url: str) -> None:
        self._vault_url = vault_url

    async def get_secret(self, name: str) -> str:
        raise NotImplementedError(
            "AzureKeyVaultProvider is a stub. Install "
            "`azure-keyvault-secrets` and implement get_secret for your "
            "deployment."
        )

    async def set_secret(self, name: str, value: str) -> None:
        raise NotImplementedError

    async def list_secrets(self) -> list[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_secret_provider(settings: Any) -> SecretProvider:
    """Return a :class:`SecretProvider` matching ``settings.SECRET_PROVIDER``.

    Unknown providers fall back to :class:`EnvSecretProvider` with a
    warning rather than raising, so a typo in config never brings down
    the process.
    """
    provider_name = getattr(settings, "SECRET_PROVIDER", "env") or "env"
    name = provider_name.lower()

    if name == "env":
        return EnvSecretProvider()

    if name == "file":
        path = getattr(settings, "SECRETS_FILE", None)
        if not path:
            logger.warning(
                "SECRET_PROVIDER=file but SECRETS_FILE is unset; "
                "falling back to env provider"
            )
            return EnvSecretProvider()
        return FileSecretProvider(path)

    if name == "vault":
        url = getattr(settings, "VAULT_URL", None)
        token_secret_name = getattr(settings, "VAULT_TOKEN_SECRET_NAME", None)
        if not url or not token_secret_name:
            logger.warning(
                "SECRET_PROVIDER=vault requires VAULT_URL and "
                "VAULT_TOKEN_SECRET_NAME; falling back to env"
            )
            return EnvSecretProvider()
        token = os.environ.get(token_secret_name, "")
        return HashiCorpVaultProvider(url=url, token=token)

    if name == "aws":
        region = os.environ.get("AWS_REGION", "us-east-1")
        return AWSSecretsManagerProvider(region=region)

    if name == "gcp":
        project = os.environ.get("GCP_PROJECT", "")
        return GCPSecretManagerProvider(project=project)

    if name == "azure":
        vault_url = os.environ.get("AZURE_KEY_VAULT_URL", "")
        return AzureKeyVaultProvider(vault_url=vault_url)

    logger.warning(
        "Unknown SECRET_PROVIDER=%r; falling back to env", provider_name
    )
    return EnvSecretProvider()


# ---------------------------------------------------------------------------
# Convenience helper for synchronous callers
# ---------------------------------------------------------------------------


async def resolve_secret(settings: Any, name: str, *, default: Optional[str] = None) -> Optional[str]:
    """Fetch *name* via the configured provider, returning *default* on miss.

    Convenience for call-sites that previously read directly from
    :class:`Settings` and cannot handle :class:`SecretNotFoundError`.
    """
    provider = get_secret_provider(settings)
    try:
        return await provider.get_secret(name)
    except SecretNotFoundError:
        return default
    except NotImplementedError:
        logger.warning("Secret provider stub raised NotImplementedError for %s", name)
        return default


__all__ = [
    "SecretProvider",
    "SecretNotFoundError",
    "EnvSecretProvider",
    "FileSecretProvider",
    "HashiCorpVaultProvider",
    "AWSSecretsManagerProvider",
    "GCPSecretManagerProvider",
    "AzureKeyVaultProvider",
    "get_secret_provider",
    "resolve_secret",
]
