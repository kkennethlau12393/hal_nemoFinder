"""Redis-backed cache with namespaced keys and graceful degradation."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class KnowledgeCache:
    """Thin async wrapper around Redis for knowledge-base caching.

    Keys are namespaced as ``{namespace}:{key}`` so different clients
    can share a single Redis instance without collisions.  All values
    are stored as JSON strings.

    If Redis is unreachable every method degrades gracefully — it logs a
    warning and returns *None* (for gets) or silently skips writes.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value or *None* on miss / error."""
        full_key = self._make_key(namespace, key)
        try:
            raw = await self._redis.get(full_key)
            if raw is None:
                return None
            return json.loads(raw)
        except aioredis.RedisError as exc:
            logger.warning("Cache GET failed for %s: %s", full_key, exc)
            return None
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Cache deserialization failed for %s: %s", full_key, exc)
            return None

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: int = 3600,
    ) -> None:
        """Store *value* (JSON-serializable) under ``namespace:key``."""
        full_key = self._make_key(namespace, key)
        try:
            raw = json.dumps(value)
            await self._redis.set(full_key, raw, ex=ttl)
        except aioredis.RedisError as exc:
            logger.warning("Cache SET failed for %s: %s", full_key, exc)
        except (TypeError, ValueError) as exc:
            logger.warning("Cache serialization failed for %s: %s", full_key, exc)

    async def invalidate(self, namespace: str, key: str) -> None:
        """Remove a single cached entry."""
        full_key = self._make_key(namespace, key)
        try:
            await self._redis.delete(full_key)
        except aioredis.RedisError as exc:
            logger.warning("Cache INVALIDATE failed for %s: %s", full_key, exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(namespace: str, key: str) -> str:
        return f"{namespace}:{key}"


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_cache(redis_url: str) -> KnowledgeCache:
    """Create a :class:`KnowledgeCache` backed by *redis_url*."""
    client = aioredis.from_url(redis_url, decode_responses=True)
    return KnowledgeCache(client)
