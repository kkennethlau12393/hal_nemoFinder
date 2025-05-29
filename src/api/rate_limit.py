"""Simple per-client fixed-bucket rate limiter.

The limiter uses Redis for shared state across API replicas. If Redis
is unavailable, it transparently falls back to a per-process in-memory
bucket — this keeps the API serving traffic during a Redis outage
instead of failing closed.  Customers on an HA deployment should treat
the in-memory path as a soft guarantee only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.observability import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Bucket:
    count: int
    window_start: float


class _InMemoryStore:
    """Per-process fallback bucket store guarded by a single lock."""

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def incr(self, key: str, window_seconds: int) -> int:
        now = time.time()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.window_start >= window_seconds:
                self._buckets[key] = _Bucket(count=1, window_start=now)
                return 1
            bucket.count += 1
            return bucket.count


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window token bucket keyed by API key or client IP.

    Two buckets are maintained per client:

    - ``global`` — every request counts against this.
    - ``analyze`` — only ``POST /api/v1/analyze*`` requests count.

    Buckets reset every minute (fixed window).  On breach the middleware
    returns ``429 Too Many Requests`` with a ``Retry-After`` header.
    """

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        rpm: int = 60,
        analyze_pm: int = 10,
        redis_url: Optional[str] = None,
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.rpm = rpm
        self.analyze_pm = analyze_pm
        self._memory = _InMemoryStore()
        self._redis_url = redis_url
        self._redis = None  # lazy-initialised async client

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis.asyncio as aioredis  # type: ignore

            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True, socket_timeout=1.0
            )
            # Probe once so we fail fast and fall through to memory.
            await self._redis.ping()
        except Exception:
            logger.warning("rate_limit.redis.unavailable", redis_url=self._redis_url)
            self._redis = None
        return self._redis

    async def _incr(self, key: str, window_seconds: int) -> int:
        redis = await self._get_redis()
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                count, _ = await pipe.execute()
                return int(count)
            except Exception:
                logger.warning("rate_limit.redis.error", key=key)
                # Drop to in-memory for this request.
        return self._memory.incr(key, window_seconds)

    @staticmethod
    def _client_key(request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        client = request.client
        host = client.host if client is not None else "unknown"
        return f"ip:{host}"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip internal endpoints — observability must always respond.
        path = request.url.path
        if path in ("/metrics",) or path.startswith("/api/v1/health"):
            return await call_next(request)

        # ---- Fixed 60-second window, epoch-aligned ---------------------
        window = int(time.time() // 60)
        client = self._client_key(request)

        global_key = f"rl:global:{client}:{window}"
        try:
            global_count = await self._incr(global_key, window_seconds=60)
        except Exception:
            # Never break request handling because of the limiter.
            logger.exception("rate_limit.incr.failed")
            return await call_next(request)

        if global_count > self.rpm:
            return self._rate_limited(self.rpm, window)

        if request.method == "POST" and path.startswith("/api/v1/analyze"):
            analyze_key = f"rl:analyze:{client}:{window}"
            analyze_count = await self._incr(analyze_key, window_seconds=60)
            if analyze_count > self.analyze_pm:
                return self._rate_limited(self.analyze_pm, window)

        return await call_next(request)

    @staticmethod
    def _rate_limited(limit: int, window: int) -> JSONResponse:
        retry_after = 60 - int(time.time()) % 60
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "limit_per_minute": limit,
            },
            headers={"Retry-After": str(max(retry_after, 1))},
        )
