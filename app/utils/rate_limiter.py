"""
Redis-backed token bucket rate limiting utilities.

Implements an async token bucket suitable for cross-process/platform rate limiting
with atomicity via Lua scripting. Prefer this over in-process limiters for
horizontal scaling as per project rules.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


LUA_TOKEN_BUCKET = """
-- KEYS[1] = bucket key
-- ARGV = now_ms, refill_rate_per_ms, capacity, tokens_needed
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local need = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if not tokens then tokens = capacity end
if not ts then ts = now end

-- refill
local elapsed = now - ts
local refill = elapsed * rate
tokens = math.min(capacity, tokens + refill)

local allowed = 0
if tokens >= need then
  tokens = tokens - need
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
-- Expire after a minute of inactivity
redis.call('PEXPIRE', key, 60000)

return {allowed, tokens}
"""


class RedisTokenBucket:
    def __init__(
        self, name: str, requests_per_minute: int, client: Optional[redis.Redis] = None
    ) -> None:
        self.name = name
        self.rpm = requests_per_minute
        self.capacity = max(1, requests_per_minute)
        # tokens per ms = rpm / 60000
        self.rate_per_ms = float(requests_per_minute) / 60000.0
        self._client = client
        self._script_sha: Optional[str] = None

    async def _client_or_create(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def _ensure_script(self, client: redis.Redis) -> str:
        if not self._script_sha:
            self._script_sha = await client.script_load(LUA_TOKEN_BUCKET)
        return self._script_sha

    async def acquire(self, tokens: int = 1) -> None:
        client = await self._client_or_create()
        sha = await self._ensure_script(client)
        key = f"rate:{self.name}"

        while True:
            try:
                now_ms = int(asyncio.get_event_loop().time() * 1000)
                res = await client.evalsha(
                    sha,
                    1,
                    key,
                    now_ms,
                    self.rate_per_ms,
                    self.capacity,
                    tokens,
                )
                allowed = int(res[0])
                if allowed == 1:
                    return
                # Not allowed; sleep for a small backoff
                await asyncio.sleep(0.05)
            except redis.RedisError as e:
                logger.error("redis_rate_limiter_error", error=str(e))
                # Fallback: sleep minimal delay to avoid burst
                await asyncio.sleep(0.05)
