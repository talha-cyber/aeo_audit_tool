from __future__ import annotations

import secrets
from typing import Optional

import redis.asyncio as redis

from app.core.config import settings


class SessionManager:
    """
    Redis-backed session store with TTL.
    """

    def __init__(self, ttl_seconds: int = 3600, prefix: str = "sess:") -> None:
        self.ttl = ttl_seconds
        self.prefix = prefix
        self._client: Optional[redis.Redis] = None

    async def _client_or_create(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def create(self, user_id: str) -> str:
        sid = secrets.token_urlsafe(32)
        key = self.prefix + sid
        client = await self._client_or_create()
        await client.set(key, user_id, ex=self.ttl)
        return sid

    async def get_user(self, session_id: str) -> Optional[str]:
        client = await self._client_or_create()
        return await client.get(self.prefix + session_id)

    async def revoke(self, session_id: str) -> None:
        client = await self._client_or_create()
        await client.delete(self.prefix + session_id)
