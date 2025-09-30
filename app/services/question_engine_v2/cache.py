"""Caching utilities for question engine v2."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class _CacheEntry:
    payload: str
    expires_at: Optional[float]


class QuestionEngineCache:
    """Redis-backed cache with graceful in-memory fallback."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        namespace: str = "qe:v2",
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        self._enabled = enabled
        self._ttl = ttl_seconds or settings.DYNAMIC_Q_CACHE_TTL
        self._namespace = namespace.rstrip(":")
        self._redis_url = redis_url or f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
        self._client: Optional[redis.Redis] = redis_client
        self._local_cache: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

        logger.debug(
            "QuestionEngineCache initialized",
            enabled=self._enabled,
            redis_url=self._redis_url,
            ttl_seconds=self._ttl,
            namespace=self._namespace,
            has_client=bool(self._client),
        )

    async def close(self) -> None:
        """Close underlying redis client if we created it."""
        if self._client and hasattr(self._client, "aclose"):
            try:
                await self._client.aclose()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to close cache client", error=str(exc))

    async def get(
        self,
        key: str,
        *,
        deserialize: Optional[Callable[[str], T]] = None,
    ) -> Optional[T]:
        """Fetch cached payload for a key."""

        if not self._enabled:
            return None

        namespaced_key = self._namespaced(key)
        entry = self._local_cache.get(namespaced_key)
        if entry and not self._is_expired(entry):
            return self._deserialize(entry.payload, deserialize)

        async with self._lock:
            entry = self._local_cache.get(namespaced_key)
            if entry and not self._is_expired(entry):
                return self._deserialize(entry.payload, deserialize)

            client = await self._ensure_client()
            if client is None:
                return None

            try:
                raw = await client.get(namespaced_key)
            except Exception as exc:
                logger.error("Cache get failed", key=namespaced_key, error=str(exc))
                return None

            if raw is None:
                return None

            payload = raw if isinstance(raw, str) else raw.decode("utf-8")
            self._local_cache[namespaced_key] = _CacheEntry(
                payload=payload,
                expires_at=self._expiry_timestamp(self._ttl),
            )

        return self._deserialize(payload, deserialize)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: Optional[int] = None,
        serializer: Optional[Callable[[Any], str]] = None,
    ) -> None:
        """Store payload in cache with optional TTL override."""

        if not self._enabled:
            return

        payload = self._serialize(value, serializer)
        expiry = ttl_seconds if ttl_seconds is not None else self._ttl
        namespaced_key = self._namespaced(key)

        self._local_cache[namespaced_key] = _CacheEntry(
            payload=payload,
            expires_at=self._expiry_timestamp(expiry),
        )

        client = await self._ensure_client()
        if client is None:
            return

        try:
            await client.set(namespaced_key, payload, ex=expiry if expiry else None)
        except Exception as exc:
            logger.error("Cache set failed", key=namespaced_key, error=str(exc))

    async def delete(self, key: str) -> None:
        """Remove a cached entry."""

        namespaced_key = self._namespaced(key)
        self._local_cache.pop(namespaced_key, None)

        client = await self._ensure_client()
        if client is None:
            return

        try:
            await client.delete(namespaced_key)
        except Exception as exc:
            logger.error("Cache delete failed", key=namespaced_key, error=str(exc))

    async def clear_local(self) -> None:
        """Clear in-memory fallback cache."""

        self._local_cache.clear()

    async def _ensure_client(self) -> Optional[redis.Redis]:
        if not self._enabled:
            return None

        if self._client is not None:
            return self._client

        try:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception as exc:
            logger.error("Failed to create redis client", error=str(exc))
            return None

        try:
            await self._client.ping()
        except Exception as exc:
            logger.error("Redis ping failed", error=str(exc))
            self._client = None
            return None

        return self._client

    def _namespaced(self, key: str) -> str:
        return f"{self._namespace}:{key}" if self._namespace else key

    @staticmethod
    def _serialize(value: Any, serializer: Optional[Callable[[Any], str]]) -> str:
        if serializer:
            return serializer(value)
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except TypeError:
            logger.warning("Cache value not JSON serializable; using repr fallback")
            return json.dumps({"__repr__": repr(value)})

    @staticmethod
    def _deserialize(payload: str, deserializer: Optional[Callable[[str], T]]) -> Optional[T]:
        if deserializer:
            return deserializer(payload)
        try:
            return json.loads(payload)  # type: ignore[return-value]
        except json.JSONDecodeError:
            return payload  # type: ignore[return-value]

    @staticmethod
    def _expiry_timestamp(ttl_seconds: Optional[int]) -> Optional[float]:
        if ttl_seconds is None or ttl_seconds <= 0:
            return None
        return time.monotonic() + ttl_seconds

    @staticmethod
    def _is_expired(entry: _CacheEntry) -> bool:
        if entry.expires_at is None:
            return False
        return time.monotonic() >= entry.expires_at


__all__ = ["QuestionEngineCache"]
