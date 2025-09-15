import hashlib
import json
import pickle
from typing import Any, List, Optional

import redis.asyncio as redis
import structlog

from app.core.config import settings
from app.services import metrics

logger = structlog.get_logger(__name__)

# Global cache instance for backward compatibility
_cache_instance = None


def get_cache_client():
    """
    Get or create a global cache client instance.

    This function provides backward compatibility for modules that expect
    a get_cache_client() function while maintaining the CacheManager architecture.

    Returns:
        CacheManager: The global cache manager instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


class CacheManager:
    """
    An asynchronous Redis cache manager.

    This utility provides a simple, async-first interface for interacting with
    a Redis cache. It handles serialization (JSON with a pickle fallback) and
    provides methods for getting, setting, and deleting cache entries, as well
    as generating cache keys based on context.
    """

    _redis_client: Optional[redis.Redis] = None

    @classmethod
    async def get_client(cls) -> redis.Redis:
        """
        Retrieves the Redis client, creating it if it doesn't exist.

        This class method ensures that a single Redis client instance is shared
        across the application, which is a best practice for managing connections.

        Returns:
            An active `redis.Redis` client instance.
        """
        if cls._redis_client is None:
            try:
                cls._redis_client = redis.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                    encoding="utf-8",
                    decode_responses=True,
                )
                await cls._redis_client.ping()
                logger.info("Successfully connected to Redis.")
            except redis.ConnectionError as e:
                logger.error("Failed to connect to Redis.", error=str(e))
                # In a real-world scenario, you might have a fallback mechanism
                # or a more robust retry logic here.
                raise
        return cls._redis_client

    @classmethod
    async def get(cls, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The deserialized item, or None if the key is not found.
        """
        client = await cls.get_client()
        try:
            cached_data = await client.get(key)
            if cached_data:
                logger.info("Cache hit", key=key)
                metrics.CACHE_ACCESS_TOTAL.labels(
                    cache_name="dynamic_question_cache", result="hit"
                ).inc()
                return cls._deserialize(cached_data)
        except Exception as e:
            logger.error("Failed to retrieve from cache", key=key, error=str(e))
        logger.info("Cache miss", key=key)
        metrics.CACHE_ACCESS_TOTAL.labels(
            cache_name="dynamic_question_cache", result="miss"
        ).inc()
        return None

    @classmethod
    async def set(cls, key: str, value: Any, ttl: int) -> None:
        """
        Sets an item in the cache with a specified time-to-live (TTL).

        Args:
            key: The key for the item.
            value: The item to cache.
            ttl: The time-to-live for the item in seconds.
        """
        client = await cls.get_client()
        try:
            serialized_value = cls._serialize(value)
            await client.set(key, serialized_value, ex=ttl)
            logger.info("Successfully set cache item", key=key, ttl=ttl)
        except Exception as e:
            logger.error("Failed to set cache item", key=key, error=str(e))

    @classmethod
    async def flush_by_pattern(cls, pattern: str) -> int:
        """
        Flushes cache keys matching a given pattern.

        Args:
            pattern: The pattern to match keys against (e.g., "dynamic_q:*").

        Returns:
            The number of keys deleted.
        """
        client = await cls.get_client()
        keys_deleted = 0
        try:
            async for key in client.scan_iter(match=pattern):
                await client.delete(key)
                keys_deleted += 1
            logger.info("Cache flushed by pattern", pattern=pattern, count=keys_deleted)
        except Exception as e:
            logger.error(
                "Failed to flush cache by pattern", pattern=pattern, error=str(e)
            )
        return keys_deleted

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Serializes a value, trying JSON first and falling back to pickle."""
        try:
            return json.dumps(value).encode("utf-8")
        except (TypeError, OverflowError):
            logger.warning(
                "Could not serialize value with JSON, falling back to pickle."
            )
            return pickle.dumps(value)

    @staticmethod
    def _deserialize(value: bytes) -> Any:
        """Deserializes a value, trying JSON first and falling back to pickle."""
        try:
            return json.loads(value.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(
                "Could not deserialize value with JSON, falling back to pickle."
            )
            return pickle.loads(value)

    @staticmethod
    def generate_cache_key(
        industry: str, product_type: str, competitors: List[str]
    ) -> str:
        """
        Generates a deterministic cache key for the dynamic provider.

        The key is based on the industry, product type, and a hash of the
        sorted competitor list to ensure that the same query context results
        in the same cache key.

        Args:
            industry: The industry for the audit.
            product_type: The product type for the audit.
            competitors: The list of competitors.

        Returns:
            A unique cache key string.
        """
        competitors_hash = hashlib.md5(
            str(sorted(c.lower() for c in competitors)).encode()
        ).hexdigest()[:8]
        key = f"dynamic_q:{industry}:{product_type}:{competitors_hash}"
        logger.debug("Generated dynamic provider cache key", key=key)
        return key
