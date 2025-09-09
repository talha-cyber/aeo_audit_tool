from __future__ import annotations

from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResilienceHealthChecker:
    """Collects basic health indicators for resilience components."""

    def __init__(self, redis_client: Optional[redis.Redis] = None) -> None:
        self._redis = redis_client

    async def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def dlq_depth(self, queue: str) -> int:
        client = await self._client()
        return await client.llen(f"dlq:{queue}")

    async def snapshot(
        self, *, dlq_queues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build a point-in-time snapshot of resilience health.
        Currently includes DLQ depths; circuits can be added by providing states.
        """
        data: Dict[str, Any] = {"dlq": {}}
        if dlq_queues:
            for q in dlq_queues:
                try:
                    data["dlq"][q] = await self.dlq_depth(q)
                except Exception as exc:  # noqa: BLE001
                    logger.error("dlq_depth_error", queue=q, error=str(exc))
                    data["dlq"][q] = None
        return data
