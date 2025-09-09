from __future__ import annotations

import json
from typing import Any, Dict

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def requeue_original(message: Dict[str, Any], original_queue: str) -> bool:
    """
    Send message back to its original queue.
    Returns True on success.
    """
    client = redis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        encoding="utf-8",
        decode_responses=True,
    )
    await client.lpush(original_queue, json.dumps(message))
    logger.info("requeued_to_original", queue=original_queue)
    return True


async def park_in_dlq(*_: Any, **__: Any) -> bool:
    """
    No-op strategy: keep the message parked in DLQ for manual inspection.
    Returns False to indicate it remains in DLQ.
    """
    return False
