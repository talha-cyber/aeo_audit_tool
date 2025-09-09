from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DLQMessage:
    original_message: Dict[str, Any]
    original_queue: str
    error: str
    timestamp: str
    retry_count: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "original_message": self.original_message,
                "original_queue": self.original_queue,
                "error": self.error,
                "timestamp": self.timestamp,
                "retry_count": self.retry_count,
            }
        )

    @staticmethod
    def from_json(data: str) -> "DLQMessage":
        o = json.loads(data)
        return DLQMessage(
            original_message=o.get("original_message", {}),
            original_queue=o.get("original_queue", ""),
            error=o.get("error", ""),
            timestamp=o.get("timestamp", datetime.now(timezone.utc).isoformat()),
            retry_count=int(o.get("retry_count", 0)),
        )


class DeadLetterQueue:
    """
    Redis-backed Dead Letter Queue implementation.

    Stores failed messages in list `dlq:{queue}` as JSON strings.
    """

    def __init__(
        self, redis_client: Optional[redis.Redis] = None, max_retries: int = 3
    ):
        self._redis = redis_client
        self.max_retries = max_retries

    async def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def send_to_dlq(
        self, message: Dict[str, Any], original_queue: str, error: Exception
    ) -> None:
        retry_count = int(message.get("retry_count", 0)) + 1
        dlq_message = DLQMessage(
            original_message=message,
            original_queue=original_queue,
            error=str(error),
            timestamp=datetime.now(timezone.utc).isoformat(),
            retry_count=retry_count,
        )
        client = await self._client()
        key = f"dlq:{original_queue}"
        await client.lpush(key, dlq_message.to_json())
        logger.error(
            "message_sent_to_dlq", queue=original_queue, retry_count=retry_count
        )

    async def process_dlq_messages(
        self,
        queue: str,
        processor: Callable[[Dict[str, Any]], Awaitable[bool]],
        *,
        max_messages: int = 100,
    ) -> int:
        """
        Process up to `max_messages` from the DLQ. For each message, `processor` is awaited.
        - If `processor` returns True, the message is considered recovered and is dropped.
        - If False, the message is re-queued to DLQ end (rpush) for later attempts.
        """
        client = await self._client()
        key = f"dlq:{queue}"
        processed = 0
        for _ in range(max_messages):
            data = await client.rpop(key)  # pop oldest
            if not data:
                break
            msg = DLQMessage.from_json(data)

            try:
                ok = await processor(msg.original_message)
                if ok:
                    logger.info("dlq_message_recovered", queue=queue)
                else:
                    await client.rpush(key, data)
                    logger.warning("dlq_message_requeued", queue=queue)
            except Exception as exc:  # noqa: BLE001
                # If processing raised, decide to requeue or park based on retry_count
                if msg.retry_count >= self.max_retries:
                    # park (keep in DLQ tail)
                    await client.rpush(key, data)
                    logger.error(
                        "dlq_message_parked_after_max_retries",
                        queue=queue,
                        error=str(exc),
                        retry_count=msg.retry_count,
                    )
                else:
                    # increment retry count and requeue to head for earlier retry later
                    msg.retry_count += 1
                    await client.lpush(key, msg.to_json())
                    logger.error(
                        "dlq_message_requeued_with_increment",
                        queue=queue,
                        error=str(exc),
                        retry_count=msg.retry_count,
                    )
            processed += 1
        return processed

    async def requeue_to_original(
        self, dlq_queue: str, *, max_messages: int = 50
    ) -> int:
        """
        Move messages from `dlq:{dlq_queue}` back to their original queues (FIFO).
        Returns number of messages moved.
        """
        client = await self._client()
        key = f"dlq:{dlq_queue}"
        moved = 0
        for _ in range(max_messages):
            data = await client.rpop(key)
            if not data:
                break
            msg = DLQMessage.from_json(data)
            await client.lpush(msg.original_queue, json.dumps(msg.original_message))
            moved += 1
            logger.info(
                "dlq_message_requeued_to_original", original_queue=msg.original_queue
            )
        return moved
