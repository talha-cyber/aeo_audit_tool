from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from app.utils.logger import get_logger

from .queue import DeadLetterQueue

logger = get_logger(__name__)


async def process_with_recovery(
    dlq: DeadLetterQueue,
    queue: str,
    handler: Callable[[Dict[str, Any]], Awaitable[bool]],
    *,
    max_messages: int = 100,
) -> int:
    """
    Convenience processor that wraps `DeadLetterQueue.process_dlq_messages`.
    """
    processed = await dlq.process_dlq_messages(
        queue, handler, max_messages=max_messages
    )
    logger.info("dlq_processed_batch", queue=queue, processed=processed)
    return processed
