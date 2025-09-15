#!/usr/bin/env python3
"""DLQ Operations Helper

Ops-friendly CLI to inspect/process DLQ messages or requeue them to original queues.

Examples:
  - Stats:    python scripts/requeue_dlq.py --queue audit:tasks --action stats
  - Process:  python scripts/requeue_dlq.py --queue audit:tasks --action process --max 100 --handler drop
  - Requeue:  python scripts/requeue_dlq.py --queue audit:tasks --action requeue --max 50
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict

import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import configure_logging, get_logger
from app.utils.resilience.dead_letter.processor import process_with_recovery
from app.utils.resilience.dead_letter.queue import DeadLetterQueue

logger = get_logger(__name__)


async def _llen(client: redis.Redis, key: str) -> int:
    try:
        n = await client.llen(key)
        return int(n or 0)
    except Exception:  # noqa: BLE001
        return 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="DLQ operations helper")
    parser.add_argument(
        "--queue", required=True, help="original queue name (e.g., audit:tasks)"
    )
    parser.add_argument(
        "--action",
        choices=["stats", "process", "requeue"],
        required=True,
        help="action to perform",
    )
    parser.add_argument("--max", type=int, default=100, help="max messages to handle")
    parser.add_argument(
        "--handler",
        choices=["drop", "requeue"],
        default="requeue",
        help="processor behavior for --action process: drop=consume, requeue=keep",
    )
    args = parser.parse_args()

    configure_logging()

    dlq = DeadLetterQueue(max_retries=settings.DLQ_MAX_RETRIES)
    client = await dlq._client()  # type: ignore[attr-defined]
    dlq_key = f"dlq:{args.queue}"

    if args.action == "stats":
        depth = await _llen(client, dlq_key)
        logger.info("dlq_stats", queue=args.queue, dlq_key=dlq_key, depth=depth)
        print(f"queue={args.queue} depth={depth}")
        return

    if args.action == "requeue":
        moved = await dlq.requeue_to_original(args.queue, max_messages=args.max)
        depth = await _llen(client, dlq_key)
        logger.info(
            "dlq_requeue_to_original", queue=args.queue, moved=moved, remaining=depth
        )
        print(f"moved={moved} remaining={depth}")
        return

    if args.action == "process":
        if args.handler == "drop":

            async def handler_drop(_: Dict[str, Any]) -> bool:
                return True  # consume

            processed = await process_with_recovery(
                dlq, args.queue, handler_drop, max_messages=args.max
            )
        else:

            async def handler_requeue(_: Dict[str, Any]) -> bool:
                return False  # keep

            processed = await process_with_recovery(
                dlq, args.queue, handler_requeue, max_messages=args.max
            )

        depth = await _llen(client, dlq_key)
        logger.info(
            "dlq_processed", queue=args.queue, processed=processed, remaining=depth
        )
        print(f"processed={processed} remaining={depth}")


if __name__ == "__main__":
    asyncio.run(main())
