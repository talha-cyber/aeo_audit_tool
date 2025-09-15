from __future__ import annotations

from typing import Optional

import psutil
import redis.asyncio as redis
from prometheus_client import Gauge

from app.core.config import settings

CPU_USAGE = Gauge("tech_cpu_usage_ratio", "Process CPU usage percent divided by 100")
MEM_USAGE_MB = Gauge("tech_memory_usage_mb", "Process memory RSS in MB")
REDIS_QUEUE_DEPTH = Gauge(
    "tech_redis_queue_depth", "Redis list length for queue", ["queue"]
)


def update_process_metrics() -> None:
    p = psutil.Process()
    try:
        CPU_USAGE.set((p.cpu_percent(interval=0.0) or 0.0) / 100.0)
    except Exception:
        # On some platforms cpu_percent needs an initial call
        CPU_USAGE.set(0.0)
    try:
        mem = p.memory_info().rss / (1024 * 1024)
        MEM_USAGE_MB.set(mem)
    except Exception:
        pass


async def update_redis_queue_depth(
    queue: str, client: Optional[redis.Redis] = None
) -> int:
    cli = client or redis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        encoding="utf-8",
        decode_responses=True,
    )
    depth = await cli.llen(queue)
    REDIS_QUEUE_DEPTH.labels(queue=queue).set(depth)
    return int(depth)
