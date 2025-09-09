from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional

from prometheus_client import Gauge

from app.utils.logger import get_logger

logger = get_logger(__name__)


class Bulkhead:
    """
    Concurrency isolator using asyncio.Semaphore.

    Limits concurrent access per resource to avoid cascading failures.
    """

    def __init__(self, name: str, max_concurrency: int = 10):
        self.name = name
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max = max_concurrency
        self._inflight_gauge = Gauge(
            "bulkhead_inflight",
            "In-flight operations under bulkhead",
            ["bulkhead"],
        )

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        try:
            if timeout is None:
                await self._sem.acquire()
            else:
                await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "bulkhead_acquire_timeout", bulkhead=self.name, timeout=timeout
            )
            raise
        try:
            self._inflight_gauge.labels(bulkhead=self.name).set(self.in_flight)
            yield
        finally:
            self._sem.release()
            self._inflight_gauge.labels(bulkhead=self.name).set(self.in_flight)

    @property
    def in_flight(self) -> int:
        return self._max - self._sem._value  # type: ignore[attr-defined]

    async def run(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        async with self.acquire(timeout=timeout):
            return await func(*args, **kwargs)
