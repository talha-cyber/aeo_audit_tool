from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Generic, Optional, TypeVar

from app.utils.logger import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class AsyncResourcePool(Generic[T]):
    """
    Minimal async resource pool with bounded size.

    Acquire with `async with pool.acquire()` to get a resource; returns it on exit.
    """

    def __init__(
        self,
        create: Callable[[], T],
        *,
        max_size: int = 10,
        warm_start: int = 0,
    ) -> None:
        self._create = create
        self._max = max_size
        self._pool: asyncio.LifoQueue[T] = asyncio.LifoQueue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()
        self._warm_start = min(warm_start, max_size)

    async def initialize(self) -> None:
        for _ in range(self._warm_start):
            await self._pool.put(self._create())
            self._created += 1

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        resource: Optional[T] = None
        try:
            if self._pool.empty():
                async with self._lock:
                    if self._created < self._max:
                        resource = self._create()
                        self._created += 1
            if resource is None:
                resource = await self._pool.get()
            yield resource
        finally:
            if resource is not None:
                try:
                    self._pool.put_nowait(resource)
                except asyncio.QueueFull:
                    # Pool saturated; drop resource
                    logger.warning("pool_full_drop_resource")
