import asyncio

import pytest

from app.utils.resilience.bulkhead.isolator import Bulkhead

pytestmark = pytest.mark.asyncio


async def test_bulkhead_limits_concurrency():
    bh = Bulkhead(name="test", max_concurrency=2)
    in_section = 0
    peak = 0

    async def task():
        nonlocal in_section, peak
        async with bh.acquire():
            in_section += 1
            peak = max(peak, in_section)
            await asyncio.sleep(0.01)
            in_section -= 1

    await asyncio.gather(*[task() for _ in range(10)])
    assert peak <= 2
