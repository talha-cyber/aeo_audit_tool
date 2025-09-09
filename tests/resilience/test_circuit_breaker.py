import asyncio

import pytest

from app.utils.resilience.circuit_breaker.breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)

pytestmark = pytest.mark.asyncio


async def test_circuit_closes_after_half_open_successes():
    cb = CircuitBreaker(
        name="test",
        failure_threshold=1,
        recovery_timeout=0.01,
        half_open_success_threshold=2,
    )

    async def failing():
        raise RuntimeError("boom")

    # First call fails -> OPEN
    with pytest.raises(RuntimeError):
        await cb.call(failing)

    assert cb._state == CircuitState.OPEN

    # Immediately blocked
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(asyncio.sleep, 0)

    # Wait for recovery timeout -> HALF_OPEN, need two successes
    await asyncio.sleep(0.02)

    calls = 0

    async def success():
        nonlocal calls
        calls += 1
        return "ok"

    assert await cb.call(success) == "ok"
    assert cb._state == CircuitState.HALF_OPEN
    assert await cb.call(success) == "ok"
    assert cb._state == CircuitState.CLOSED
    assert calls == 2


async def test_half_open_failure_reopens():
    cb = CircuitBreaker(
        name="test2",
        failure_threshold=1,
        recovery_timeout=0.01,
        half_open_success_threshold=2,
    )

    async def failing():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cb.call(failing)

    assert cb._state == CircuitState.OPEN
    await asyncio.sleep(0.02)

    # First call in HALF_OPEN fails -> back to OPEN
    with pytest.raises(RuntimeError):
        await cb.call(failing)
    assert cb._state == CircuitState.OPEN
