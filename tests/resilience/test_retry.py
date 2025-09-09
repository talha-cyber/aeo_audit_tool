import asyncio

import pytest

from app.utils.resilience.retry.decorators import retry

pytestmark = pytest.mark.asyncio


async def test_retry_retries_on_exception(monkeypatch):
    calls = {"n": 0}

    async def sleeper(delay):
        # speed up tests
        return None

    monkeypatch.setattr(asyncio, "sleep", sleeper)

    @retry(max_attempts=3, exceptions=(RuntimeError,))
    async def sometimes_fails():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("fail")
        return "ok"

    result = await sometimes_fails()
    assert result == "ok"
    assert calls["n"] == 3


async def test_retry_gives_up(monkeypatch):
    calls = {"n": 0}

    async def sleeper(delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", sleeper)

    @retry(max_attempts=2, exceptions=(RuntimeError,))
    async def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await always_fails()
    assert calls["n"] == 2
