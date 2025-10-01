import pytest

from app.services.question_engine_v2.cache import QuestionEngineCache


class DummyRedisClient:
    def __init__(self) -> None:
        self.store = {}

    async def ping(self) -> bool:  # pragma: no cover - trivial
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value, ex=None):
        self.store[key] = value

    async def delete(self, key: str):
        self.store.pop(key, None)

    async def aclose(self) -> None:  # pragma: no cover - compatibility
        return None


@pytest.mark.asyncio
async def test_cache_round_trip_with_injected_client():
    client = DummyRedisClient()
    cache = QuestionEngineCache(redis_client=client, ttl_seconds=60)

    payload = {"persona": "marketing_manager", "questions": ["Q1", "Q2"]}
    await cache.set("audit-1", payload)

    cached = await cache.get("audit-1")
    assert cached == payload

    await cache.delete("audit-1")
    assert await cache.get("audit-1") is None

    await cache.close()


@pytest.mark.asyncio
async def test_cache_custom_serializer_deserializer():
    client = DummyRedisClient()
    cache = QuestionEngineCache(redis_client=client, ttl_seconds=60)

    await cache.set(
        "serialized",
        {"value": 42},
        serializer=lambda payload: "|".join(f"{k}={v}" for k, v in payload.items()),
    )

    result = await cache.get(
        "serialized",
        deserialize=lambda raw: {
            pair.split("=")[0]: int(pair.split("=")[1]) for pair in raw.split("|")
        },
    )

    assert result == {"value": 42}
    await cache.close()
