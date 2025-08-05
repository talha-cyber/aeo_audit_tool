from unittest.mock import AsyncMock, patch

import pytest

from app.utils.cache import CacheManager


@pytest.fixture
def mock_redis_client():
    """Fixture to mock the redis.asyncio.Redis client."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    return mock_client


@pytest.mark.asyncio
@patch("app.utils.cache.redis.from_url")
async def test_get_client(mock_from_url, mock_redis_client):
    """Test that the Redis client is created and reused."""
    mock_from_url.return_value = mock_redis_client
    CacheManager._redis_client = None  # Reset client

    client1 = await CacheManager.get_client()
    client2 = await CacheManager.get_client()

    assert client1 is client2
    mock_from_url.assert_called_once()
    mock_redis_client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_get_miss(mock_redis_client):
    """Test cache miss scenario."""
    with patch(
        "app.utils.cache.CacheManager.get_client", new_callable=AsyncMock
    ) as mock_get_client:
        mock_get_client.return_value = mock_redis_client

        result = await CacheManager.get("non_existent_key")

        assert result is None
        mock_redis_client.get.assert_awaited_with("non_existent_key")


@pytest.mark.asyncio
async def test_cache_get_hit(mock_redis_client):
    """Test cache hit scenario."""
    mock_redis_client.get.return_value = b'{"data": "some_value"}'

    with patch(
        "app.utils.cache.CacheManager.get_client", new_callable=AsyncMock
    ) as mock_get_client:
        mock_get_client.return_value = mock_redis_client

        result = await CacheManager.get("existent_key")

        assert result == {"data": "some_value"}
        mock_redis_client.get.assert_awaited_with("existent_key")


@pytest.mark.asyncio
async def test_cache_set(mock_redis_client):
    """Test setting a value in the cache."""
    with patch(
        "app.utils.cache.CacheManager.get_client", new_callable=AsyncMock
    ) as mock_get_client:
        mock_get_client.return_value = mock_redis_client

        value_to_cache = {"data": "test_data"}
        await CacheManager.set("my_key", value_to_cache, ttl=3600)

        mock_redis_client.set.assert_awaited_with(
            "my_key", b'{"data": "test_data"}', ex=3600
        )


def test_generate_cache_key():
    """Test the cache key generation logic."""
    key = CacheManager.generate_cache_key(
        industry="SaaS", product_type="CRM", competitors=["Salesforce", "HubSpot"]
    )

    # Check that the key is deterministic
    key2 = CacheManager.generate_cache_key(
        industry="SaaS",
        product_type="CRM",
        competitors=["HubSpot", "Salesforce"],  # Order shouldn't matter
    )

    assert key.startswith("dynamic_q:SaaS:CRM:")
    assert key == key2
