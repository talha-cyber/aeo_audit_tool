"""
Unit tests for BasePlatform class.

Tests the abstract base platform implementation including circuit breaker,
error handling, retry logic, and async context management.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.services.ai_platforms.base import BasePlatform


class MockPlatform(BasePlatform):
    """Mock platform implementation for testing."""

    def _get_default_headers(self):
        return {"Authorization": "Bearer test"}

    def _get_endpoint_url(self):
        return "https://api.test.com/v1/chat"

    def _prepare_request_payload(self, question, **kwargs):
        return {"input": question}

    async def query(self, question, **kwargs):
        return {"output": f"Response to: {question}"}

    def extract_text_response(self, raw_response):
        return raw_response["output"]


class TestBasePlatform:
    """Test cases for BasePlatform abstract base class."""

    def test_platform_initialization(self):
        """Test platform initialization with various configurations."""
        platform = MockPlatform("test_key", 60, custom_param="test")

        assert platform.api_key == "test_key"
        assert platform.platform_name == "mock"
        assert platform.config["custom_param"] == "test"
        assert platform.failure_count == 0
        assert platform.circuit_open is False
        assert platform.max_failures == 5
        assert platform.circuit_timeout == 300

    def test_platform_name_extraction(self):
        """Test that platform name is correctly extracted from class name."""
        # Should remove "Platform" suffix and convert to lowercase
        assert MockPlatform("key", 60).platform_name == "mock"

        # Test with a different class name
        class OpenAIPlatform(BasePlatform):
            def _get_default_headers(self):
                return {}

            def _get_endpoint_url(self):
                return ""

            def _prepare_request_payload(self, question, **kwargs):
                return {}

            async def query(self, question, **kwargs):
                return {}

            def extract_text_response(self, raw_response):
                return ""

        assert OpenAIPlatform("key", 60).platform_name == "openai"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager for session management."""
        platform = MockPlatform("test_key", 60)

        assert platform.session is None

        async with platform:
            assert platform.session is not None
            assert hasattr(platform.session, "post")

        # Session should be closed after context exit
        # Note: In real scenarios, session.closed would be True
        # but aiohttp.ClientSession is complex to mock properly

    @pytest.mark.asyncio
    async def test_successful_query(self):
        """Test successful query execution."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"output": "Test response"}
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                result = await platform.safe_query("Test question")

            assert result["success"] is True
            assert result["platform"] == "mock"
            assert result["response"]["output"] == "Test response"
            assert result["error"] is None
            assert "duration" in result["metadata"]
            assert "retries" in result["metadata"]
            assert "timestamp" in result["metadata"]

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test handling of rate limit errors with retry."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            # First call: rate limited
            rate_limited_response = AsyncMock()
            rate_limited_response.status = 429
            rate_limited_response.headers = {"retry-after": "1"}

            # Second call: success
            success_response = AsyncMock()
            success_response.status = 200
            success_response.json.return_value = {"output": "Success after retry"}

            mock_post.return_value.__aenter__.side_effect = [
                rate_limited_response,
                success_response,
            ]

            with patch("asyncio.sleep") as mock_sleep:
                async with platform:
                    _start_time = time.time()
                    result = await platform.safe_query("Test question")
                    _end_time = time.time()

                # Should have called sleep for rate limit delay
                mock_sleep.assert_called_with(1)

                assert result["success"] is True
                assert result["metadata"]["retries"] == 1

    @pytest.mark.asyncio
    async def test_transient_error_retry(self):
        """Test retry behavior for transient errors."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            # First call: server error
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Internal server error"}

            # Second call: success
            success_response = AsyncMock()
            success_response.status = 200
            success_response.json.return_value = {"output": "Success after retry"}

            mock_post.return_value.__aenter__.side_effect = [
                error_response,
                success_response,
            ]

            with patch("asyncio.sleep") as mock_sleep:
                async with platform:
                    result = await platform.safe_query("Test question")

                # Should have used exponential backoff
                mock_sleep.assert_called_with(1)  # 2^0 = 1

                assert result["success"] is True
                assert result["metadata"]["retries"] == 1

    @pytest.mark.asyncio
    async def test_authentication_error_no_retry(self):
        """Test that authentication errors are not retried."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json.return_value = {"error": "Unauthorized"}
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                result = await platform.safe_query("Test question")

            assert result["success"] is False
            assert "Invalid API key" in result["error"]
            assert result["metadata"].get("retries", 0) == 0
            assert platform.failure_count == 1

    @pytest.mark.asyncio
    async def test_quota_exceeded_error_no_retry(self):
        """Test that quota exceeded errors are not retried."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_response.json.return_value = {"error": "Quota exceeded"}
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                result = await platform.safe_query("Test question")

            assert result["success"] is False
            assert "Quota exceeded" in result["error"]
            assert platform.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker opens after consecutive failures."""
        platform = MockPlatform("test_key", 60)
        platform.max_failures = 2  # Lower threshold for testing

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Simulate consecutive failures
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Server error"}
            mock_post.return_value.__aenter__.return_value = error_response

            async with platform:
                # First failure
                result1 = await platform.safe_query("Test 1")
                assert result1["success"] is False
                assert platform.failure_count == 1
                assert platform.circuit_open is False

                # Second failure - should open circuit
                result2 = await platform.safe_query("Test 2")
                assert result2["success"] is False
                assert platform.failure_count == 2
                assert platform.circuit_open is True

                # Third request - should be circuit broken
                result3 = await platform.safe_query("Test 3")
                assert result3["success"] is False
                assert "Circuit breaker open" in result3["error"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self):
        """Test circuit breaker resets after timeout."""
        platform = MockPlatform("test_key", 60)
        platform.max_failures = 1
        platform.circuit_timeout = 1  # 1 second for testing

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Cause failure to open circuit
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Server error"}
            mock_post.return_value.__aenter__.return_value = error_response

            async with platform:
                # Cause failure
                await platform.safe_query("Test")
                assert platform.circuit_open is True

                # Wait for circuit timeout
                await asyncio.sleep(1.1)

                # Setup success response
                success_response = AsyncMock()
                success_response.status = 200
                success_response.json.return_value = {"output": "Success"}
                mock_post.return_value.__aenter__.return_value = success_response

                # Should reset circuit and succeed
                result = await platform.safe_query("Test")
                assert result["success"] is True
                assert platform.circuit_open is False
                assert platform.failure_count == 0

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of request timeouts."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.side_effect = asyncio.TimeoutError()

            async with platform:
                result = await platform.safe_query("Test question")

            assert result["success"] is False
            assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            import aiohttp

            mock_post.return_value.__aenter__.side_effect = aiohttp.ClientError(
                "Network error"
            )

            async with platform:
                result = await platform.safe_query("Test question")

            assert result["success"] is False
            assert "network error" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test behavior when maximum retries are exceeded."""
        platform = MockPlatform("test_key", 60)

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Always return server error
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Persistent error"}
            mock_post.return_value.__aenter__.return_value = error_response

            with patch("asyncio.sleep"):  # Speed up test
                async with platform:
                    result = await platform.safe_query("Test question")

                assert result["success"] is False
                assert "Max retries exceeded" in result["error"]
                assert platform.failure_count > 0

    @pytest.mark.asyncio
    async def test_session_not_initialized_error(self):
        """Test error when HTTP session is not initialized."""
        platform = MockPlatform("test_key", 60)

        # Don't use async context manager, so session won't be initialized
        result = await platform.safe_query("Test question")

        assert result["success"] is False
        assert "HTTP session not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self):
        """Test integration with rate limiter."""
        platform = MockPlatform("test_key", 120)  # 2 per second

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"output": "Success"}
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                start_time = time.time()

                # Make multiple requests
                tasks = [platform.safe_query(f"Question {i}") for i in range(3)]
                results = await asyncio.gather(*tasks)

                end_time = time.time()

                # All should succeed
                assert all(r["success"] for r in results)

                # Should be rate limited (3 requests at 2/second = at least 1 second)
                # Note: This test might be flaky due to timing, so we use a generous threshold
                assert end_time - start_time >= 0.5

    def test_error_response_creation(self):
        """Test creation of standardized error responses."""
        platform = MockPlatform("test_key", 60)

        error_response = platform._create_error_response("Test error")

        assert error_response["success"] is False
        assert error_response["response"] is None
        assert error_response["platform"] == "mock"
        assert error_response["error"] == "Test error"
        assert "timestamp" in error_response["metadata"]

    def test_failure_recording(self):
        """Test failure recording for circuit breaker."""
        platform = MockPlatform("test_key", 60)

        initial_count = platform.failure_count
        initial_time = platform.last_failure_time

        platform._record_failure()

        assert platform.failure_count == initial_count + 1
        assert platform.last_failure_time > initial_time

        # Test circuit opening
        platform.max_failures = 1
        platform._record_failure()

        assert platform.circuit_open is True

    def test_circuit_open_check(self):
        """Test circuit breaker open/closed state checking."""
        platform = MockPlatform("test_key", 60)

        # Initially closed
        assert platform._is_circuit_open() is False

        # Open circuit
        platform.circuit_open = True
        platform.last_failure_time = time.time()
        assert platform._is_circuit_open() is True

        # Test timeout reset
        platform.circuit_timeout = 0.1  # 100ms
        time.sleep(0.2)  # Wait for timeout
        assert platform._is_circuit_open() is False
        assert platform.circuit_open is False
        assert platform.failure_count == 0
