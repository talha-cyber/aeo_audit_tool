"""
Performance tests for AI platform clients.

Tests concurrent query handling, rate limiting performance, and scalability
under various load conditions.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.services.ai_platforms.base import AIRateLimiter
from app.services.ai_platforms.openai_client import OpenAIPlatform


class TestRateLimiterPerformance:
    """Performance tests for AIRateLimiter."""

    @pytest.mark.asyncio
    async def test_high_concurrency_rate_limiting(self):
        """Test rate limiter performance with high concurrency."""
        limiter = AIRateLimiter(requests_per_minute=120, burst_limit=10)  # 2 per second

        start_time = time.time()

        # Create 50 concurrent requests
        tasks = [limiter.acquire() for _ in range(50)]
        await asyncio.gather(*tasks)

        end_time = time.time()

        # Should take approximately 20 seconds (10 burst + 40 at 2/sec = 20 seconds)
        expected_min_time = 19.0  # Account for timing variations
        assert end_time - start_time >= expected_min_time

        # All requests should be recorded
        assert len(limiter.requests) == 50

    @pytest.mark.asyncio
    async def test_rate_limiter_memory_usage(self):
        """Test that rate limiter doesn't accumulate excessive memory."""
        limiter = AIRateLimiter(
            requests_per_minute=600, burst_limit=50
        )  # 10 per second

        # Make many requests over time
        for batch in range(10):
            # Make 20 requests quickly
            tasks = [limiter.acquire() for _ in range(20)]
            await asyncio.gather(*tasks)

            # Check that old requests are cleaned up
            # After each batch, the deque should not grow indefinitely
            assert len(limiter.requests) <= 200  # Should clean up old requests

            # Small delay between batches to allow cleanup
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_burst_performance(self):
        """Test performance of burst request handling."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=20)

        start_time = time.time()

        # Burst 20 requests - should all be immediate
        tasks = [limiter.acquire() for _ in range(20)]
        await asyncio.gather(*tasks)

        end_time = time.time()

        # Burst should be very fast (under 100ms)
        assert end_time - start_time < 0.1
        assert limiter.tokens == 0

    @pytest.mark.asyncio
    async def test_lock_contention_performance(self):
        """Test performance under high lock contention."""
        limiter = AIRateLimiter(requests_per_minute=600, burst_limit=100)  # High limits

        start_time = time.time()

        # Create many concurrent tasks that will contend for the lock
        tasks = [limiter.acquire() for _ in range(100)]
        await asyncio.gather(*tasks)

        end_time = time.time()

        # Should complete reasonably quickly with high limits
        assert end_time - start_time < 5.0

        # All requests should be processed
        assert len(limiter.requests) == 100


class TestPlatformPerformance:
    """Performance tests for platform implementations."""

    @pytest.mark.asyncio
    async def test_concurrent_platform_queries(self):
        """Test concurrent query performance on a single platform."""
        platform = OpenAIPlatform("test_key", rate_limit=120)  # 2 per second

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock fast responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                start_time = time.time()

                # Execute 20 concurrent queries
                tasks = [platform.safe_query(f"Question {i}") for i in range(20)]
                results = await asyncio.gather(*tasks)

                end_time = time.time()

            # All should succeed
            assert all(r["success"] for r in results)

            # Should take at least 9 seconds due to rate limiting
            # (2 burst + 18 at 2/second = 9 seconds minimum)
            assert end_time - start_time >= 8.0

    @pytest.mark.asyncio
    async def test_platform_memory_efficiency(self):
        """Test that platforms don't leak memory during operation."""
        platform = OpenAIPlatform("test_key", rate_limit=600)  # High rate limit

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_response

            async with platform:
                # Make many requests in batches
                for batch in range(5):
                    tasks = [
                        platform.safe_query(f"Batch {batch} Question {i}")
                        for i in range(20)
                    ]
                    results = await asyncio.gather(*tasks)

                    # All should succeed
                    assert all(r["success"] for r in results)

                    # Rate limiter should clean up old requests
                    request_count = len(platform.rate_limiter.requests)
                    assert request_count <= 100  # Should not accumulate indefinitely

    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test performance of error handling and retry logic."""
        platform = OpenAIPlatform("test_key", rate_limit=600)  # High rate limit

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock errors that require retries
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Server error"}

            success_response = AsyncMock()
            success_response.status = 200
            success_response.json.return_value = {
                "choices": [{"message": {"content": "Success"}}]
            }

            # Fail first two times, succeed on third
            mock_post.return_value.__aenter__.side_effect = [
                error_response,  # First failure
                error_response,  # Second failure
                success_response,  # Success
            ]

            with patch("asyncio.sleep"):  # Speed up retries
                async with platform:
                    start_time = time.time()

                    result = await platform.safe_query("Test question")

                    end_time = time.time()

                # Should eventually succeed
                assert result["success"] is True
                assert result["metadata"]["retries"] == 2

                # Should not take too long despite retries
                assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance under failure conditions."""
        platform = OpenAIPlatform("test_key", rate_limit=600)
        platform.max_failures = 3  # Lower threshold for testing

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock consistent failures
            error_response = AsyncMock()
            error_response.status = 500
            error_response.json.return_value = {"error": "Persistent error"}
            mock_post.return_value.__aenter__.return_value = error_response

            with patch("asyncio.sleep"):  # Speed up retries
                async with platform:
                    start_time = time.time()

                    # Make requests until circuit opens
                    results = []
                    for i in range(10):
                        result = await platform.safe_query(f"Question {i}")
                        results.append(result)

                        if platform.circuit_open:
                            break

                    end_time = time.time()

                # Circuit should be open
                assert platform.circuit_open is True

                # Some requests should fail due to circuit breaker
                circuit_breaker_errors = [
                    r
                    for r in results
                    if not r["success"] and "Circuit breaker open" in r["error"]
                ]
                assert len(circuit_breaker_errors) > 0

                # Should fail fast once circuit is open
                assert end_time - start_time < 5.0


class TestMultiPlatformPerformance:
    """Performance tests for multiple platforms working together."""

    @pytest.mark.asyncio
    async def test_multi_platform_concurrent_queries(self):
        """Test concurrent queries across multiple platforms."""
        platforms = {
            "openai": OpenAIPlatform("key1", rate_limit=120),
            "anthropic": OpenAIPlatform(
                "key2", rate_limit=120
            ),  # Using OpenAI for simplicity
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_response

            async def query_platform(platform_name, platform, question):
                async with platform:
                    return await platform.safe_query(question)

            start_time = time.time()

            # Create concurrent queries across platforms
            tasks = []
            for i in range(20):
                platform_name = "openai" if i % 2 == 0 else "anthropic"
                platform = platforms[platform_name]
                task = query_platform(platform_name, platform, f"Question {i}")
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            end_time = time.time()

            # All should succeed
            assert all(r["success"] for r in results)

            # Should be faster than single platform due to parallel rate limiting
            # Each platform handles 10 requests at 2/second = 4.5 seconds each
            # But they run in parallel, so total should be around 4.5 seconds
            assert end_time - start_time >= 4.0
            assert (
                end_time - start_time < 8.0
            )  # Much faster than 9 seconds for single platform

    @pytest.mark.asyncio
    async def test_platform_isolation(self):
        """Test that platform failures don't affect other platforms."""
        good_platform = OpenAIPlatform("good_key", rate_limit=600)
        bad_platform = OpenAIPlatform("bad_key", rate_limit=600)
        bad_platform.max_failures = 1  # Quick circuit breaker

        with patch("aiohttp.ClientSession.post") as mock_post:

            def side_effect(*args, **kwargs):
                # Determine which platform is calling based on headers
                session = args[0] if args else None
                if hasattr(session, "_default_headers"):
                    auth_header = session._default_headers.get("Authorization", "")
                    if "bad_key" in auth_header:
                        # Bad platform always fails
                        error_response = AsyncMock()
                        error_response.status = 500
                        error_response.json.return_value = {"error": "Auth failed"}
                        return error_response

                # Good platform succeeds
                success_response = AsyncMock()
                success_response.status = 200
                success_response.json.return_value = {
                    "choices": [{"message": {"content": "Success"}}]
                }
                return success_response

            mock_post.return_value.__aenter__.side_effect = side_effect

            with patch("asyncio.sleep"):  # Speed up retries
                # Test that good platform still works when bad platform fails
                async with good_platform, bad_platform:
                    # Make bad platform fail and open circuit
                    bad_result = await bad_platform.safe_query("Test")
                    assert bad_result["success"] is False
                    assert bad_platform.circuit_open is True

                    # Good platform should still work
                    good_result = await good_platform.safe_query("Test")
                    assert good_result["success"] is True
                    assert good_platform.circuit_open is False


class TestScalabilityLimits:
    """Tests to determine scalability limits and performance boundaries."""

    @pytest.mark.asyncio
    async def test_maximum_concurrent_requests(self):
        """Test handling of very high concurrent request counts."""
        limiter = AIRateLimiter(
            requests_per_minute=3600, burst_limit=100
        )  # 60 per second

        start_time = time.time()

        # Test with 500 concurrent requests
        tasks = [limiter.acquire() for _ in range(500)]
        await asyncio.gather(*tasks)

        end_time = time.time()

        # Should complete in reasonable time
        # 100 burst + 400 at 60/second = 100 + 6.67 = ~7 seconds minimum
        assert end_time - start_time >= 6.0
        assert end_time - start_time < 15.0  # Should not be excessively slow

        # All requests should be processed
        assert len(limiter.requests) == 500

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self):
        """Test performance under sustained load over time."""
        limiter = AIRateLimiter(
            requests_per_minute=600, burst_limit=50
        )  # 10 per second

        total_requests = 0
        start_time = time.time()

        # Run sustained load for multiple rounds
        for round_num in range(5):
            # Make 30 requests per round
            tasks = [limiter.acquire() for _ in range(30)]
            await asyncio.gather(*tasks)
            total_requests += 30

            # Check that performance remains consistent
            elapsed = time.time() - start_time
            expected_min_time = (total_requests - 50) / 10  # Account for burst

            if total_requests > 50:  # After burst is exhausted
                assert elapsed >= expected_min_time * 0.9  # Allow 10% variance

        end_time = time.time()

        # Total time should be predictable
        # 50 burst + 100 at 10/second = 10 seconds minimum
        assert end_time - start_time >= 9.0
        assert total_requests == 150

    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self):
        """Test memory stability during extended operation."""
        limiter = AIRateLimiter(
            requests_per_minute=1200, burst_limit=100
        )  # 20 per second

        # Track memory-related metrics
        max_deque_size = 0

        for batch in range(20):  # 20 batches of 25 requests each
            tasks = [limiter.acquire() for _ in range(25)]
            await asyncio.gather(*tasks)

            # Track maximum deque size
            current_size = len(limiter.requests)
            max_deque_size = max(max_deque_size, current_size)

            # Deque should not grow without bounds
            assert current_size <= 300  # Reasonable upper bound

            # Small delay to allow cleanup
            await asyncio.sleep(0.05)

        # Final check - should have cleaned up old requests
        final_size = len(limiter.requests)
        assert final_size <= 100  # Should clean up aggressively
