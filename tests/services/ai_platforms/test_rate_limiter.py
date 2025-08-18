"""
Unit tests for AIRateLimiter class.

Tests the token bucket rate limiting algorithm with burst capability,
including edge cases and concurrent usage scenarios.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from app.services.ai_platforms.base import AIRateLimiter


class TestAIRateLimiter:
    """Test cases for AIRateLimiter token bucket implementation."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with default and custom burst limits."""
        # Test with default burst limit
        limiter = AIRateLimiter(requests_per_minute=60)
        assert limiter.requests_per_minute == 60
        assert limiter.burst_limit == 10  # min(60/4, 10) = 10
        assert limiter.tokens == 10

        # Test with custom burst limit
        limiter = AIRateLimiter(requests_per_minute=120, burst_limit=5)
        assert limiter.requests_per_minute == 120
        assert limiter.burst_limit == 5
        assert limiter.tokens == 5

    @pytest.mark.asyncio
    async def test_burst_limit_calculation(self):
        """Test burst limit calculation for different RPM values."""
        # High RPM should be capped at 10
        limiter = AIRateLimiter(requests_per_minute=200)
        assert limiter.burst_limit == 10  # min(200/4, 10) = 10

        # Low RPM should use RPM/4
        limiter = AIRateLimiter(requests_per_minute=20)
        assert limiter.burst_limit == 5  # min(20/4, 10) = 5

        # Very low RPM
        limiter = AIRateLimiter(requests_per_minute=8)
        assert limiter.burst_limit == 2  # min(8/4, 10) = 2

    @pytest.mark.asyncio
    async def test_single_token_acquisition(self):
        """Test acquiring single tokens."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=5)

        initial_tokens = limiter.tokens
        start_time = time.time()

        await limiter.acquire()

        end_time = time.time()

        # Should consume one token
        assert limiter.tokens == initial_tokens - 1
        # Should be immediate (burst)
        assert end_time - start_time < 0.1

    @pytest.mark.asyncio
    async def test_burst_capability(self):
        """Test that burst requests are allowed up to burst limit."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=5)

        start_time = time.time()

        # First 5 requests should be immediate (burst)
        for _ in range(5):
            await limiter.acquire()

        mid_time = time.time()

        # Burst should be immediate
        assert mid_time - start_time < 0.1
        assert limiter.tokens == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_after_burst(self):
        """Test that requests are rate limited after burst is exhausted."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=3)  # 1 per second

        start_time = time.time()

        # Exhaust burst
        for _ in range(3):
            await limiter.acquire()

        # Next request should be delayed
        await limiter.acquire()

        end_time = time.time()

        # Should take at least 1 second (rate limited)
        assert end_time - start_time >= 0.9

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled based on time passed."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=5)

        # Exhaust all tokens
        for _ in range(5):
            await limiter.acquire()

        assert limiter.tokens == 0

        # Mock time to simulate 30 seconds passing
        with patch("time.time") as mock_time:
            mock_time.side_effect = [
                limiter.last_refill + 30,  # 30 seconds later
                limiter.last_refill + 30,  # Called again in _refill_tokens
            ]

            limiter._refill_tokens()

            # Should have refilled 30 tokens (30 seconds * 1 token/second)
            # But capped at burst_limit
            assert limiter.tokens == 5

    @pytest.mark.asyncio
    async def test_old_requests_cleanup(self):
        """Test that old requests are cleaned up from the deque."""
        limiter = AIRateLimiter(requests_per_minute=60)

        # Add some old requests
        old_time = time.time() - 120  # 2 minutes ago
        limiter.requests.extend([old_time, old_time + 10, old_time + 20])

        # Add a recent request
        recent_time = time.time() - 30  # 30 seconds ago
        limiter.requests.append(recent_time)

        # Trigger cleanup
        limiter._refill_tokens()

        # Should only have the recent request
        assert len(limiter.requests) == 1
        assert limiter.requests[0] == recent_time

    @pytest.mark.asyncio
    async def test_sleep_time_calculation(self):
        """Test sleep time calculation for rate limiting."""
        limiter = AIRateLimiter(requests_per_minute=60)  # 1 per second
        sleep_time = limiter._calculate_sleep_time()
        assert sleep_time == 1.0

        limiter = AIRateLimiter(requests_per_minute=120)  # 2 per second
        sleep_time = limiter._calculate_sleep_time()
        assert sleep_time == 0.5

        limiter = AIRateLimiter(requests_per_minute=30)  # 0.5 per second
        sleep_time = limiter._calculate_sleep_time()
        assert sleep_time == 2.0

    @pytest.mark.asyncio
    async def test_multiple_token_acquisition(self):
        """Test acquiring multiple tokens at once."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=5)

        start_time = time.time()

        # Acquire 3 tokens at once
        await limiter.acquire(tokens=3)

        end_time = time.time()

        # Should consume 3 tokens
        assert limiter.tokens == 2
        # Should be immediate (burst)
        assert end_time - start_time < 0.1

    @pytest.mark.asyncio
    async def test_token_acquisition_more_than_available(self):
        """Test acquiring more tokens than currently available."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=3)

        # Consume 2 tokens
        await limiter.acquire(tokens=2)
        assert limiter.tokens == 1

        start_time = time.time()

        # Try to acquire 2 more tokens (need to wait for refill)
        await limiter.acquire(tokens=2)

        end_time = time.time()

        # Should have waited for tokens to refill
        assert end_time - start_time >= 0.9

    @pytest.mark.asyncio
    async def test_concurrent_token_acquisition(self):
        """Test that concurrent token acquisition is properly synchronized."""
        limiter = AIRateLimiter(requests_per_minute=120, burst_limit=10)  # 2 per second

        start_time = time.time()

        # Create 15 concurrent requests (more than burst limit)
        tasks = [limiter.acquire() for _ in range(15)]
        await asyncio.gather(*tasks)

        end_time = time.time()

        # Should take at least 2.5 seconds due to rate limiting
        # 10 burst + 5 additional at 2/second = 2.5 seconds
        assert end_time - start_time >= 2.0

        # All requests should be recorded
        assert len(limiter.requests) == 15

    @pytest.mark.asyncio
    async def test_zero_tokens_wait(self):
        """Test behavior when no tokens are available and must wait."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=1)

        # Use the only token
        await limiter.acquire()
        assert limiter.tokens == 0

        start_time = time.time()

        # Next request should wait for refill
        await limiter.acquire()

        end_time = time.time()

        # Should wait approximately 1 second
        assert end_time - start_time >= 0.9
        assert end_time - start_time < 1.5

    @pytest.mark.asyncio
    async def test_lock_contention(self):
        """Test that the async lock prevents race conditions."""
        limiter = AIRateLimiter(requests_per_minute=60, burst_limit=1)

        # Create multiple concurrent tasks that will contend for the lock
        async def acquire_token():
            await limiter.acquire()
            return time.time()

        # Run 5 concurrent acquisitions
        tasks = [acquire_token() for _ in range(5)]
        timestamps = await asyncio.gather(*tasks)

        # Timestamps should be ordered (no race conditions)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

        # Should have processed all requests
        assert len(limiter.requests) == 5

    @pytest.mark.asyncio
    async def test_refill_timing_accuracy(self):
        """Test that token refill timing is accurate."""
        limiter = AIRateLimiter(requests_per_minute=120, burst_limit=2)  # 2 per second

        # Exhaust tokens
        await limiter.acquire(tokens=2)

        # Wait a specific amount of time
        await asyncio.sleep(1.0)  # 1 second

        # Check if tokens were properly refilled
        limiter._refill_tokens()

        # Should have approximately 2 tokens refilled (2 per second * 1 second)
        # Capped at burst_limit
        assert limiter.tokens == 2
