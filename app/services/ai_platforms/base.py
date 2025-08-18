"""
Base classes for AI platform clients.

Provides:
- AIRateLimiter: Token bucket rate limiting with burst capability
- BasePlatform: Abstract base class for all platform implementations
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional

import aiohttp

from app.utils.logger import get_logger

from .exceptions import (
    AuthenticationError,
    PlatformError,
    QuotaExceededError,
    RateLimitError,
    TransientError,
)

logger = get_logger(__name__)


class AIRateLimiter:
    """
    Token bucket rate limiter with burst capability.

    Implements smooth rate limiting allowing for burst requests up to a configurable
    limit while maintaining an average rate over time.
    """

    def __init__(self, requests_per_minute: int, burst_limit: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            burst_limit: Maximum burst requests allowed (defaults to RPM/4 or 10,
                whichever is smaller)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit or min(requests_per_minute // 4, 10)
        self.requests = deque()
        self.tokens = self.burst_limit
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens for request, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (usually 1)
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._consume_tokens(tokens)
            self.requests.append(time.time())

    async def _wait_for_tokens(self, needed_tokens: int) -> None:
        """Wait until enough tokens are available."""
        while self.tokens < needed_tokens:
            self._refill_tokens()
            if self.tokens < needed_tokens:
                sleep_time = self._calculate_sleep_time()
                await asyncio.sleep(sleep_time)

    def _refill_tokens(self) -> None:
        """Refill tokens based on time passed."""
        now = time.time()
        time_passed = now - self.last_refill

        # Refill tokens based on time passed
        tokens_to_add = int(time_passed * (self.requests_per_minute / 60))
        self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)
        self.last_refill = now

        # Clean old requests (older than 1 minute)
        cutoff = now - 60
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def _calculate_sleep_time(self) -> float:
        """Calculate how long to sleep before next token is available."""
        return 60 / self.requests_per_minute

    def _consume_tokens(self, tokens: int) -> None:
        """Consume the specified number of tokens."""
        self.tokens -= tokens


class BasePlatform(ABC):
    """
    Abstract base class for all AI platform implementations.

    Provides:
    - Rate limiting per platform
    - Circuit breaker pattern for resilience
    - Standardized error handling and retries
    - Async context manager for session management
    - Unified response format
    """

    def __init__(self, api_key: str, rate_limit: int = 60, **config):
        """
        Initialize platform client.

        Args:
            api_key: API key for the platform
            rate_limit: Requests per minute limit
            **config: Additional platform-specific configuration
        """
        self.api_key = api_key
        self.rate_limiter = AIRateLimiter(rate_limit)
        self.platform_name = self.__class__.__name__.lower().replace("platform", "")
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        self.max_failures = 5
        self.circuit_timeout = 300  # 5 minutes

    async def __aenter__(self):
        """Create HTTP session when entering async context."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), headers=self._get_default_headers()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up HTTP session when exiting async context."""
        if self.session:
            await self.session.close()

    @abstractmethod
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Platform-specific implementation of query.

        Note: This method should not be called directly. Use safe_query() instead.
        """
        pass

    @abstractmethod
    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract clean text from platform-specific response.

        Args:
            raw_response: Raw response from the platform API

        Returns:
            Clean text content from the response

        Raises:
            ValueError: If response format is invalid
        """
        pass

    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for HTTP requests."""
        pass

    @abstractmethod
    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        """Prepare platform-specific request payload."""
        pass

    @abstractmethod
    def _get_endpoint_url(self) -> str:
        """Get the API endpoint URL for this platform."""
        pass

    async def safe_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Public interface with error handling, rate limiting, and retries.

        Args:
            question: Question to send to the platform
            **kwargs: Additional query parameters

        Returns:
            Standardized response dictionary with keys:
            - success: bool
            - response: Dict[str, Any] or None
            - platform: str
            - error: str or None
            - metadata: Dict[str, Any]
        """
        if self._is_circuit_open():
            return self._create_error_response("Circuit breaker open")

        await self.rate_limiter.acquire()

        start_time = time.time()
        retries = 0
        max_retries = 3

        while retries <= max_retries:
            try:
                response = await self._execute_query_with_timeout(question, **kwargs)

                # Reset circuit breaker on success
                self.failure_count = 0
                self.circuit_open = False

                return {
                    "success": True,
                    "response": response,
                    "platform": self.platform_name,
                    "error": None,
                    "metadata": {
                        "duration": time.time() - start_time,
                        "retries": retries,
                        "timestamp": time.time(),
                    },
                }

            except RateLimitError as e:
                logger.warning(
                    "Rate limited by platform",
                    platform=self.platform_name,
                    retry_after=e.retry_after,
                    operation="rate_limit",
                )
                await asyncio.sleep(e.retry_after or 60)
                retries += 1

            except TransientError as e:
                logger.warning(
                    "Transient error from platform",
                    platform=self.platform_name,
                    error_message=str(e),
                    retries=retries,
                    operation="transient_error",
                )
                if retries < max_retries:
                    await asyncio.sleep(2**retries)  # Exponential backoff
                retries += 1

            except (AuthenticationError, QuotaExceededError) as e:
                logger.error(
                    "Non-retryable error from platform",
                    platform=self.platform_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    operation="auth_error",
                )
                self._record_failure()
                return self._create_error_response(str(e))

            except Exception as e:
                logger.error(
                    "Unexpected error from platform",
                    platform=self.platform_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    retries=retries,
                    operation="unexpected_error",
                )
                self._record_failure()
                if retries >= max_retries:
                    return self._create_error_response(f"Max retries exceeded: {e}")
                retries += 1

        self._record_failure()
        return self._create_error_response("Max retries exceeded")

    async def _execute_query_with_timeout(
        self, question: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute query with timeout and error handling.

        Args:
            question: Question to send
            **kwargs: Additional parameters

        Returns:
            Raw platform response

        Raises:
            Various platform-specific errors
        """
        if not self.session:
            raise PlatformError("HTTP session not initialized")

        payload = self._prepare_request_payload(question, **kwargs)

        try:
            async with self.session.post(
                self._get_endpoint_url(), json=payload
            ) as response:
                response_data = await response.json()

                if response.status == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    raise RateLimitError("Rate limited", retry_after=retry_after)
                elif response.status == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status == 403:
                    raise QuotaExceededError("Quota exceeded")
                elif response.status >= 500:
                    raise TransientError(f"Server error: {response.status}")
                elif response.status != 200:
                    raise PlatformError(f"HTTP {response.status}: {response_data}")

                return response_data

        except asyncio.TimeoutError:
            raise TransientError("Request timeout")
        except aiohttp.ClientError as e:
            raise TransientError(f"Network error: {e}")

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        if not self.circuit_open:
            return False

        # Check if circuit timeout has passed
        if time.time() - self.last_failure_time > self.circuit_timeout:
            self.circuit_open = False
            self.failure_count = 0
            logger.info(
                "Circuit breaker reset",
                platform=self.platform_name,
                operation="circuit_breaker_reset",
            )
            return False

        return True

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker tracking."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(
                "Circuit breaker opened",
                platform=self.platform_name,
                failure_count=self.failure_count,
                operation="circuit_breaker_open",
            )

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "response": None,
            "platform": self.platform_name,
            "error": error_message,
            "metadata": {"timestamp": time.time()},
        }
