"""
Retry policy implementation for job scheduling.

Provides sophisticated retry mechanisms with exponential backoff,
circuit breakers, and failure analysis for robust job execution.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetryStrategy(str, Enum):
    """Available retry strategies"""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""

    attempt_number: int
    execution_id: str
    failed_at: datetime
    error_message: str
    error_type: str
    next_retry_at: datetime
    delay_seconds: float


class RetryPolicy(ABC):
    """
    Abstract base class for retry policies.

    Defines the interface for implementing various retry strategies
    with customizable behavior for different failure scenarios.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 60.0,
        max_delay_seconds: float = 3600.0,  # 1 hour max
        jitter: bool = True,
        retry_conditions: Optional[List[str]] = None,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay_seconds: Base delay between retries
            max_delay_seconds: Maximum delay between retries
            jitter: Add random jitter to prevent thundering herd
            retry_conditions: Error types/patterns that should trigger retries
        """
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter = jitter
        self.retry_conditions = retry_conditions or [
            "TimeoutError",
            "ConnectionError",
            "TemporaryError",
            "ServiceUnavailableError",
        ]

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None

        # Statistics
        self._total_retries = 0
        self._successful_retries = 0

    @abstractmethod
    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Calculate delay for the given retry attempt"""
        pass

    def should_retry(
        self,
        attempt_number: int,
        error_message: str,
        error_type: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if job should be retried based on failure analysis.

        Args:
            attempt_number: Current retry attempt number
            error_message: Error message from failed execution
            error_type: Type/class of the error
            execution_context: Additional context about the execution

        Returns:
            True if job should be retried, False otherwise
        """
        # Check max retries limit
        if attempt_number >= self.max_retries:
            logger.info(
                "Max retries exceeded",
                attempt=attempt_number,
                max_retries=self.max_retries,
                error_type=error_type,
            )
            return False

        # Check circuit breaker
        if self._is_circuit_open():
            logger.warning(
                "Circuit breaker open, skipping retry",
                attempt=attempt_number,
                error_type=error_type,
            )
            return False

        # Check if error type is retryable
        if not self._is_retryable_error(error_type, error_message):
            logger.info(
                "Error not retryable",
                error_type=error_type,
                error_message=error_message[:100],
            )
            return False

        # Check execution context constraints
        if execution_context:
            # Don't retry if execution took too long (resource exhaustion)
            runtime_seconds = execution_context.get("runtime_seconds", 0)
            if runtime_seconds > 1800:  # 30 minutes
                logger.info(
                    "Execution too long, skipping retry", runtime=runtime_seconds
                )
                return False

            # Don't retry certain job types during business hours
            job_type = execution_context.get("job_type", "")
            if job_type in ["heavy_processing", "batch_export"]:
                now = datetime.now(timezone.utc)
                # Simple business hours check (can be made more sophisticated)
                if 9 <= now.hour <= 17:
                    logger.info(
                        "Heavy job skipped during business hours", job_type=job_type
                    )
                    return False

        logger.info(
            "Job will be retried",
            attempt=attempt_number,
            max_retries=self.max_retries,
            error_type=error_type,
        )

        return True

    def schedule_retry(
        self,
        attempt_number: int,
        execution_id: str,
        error_message: str,
        error_type: str,
        failed_at: Optional[datetime] = None,
    ) -> RetryAttempt:
        """
        Schedule a retry attempt.

        Args:
            attempt_number: Retry attempt number
            execution_id: ID of failed execution
            error_message: Error message from failure
            error_type: Type of error
            failed_at: When the execution failed

        Returns:
            RetryAttempt with scheduling information
        """
        if failed_at is None:
            failed_at = datetime.now(timezone.utc)

        # Calculate delay
        delay_seconds = self.calculate_delay(attempt_number)

        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay_seconds * 0.1  # 10% jitter
            delay_seconds += random.uniform(-jitter_amount, jitter_amount)

        # Ensure delay is within bounds
        delay_seconds = max(1.0, min(delay_seconds, self.max_delay_seconds))

        next_retry_at = failed_at + timedelta(seconds=delay_seconds)

        retry_attempt = RetryAttempt(
            attempt_number=attempt_number,
            execution_id=execution_id,
            failed_at=failed_at,
            error_message=error_message,
            error_type=error_type,
            next_retry_at=next_retry_at,
            delay_seconds=delay_seconds,
        )

        # Update statistics
        self._total_retries += 1

        logger.info(
            "Scheduled retry attempt",
            attempt=attempt_number,
            execution_id=execution_id,
            delay_seconds=delay_seconds,
            next_retry=next_retry_at.isoformat(),
            error_type=error_type,
        )

        return retry_attempt

    def record_retry_outcome(self, successful: bool, attempt_number: int) -> None:
        """Record the outcome of a retry attempt"""
        if successful:
            self._successful_retries += 1
            self._reset_circuit_breaker()
            logger.info("Retry attempt succeeded", attempt=attempt_number)
        else:
            self._record_failure()
            logger.warning("Retry attempt failed", attempt=attempt_number)

    def _is_retryable_error(self, error_type: str, error_message: str) -> bool:
        """Check if error should trigger retry"""
        # Check error type patterns
        for condition in self.retry_conditions:
            if condition.lower() in error_type.lower():
                return True

        # Check error message patterns (for specific scenarios)
        retryable_patterns = [
            "connection timeout",
            "temporary failure",
            "service unavailable",
            "rate limit exceeded",
            "database connection lost",
        ]

        error_msg_lower = error_message.lower()
        for pattern in retryable_patterns:
            if pattern in error_msg_lower:
                return True

        return False

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self._circuit_open:
            return False

        # Check if circuit should be reset
        if (
            self._circuit_open_until
            and datetime.now(timezone.utc) >= self._circuit_open_until
        ):
            self._reset_circuit_breaker()
            return False

        return True

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker"""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        # Open circuit if too many failures
        if self._failure_count >= 5:  # Configurable threshold
            self._open_circuit_breaker()

    def _open_circuit_breaker(self) -> None:
        """Open the circuit breaker"""
        self._circuit_open = True
        self._circuit_open_until = datetime.now(timezone.utc) + timedelta(
            minutes=15
        )  # 15 minute timeout

        logger.warning(
            "Circuit breaker opened due to repeated failures",
            failure_count=self._failure_count,
            open_until=self._circuit_open_until.isoformat(),
        )

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after successful execution"""
        if self._circuit_open:
            logger.info("Circuit breaker reset after successful execution")

        self._circuit_open = False
        self._circuit_open_until = None
        self._failure_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get retry policy statistics"""
        success_rate = 0.0
        if self._total_retries > 0:
            success_rate = (self._successful_retries / self._total_retries) * 100

        return {
            "strategy": self.__class__.__name__,
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "total_retries": self._total_retries,
            "successful_retries": self._successful_retries,
            "success_rate": success_rate,
            "circuit_open": self._circuit_open,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time.isoformat()
            if self._last_failure_time
            else None,
        }


class FixedDelayRetry(RetryPolicy):
    """Fixed delay retry strategy"""

    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Always return the base delay"""
        return self.base_delay_seconds


class ExponentialBackoffRetry(RetryPolicy):
    """Exponential backoff retry strategy"""

    def __init__(self, multiplier: float = 2.0, **kwargs):
        """
        Initialize exponential backoff retry.

        Args:
            multiplier: Multiplier for exponential growth
            **kwargs: Base retry policy arguments
        """
        super().__init__(**kwargs)
        self.multiplier = multiplier

    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Calculate exponential backoff delay"""
        delay = self.base_delay_seconds * (self.multiplier ** (attempt_number - 1))
        return min(delay, self.max_delay_seconds)


class LinearBackoffRetry(RetryPolicy):
    """Linear backoff retry strategy"""

    def __init__(self, increment_seconds: float = 60.0, **kwargs):
        """
        Initialize linear backoff retry.

        Args:
            increment_seconds: Amount to increase delay each attempt
            **kwargs: Base retry policy arguments
        """
        super().__init__(**kwargs)
        self.increment_seconds = increment_seconds

    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Calculate linear backoff delay"""
        delay = self.base_delay_seconds + (
            self.increment_seconds * (attempt_number - 1)
        )
        return min(delay, self.max_delay_seconds)


class FibonacciBackoffRetry(RetryPolicy):
    """Fibonacci sequence backoff retry strategy"""

    def __init__(self, **kwargs):
        """Initialize fibonacci backoff retry"""
        super().__init__(**kwargs)
        self._fib_cache = {0: 1, 1: 1}

    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number with caching"""
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        return self._fib_cache[n]

    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Calculate fibonacci backoff delay"""
        fib_multiplier = self._fibonacci(attempt_number)
        delay = self.base_delay_seconds * fib_multiplier
        return min(delay, self.max_delay_seconds)


class CustomRetry(RetryPolicy):
    """Custom retry strategy with user-defined delay function"""

    def __init__(self, delay_function: Callable[[int], float], **kwargs):
        """
        Initialize custom retry strategy.

        Args:
            delay_function: Function that takes attempt number and returns delay in seconds
            **kwargs: Base retry policy arguments
        """
        super().__init__(**kwargs)
        self.delay_function = delay_function

    def calculate_delay(
        self, attempt_number: int, previous_delay: Optional[float] = None
    ) -> float:
        """Use custom delay function"""
        try:
            delay = self.delay_function(attempt_number)
            return min(max(delay, 1.0), self.max_delay_seconds)
        except Exception as e:
            logger.error(f"Custom delay function failed: {e}", exc_info=True)
            # Fall back to base delay
            return self.base_delay_seconds


def create_retry_policy(
    strategy: RetryStrategy,
    max_retries: int = 3,
    base_delay_seconds: float = 60.0,
    **kwargs,
) -> RetryPolicy:
    """
    Factory function to create retry policies.

    Args:
        strategy: Retry strategy to use
        max_retries: Maximum number of retries
        base_delay_seconds: Base delay between retries
        **kwargs: Strategy-specific arguments

    Returns:
        Configured retry policy instance
    """
    common_args = {
        "max_retries": max_retries,
        "base_delay_seconds": base_delay_seconds,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["max_delay_seconds", "jitter", "retry_conditions"]
        },
    }

    if strategy == RetryStrategy.FIXED_DELAY:
        return FixedDelayRetry(**common_args)

    elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        return ExponentialBackoffRetry(
            multiplier=kwargs.get("multiplier", 2.0), **common_args
        )

    elif strategy == RetryStrategy.LINEAR_BACKOFF:
        return LinearBackoffRetry(
            increment_seconds=kwargs.get("increment_seconds", 60.0), **common_args
        )

    elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
        return FibonacciBackoffRetry(**common_args)

    elif strategy == RetryStrategy.CUSTOM:
        delay_function = kwargs.get("delay_function")
        if not delay_function:
            raise ValueError("Custom retry strategy requires 'delay_function' argument")
        return CustomRetry(delay_function=delay_function, **common_args)

    else:
        raise ValueError(f"Unknown retry strategy: {strategy}")
