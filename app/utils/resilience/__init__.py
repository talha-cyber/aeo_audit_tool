"""
Resilience utilities: circuit breakers, retries, dead-letter queues, bulkheads.

This package provides production-grade building blocks to improve fault tolerance
and stability across external integrations and internal subsystems. All modules
are async-first, structured-logging enabled, and export Prometheus metrics.

Keep modules small, composable, and configuration-driven via `app.core.config`.
"""

from .bulkhead.isolator import Bulkhead
from .circuit_breaker.breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from .dead_letter.queue import DeadLetterQueue
from .retry.decorators import retry
from .retry.strategies import ExponentialBackoffStrategy

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "retry",
    "ExponentialBackoffStrategy",
    "DeadLetterQueue",
    "Bulkhead",
]
