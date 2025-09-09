from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Sequence, Type, Union

from app.utils.logger import get_logger

from .monitoring import circuit_metrics

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when attempting to call through an OPEN circuit."""


ExceptionTypes = Union[Type[BaseException], Sequence[Type[BaseException]]]


@dataclass
class CircuitBreaker:
    """
    Async-first circuit breaker with HALF-OPEN trial window and metrics.

    - CLOSED: calls pass through; on failures, increment counters.
    - OPEN: calls fail-fast until `recovery_timeout` elapses.
    - HALF_OPEN: allow limited test calls; on success threshold -> CLOSED; on any failure -> OPEN.

    Configurable via constructor or env-backed defaults in `app.core.config`.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_success_threshold: int = 2
    monitored_exceptions: Iterable[Type[BaseException]] = field(
        default_factory=lambda: (Exception,)
    )
    ignored_exceptions: Iterable[Type[BaseException]] = field(default_factory=tuple)

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def _transition(self, new_state: CircuitState) -> None:
        if self._state == new_state:
            return
        prev = self._state
        self._state = new_state
        logger.warning(
            "circuit_state_change",
            circuit=self.name,
            from_state=prev.value,
            to_state=new_state.value,
        )
        circuit_metrics.set_state(self.name, new_state)

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    def _is_ignored(self, exc: BaseException) -> bool:
        return isinstance(exc, tuple(self.ignored_exceptions))

    def _is_monitored(self, exc: BaseException) -> bool:
        return isinstance(exc, tuple(self.monitored_exceptions))

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke an async function through the circuit breaker."""
        async with self._lock:
            # Evaluate OPEN state fast-fail, with time-based probe window
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition(CircuitState.HALF_OPEN)
                else:
                    circuit_metrics.inc_blocked(self.name)
                    raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")

        try:
            result = await func(*args, **kwargs)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 - we need to catch broadly and re-raise
            # Ignore non-monitored exceptions for circuit accounting
            if self._is_ignored(exc) or not self._is_monitored(exc):
                raise

            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                circuit_metrics.inc_failure(self.name)

                if self._state == CircuitState.HALF_OPEN:
                    # Any failure in HALF_OPEN flips back to OPEN immediately
                    self._transition(CircuitState.OPEN)
                elif (
                    self._state == CircuitState.CLOSED
                    and self._failure_count >= self.failure_threshold
                ):
                    self._transition(CircuitState.OPEN)

            raise

        # Success path
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_success_threshold:
                    # Reset counters and close circuit
                    self._failure_count = 0
                    self._half_open_successes = 0
                    self._last_failure_time = None
                    self._transition(CircuitState.CLOSED)
            else:
                # In CLOSED, success reduces failure pressure
                if self._failure_count > 0:
                    self._failure_count -= 1

            circuit_metrics.inc_success(self.name)

        return result

    async def __call__(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.call(func, *args, **kwargs)

    def decorate(self, func: Callable[..., Awaitable[Any]]):
        """Decorator for async call-sites."""

        async def wrapper(*args: Any, **kwargs: Any):
            return await self.call(func, *args, **kwargs)

        # Preserve name/docs for introspection
        wrapper.__name__ = getattr(func, "__name__", "circuit_wrapper")
        wrapper.__doc__ = func.__doc__
        return wrapper

    # Optional sync-compatible wrapper (executes in threadpool)
    async def call_sync(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await self.call(
            lambda *a, **kw: loop.run_in_executor(None, func, *a, **kw), *args, **kwargs
        )
