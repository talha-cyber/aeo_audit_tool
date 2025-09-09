from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from app.utils.logger import get_logger

from .backoff import full_jitter
from .strategies import ExponentialBackoffStrategy

logger = get_logger(__name__)


ExceptionTypes = Union[Type[BaseException], Sequence[Type[BaseException]]]


@dataclass
class RetryConfig:
    max_attempts: int = 3
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    backoff_strategy: Any = ExponentialBackoffStrategy()  # type: ignore[assignment]
    use_jitter: bool = True


def retry(
    max_attempts: int = 3,
    exceptions: ExceptionTypes = (Exception,),
    backoff_strategy: Any = None,
    use_jitter: bool = True,
    result_retry_predicate: Optional[Callable[[Any], bool]] = None,
):
    """
    Retry decorator supporting async and sync callables.

    - Retries on specified exceptions; ignores others.
    - Optional result predicate triggers retry when True (e.g. retry on None).
    - Exponential backoff with jitter by default.
    """

    cfg = RetryConfig(
        max_attempts=max_attempts,
        exceptions=(exceptions if isinstance(exceptions, tuple) else tuple(exceptions)),
        backoff_strategy=backoff_strategy or ExponentialBackoffStrategy(),
        use_jitter=use_jitter,
    )

    def _delays():
        delays = cfg.backoff_strategy.delays(
            cfg.max_attempts - 1
        )  # n-1 waits between attempts
        return full_jitter(delays) if cfg.use_jitter else delays

    def decorator(func: Callable[..., Any]):
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any):
                attempt = 1
                delays_iter = _delays()
                while True:
                    try:
                        result = await func(*args, **kwargs)
                        if result_retry_predicate and result_retry_predicate(result):
                            raise RuntimeError("Retry predicate requested retry")
                        return result
                    except cfg.exceptions as exc:  # type: ignore[misc]
                        if attempt >= cfg.max_attempts:
                            logger.error(
                                "retry_exhausted",
                                function=getattr(func, "__name__", "unknown"),
                                attempts=attempt,
                                error=str(exc),
                            )
                            raise
                        delay = next(delays_iter, 0.0)
                        logger.warning(
                            "retry_attempt",
                            function=getattr(func, "__name__", "unknown"),
                            attempt=attempt,
                            next_delay_s=round(delay, 3),
                            error=str(exc),
                        )
                        attempt += 1
                        await asyncio.sleep(delay)

            return async_wrapper

        # sync path runs in-place with time.sleep inside an async-less environment
        # however our stack is async-first; prefer the async variant
        def sync_wrapper(*args: Any, **kwargs: Any):
            import time as _time

            attempt = 1
            delays_iter = _delays()
            while True:
                try:
                    result = func(*args, **kwargs)
                    if result_retry_predicate and result_retry_predicate(result):
                        raise RuntimeError("Retry predicate requested retry")
                    return result
                except cfg.exceptions as exc:  # type: ignore[misc]
                    if attempt >= cfg.max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=getattr(func, "__name__", "unknown"),
                            attempts=attempt,
                            error=str(exc),
                        )
                        raise
                    delay = next(delays_iter, 0.0)
                    logger.warning(
                        "retry_attempt",
                        function=getattr(func, "__name__", "unknown"),
                        attempt=attempt,
                        next_delay_s=round(delay, 3),
                        error=str(exc),
                    )
                    attempt += 1
                    _time.sleep(delay)

        return sync_wrapper

    return decorator
