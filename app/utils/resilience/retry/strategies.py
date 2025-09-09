from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class FixedBackoffStrategy:
    """Yields a constant delay in seconds for each attempt."""

    delay_seconds: float = 0.5

    def delays(self, max_attempts: int) -> Iterator[float]:
        for _ in range(max_attempts):
            yield self.delay_seconds


@dataclass
class ExponentialBackoffStrategy:
    """Exponential backoff with optional cap and multiplier."""

    base_delay_seconds: float = 0.2
    multiplier: float = 2.0
    max_delay_seconds: float = 5.0

    def delays(self, max_attempts: int) -> Iterator[float]:
        delay = self.base_delay_seconds
        for _ in range(max_attempts):
            yield min(delay, self.max_delay_seconds)
            delay *= self.multiplier
