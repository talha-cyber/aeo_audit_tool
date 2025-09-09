from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class FailureThresholdPolicy:
    """
    Simple consecutive-failure threshold policy.

    When failure count reaches `threshold`, the circuit should OPEN.
    """

    threshold: int = 5

    def should_open(self, consecutive_failures: int) -> bool:
        return consecutive_failures >= self.threshold


@dataclass
class ErrorWindowPolicy:
    """
    Time-window error rate policy.

    If failures in the past `window_seconds` exceed `max_failures`, the circuit should OPEN.
    """

    window_seconds: float = 60.0
    max_failures: int = 10
    _failures: Deque[float] = None

    def __post_init__(self) -> None:
        self._failures = deque()

    def record_failure(self) -> None:
        now = time.time()
        self._failures.append(now)
        self._prune(now)

    def should_open(self) -> bool:
        now = time.time()
        self._prune(now)
        return len(self._failures) >= self.max_failures

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
