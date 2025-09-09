from __future__ import annotations

import random
from typing import Iterable, Iterator


def full_jitter(delays: Iterable[float]) -> Iterator[float]:
    """
    Apply 'full jitter' to a sequence of base delays.

    For each base delay d, yield a random value in [0, d].
    """
    for d in delays:
        yield random.random() * max(0.0, d)
