from __future__ import annotations

from prometheus_client import Counter

AUTH_FAILURES_TOTAL = Counter(
    "auth_failures_total", "Total authentication failures", ["reason"]
)

BLOCKED_REQUESTS_TOTAL = Counter(
    "blocked_requests_total", "Total requests blocked by security controls", ["control"]
)
