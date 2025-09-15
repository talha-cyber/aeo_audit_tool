from __future__ import annotations

from prometheus_client import Counter, Gauge

# Example business KPIs
AUDIT_SUCCESS_RATE = Gauge(
    "business_audit_success_rate",
    "Audit success rate (0..1)",
)

AUDIT_RUNTIME_SECONDS = Gauge(
    "business_audit_avg_runtime_seconds",
    "Average audit runtime duration in seconds",
)

AUDITS_BY_CLIENT_TOTAL = Counter(
    "business_audits_by_client_total",
    "Total audits triggered per client",
    ["client_id"],
)


def set_success_rate(value: float) -> None:
    AUDIT_SUCCESS_RATE.set(max(0.0, min(1.0, value)))


def set_avg_runtime(seconds: float) -> None:
    AUDIT_RUNTIME_SECONDS.set(max(0.0, seconds))


def inc_client_audits(client_id: str, count: int = 1) -> None:
    AUDITS_BY_CLIENT_TOTAL.labels(client_id=client_id).inc(count)
