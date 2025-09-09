from __future__ import annotations

from prometheus_client import Counter, Gauge

_STATE = {
    "closed": 0,
    "half_open": 0.5,
    "open": 1,
}


class _CircuitMetrics:
    def __init__(self) -> None:
        self.circuit_state = Gauge(
            "circuit_state",
            "Circuit state (0=closed,0.5=half,1=open)",
            ["circuit"],
        )
        self.circuit_success_total = Counter(
            "circuit_success_total",
            "Total successful calls through circuit",
            ["circuit"],
        )
        self.circuit_failure_total = Counter(
            "circuit_failure_total",
            "Total failed calls counted by circuit",
            ["circuit"],
        )
        self.circuit_blocked_total = Counter(
            "circuit_blocked_total",
            "Calls blocked due to OPEN circuit",
            ["circuit"],
        )

    def set_state(self, circuit: str, state) -> None:
        # defer import to avoid circular dep

        self.circuit_state.labels(circuit=circuit).set(_STATE[state.value])

    def inc_success(self, circuit: str) -> None:
        self.circuit_success_total.labels(circuit=circuit).inc()

    def inc_failure(self, circuit: str) -> None:
        self.circuit_failure_total.labels(circuit=circuit).inc()

    def inc_blocked(self, circuit: str) -> None:
        self.circuit_blocked_total.labels(circuit=circuit).inc()


circuit_metrics = _CircuitMetrics()
