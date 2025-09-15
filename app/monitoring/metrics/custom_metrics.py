from __future__ import annotations

from typing import Dict

from prometheus_client import Counter, Gauge, Histogram


class MetricFactory:
    """Factory to create namespaced custom metrics on demand.

    Usage:
        mf = MetricFactory(namespace="aeo")
        counter = mf.counter("batch_processed_total", "Batches processed")
        counter.inc()
    """

    def __init__(self, namespace: str = "aeo") -> None:
        self.ns = namespace
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._hist: Dict[str, Histogram] = {}

    def counter(self, name: str, desc: str, labels=None) -> Counter:
        key = f"{self.ns}_{name}"
        if key not in self._counters:
            self._counters[key] = Counter(key, desc, labels or [])
        return self._counters[key]

    def gauge(self, name: str, desc: str, labels=None) -> Gauge:
        key = f"{self.ns}_{name}"
        if key not in self._gauges:
            self._gauges[key] = Gauge(key, desc, labels or [])
        return self._gauges[key]

    def histogram(self, name: str, desc: str, labels=None, buckets=None) -> Histogram:
        key = f"{self.ns}_{name}"
        if key not in self._hist:
            self._hist[key] = Histogram(key, desc, labels or [], buckets=buckets)
        return self._hist[key]
