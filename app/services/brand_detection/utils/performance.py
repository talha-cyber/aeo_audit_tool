import functools
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""

    operation: str
    duration_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Thread-safe performance monitoring for brand detection operations"""

    def __init__(self, max_metrics_per_operation: int = 1000):
        self.max_metrics_per_operation = max_metrics_per_operation
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_metrics_per_operation)
        )
        self._lock = threading.RLock()
        self._operation_counts = defaultdict(int)

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric thread-safely"""
        with self._lock:
            self._metrics[metric.operation].append(metric)
            self._operation_counts[metric.operation] += 1

    @contextmanager
    def measure_operation(
        self,
        operation: str,
        metadata: Dict[str, Any] = None,
        log_slow_operations: bool = True,
        slow_threshold_ms: float = 1000.0,
    ):
        """Context manager for measuring operation performance"""
        start_time = time.perf_counter()
        timestamp = time.time()
        error_message = None
        success = True

        try:
            yield
        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"Operation {operation} failed: {error_message}")
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=timestamp,
                metadata=metadata or {},
                success=success,
                error_message=error_message,
            )

            self.record_metric(metric)

            # Log slow operations
            if log_slow_operations and duration_ms > slow_threshold_ms:
                logger.warning(
                    f"Slow operation detected: {operation} took {duration_ms:.2f}ms "
                    f"(threshold: {slow_threshold_ms}ms)"
                )

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        with self._lock:
            metrics = list(self._metrics.get(operation, []))

            if not metrics:
                return {
                    "operation": operation,
                    "total_calls": 0,
                    "avg_duration_ms": 0,
                    "min_duration_ms": 0,
                    "max_duration_ms": 0,
                    "success_rate": 0,
                    "error_count": 0,
                }

            durations = [m.duration_ms for m in metrics]
            successes = [m for m in metrics if m.success]

            return {
                "operation": operation,
                "total_calls": len(metrics),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "median_duration_ms": sorted(durations)[len(durations) // 2],
                "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
                "success_rate": len(successes) / len(metrics) * 100,
                "error_count": len(metrics) - len(successes),
                "recent_errors": [
                    m.error_message
                    for m in metrics[-10:]
                    if not m.success and m.error_message
                ],
            }


def performance_monitor(
    operation: str = None,
    metadata: Dict[str, Any] = None,
    monitor_instance: PerformanceMonitor = None,
):
    """Decorator for monitoring function performance"""

    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__qualname__}"
        monitor = monitor_instance or _global_monitor

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with monitor.measure_operation(op_name, metadata):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with monitor.measure_operation(op_name, metadata):
                return await func(*args, **kwargs)

        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


class PerformanceProfiler:
    """Detailed profiler for complex operations"""

    def __init__(self, name: str):
        self.name = name
        self.checkpoints: List[Dict[str, Any]] = []
        self.start_time = None
        self.metadata = {}

    def start(self, metadata: Dict[str, Any] = None):
        """Start profiling"""
        self.start_time = time.perf_counter()
        self.metadata = metadata or {}
        self.checkpoints = []

    def checkpoint(self, label: str, metadata: Dict[str, Any] = None):
        """Add a checkpoint"""
        if self.start_time is None:
            raise ValueError("Profiler not started")

        current_time = time.perf_counter()
        elapsed_ms = (current_time - self.start_time) * 1000

        self.checkpoints.append(
            {
                "label": label,
                "elapsed_ms": elapsed_ms,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

    def finish(self) -> Dict[str, Any]:
        """Finish profiling and return results"""
        if self.start_time is None:
            raise ValueError("Profiler not started")

        total_time = (time.perf_counter() - self.start_time) * 1000

        # Calculate intervals between checkpoints
        intervals = []
        prev_time = 0
        for checkpoint in self.checkpoints:
            interval_ms = checkpoint["elapsed_ms"] - prev_time
            intervals.append(
                {
                    "label": checkpoint["label"],
                    "interval_ms": interval_ms,
                    "cumulative_ms": checkpoint["elapsed_ms"],
                    "metadata": checkpoint["metadata"],
                }
            )
            prev_time = checkpoint["elapsed_ms"]

        result = {
            "operation": self.name,
            "total_duration_ms": total_time,
            "checkpoints": intervals,
            "metadata": self.metadata,
            "timestamp": time.time(),
        }

        # Record in global monitor
        metric = PerformanceMetric(
            operation=f"profile_{self.name}",
            duration_ms=total_time,
            timestamp=time.time(),
            metadata={"profile_data": result},
        )
        _global_monitor.record_metric(metric)

        return result


@contextmanager
def profile_operation(name: str, metadata: Dict[str, Any] = None):
    """Context manager for detailed profiling"""
    profiler = PerformanceProfiler(name)
    profiler.start(metadata)

    try:
        yield profiler
    finally:
        result = profiler.finish()
        logger.info(f"Profile complete for {name}: {result['total_duration_ms']:.2f}ms")


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _global_monitor
