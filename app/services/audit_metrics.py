"""
Audit-specific metrics collection and monitoring.
"""

import time
from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram

# === Audit Run Metrics ===

# Counter for tracking audit runs
audit_runs_started_total = Counter(
    "audit_runs_started_total", "Total number of audit runs started"
)

audit_runs_completed_total = Counter(
    "audit_runs_completed_total", "Total number of audit runs completed successfully"
)

audit_runs_failed_total = Counter(
    "audit_runs_failed_total", "Total number of audit runs that failed"
)

audit_runs_cancelled_total = Counter(
    "audit_runs_cancelled_total", "Total number of audit runs that were cancelled"
)

# === Question Generation Metrics ===

question_generation_duration_seconds = Histogram(
    "question_generation_duration_seconds",
    "Time spent generating questions for audit runs",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

questions_generated_total = Counter(
    "questions_generated_total",
    "Total number of questions generated",
    ["category", "question_type"],
)

# === Batch Processing Metrics ===

audit_batch_duration_seconds = Histogram(
    "audit_batch_duration_seconds",
    "Time spent processing audit question batches",
    buckets=[5, 15, 30, 60, 120, 300, 600],
)

audit_batch_size_distribution = Histogram(
    "audit_batch_size_distribution",
    "Distribution of batch sizes processed",
    buckets=[1, 5, 10, 15, 20, 25, 50, 100],
)

# === Platform Query Metrics ===

platform_query_latency_seconds = Histogram(
    "platform_query_latency_seconds",
    "Platform query response time",
    ["platform"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

platform_queries_total = Counter(
    "platform_queries_total", "Total platform queries by status", ["platform", "status"]
)

platform_errors_total = Counter(
    "platform_errors_total", "Total platform errors by type", ["platform", "reason"]
)

platform_rate_limit_hits_total = Counter(
    "platform_rate_limit_hits_total", "Total rate limit hits per platform", ["platform"]
)

# === Brand Detection Metrics ===

brand_detection_duration_seconds = Histogram(
    "brand_detection_duration_seconds",
    "Time spent on brand detection per response",
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

brands_detected_total = Counter(
    "brands_detected_total", "Total brands detected", ["brand_name", "confidence_level"]
)

brand_detection_errors_total = Counter(
    "brand_detection_errors_total", "Total brand detection errors"
)

# === Progress and State Metrics ===

audit_progress_gauge = Gauge(
    "audit_progress_ratio",
    "Current audit progress ratio (0.0 to 1.0)",
    ["audit_run_id"],
)

audit_inflight_tasks = Gauge(
    "audit_inflight_tasks", "Number of audit tasks currently running"
)

audit_queue_size = Gauge(
    "audit_queue_size", "Number of audit runs waiting to be processed"
)

# === Resource Usage Metrics ===

audit_memory_usage_mb = Gauge(
    "audit_memory_usage_mb", "Memory usage by audit processor in MB", ["audit_run_id"]
)

audit_database_operations_total = Counter(
    "audit_database_operations_total",
    "Total database operations",
    ["operation_type", "table"],
)

# === Cost and Token Metrics ===

audit_platform_tokens_total = Counter(
    "audit_platform_tokens_total",
    "Total tokens used per platform",
    ["platform", "token_type"],
)

audit_platform_cost_total = Counter(
    "audit_platform_cost_total",
    "Total cost incurred per platform (in USD)",
    ["platform"],
)


class AuditMetrics:
    """
    Audit-specific metrics collection and management.

    This class provides a high-level interface for collecting metrics
    during audit processing with automatic timing and error handling.
    """

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.active_timers: Dict[str, float] = {}

    # === Audit Run Metrics ===

    def increment_audit_started(self):
        """Record the start of an audit run"""
        audit_runs_started_total.inc()
        audit_inflight_tasks.inc()

    def increment_audit_completed(self):
        """Record successful completion of an audit run"""
        audit_runs_completed_total.inc()
        audit_inflight_tasks.dec()

    def increment_audit_failed(self):
        """Record failure of an audit run"""
        audit_runs_failed_total.inc()
        audit_inflight_tasks.dec()

    def increment_audit_cancelled(self):
        """Record cancellation of an audit run"""
        audit_runs_cancelled_total.inc()
        audit_inflight_tasks.dec()

    # === Question Generation Metrics ===

    def record_question_generation_time(self, duration_ms: float):
        """Record question generation timing"""
        question_generation_duration_seconds.observe(duration_ms / 1000)

    def increment_questions_generated(
        self, category: str, question_type: str, count: int = 1
    ):
        """Record questions generated"""
        questions_generated_total.labels(
            category=category, question_type=question_type
        ).inc(count)

    # === Batch Processing Metrics ===

    def record_batch_processing_time(self, duration_ms: int):
        """Record batch processing timing"""
        audit_batch_duration_seconds.observe(duration_ms / 1000)

    def record_batch_size(self, batch_size: int):
        """Record batch size distribution"""
        audit_batch_size_distribution.observe(batch_size)

    # === Platform Query Metrics ===

    def record_platform_query_time(self, platform: str, duration_ms: int):
        """Record platform-specific query timing"""
        platform_query_latency_seconds.labels(platform=platform).observe(
            duration_ms / 1000
        )

    def increment_successful_queries(self, platform: str):
        """Record successful platform query"""
        platform_queries_total.labels(platform=platform, status="success").inc()

    def increment_failed_queries(self, platform: str):
        """Record failed platform query"""
        platform_queries_total.labels(platform=platform, status="error").inc()

    def increment_platform_error(self, platform: str, error_type: str):
        """Record platform-specific error"""
        platform_errors_total.labels(platform=platform, reason=error_type).inc()

    def increment_rate_limit_hit(self, platform: str):
        """Record rate limit hit"""
        platform_rate_limit_hits_total.labels(platform=platform).inc()

    # === Brand Detection Metrics ===

    def record_brand_detection_time(self, duration_ms: float):
        """Record brand detection timing"""
        brand_detection_duration_seconds.observe(duration_ms / 1000)

    def increment_brands_detected(self, brand_name: str, confidence: float):
        """Record brand detection"""
        confidence_level = (
            "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        )
        brands_detected_total.labels(
            brand_name=brand_name, confidence_level=confidence_level
        ).inc()

    def increment_brand_detection_error(self):
        """Record brand detection error"""
        brand_detection_errors_total.inc()

    # === Progress Tracking ===

    def update_progress(self, audit_run_id: str, progress_ratio: float):
        """Update audit progress gauge"""
        audit_progress_gauge.labels(audit_run_id=audit_run_id).set(progress_ratio)

    def clear_progress(self, audit_run_id: str):
        """Clear progress gauge for completed audit"""
        try:
            audit_progress_gauge.remove(audit_run_id)
        except KeyError:
            pass  # Gauge may not exist

    def update_queue_size(self, size: int):
        """Update audit queue size"""
        audit_queue_size.set(size)

    # === Resource Usage ===

    def update_memory_usage(self, audit_run_id: str, memory_mb: float):
        """Update memory usage for audit run"""
        audit_memory_usage_mb.labels(audit_run_id=audit_run_id).set(memory_mb)

    def increment_database_operation(self, operation: str, table: str):
        """Record database operation"""
        audit_database_operations_total.labels(
            operation_type=operation, table=table
        ).inc()

    # === Cost and Token Tracking ===

    def record_platform_tokens(
        self, platform: str, input_tokens: int, output_tokens: int
    ):
        """Record token usage"""
        audit_platform_tokens_total.labels(platform=platform, token_type="input").inc(
            input_tokens
        )
        audit_platform_tokens_total.labels(platform=platform, token_type="output").inc(
            output_tokens
        )

    def record_platform_cost(self, platform: str, cost_usd: float):
        """Record platform cost"""
        audit_platform_cost_total.labels(platform=platform).inc(cost_usd)

    # === Timing Helpers ===

    def start_timer(self, timer_name: str) -> str:
        """Start a named timer"""
        self.start_times[timer_name] = time.time()
        return timer_name

    def end_timer(self, timer_name: str) -> float:
        """End a named timer and return duration in milliseconds"""
        if timer_name not in self.start_times:
            return 0.0

        duration_ms = (time.time() - self.start_times[timer_name]) * 1000
        del self.start_times[timer_name]
        return duration_ms

    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return TimingContext(self, operation_name)

    # === Summary Methods ===

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit metrics"""
        return {
            "audits_started": audit_runs_started_total._value.get(),
            "audits_completed": audit_runs_completed_total._value.get(),
            "audits_failed": audit_runs_failed_total._value.get(),
            "audits_cancelled": audit_runs_cancelled_total._value.get(),
            "inflight_tasks": audit_inflight_tasks._value.get(),
            "queue_size": audit_queue_size._value.get(),
        }

    def get_platform_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of platform metrics"""
        # This would extract current values from Prometheus metrics
        # Implementation depends on how you want to access the metrics values
        return {}


class TimingContext:
    """Context manager for automatic timing of operations"""

    def __init__(self, metrics: AuditMetrics, operation_name: str):
        self.metrics = metrics
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            # Route to appropriate metric based on operation name
            if "question_generation" in self.operation_name:
                self.metrics.record_question_generation_time(duration_ms)
            elif "batch_processing" in self.operation_name:
                self.metrics.record_batch_processing_time(int(duration_ms))
            elif "brand_detection" in self.operation_name:
                self.metrics.record_brand_detection_time(duration_ms)


# Global metrics instance
metrics = AuditMetrics()


def get_audit_metrics() -> AuditMetrics:
    """Get the global audit metrics instance"""
    return metrics
