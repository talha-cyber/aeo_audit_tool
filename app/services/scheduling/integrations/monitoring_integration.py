"""
Monitoring integration for the scheduling system.

Provides comprehensive monitoring, metrics collection, alerting,
and integration with external monitoring systems.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""

    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class Alert:
    """Monitoring alert"""

    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    tags: Optional[Dict[str, str]] = None
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and aggregates scheduler metrics.

    Provides metric collection, aggregation, and export
    for monitoring and observability.
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self._metrics: List[MetricPoint] = []
        self._lock = asyncio.Lock()

        # Metric aggregations
        self._aggregations = {
            "scheduler.jobs.scheduled": {"count": 0, "rate": 0.0},
            "scheduler.jobs.executed": {"count": 0, "rate": 0.0},
            "scheduler.jobs.failed": {"count": 0, "rate": 0.0},
            "scheduler.executions.active": {"current": 0, "max": 0},
            "scheduler.queue.size": {"current": 0, "max": 0},
            "scheduler.performance.execution_time": {
                "avg": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            },
        }

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a metric data point"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if tags is None:
            tags = {}

        metric = MetricPoint(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            tags=tags,
            unit=unit,
        )

        async with self._lock:
            self._metrics.append(metric)
            await self._update_aggregations(metric)
            await self._cleanup_old_metrics()

    async def _update_aggregations(self, metric: MetricPoint) -> None:
        """Update metric aggregations"""
        metric_name = metric.metric_name

        if metric_name in self._aggregations:
            agg = self._aggregations[metric_name]

            if "count" in agg:
                agg["count"] += 1

            if "current" in agg:
                agg["current"] = metric.value
                agg["max"] = max(agg["max"], metric.value)

    async def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        self._metrics = [m for m in self._metrics if m.timestamp >= cutoff]

    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        relevant_metrics = [
            m
            for m in self._metrics
            if m.metric_name == metric_name and m.timestamp >= cutoff
        ]

        if not relevant_metrics:
            return {"count": 0, "values": []}

        values = [m.value for m in relevant_metrics]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "total": sum(values),
            "latest": values[-1],
            "values": values[-10:],  # Last 10 values
        }

    def get_all_aggregations(self) -> Dict[str, Any]:
        """Get current aggregations for all metrics"""
        return dict(self._aggregations)

    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format == "prometheus":
            return await self._export_prometheus_format()
        elif format == "json":
            return await self._export_json_format()
        else:
            raise ValueError(f"Unknown export format: {format}")

    async def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for metric_name, agg in self._aggregations.items():
            prometheus_name = metric_name.replace(".", "_")

            for key, value in agg.items():
                full_name = f"{prometheus_name}_{key}"
                lines.append(f"# TYPE {full_name} gauge")
                lines.append(f"{full_name} {value}")

        return "\n".join(lines)

    async def _export_json_format(self) -> str:
        """Export metrics in JSON format"""
        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregations": self._aggregations,
            "recent_metrics": [
                asdict(m) for m in self._metrics[-100:]  # Last 100 metrics
            ],
        }

        # Convert datetime objects to strings for JSON serialization
        for metric in export_data["recent_metrics"]:
            metric["timestamp"] = metric["timestamp"].isoformat()

        return json.dumps(export_data, indent=2)


class AlertManager:
    """
    Manages monitoring alerts and notifications.

    Provides alert generation, threshold monitoring,
    and notification dispatch.
    """

    def __init__(self):
        """Initialize alert manager"""
        self._alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._notification_handlers: List[Callable] = []

        # Setup default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self) -> None:
        """Setup default alerting rules"""
        self._alert_rules.update(
            {
                "scheduler.jobs.failed_rate": {
                    "threshold": 0.1,  # 10% failure rate
                    "operator": ">",
                    "severity": AlertSeverity.WARNING,
                    "title": "High Job Failure Rate",
                    "description": "Job failure rate is above 10%",
                },
                "scheduler.executions.active": {
                    "threshold": 50,  # 50 active executions
                    "operator": ">",
                    "severity": AlertSeverity.WARNING,
                    "title": "High Number of Active Executions",
                    "description": "Too many jobs running concurrently",
                },
                "scheduler.queue.size": {
                    "threshold": 100,  # 100 queued jobs
                    "operator": ">",
                    "severity": AlertSeverity.ERROR,
                    "title": "Job Queue Backlog",
                    "description": "Large number of jobs queued for execution",
                },
            }
        )

    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        operator: str,
        severity: AlertSeverity,
        title: str,
        description: str,
    ) -> None:
        """Add custom alert rule"""
        self._alert_rules[metric_name] = {
            "threshold": threshold,
            "operator": operator,
            "severity": severity,
            "title": title,
            "description": description,
        }

        logger.info(
            f"Added alert rule for {metric_name}",
            threshold=threshold,
            severity=severity.value,
        )

    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add notification handler for alerts"""
        self._notification_handlers.append(handler)

    async def evaluate_metrics(
        self, metrics_collector: MetricsCollector
    ) -> List[Alert]:
        """Evaluate metrics against alert rules"""
        new_alerts = []

        for metric_name, rule in self._alert_rules.items():
            try:
                # Get metric summary
                summary = metrics_collector.get_metric_summary(metric_name, hours=1)

                if summary["count"] == 0:
                    continue

                current_value = summary["latest"]
                threshold = rule["threshold"]
                operator = rule["operator"]

                # Check if threshold is breached
                threshold_breached = False
                if operator == ">":
                    threshold_breached = current_value > threshold
                elif operator == "<":
                    threshold_breached = current_value < threshold
                elif operator == ">=":
                    threshold_breached = current_value >= threshold
                elif operator == "<=":
                    threshold_breached = current_value <= threshold
                elif operator == "==":
                    threshold_breached = current_value == threshold

                alert_key = f"{metric_name}_{operator}_{threshold}"

                if threshold_breached:
                    # Check if alert already exists
                    if (
                        alert_key not in self._alerts
                        or self._alerts[alert_key].resolved_at
                    ):
                        # Create new alert
                        alert = Alert(
                            alert_id=alert_key,
                            severity=AlertSeverity(rule["severity"]),
                            title=rule["title"],
                            description=rule["description"],
                            triggered_at=datetime.now(timezone.utc),
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold=threshold,
                            tags={"operator": operator},
                        )

                        self._alerts[alert_key] = alert
                        new_alerts.append(alert)

                        logger.warning(
                            f"Alert triggered: {alert.title}",
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold=threshold,
                            severity=alert.severity.value,
                        )

                        # Send notifications
                        await self._send_notifications(alert)

                else:
                    # Resolve existing alert if threshold is no longer breached
                    if (
                        alert_key in self._alerts
                        and not self._alerts[alert_key].resolved_at
                    ):
                        self._alerts[alert_key].resolved_at = datetime.now(timezone.utc)

                        logger.info(
                            f"Alert resolved: {self._alerts[alert_key].title}",
                            metric_name=metric_name,
                            current_value=current_value,
                        )

                        # Could send resolution notifications here

            except Exception as e:
                logger.error(
                    f"Failed to evaluate alert rule for {metric_name}: {e}",
                    exc_info=True,
                )

        return new_alerts

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for alert"""
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}", exc_info=True)

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active (unresolved) alerts"""
        return [alert for alert in self._alerts.values() if not alert.resolved_at]

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            alert for alert in self._alerts.values() if alert.triggered_at >= cutoff
        ]


class MonitoringIntegration:
    """
    Main monitoring integration class.

    Coordinates metrics collection, alerting, and integration
    with external monitoring systems.
    """

    def __init__(
        self,
        scheduler_engine,
        collection_interval: int = 60,  # seconds
        enable_alerts: bool = True,
    ):
        """
        Initialize monitoring integration.

        Args:
            scheduler_engine: Main scheduler engine
            collection_interval: How often to collect metrics
            enable_alerts: Enable alerting functionality
        """
        self.scheduler_engine = scheduler_engine
        self.collection_interval = collection_interval
        self.enable_alerts = enable_alerts

        # Components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager() if enable_alerts else None

        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Setup default notification handlers
        if self.alert_manager:
            self.alert_manager.add_notification_handler(self._log_alert_notification)

    async def start_monitoring(self) -> None:
        """Start monitoring collection"""
        if self._is_monitoring:
            logger.warning("Monitoring already started")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info(
            "Started scheduler monitoring",
            collection_interval=self.collection_interval,
            alerts_enabled=self.enable_alerts,
        )

    async def stop_monitoring(self) -> None:
        """Stop monitoring collection"""
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped scheduler monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring collection loop"""
        while self._is_monitoring:
            try:
                await self._collect_scheduler_metrics()

                if self.alert_manager:
                    await self.alert_manager.evaluate_metrics(self.metrics_collector)

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Monitoring collection failed: {e}", exc_info=True)
                await asyncio.sleep(self.collection_interval)

    async def _collect_scheduler_metrics(self) -> None:
        """Collect metrics from scheduler engine"""
        try:
            # Get scheduler status
            status = self.scheduler_engine.get_scheduler_status()

            # Record basic metrics
            await self.metrics_collector.record_metric(
                "scheduler.uptime_seconds",
                status["uptime_seconds"],
                tags={"scheduler_id": status["scheduler_id"]},
            )

            await self.metrics_collector.record_metric(
                "scheduler.active_executions",
                status["active_executions"],
                tags={"scheduler_id": status["scheduler_id"]},
            )

            # Record statistics
            stats = status["statistics"]
            await self.metrics_collector.record_metric(
                "scheduler.jobs.scheduled",
                stats["jobs_scheduled"],
                tags={"type": "total"},
            )

            await self.metrics_collector.record_metric(
                "scheduler.jobs.executed",
                stats["jobs_executed"],
                tags={"type": "total"},
            )

            await self.metrics_collector.record_metric(
                "scheduler.jobs.failed", stats["jobs_failed"], tags={"type": "total"}
            )

            # Get health status
            health = await self.scheduler_engine.get_health_status()

            await self.metrics_collector.record_metric(
                "scheduler.health.overall",
                1.0 if health["overall_healthy"] else 0.0,
                tags={"component": "scheduler"},
            )

            await self.metrics_collector.record_metric(
                "scheduler.health.database",
                1.0 if health["repository"]["database_connection"] else 0.0,
                tags={"component": "database"},
            )

            # Get execution manager metrics if available
            if hasattr(self.scheduler_engine, "execution_manager"):
                exec_health = (
                    self.scheduler_engine.execution_manager.get_health_status()
                )

                await self.metrics_collector.record_metric(
                    "scheduler.executions.stuck",
                    exec_health["stuck_executions"],
                    tags={"type": "stuck_count"},
                )

        except Exception as e:
            logger.error(f"Failed to collect scheduler metrics: {e}", exc_info=True)

    async def _log_alert_notification(self, alert: Alert) -> None:
        """Default alert notification handler that logs alerts"""
        logger.warning(
            f"ALERT: {alert.title}",
            severity=alert.severity.value,
            description=alert.description,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold,
        )

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler_status": self.scheduler_engine.get_scheduler_status(),
            "metrics": {
                "jobs_scheduled_1h": self.metrics_collector.get_metric_summary(
                    "scheduler.jobs.scheduled", 1
                ),
                "jobs_executed_1h": self.metrics_collector.get_metric_summary(
                    "scheduler.jobs.executed", 1
                ),
                "jobs_failed_1h": self.metrics_collector.get_metric_summary(
                    "scheduler.jobs.failed", 1
                ),
                "active_executions": self.metrics_collector.get_metric_summary(
                    "scheduler.active_executions", 1
                ),
            },
            "aggregations": self.metrics_collector.get_all_aggregations(),
        }

        if self.alert_manager:
            dashboard_data["alerts"] = {
                "active": [
                    asdict(alert) for alert in self.alert_manager.get_active_alerts()
                ],
                "recent": [
                    asdict(alert) for alert in self.alert_manager.get_alert_history(24)
                ],
            }

            # Convert datetime objects to strings
            for alert_list in [
                dashboard_data["alerts"]["active"],
                dashboard_data["alerts"]["recent"],
            ]:
                for alert in alert_list:
                    alert["triggered_at"] = (
                        alert["triggered_at"].isoformat()
                        if isinstance(alert["triggered_at"], datetime)
                        else alert["triggered_at"]
                    )
                    if alert["resolved_at"]:
                        alert["resolved_at"] = (
                            alert["resolved_at"].isoformat()
                            if isinstance(alert["resolved_at"], datetime)
                            else alert["resolved_at"]
                        )

        return dashboard_data

    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics for external monitoring systems"""
        return await self.metrics_collector.export_metrics(format)

    def add_custom_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        operator: str,
        severity: AlertSeverity,
        title: str,
        description: str,
    ) -> None:
        """Add custom alert rule"""
        if self.alert_manager:
            self.alert_manager.add_alert_rule(
                metric_name, threshold, operator, severity, title, description
            )
        else:
            logger.warning("Alert manager not enabled, cannot add alert rule")

    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add custom notification handler"""
        if self.alert_manager:
            self.alert_manager.add_notification_handler(handler)
        else:
            logger.warning("Alert manager not enabled, cannot add notification handler")
