"""
Comprehensive health checking system for the AEO Audit Tool.

This module provides system health monitoring, dependency checks, performance metrics,
and alerting capabilities for proactive system maintenance.
"""

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import psutil
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.audit_config import get_audit_settings
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.audit_metrics import get_audit_metrics
from app.services.platform_manager import PlatformManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """System component types"""

    DATABASE = "database"
    PLATFORM = "platform"
    CACHE = "cache"
    QUEUE = "queue"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SYSTEM_RESOURCES = "system_resources"
    EXTERNAL_API = "external_api"


@dataclass
class HealthMetric:
    """Individual health metric data"""

    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: HealthStatus = HealthStatus.HEALTHY
    message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

        # Determine status based on thresholds
        if self.threshold_critical and self.value >= self.threshold_critical:
            self.status = HealthStatus.UNHEALTHY
        elif self.threshold_warning and self.value >= self.threshold_warning:
            self.status = HealthStatus.DEGRADED


@dataclass
class ComponentHealth:
    """Health status for a system component"""

    name: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: Optional[float] = None
    metrics: List[HealthMetric] = None
    error_message: Optional[str] = None
    last_check: datetime = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now(timezone.utc)
        if self.metrics is None:
            self.metrics = []
        if self.dependencies is None:
            self.dependencies = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result["status"] = self.status.value
        result["component_type"] = self.component_type.value
        result["last_check"] = self.last_check.isoformat()

        # Convert metrics
        result["metrics"] = [
            {
                **asdict(metric),
                "status": metric.status.value,
                "timestamp": metric.timestamp.isoformat(),
            }
            for metric in self.metrics
        ]

        return result


@dataclass
class SystemHealth:
    """Overall system health status"""

    status: HealthStatus
    timestamp: datetime
    components: List[ComponentHealth]
    summary: Dict[str, Any]
    alert_level: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [comp.to_dict() for comp in self.components],
            "summary": self.summary,
            "alert_level": self.alert_level,
        }


class HealthChecker:
    """Comprehensive system health checker"""

    def __init__(self):
        self.platform_manager = PlatformManager()
        self.metrics = get_audit_metrics()
        self.settings = get_audit_settings()
        self._last_check = None
        self._cached_health = None
        self._cache_duration = timedelta(minutes=2)  # Cache health for 2 minutes
        self._history: Deque[SystemHealth] = deque(maxlen=720)

    async def check_system_health(self, force_refresh: bool = False) -> SystemHealth:
        """
        Perform comprehensive system health check.

        Args:
            force_refresh: Skip cache and force fresh health check

        Returns:
            SystemHealth with detailed component status
        """
        now = datetime.now(timezone.utc)

        # Use cached result if available and not expired
        if (
            not force_refresh
            and self._cached_health
            and self._last_check
            and now - self._last_check < self._cache_duration
        ):
            return self._cached_health

        logger.info("Starting comprehensive system health check")

        # Run all health checks concurrently
        health_checks = [
            self._check_database_health(),
            self._check_platform_health(),
            self._check_system_resources(),
            self._check_filesystem_health(),
            self._check_cache_health(),
            self._check_queue_health(),
        ]

        component_results = await asyncio.gather(*health_checks, return_exceptions=True)

        # Process results and handle exceptions
        components = []
        for result in component_results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}", exc_info=True)
                components.append(
                    ComponentHealth(
                        name="unknown",
                        component_type=ComponentType.SYSTEM_RESOURCES,
                        status=HealthStatus.UNHEALTHY,
                        error_message=str(result),
                    )
                )
            else:
                components.extend(result if isinstance(result, list) else [result])

        # Determine overall system health
        overall_status = self._determine_overall_status(components)

        # Generate summary
        summary = self._generate_summary(components)

        # Determine alert level
        alert_level = self._determine_alert_level(overall_status, components)

        system_health = SystemHealth(
            status=overall_status,
            timestamp=now,
            components=components,
            summary=summary,
            alert_level=alert_level,
        )

        # Cache result and retain history
        self._cached_health = system_health
        self._last_check = now
        self._history.append(system_health)

        logger.info(
            "System health check completed",
            overall_status=overall_status.value,
            components_count=len(components),
            alert_level=alert_level,
        )

        return system_health

    async def _check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        start_time = time.time()

        try:
            db = SessionLocal()

            # Test basic connectivity
            db.execute(text("SELECT 1"))

            # Test performance with a more complex query
            result = db.execute(
                text(
                    """
                SELECT
                    COUNT(*) as audit_count,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
                FROM auditrun
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """
                )
            ).fetchone()

            response_time = (time.time() - start_time) * 1000

            # Get connection pool stats if available
            pool_stats = None
            if hasattr(db.get_bind().pool, "status"):
                pool = db.get_bind().pool
                pool_stats = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                }

            db.close()

            # Create metrics
            metrics = [
                HealthMetric(
                    name="response_time",
                    value=response_time,
                    unit="ms",
                    threshold_warning=1000,  # 1 second
                    threshold_critical=5000,  # 5 seconds
                ),
                HealthMetric(
                    name="recent_audits",
                    value=result.audit_count if result else 0,
                    unit="count",
                ),
            ]

            if pool_stats:
                metrics.append(
                    HealthMetric(
                        name="connection_utilization",
                        value=(pool_stats["checked_out"] / pool_stats["size"]) * 100,
                        unit="percent",
                        threshold_warning=70,
                        threshold_critical=90,
                    )
                )

            # Determine status based on metrics
            status = HealthStatus.HEALTHY
            for metric in metrics:
                if metric.status == HealthStatus.UNHEALTHY:
                    status = HealthStatus.UNHEALTHY
                    break
                elif (
                    metric.status == HealthStatus.DEGRADED
                    and status == HealthStatus.HEALTHY
                ):
                    status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="database",
                component_type=ComponentType.DATABASE,
                status=status,
                response_time_ms=response_time,
                metrics=metrics,
            )

        except SQLAlchemyError as e:
            return ComponentHealth(
                name="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Database error: {str(e)}",
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def _check_platform_health(self) -> List[ComponentHealth]:
        """Check AI platform availability and performance"""
        platform_health = []

        available_platforms = self.platform_manager.get_available_platforms()

        for platform_name in available_platforms:
            start_time = time.time()

            try:
                platform = self.platform_manager.get_platform(platform_name)

                # Perform health check
                is_healthy = await platform.health_check()
                response_time = (time.time() - start_time) * 1000

                # Get rate limit info if available
                rate_limit_info = None
                if hasattr(platform, "get_rate_limit_status"):
                    try:
                        rate_limit_info = await platform.get_rate_limit_status()
                    except:
                        pass  # Rate limit info not available

                metrics = [
                    HealthMetric(
                        name="response_time",
                        value=response_time,
                        unit="ms",
                        threshold_warning=2000,
                        threshold_critical=10000,
                    )
                ]

                if rate_limit_info:
                    metrics.append(
                        HealthMetric(
                            name="rate_limit_remaining",
                            value=rate_limit_info.get("remaining", 0),
                            unit="requests",
                            threshold_warning=100,
                            threshold_critical=10,
                        )
                    )

                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

                # Check metrics for degraded status
                for metric in metrics:
                    if metric.status == HealthStatus.UNHEALTHY:
                        status = HealthStatus.UNHEALTHY
                        break
                    elif (
                        metric.status == HealthStatus.DEGRADED
                        and status == HealthStatus.HEALTHY
                    ):
                        status = HealthStatus.DEGRADED

                platform_health.append(
                    ComponentHealth(
                        name=f"platform_{platform_name}",
                        component_type=ComponentType.PLATFORM,
                        status=status,
                        response_time_ms=response_time,
                        metrics=metrics,
                    )
                )

            except Exception as e:
                platform_health.append(
                    ComponentHealth(
                        name=f"platform_{platform_name}",
                        component_type=ComponentType.PLATFORM,
                        status=HealthStatus.UNHEALTHY,
                        error_message=f"Platform error: {str(e)}",
                    )
                )

        return platform_health

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # Load average (Unix only)
            load_avg = None
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()[0]  # 1-minute load average

            metrics = [
                HealthMetric(
                    name="cpu_usage",
                    value=cpu_percent,
                    unit="percent",
                    threshold_warning=70,
                    threshold_critical=90,
                ),
                HealthMetric(
                    name="memory_usage",
                    value=memory_percent,
                    unit="percent",
                    threshold_warning=80,
                    threshold_critical=95,
                ),
                HealthMetric(
                    name="disk_usage",
                    value=disk_percent,
                    unit="percent",
                    threshold_warning=80,
                    threshold_critical=95,
                ),
            ]

            if load_avg is not None:
                cpu_count = psutil.cpu_count()
                load_avg_percent = (load_avg / cpu_count) * 100
                metrics.append(
                    HealthMetric(
                        name="load_average",
                        value=load_avg_percent,
                        unit="percent",
                        threshold_warning=80,
                        threshold_critical=100,
                    )
                )

            # Determine overall status
            status = HealthStatus.HEALTHY
            for metric in metrics:
                if metric.status == HealthStatus.UNHEALTHY:
                    status = HealthStatus.UNHEALTHY
                    break
                elif (
                    metric.status == HealthStatus.DEGRADED
                    and status == HealthStatus.HEALTHY
                ):
                    status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCES,
                status=status,
                metrics=metrics,
            )

        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCES,
                status=HealthStatus.UNHEALTHY,
                error_message=f"System monitoring error: {str(e)}",
            )

    async def _check_filesystem_health(self) -> ComponentHealth:
        """Check filesystem health and permissions"""
        try:
            import os
            import tempfile

            # Test write permissions in temp directory
            start_time = time.time()
            with tempfile.NamedTemporaryFile(delete=True) as f:
                f.write(b"health check")
                f.flush()
                os.fsync(f.fileno())
            write_time = (time.time() - start_time) * 1000

            # Check reports directory if it exists
            reports_writable = True
            reports_path = "reports"
            if os.path.exists(reports_path):
                reports_writable = os.access(reports_path, os.W_OK)

            metrics = [
                HealthMetric(
                    name="filesystem_write_time",
                    value=write_time,
                    unit="ms",
                    threshold_warning=100,
                    threshold_critical=1000,
                )
            ]

            status = HealthStatus.HEALTHY
            if not reports_writable:
                status = HealthStatus.DEGRADED

            for metric in metrics:
                if metric.status == HealthStatus.UNHEALTHY:
                    status = HealthStatus.UNHEALTHY
                    break
                elif (
                    metric.status == HealthStatus.DEGRADED
                    and status == HealthStatus.HEALTHY
                ):
                    status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="filesystem",
                component_type=ComponentType.FILESYSTEM,
                status=status,
                metrics=metrics,
                error_message=None
                if reports_writable
                else "Reports directory not writable",
            )

        except Exception as e:
            return ComponentHealth(
                name="filesystem",
                component_type=ComponentType.FILESYSTEM,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Filesystem error: {str(e)}",
            )

    async def _check_cache_health(self) -> ComponentHealth:
        """Check cache system health (Redis if configured)."""

        start = time.perf_counter()
        client: Optional[redis.Redis] = None

        try:
            client = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
                encoding="utf-8",
                decode_responses=True,
            )

            await client.ping()
            response_time = (time.perf_counter() - start) * 1000

            metrics = [
                HealthMetric(
                    name="redis_ping_ms",
                    value=response_time,
                    unit="ms",
                    threshold_warning=250,
                    threshold_critical=1000,
                )
            ]

            try:
                info = await client.info(section="memory")
                used_memory = info.get("used_memory", 0) / (1024 * 1024)
                metrics.append(
                    HealthMetric(
                        name="redis_used_memory_mb",
                        value=used_memory,
                        unit="MB",
                        threshold_warning=768,
                        threshold_critical=1024,
                    )
                )
            except Exception as info_error:  # pragma: no cover - optional metric
                logger.debug("Redis memory info unavailable", error=str(info_error))

            status = HealthStatus.HEALTHY
            for metric in metrics:
                if metric.status == HealthStatus.UNHEALTHY:
                    status = HealthStatus.UNHEALTHY
                    break
                if (
                    metric.status == HealthStatus.DEGRADED
                    and status == HealthStatus.HEALTHY
                ):
                    status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="cache",
                component_type=ComponentType.CACHE,
                status=status,
                response_time_ms=response_time,
                metrics=metrics,
            )

        except Exception as exc:
            return ComponentHealth(
                name="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Cache error: {str(exc)}",
            )
        finally:
            if client is not None and hasattr(client, "aclose"):
                try:
                    await client.aclose()  # type: ignore[attr-defined]
                except Exception as close_error:  # pragma: no cover - defensive log
                    logger.debug("Failed to close redis client", error=str(close_error))

    async def _check_queue_health(self) -> ComponentHealth:
        """Check Celery queue health."""

        try:
            from app.core.celery_app import celery_app
        except Exception as exc:
            return ComponentHealth(
                name="queue",
                component_type=ComponentType.QUEUE,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Celery app unavailable: {str(exc)}",
            )

        start = time.perf_counter()

        try:
            ping_result = await asyncio.to_thread(celery_app.control.ping, timeout=1.0)
            worker_count = len(ping_result) if ping_result else 0

            def _active_tasks() -> Dict[str, List[Any]]:
                snapshot = celery_app.control.inspect().active() or {}
                return {worker: tasks or [] for worker, tasks in snapshot.items()}

            active = await asyncio.to_thread(_active_tasks)
            total_active = sum(len(tasks) for tasks in active.values())

            metrics = [
                HealthMetric(
                    name="worker_count",
                    value=worker_count,
                    unit="workers",
                    status=HealthStatus.HEALTHY
                    if worker_count > 0
                    else HealthStatus.UNHEALTHY,
                ),
                HealthMetric(
                    name="active_tasks",
                    value=total_active,
                    unit="tasks",
                    threshold_warning=50,
                    threshold_critical=150,
                ),
            ]

            status = HealthStatus.HEALTHY
            for metric in metrics:
                if metric.status == HealthStatus.UNHEALTHY:
                    status = HealthStatus.UNHEALTHY
                    break
                if (
                    metric.status == HealthStatus.DEGRADED
                    and status == HealthStatus.HEALTHY
                ):
                    status = HealthStatus.DEGRADED

            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="queue",
                component_type=ComponentType.QUEUE,
                status=status,
                response_time_ms=response_time,
                metrics=metrics,
            )

        except Exception as exc:
            return ComponentHealth(
                name="queue",
                component_type=ComponentType.QUEUE,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Queue error: {str(exc)}",
            )

    def _determine_overall_status(
        self, components: List[ComponentHealth]
    ) -> HealthStatus:
        """Determine overall system status from component health"""
        if not components:
            return HealthStatus.UNKNOWN

        # Count statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0,
        }

        for component in components:
            status_counts[component.status] += 1

        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            # Critical components unhealthy
            critical_components = [
                "database",
                "platform_openai",
            ]  # Define critical components
            for component in components:
                if (
                    component.name in critical_components
                    and component.status == HealthStatus.UNHEALTHY
                ):
                    return HealthStatus.UNHEALTHY

            # Non-critical components unhealthy
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _generate_summary(self, components: List[ComponentHealth]) -> Dict[str, Any]:
        """Generate health summary"""
        total_components = len(components)
        healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(
            1 for c in components if c.status == HealthStatus.UNHEALTHY
        )

        return {
            "total_components": total_components,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "health_percentage": (healthy_count / total_components) * 100
            if total_components > 0
            else 0,
        }

    def _determine_alert_level(
        self, overall_status: HealthStatus, components: List[ComponentHealth]
    ) -> Optional[str]:
        """Determine alert level based on health status"""
        if overall_status == HealthStatus.UNHEALTHY:
            return "critical"
        elif overall_status == HealthStatus.DEGRADED:
            # Check if any critical components are affected
            critical_components = ["database", "platform_openai"]
            for component in components:
                if component.name in critical_components and component.status in [
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY,
                ]:
                    return "high"
            return "medium"
        else:
            return None

    async def get_component_health(
        self, component_name: str
    ) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        system_health = await self.check_system_health()

        for component in system_health.components:
            if component.name == component_name:
                return component

        return None

    async def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent health check history within the requested window."""

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        history: List[Dict[str, Any]] = []

        for record in reversed(self._history):
            if record.timestamp < cutoff:
                break
            history.append(record.to_dict())

        return history


# === Global Health Checker Instance ===

health_checker = HealthChecker()


# === Utility Functions ===


async def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check for basic monitoring"""
    try:
        system_health = await health_checker.check_system_health()

        return {
            "status": system_health.status.value,
            "timestamp": system_health.timestamp.isoformat(),
            "summary": system_health.summary,
        }
    except Exception as e:
        logger.error(f"Quick health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


async def check_readiness() -> bool:
    """Check if system is ready to handle requests"""
    try:
        system_health = await health_checker.check_system_health()

        # System is ready if overall status is not unhealthy
        return system_health.status != HealthStatus.UNHEALTHY
    except Exception:
        return False


async def check_liveness() -> bool:
    """Check if system is alive (basic connectivity)"""
    try:
        # Basic database connectivity check
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception:
        return False
