"""
Health monitoring system for the scheduler.

Provides comprehensive health checks, system diagnostics,
and automated recovery mechanisms.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from app.utils.logger import get_logger

from .engine import SchedulerEngine
from .execution_manager import ExecutionManager
from .repository import SchedulingRepository

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    checked_at: datetime
    response_time_ms: Optional[float] = None


class SystemHealthChecker:
    """
    Performs system-level health checks.

    Monitors system resources, database connectivity,
    and external service dependencies.
    """

    def __init__(self):
        """Initialize system health checker"""
        self.warning_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "load_average_1m": 4.0,
        }

        self.critical_thresholds = {
            "cpu_percent": 95.0,
            "memory_percent": 95.0,
            "disk_percent": 98.0,
            "load_average_1m": 8.0,
        }

    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization"""
        start_time = datetime.now(timezone.utc)

        try:
            if psutil is None:
                details = {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "memory_available_mb": None,
                    "disk_percent": None,
                    "disk_free_gb": None,
                    "load_average_1m": os.getloadavg()[0]
                    if hasattr(os, "getloadavg")
                    else 0.0,
                    "python_version": sys.version,
                    "note": "psutil not installed; metrics unavailable",
                }

                response_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                return HealthCheckResult(
                    component="system_resources",
                    status=HealthStatus.UNKNOWN,
                    message="System metrics unavailable (psutil missing)",
                    details=details,
                    checked_at=start_time,
                    response_time_ms=response_time,
                )

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // 1024 // 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // 1024 // 1024 // 1024,
                "load_average_1m": load_avg,
                "python_version": sys.version,
            }

            # Determine health status
            status = HealthStatus.HEALTHY
            issues = []

            # Check critical thresholds
            if cpu_percent >= self.critical_thresholds["cpu_percent"]:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.warning_thresholds["cpu_percent"]:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory.percent >= self.critical_thresholds["memory_percent"]:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.warning_thresholds["memory_percent"]:
                status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory.percent:.1f}%")

            if disk.percent >= self.critical_thresholds["disk_percent"]:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent >= self.warning_thresholds["disk_percent"]:
                status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk.percent:.1f}%")

            if load_avg >= self.critical_thresholds["load_average_1m"]:
                status = HealthStatus.CRITICAL
                issues.append(f"Load average critical: {load_avg:.2f}")
            elif load_avg >= self.warning_thresholds["load_average_1m"]:
                status = HealthStatus.WARNING
                issues.append(f"Load average high: {load_avg:.2f}")

            message = (
                "System resources healthy"
                if status == HealthStatus.HEALTHY
                else "; ".join(issues)
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                details=details,
                checked_at=start_time,
                response_time_ms=response_time,
            )

        except Exception as e:
            logger.error(f"System resource health check failed: {e}", exc_info=True)

            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {e}",
                details={},
                checked_at=start_time,
            )

    async def check_database_connectivity(
        self, repository: SchedulingRepository
    ) -> HealthCheckResult:
        """Check database connectivity and performance"""
        start_time = datetime.now(timezone.utc)

        try:
            # Perform database health check
            health_result = repository.health_check()

            # Determine status based on results
            if health_result["scheduler_healthy"]:
                if len(health_result.get("issues", [])) == 0:
                    status = HealthStatus.HEALTHY
                    message = "Database connectivity healthy"
                else:
                    status = HealthStatus.WARNING
                    message = f"Database issues detected: {'; '.join(health_result['issues'])}"
            else:
                status = HealthStatus.CRITICAL
                message = "Database connectivity failed"

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return HealthCheckResult(
                component="database",
                status=status,
                message=message,
                details=health_result,
                checked_at=start_time,
                response_time_ms=response_time,
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=True)

            return HealthCheckResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {e}",
                details={},
                checked_at=start_time,
            )


class SchedulerHealthChecker:
    """
    Performs scheduler-specific health checks.

    Monitors scheduler engine, job execution, queue health,
    and scheduling performance.
    """

    def __init__(self):
        """Initialize scheduler health checker"""
        pass

    async def check_scheduler_engine(
        self, engine: SchedulerEngine
    ) -> HealthCheckResult:
        """Check scheduler engine health"""
        start_time = datetime.now(timezone.utc)

        try:
            status_info = engine.get_scheduler_status()

            details = {
                "status": status_info["status"],
                "uptime_seconds": status_info["uptime_seconds"],
                "active_executions": status_info["active_executions"],
                "statistics": status_info["statistics"],
                "last_poll": status_info.get("last_poll"),
            }

            # Determine health status
            scheduler_status = status_info["status"]

            if scheduler_status == "running":
                status = HealthStatus.HEALTHY
                message = "Scheduler engine running normally"

                # Check for concerning patterns
                stats = status_info["statistics"]

                if stats["jobs_failed"] > 0:
                    failure_rate = stats["jobs_failed"] / max(stats["jobs_executed"], 1)
                    if failure_rate > 0.5:  # > 50% failure rate
                        status = HealthStatus.CRITICAL
                        message = f"High job failure rate: {failure_rate:.1%}"
                    elif failure_rate > 0.1:  # > 10% failure rate
                        status = HealthStatus.WARNING
                        message = f"Elevated job failure rate: {failure_rate:.1%}"

                # Check if scheduler is polling recently
                if status_info.get("last_poll"):
                    last_poll = datetime.fromisoformat(
                        status_info["last_poll"].replace("Z", "+00:00")
                    )
                    time_since_poll = (
                        datetime.now(timezone.utc) - last_poll
                    ).total_seconds()

                    if time_since_poll > 300:  # 5 minutes
                        status = HealthStatus.CRITICAL
                        message = f"Scheduler not polling (last poll: {time_since_poll:.0f}s ago)"
                    elif time_since_poll > 120:  # 2 minutes
                        status = HealthStatus.WARNING
                        message = f"Scheduler polling delayed (last poll: {time_since_poll:.0f}s ago)"

            elif scheduler_status in ["paused", "pausing"]:
                status = HealthStatus.WARNING
                message = f"Scheduler is {scheduler_status}"

            else:
                status = HealthStatus.CRITICAL
                message = f"Scheduler in unhealthy state: {scheduler_status}"

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return HealthCheckResult(
                component="scheduler_engine",
                status=status,
                message=message,
                details=details,
                checked_at=start_time,
                response_time_ms=response_time,
            )

        except Exception as e:
            logger.error(f"Scheduler engine health check failed: {e}", exc_info=True)

            return HealthCheckResult(
                component="scheduler_engine",
                status=HealthStatus.CRITICAL,
                message=f"Scheduler engine health check failed: {e}",
                details={},
                checked_at=start_time,
            )

    async def check_execution_manager(
        self, exec_manager: ExecutionManager
    ) -> HealthCheckResult:
        """Check execution manager health"""
        start_time = datetime.now(timezone.utc)

        try:
            health_info = exec_manager.get_health_status()

            details = {
                "active_executions": health_info["active_executions"],
                "stuck_executions": health_info["stuck_executions"],
                "stuck_execution_details": health_info["stuck_execution_details"],
            }

            # Determine health status
            if health_info["is_healthy"]:
                status = HealthStatus.HEALTHY
                message = "Execution manager healthy"
            else:
                if health_info["stuck_executions"] > 5:
                    status = HealthStatus.CRITICAL
                    message = (
                        f"Many stuck executions: {health_info['stuck_executions']}"
                    )
                else:
                    status = HealthStatus.WARNING
                    message = (
                        f"Some stuck executions: {health_info['stuck_executions']}"
                    )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return HealthCheckResult(
                component="execution_manager",
                status=status,
                message=message,
                details=details,
                checked_at=start_time,
                response_time_ms=response_time,
            )

        except Exception as e:
            logger.error(f"Execution manager health check failed: {e}", exc_info=True)

            return HealthCheckResult(
                component="execution_manager",
                status=HealthStatus.CRITICAL,
                message=f"Execution manager health check failed: {e}",
                details={},
                checked_at=start_time,
            )


class HealthMonitor:
    """
    Comprehensive health monitoring system.

    Coordinates health checks, maintains health history,
    and provides recovery recommendations.
    """

    def __init__(
        self,
        scheduler_engine: SchedulerEngine,
        check_interval: int = 60,  # seconds
        history_retention_hours: int = 24,
    ):
        """
        Initialize health monitor.

        Args:
            scheduler_engine: Main scheduler engine
            check_interval: How often to perform health checks
            history_retention_hours: How long to keep health history
        """
        self.scheduler_engine = scheduler_engine
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours

        # Health checkers
        self.system_checker = SystemHealthChecker()
        self.scheduler_checker = SchedulerHealthChecker()

        # Health history
        self._health_history: List[List[HealthCheckResult]] = []
        self._lock = asyncio.Lock()

        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Recovery handlers
        self._recovery_handlers: Dict[str, Callable] = {}
        self._setup_default_recovery_handlers()

    def _setup_default_recovery_handlers(self) -> None:
        """Setup default recovery handlers"""
        self._recovery_handlers.update(
            {
                "stuck_executions": self._handle_stuck_executions,
                "high_failure_rate": self._handle_high_failure_rate,
                "database_issues": self._handle_database_issues,
                "resource_exhaustion": self._handle_resource_exhaustion,
            }
        )

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self._is_monitoring:
            logger.warning("Health monitoring already started")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info(
            "Started health monitoring",
            check_interval=self.check_interval,
            retention_hours=self.history_retention_hours,
        )

    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main health monitoring loop"""
        while self._is_monitoring:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Health monitoring loop failed: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def perform_health_check(self) -> List[HealthCheckResult]:
        """Perform comprehensive health check"""
        results = []

        try:
            # System health checks
            system_result = await self.system_checker.check_system_resources()
            results.append(system_result)

            # Database health check
            db_result = await self.system_checker.check_database_connectivity(
                self.scheduler_engine.repository
            )
            results.append(db_result)

            # Scheduler engine health check
            engine_result = await self.scheduler_checker.check_scheduler_engine(
                self.scheduler_engine
            )
            results.append(engine_result)

            # Execution manager health check
            exec_result = await self.scheduler_checker.check_execution_manager(
                self.scheduler_engine.execution_manager
            )
            results.append(exec_result)

            # Store results in history
            async with self._lock:
                self._health_history.append(results)
                await self._cleanup_old_history()

            # Trigger recovery actions if needed
            await self._evaluate_recovery_actions(results)

            # Log overall health status
            overall_status = self._calculate_overall_status(results)
            if overall_status != HealthStatus.HEALTHY:
                logger.warning(
                    f"Health check completed with {overall_status.value} status",
                    results_summary=[
                        f"{r.component}: {r.status.value}" for r in results
                    ],
                )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)

            # Add error result
            error_result = HealthCheckResult(
                component="health_monitor",
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {e}",
                details={},
                checked_at=datetime.now(timezone.utc),
            )
            results.append(error_result)

        return results

    async def _cleanup_old_history(self) -> None:
        """Remove old health check history"""
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.history_retention_hours
        )

        # Filter out old entries
        self._health_history = [
            results
            for results in self._health_history
            if any(r.checked_at >= cutoff for r in results)
        ]

    def _calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate overall health status from individual results"""
        if not results:
            return HealthStatus.UNKNOWN

        # Priority: CRITICAL > WARNING > HEALTHY > UNKNOWN
        if any(r.status == HealthStatus.CRITICAL for r in results):
            return HealthStatus.CRITICAL
        elif any(r.status == HealthStatus.WARNING for r in results):
            return HealthStatus.WARNING
        elif all(r.status == HealthStatus.HEALTHY for r in results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    async def _evaluate_recovery_actions(
        self, results: List[HealthCheckResult]
    ) -> None:
        """Evaluate if recovery actions are needed"""
        for result in results:
            if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                # Check for specific recovery scenarios
                if (
                    result.component == "execution_manager"
                    and "stuck executions" in result.message.lower()
                ):
                    await self._trigger_recovery("stuck_executions", result)

                elif (
                    result.component == "scheduler_engine"
                    and "failure rate" in result.message.lower()
                ):
                    await self._trigger_recovery("high_failure_rate", result)

                elif result.component == "database" and not result.details.get(
                    "database_connection", True
                ):
                    await self._trigger_recovery("database_issues", result)

                elif result.component == "system_resources":
                    if (
                        result.details.get("memory_percent", 0) > 95
                        or result.details.get("cpu_percent", 0) > 95
                    ):
                        await self._trigger_recovery("resource_exhaustion", result)

    async def _trigger_recovery(
        self, recovery_type: str, health_result: HealthCheckResult
    ) -> None:
        """Trigger recovery action"""
        handler = self._recovery_handlers.get(recovery_type)
        if handler:
            try:
                logger.info(f"Triggering recovery action: {recovery_type}")
                await handler(health_result)
            except Exception as e:
                logger.error(
                    f"Recovery action failed: {recovery_type}: {e}", exc_info=True
                )
        else:
            logger.warning(f"No recovery handler for: {recovery_type}")

    # Recovery handlers

    async def _handle_stuck_executions(self, health_result: HealthCheckResult) -> None:
        """Handle stuck executions"""
        stuck_details = health_result.details.get("stuck_execution_details", [])

        for stuck_execution in stuck_details:
            execution_id = stuck_execution.get("execution_id")
            runtime = stuck_execution.get("runtime_seconds", 0)

            if runtime > 7200:  # 2 hours - force cancel
                logger.warning(f"Force cancelling stuck execution: {execution_id}")
                success = (
                    await self.scheduler_engine.execution_manager.cancel_execution(
                        execution_id,
                        reason="Automatically cancelled due to excessive runtime",
                    )
                )
                if success:
                    logger.info(
                        f"Successfully cancelled stuck execution: {execution_id}"
                    )

    async def _handle_high_failure_rate(self, health_result: HealthCheckResult) -> None:
        """Handle high job failure rate"""
        logger.warning("High failure rate detected - implementing recovery measures")

        # Could pause non-critical jobs, increase retry delays, etc.
        # For now, just log the issue
        stats = health_result.details.get("statistics", {})
        logger.info(
            "Failure rate analysis",
            jobs_executed=stats.get("jobs_executed", 0),
            jobs_failed=stats.get("jobs_failed", 0),
        )

    async def _handle_database_issues(self, health_result: HealthCheckResult) -> None:
        """Handle database connectivity issues"""
        logger.error("Database connectivity issues detected")

        # Could implement connection retry, failover, etc.
        # For now, just clean up expired locks which might help
        try:
            cleaned_locks = self.scheduler_engine.repository.cleanup_expired_locks()
            logger.info(
                f"Cleaned up {cleaned_locks} expired locks during database recovery"
            )
        except Exception as e:
            logger.error(f"Failed to cleanup locks during database recovery: {e}")

    async def _handle_resource_exhaustion(
        self, health_result: HealthCheckResult
    ) -> None:
        """Handle system resource exhaustion"""
        details = health_result.details

        logger.warning(
            "Resource exhaustion detected",
            cpu_percent=details.get("cpu_percent"),
            memory_percent=details.get("memory_percent"),
        )

        # Could pause scheduler, reduce concurrency, etc.
        if self.scheduler_engine.status.value == "running":
            logger.info("Pausing scheduler due to resource exhaustion")
            await self.scheduler_engine.pause()

            # Schedule resume after some time
            asyncio.create_task(self._delayed_resume(300))  # Resume after 5 minutes

    async def _delayed_resume(self, delay_seconds: int) -> None:
        """Resume scheduler after delay"""
        await asyncio.sleep(delay_seconds)

        if self.scheduler_engine.status.value == "paused":
            logger.info("Resuming scheduler after resource recovery period")
            await self.scheduler_engine.resume()

    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self._health_history:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "components": [],
                "last_check": None,
            }

        latest_results = self._health_history[-1]
        overall_status = self._calculate_overall_status(latest_results)

        return {
            "overall_status": overall_status.value,
            "message": self._get_overall_message(latest_results),
            "components": [
                {
                    "component": r.component,
                    "status": r.status.value,
                    "message": r.message,
                    "response_time_ms": r.response_time_ms,
                }
                for r in latest_results
            ],
            "last_check": latest_results[0].checked_at.isoformat()
            if latest_results
            else None,
        }

    def _get_overall_message(self, results: List[HealthCheckResult]) -> str:
        """Get overall health message"""
        critical_issues = [r for r in results if r.status == HealthStatus.CRITICAL]
        warning_issues = [r for r in results if r.status == HealthStatus.WARNING]

        if critical_issues:
            return f"{len(critical_issues)} critical issues detected"
        elif warning_issues:
            return f"{len(warning_issues)} warnings detected"
        else:
            return "All systems healthy"

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter relevant history
        relevant_history = [
            results
            for results in self._health_history
            if any(r.checked_at >= cutoff for r in results)
        ]

        if not relevant_history:
            return {"trend_data": [], "summary": "No data available"}

        # Analyze trends
        trend_data = []
        status_counts = {status.value: 0 for status in HealthStatus}

        for results in relevant_history:
            timestamp = results[0].checked_at.isoformat()
            overall_status = self._calculate_overall_status(results)

            status_counts[overall_status.value] += 1

            trend_data.append(
                {
                    "timestamp": timestamp,
                    "overall_status": overall_status.value,
                    "component_statuses": {
                        r.component: r.status.value for r in results
                    },
                }
            )

        total_checks = len(relevant_history)
        health_percentage = (
            (status_counts[HealthStatus.HEALTHY.value] / total_checks * 100)
            if total_checks > 0
            else 0
        )

        return {
            "trend_data": trend_data,
            "summary": {
                "total_checks": total_checks,
                "health_percentage": health_percentage,
                "status_distribution": status_counts,
                "period_hours": hours,
            },
        }
