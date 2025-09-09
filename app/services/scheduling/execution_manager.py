"""
Job execution history and lifecycle management.

Provides comprehensive tracking of job executions with proper state management,
error handling, and performance monitoring.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from app.models.scheduling import ExecutionStatus, ScheduledJob
from app.utils.logger import get_logger

from .repository import SchedulingRepository

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """Context information for job execution"""

    execution_id: str
    job_id: str
    job_name: str
    trigger_type: str
    scheduled_time: datetime
    started_at: datetime
    metadata: Dict[str, Any]
    timeout_seconds: Optional[int] = None


class ExecutionError(Exception):
    """Base exception for execution errors"""

    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when execution times out"""

    pass


class ExecutionManager:
    """
    Manages job execution lifecycle and history.

    Handles execution tracking, state management, timeout handling,
    and coordination with the scheduling repository.
    """

    def __init__(self, repository: Optional[SchedulingRepository] = None):
        """Initialize execution manager"""
        self.repository = repository or SchedulingRepository()
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_success": [],
            "on_failure": [],
            "on_timeout": [],
            "on_complete": [],
        }

    def add_execution_callback(self, event: str, callback: Callable) -> None:
        """Add callback for execution events"""
        if event in self._execution_callbacks:
            self._execution_callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown execution event: {event}")

    async def _notify_callbacks(
        self, event: str, context: ExecutionContext, **kwargs
    ) -> None:
        """Notify all callbacks for an event"""
        for callback in self._execution_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, **kwargs)
                else:
                    callback(context, **kwargs)
            except Exception as e:
                logger.error(
                    "Execution callback failed",
                    event=event,
                    execution_id=context.execution_id,
                    error=str(e),
                    exc_info=True,
                )

    async def start_execution(
        self,
        job: ScheduledJob,
        scheduled_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        Start tracking a new job execution.

        Args:
            job: Job being executed
            scheduled_time: Time job was scheduled to run
            metadata: Additional execution metadata

        Returns:
            ExecutionContext for this execution
        """
        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        if scheduled_time is None:
            scheduled_time = now

        if metadata is None:
            metadata = {}

        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            job_id=job.job_id,
            job_name=job.name,
            trigger_type=job.trigger_type.value,
            scheduled_time=scheduled_time,
            started_at=now,
            metadata=metadata,
            timeout_seconds=job.timeout_seconds,
        )

        try:
            # Create database record
            execution_data = {
                "execution_id": execution_id,
                "job_id": job.job_id,
                "status": ExecutionStatus.RUNNING,
                "scheduled_time": scheduled_time,
                "started_at": now,
                "metadata": metadata,
                "trigger_info": job.trigger_config,
            }

            execution = self.repository.create_execution(execution_data)

            # Track active execution
            self._active_executions[execution_id] = context

            # Notify callbacks
            await self._notify_callbacks("on_start", context)

            logger.info(
                "Started job execution",
                execution_id=execution_id,
                job_id=job.job_id,
                job_name=job.name,
                scheduled_time=scheduled_time.isoformat(),
            )

            return context

        except Exception as e:
            logger.error(
                "Failed to start execution tracking",
                job_id=job.job_id,
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )
            raise ExecutionError(f"Failed to start execution: {e}") from e

    async def complete_execution(
        self,
        execution_id: str,
        status: ExecutionStatus,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark execution as completed with given status.

        Args:
            execution_id: ID of execution to complete
            status: Final execution status
            result: Execution result data
            error_message: Error message if failed
            error_details: Additional error information
        """
        if execution_id not in self._active_executions:
            logger.warning(f"Attempted to complete unknown execution: {execution_id}")
            return

        context = self._active_executions[execution_id]
        finished_at = datetime.now(timezone.utc)
        runtime_seconds = (finished_at - context.started_at).total_seconds()

        try:
            # Update database record
            updates = {
                "status": status,
                "finished_at": finished_at,
                "runtime_seconds": runtime_seconds,
                "result": result,
                "error_message": error_message,
                "error_details": error_details,
            }

            self.repository.update_execution(execution_id, updates)

            # Remove from active tracking
            del self._active_executions[execution_id]

            # Notify status-specific callbacks
            if status == ExecutionStatus.SUCCESS:
                await self._notify_callbacks("on_success", context, result=result)
            elif status == ExecutionStatus.FAILURE:
                await self._notify_callbacks(
                    "on_failure",
                    context,
                    error_message=error_message,
                    error_details=error_details,
                )
            elif status == ExecutionStatus.TIMEOUT:
                await self._notify_callbacks("on_timeout", context)

            # Always notify completion
            await self._notify_callbacks(
                "on_complete", context, status=status, runtime_seconds=runtime_seconds
            )

            logger.info(
                "Completed job execution",
                execution_id=execution_id,
                job_id=context.job_id,
                status=status.value,
                runtime_seconds=runtime_seconds,
                success=status == ExecutionStatus.SUCCESS,
            )

        except Exception as e:
            logger.error(
                "Failed to complete execution tracking",
                execution_id=execution_id,
                job_id=context.job_id,
                error=str(e),
                exc_info=True,
            )
            raise ExecutionError(f"Failed to complete execution: {e}") from e

    async def update_execution_progress(
        self, execution_id: str, progress_data: Dict[str, Any]
    ) -> None:
        """Update execution progress information"""
        if execution_id not in self._active_executions:
            logger.warning(f"Attempted to update unknown execution: {execution_id}")
            return

        try:
            # Merge with existing metadata
            updates = {
                "metadata": {
                    **self._active_executions[execution_id].metadata,
                    "progress": progress_data,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            }

            self.repository.update_execution(execution_id, updates)

            logger.debug(
                "Updated execution progress",
                execution_id=execution_id,
                progress_keys=list(progress_data.keys()),
            )

        except Exception as e:
            logger.error(
                "Failed to update execution progress",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )

    @asynccontextmanager
    async def track_execution(
        self,
        job: ScheduledJob,
        scheduled_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for automatic execution tracking.

        Usage:
            async with execution_manager.track_execution(job) as context:
                # Execute job logic
                result = await job_function()
                return result
        """
        context = await self.start_execution(job, scheduled_time, metadata)

        # Set up timeout if specified
        timeout_task = None
        if context.timeout_seconds:
            timeout_task = asyncio.create_task(
                self._handle_execution_timeout(
                    context.execution_id, context.timeout_seconds
                )
            )

        try:
            yield context

            # If we get here, execution completed successfully
            await self.complete_execution(context.execution_id, ExecutionStatus.SUCCESS)

        except asyncio.TimeoutError:
            await self.complete_execution(
                context.execution_id,
                ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {context.timeout_seconds} seconds",
            )
            raise ExecutionTimeoutError("Job execution timed out")

        except Exception as e:
            await self.complete_execution(
                context.execution_id,
                ExecutionStatus.FAILURE,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )
            raise

        finally:
            # Cancel timeout task if it's still running
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
                try:
                    await timeout_task
                except asyncio.CancelledError:
                    pass

    async def _handle_execution_timeout(
        self, execution_id: str, timeout_seconds: int
    ) -> None:
        """Handle execution timeout"""
        try:
            await asyncio.sleep(timeout_seconds)

            # If we're still here, the execution timed out
            if execution_id in self._active_executions:
                context = self._active_executions[execution_id]

                logger.warning(
                    "Job execution timed out",
                    execution_id=execution_id,
                    job_id=context.job_id,
                    timeout_seconds=timeout_seconds,
                )

                await self.complete_execution(
                    execution_id,
                    ExecutionStatus.TIMEOUT,
                    error_message=f"Execution timed out after {timeout_seconds} seconds",
                )

        except asyncio.CancelledError:
            # Timeout was cancelled - execution completed normally
            pass

    def get_active_executions(self) -> List[ExecutionContext]:
        """Get all currently active executions"""
        return list(self._active_executions.values())

    def get_execution_context(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get execution context by ID"""
        return self._active_executions.get(execution_id)

    async def get_execution_history(
        self,
        job_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100,
        include_running: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get execution history with optional filtering.

        Args:
            job_id: Filter by specific job
            status: Filter by execution status
            limit: Maximum number of records
            include_running: Include currently running executions

        Returns:
            List of execution records with enhanced information
        """
        try:
            # Get database records
            if job_id:
                executions = self.repository.get_job_executions(job_id, limit, status)
            else:
                # Get all recent executions
                executions = []  # Would need additional repository method

            history = []
            for execution in executions:
                record = {
                    "execution_id": execution.execution_id,
                    "job_id": execution.job_id,
                    "status": execution.status.value,
                    "scheduled_time": execution.scheduled_time.isoformat(),
                    "started_at": execution.started_at.isoformat(),
                    "finished_at": execution.finished_at.isoformat()
                    if execution.finished_at
                    else None,
                    "runtime_seconds": execution.runtime_seconds,
                    "result": execution.result,
                    "error_message": execution.error_message,
                    "metadata": execution.metadata,
                }
                history.append(record)

            # Add currently running executions if requested
            if include_running:
                for context in self._active_executions.values():
                    if not job_id or context.job_id == job_id:
                        runtime = (
                            datetime.now(timezone.utc) - context.started_at
                        ).total_seconds()

                        record = {
                            "execution_id": context.execution_id,
                            "job_id": context.job_id,
                            "status": "RUNNING",
                            "scheduled_time": context.scheduled_time.isoformat(),
                            "started_at": context.started_at.isoformat(),
                            "finished_at": None,
                            "runtime_seconds": runtime,
                            "result": None,
                            "error_message": None,
                            "metadata": context.metadata,
                            "is_active": True,
                        }
                        history.append(record)

            # Sort by started_at descending
            history.sort(key=lambda x: x["started_at"], reverse=True)

            return history[:limit]

        except Exception as e:
            logger.error(f"Failed to get execution history: {e}", exc_info=True)
            raise ExecutionError(f"Failed to get execution history: {e}")

    async def get_execution_statistics(
        self, job_id: Optional[str] = None, time_period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get execution statistics for monitoring"""
        try:
            stats = self.repository.get_job_statistics(job_id)

            # Add real-time information
            active_count = len(
                [
                    ctx
                    for ctx in self._active_executions.values()
                    if not job_id or ctx.job_id == job_id
                ]
            )

            stats.update(
                {
                    "currently_running": active_count,
                    "time_period_hours": time_period_hours,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to get execution statistics: {e}", exc_info=True)
            raise ExecutionError(f"Failed to get execution statistics: {e}")

    async def cleanup_old_executions(
        self, retention_days: int = 30, keep_failures_days: int = 90
    ) -> Dict[str, int]:
        """
        Clean up old execution records based on retention policy.

        Args:
            retention_days: Days to keep successful executions
            keep_failures_days: Days to keep failed executions (usually longer)

        Returns:
            Cleanup statistics
        """
        try:
            now = datetime.now(timezone.utc)
            success_cutoff = now - timedelta(days=retention_days)
            failure_cutoff = now - timedelta(days=keep_failures_days)

            # Clean up successful executions older than retention_days
            success_deleted = 0  # Would need additional repository methods

            # Clean up failed executions older than keep_failures_days
            failure_deleted = 0

            # For now, use the general cleanup method
            total_deleted = self.repository.cleanup_old_executions(failure_cutoff)

            cleanup_stats = {
                "total_deleted": total_deleted,
                "success_deleted": success_deleted,
                "failure_deleted": failure_deleted,
                "retention_days": retention_days,
                "keep_failures_days": keep_failures_days,
            }

            logger.info("Cleaned up old execution records", **cleanup_stats)

            return cleanup_stats

        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}", exc_info=True)
            raise ExecutionError(f"Failed to cleanup executions: {e}")

    async def cancel_execution(
        self, execution_id: str, reason: str = "Cancelled"
    ) -> bool:
        """Cancel a running execution"""
        if execution_id not in self._active_executions:
            logger.warning(f"Attempted to cancel unknown execution: {execution_id}")
            return False

        try:
            await self.complete_execution(
                execution_id, ExecutionStatus.CANCELLED, error_message=reason
            )

            logger.info(
                "Cancelled job execution", execution_id=execution_id, reason=reason
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to cancel execution",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of execution manager"""
        now = datetime.now(timezone.utc)

        # Check for stuck executions
        stuck_executions = []
        for execution_id, context in self._active_executions.items():
            runtime = (now - context.started_at).total_seconds()
            if runtime > 3600:  # 1 hour
                stuck_executions.append(
                    {
                        "execution_id": execution_id,
                        "job_id": context.job_id,
                        "runtime_seconds": runtime,
                    }
                )

        return {
            "active_executions": len(self._active_executions),
            "stuck_executions": len(stuck_executions),
            "stuck_execution_details": stuck_executions,
            "is_healthy": len(stuck_executions) == 0,
            "checked_at": now.isoformat(),
        }
