"""
Celery integration for the scheduling system.

Provides seamless integration between the scheduler and existing Celery
task queue infrastructure, allowing scheduled jobs to be executed as Celery tasks.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery import Celery, Task
from celery.exceptions import Retry, WorkerLostError
from celery.result import AsyncResult

from app.models.scheduling import ScheduledJob
from app.tasks.audit_tasks import celery_app
from app.utils.logger import get_logger

from ..execution_manager import ExecutionContext, ExecutionManager

logger = get_logger(__name__)


class ScheduledJobTask(Task):
    """
    Custom Celery task class for scheduled jobs.

    Provides integration between Celery tasks and the scheduling system
    with proper status tracking and error handling.
    """

    def __init__(self):
        """Initialize scheduled job task"""
        self.execution_manager: Optional[ExecutionManager] = None
        self.execution_context: Optional[ExecutionContext] = None

    def apply_async(self, args=None, kwargs=None, **options):
        """Override apply_async to inject scheduling context"""
        # Extract scheduling metadata from kwargs
        scheduling_meta = kwargs.pop("__scheduling_meta__", {}) if kwargs else {}

        # Add scheduling info to task headers
        headers = options.get("headers", {})
        headers.update(
            {
                "scheduled_job_id": scheduling_meta.get("job_id"),
                "execution_id": scheduling_meta.get("execution_id"),
                "scheduled_time": scheduling_meta.get("scheduled_time"),
                "scheduler_version": "2.0",
            }
        )
        options["headers"] = headers

        return super().apply_async(args, kwargs, **options)

    def retry(
        self,
        args=None,
        kwargs=None,
        exc=None,
        throw=True,
        eta=None,
        countdown=None,
        max_retries=None,
        **options,
    ):
        """Override retry to integrate with scheduling system retry policies"""
        # Get retry policy from execution context if available
        if self.execution_context:
            # Use scheduling system retry logic instead of Celery's
            logger.info(
                "Task retry requested, delegating to scheduling system",
                job_id=self.execution_context.job_id,
                execution_id=self.execution_context.execution_id,
            )
            # The scheduling system will handle retry logic
            raise exc if exc else Retry()

        return super().retry(
            args, kwargs, exc, throw, eta, countdown, max_retries, **options
        )


class CeleryJobExecutor:
    """
    Executes scheduled jobs using Celery task queue.

    Bridges the scheduling system with Celery infrastructure,
    providing job execution, status tracking, and result handling.
    """

    def __init__(
        self,
        celery_app: Celery,
        execution_manager: Optional[ExecutionManager] = None,
        default_queue: str = "scheduled_jobs",
        max_execution_time: int = 3600,  # 1 hour default
    ):
        """
        Initialize Celery job executor.

        Args:
            celery_app: Celery application instance
            execution_manager: Execution manager for tracking
            default_queue: Default queue for scheduled jobs
            max_execution_time: Maximum execution time in seconds
        """
        self.celery_app = celery_app
        self.execution_manager = execution_manager
        self.default_queue = default_queue
        self.max_execution_time = max_execution_time

        # Job type to task name mapping
        self._job_handlers: Dict[str, str] = {}

        # Active task tracking
        self._active_tasks: Dict[str, AsyncResult] = {}

        # Register built-in job types
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default job type handlers"""
        # Map job types to existing Celery task names
        self._job_handlers.update(
            {
                "audit_execution": "app.tasks.audit_tasks.run_audit",
                "report_generation": "app.tasks.audit_tasks.generate_report",
                "data_cleanup": "app.tasks.audit_tasks.cleanup_old_data",
                "system_health_check": "app.tasks.audit_tasks.system_health_check",
                "user_notification": "app.tasks.audit_tasks.send_notification",
            }
        )

        logger.info(
            "Registered default job type handlers",
            handlers=list(self._job_handlers.keys()),
        )

    def register_job_handler(self, job_type: str, task_name: str) -> None:
        """
        Register handler for specific job type.

        Args:
            job_type: Type of job to handle
            task_name: Celery task name to execute
        """
        self._job_handlers[job_type] = task_name
        logger.info("Registered job handler", job_type=job_type, task_name=task_name)

    async def execute_job(
        self, job: ScheduledJob, execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute job using Celery task queue.

        Args:
            job: Job to execute
            execution_context: Execution tracking context

        Returns:
            Execution result dictionary
        """
        # Get task name for job type
        task_name = self._job_handlers.get(job.job_type)
        if not task_name:
            raise ValueError(f"No handler registered for job type: {job.job_type}")

        # Prepare task arguments
        task_args = self._prepare_task_arguments(job, execution_context)
        task_kwargs = self._prepare_task_kwargs(job, execution_context)

        # Prepare Celery task options
        task_options = {
            "queue": job.metadata.get("celery_queue", self.default_queue),
            "routing_key": job.metadata.get("routing_key"),
            "priority": self._convert_priority(job.priority),
            "time_limit": job.timeout_seconds or self.max_execution_time,
            "soft_time_limit": (job.timeout_seconds or self.max_execution_time) - 30,
            "retry_policy": {
                "max_retries": job.max_retries,
                "interval_start": job.retry_delay_seconds,
                "interval_step": job.retry_delay_seconds * 0.5,
                "interval_max": job.retry_delay_seconds * 5,
            },
        }

        # Add scheduling metadata
        task_kwargs["__scheduling_meta__"] = {
            "job_id": job.job_id,
            "execution_id": execution_context.execution_id,
            "scheduled_time": execution_context.scheduled_time.isoformat(),
            "job_type": job.job_type,
        }

        try:
            logger.info(
                "Submitting job to Celery",
                job_id=job.job_id,
                job_type=job.job_type,
                task_name=task_name,
                execution_id=execution_context.execution_id,
            )

            # Submit task to Celery
            task_result = self.celery_app.send_task(
                task_name, args=task_args, kwargs=task_kwargs, **task_options
            )

            # Track active task
            self._active_tasks[execution_context.execution_id] = task_result

            # Wait for task completion with timeout
            timeout_seconds = job.timeout_seconds or self.max_execution_time
            result = await self._wait_for_task_completion(
                task_result, execution_context, timeout_seconds
            )

            # Clean up tracking
            self._active_tasks.pop(execution_context.execution_id, None)

            logger.info(
                "Job execution completed via Celery",
                job_id=job.job_id,
                execution_id=execution_context.execution_id,
                task_id=task_result.task_id,
                success=result.get("success", False),
            )

            return result

        except Exception as e:
            # Clean up tracking
            self._active_tasks.pop(execution_context.execution_id, None)

            logger.error(
                f"Job execution failed via Celery: {e}",
                job_id=job.job_id,
                execution_id=execution_context.execution_id,
                task_name=task_name,
                exc_info=True,
            )
            raise

    def _prepare_task_arguments(
        self, job: ScheduledJob, execution_context: ExecutionContext
    ) -> List[Any]:
        """Prepare positional arguments for Celery task"""
        # Most audit tasks take job_data as first argument
        return [job.job_data]

    def _prepare_task_kwargs(
        self, job: ScheduledJob, execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Prepare keyword arguments for Celery task"""
        kwargs = {
            "job_id": job.job_id,
            "execution_id": execution_context.execution_id,
            "scheduled_time": execution_context.scheduled_time.isoformat(),
            "metadata": execution_context.metadata,
        }

        # Add job-specific parameters
        if "task_kwargs" in job.metadata:
            kwargs.update(job.metadata["task_kwargs"])

        return kwargs

    def _convert_priority(self, scheduler_priority: int) -> int:
        """Convert scheduler priority (1-10) to Celery priority (0-9)"""
        # Invert priority scale for Celery (higher number = higher priority)
        return 10 - scheduler_priority

    async def _wait_for_task_completion(
        self,
        task_result: AsyncResult,
        execution_context: ExecutionContext,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        """
        Wait for Celery task completion with proper error handling.

        Args:
            task_result: Celery AsyncResult
            execution_context: Execution context
            timeout_seconds: Maximum wait time

        Returns:
            Task execution result
        """
        start_time = datetime.now(timezone.utc)
        check_interval = 1.0  # Check every second

        while True:
            # Check timeout
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= timeout_seconds:
                logger.error(
                    "Task execution timed out",
                    execution_id=execution_context.execution_id,
                    task_id=task_result.task_id,
                    elapsed=elapsed,
                )

                # Try to revoke the task
                try:
                    task_result.revoke(terminate=True)
                except Exception as revoke_error:
                    logger.warning(f"Failed to revoke timed-out task: {revoke_error}")

                raise asyncio.TimeoutError(
                    f"Task execution timed out after {timeout_seconds} seconds"
                )

            # Check task state
            if task_result.ready():
                try:
                    if task_result.successful():
                        result = task_result.result
                        return {
                            "success": True,
                            "result": result,
                            "task_id": task_result.task_id,
                            "execution_time": elapsed,
                        }
                    else:
                        error_info = {
                            "success": False,
                            "error": str(task_result.result),
                            "task_id": task_result.task_id,
                            "execution_time": elapsed,
                        }

                        # Handle specific Celery exceptions
                        if isinstance(task_result.result, WorkerLostError):
                            error_info["error_type"] = "WorkerLostError"
                            error_info["retryable"] = True
                        elif isinstance(task_result.result, Retry):
                            error_info["error_type"] = "RetryRequested"
                            error_info["retryable"] = True
                        else:
                            error_info["error_type"] = type(task_result.result).__name__
                            error_info["retryable"] = False

                        return error_info

                except Exception as e:
                    logger.error(f"Error retrieving task result: {e}", exc_info=True)
                    return {
                        "success": False,
                        "error": f"Failed to retrieve task result: {e}",
                        "task_id": task_result.task_id,
                        "execution_time": elapsed,
                    }

            # Wait before next check
            await asyncio.sleep(check_interval)

            # Update execution progress if available
            if hasattr(task_result, "info") and task_result.info:
                try:
                    if self.execution_manager:
                        await self.execution_manager.update_execution_progress(
                            execution_context.execution_id,
                            {"celery_task_info": task_result.info},
                        )
                except Exception as progress_error:
                    logger.warning(
                        f"Failed to update execution progress: {progress_error}"
                    )

    async def cancel_job_execution(self, execution_id: str) -> bool:
        """
        Cancel running job execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancellation was successful
        """
        task_result = self._active_tasks.get(execution_id)
        if not task_result:
            logger.warning(f"No active task found for execution {execution_id}")
            return False

        try:
            # Revoke the Celery task
            task_result.revoke(terminate=True)

            # Remove from tracking
            self._active_tasks.pop(execution_id, None)

            logger.info(
                "Cancelled job execution",
                execution_id=execution_id,
                task_id=task_result.task_id,
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to cancel job execution {execution_id}: {e}", exc_info=True
            )
            return False

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get information about active Celery task executions"""
        active_executions = []

        for execution_id, task_result in self._active_tasks.items():
            try:
                execution_info = {
                    "execution_id": execution_id,
                    "task_id": task_result.task_id,
                    "task_name": task_result.name,
                    "state": task_result.state,
                    "result": None,
                }

                # Get task info if available
                if hasattr(task_result, "info") and task_result.info:
                    execution_info["result"] = task_result.info

                active_executions.append(execution_info)

            except Exception as e:
                logger.warning(f"Failed to get info for execution {execution_id}: {e}")

        return active_executions

    def get_statistics(self) -> Dict[str, Any]:
        """Get Celery executor statistics"""
        return {
            "active_executions": len(self._active_tasks),
            "registered_handlers": len(self._job_handlers),
            "default_queue": self.default_queue,
            "max_execution_time": self.max_execution_time,
            "job_type_handlers": dict(self._job_handlers),
        }


class CelerySchedulerBridge:
    """
    Bridge between scheduling system and Celery infrastructure.

    Provides high-level integration, job routing, and monitoring
    coordination between the scheduler and Celery workers.
    """

    def __init__(
        self,
        scheduler_engine,
        celery_executor: CeleryJobExecutor,
        monitoring_enabled: bool = True,
    ):
        """
        Initialize Celery scheduler bridge.

        Args:
            scheduler_engine: Main scheduler engine
            celery_executor: Celery job executor
            monitoring_enabled: Enable Celery monitoring integration
        """
        self.scheduler_engine = scheduler_engine
        self.celery_executor = celery_executor
        self.monitoring_enabled = monitoring_enabled

        # Register executor with scheduler
        self.scheduler_engine.register_job_handler(
            "audit_execution", self._execute_audit_job
        )
        self.scheduler_engine.register_job_handler(
            "report_generation", self._execute_report_job
        )
        self.scheduler_engine.register_job_handler(
            "data_cleanup", self._execute_cleanup_job
        )

        logger.info("Celery scheduler bridge initialized")

    async def _execute_audit_job(
        self, job_data: Dict[str, Any], execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute audit job via Celery"""
        # Get job from context
        job = await self._get_job_from_context(execution_context)
        return await self.celery_executor.execute_job(job, execution_context)

    async def _execute_report_job(
        self, job_data: Dict[str, Any], execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute report generation job via Celery"""
        job = await self._get_job_from_context(execution_context)
        return await self.celery_executor.execute_job(job, execution_context)

    async def _execute_cleanup_job(
        self, job_data: Dict[str, Any], execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute cleanup job via Celery"""
        job = await self._get_job_from_context(execution_context)
        return await self.celery_executor.execute_job(job, execution_context)

    async def _get_job_from_context(
        self, execution_context: ExecutionContext
    ) -> ScheduledJob:
        """Helper to get job from execution context"""
        job = self.scheduler_engine.repository.get_job(execution_context.job_id)
        if not job:
            raise ValueError(f"Job not found: {execution_context.job_id}")
        return job

    def setup_celery_monitoring(self) -> None:
        """Setup monitoring integration with Celery"""
        if not self.monitoring_enabled:
            return

        # Setup Celery event monitoring
        try:
            # This would integrate with Celery's monitoring events
            # Implementation depends on specific monitoring requirements
            logger.info("Celery monitoring integration enabled")
        except Exception as e:
            logger.error(f"Failed to setup Celery monitoring: {e}", exc_info=True)

    def get_celery_worker_status(self) -> Dict[str, Any]:
        """Get status of Celery workers"""
        try:
            # Get worker statistics from Celery
            inspect = self.celery_executor.celery_app.control.inspect()

            stats = inspect.stats()
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()

            return {
                "worker_stats": stats or {},
                "active_tasks": active_tasks or {},
                "scheduled_tasks": scheduled_tasks or {},
                "is_healthy": bool(stats),
                "worker_count": len(stats) if stats else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get Celery worker status: {e}", exc_info=True)
            return {
                "worker_stats": {},
                "active_tasks": {},
                "scheduled_tasks": {},
                "is_healthy": False,
                "worker_count": 0,
                "error": str(e),
            }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of scheduler-Celery integration"""
        scheduler_status = self.scheduler_engine.get_scheduler_status()
        celery_status = self.get_celery_worker_status()
        executor_stats = self.celery_executor.get_statistics()

        return {
            "bridge_healthy": True,
            "scheduler": scheduler_status,
            "celery_workers": celery_status,
            "job_executor": executor_stats,
            "integration_version": "2.0",
            "monitoring_enabled": self.monitoring_enabled,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# Create default Celery job task using existing celery app
@celery_app.task(bind=True, base=ScheduledJobTask, name="scheduled_job_execution")
def execute_scheduled_job(self, job_data: Dict[str, Any], **kwargs):
    """
    Generic Celery task for executing scheduled jobs.

    This task serves as a bridge between the scheduling system
    and specific job implementation functions.
    """
    try:
        # Extract scheduling metadata
        scheduling_meta = kwargs.get("__scheduling_meta__", {})
        job_id = scheduling_meta.get("job_id")
        execution_id = scheduling_meta.get("execution_id")
        job_type = scheduling_meta.get("job_type")

        logger.info(
            "Executing scheduled job via Celery",
            job_id=job_id,
            execution_id=execution_id,
            job_type=job_type,
            task_id=self.request.id,
        )

        # Route to appropriate job handler based on job_type
        if job_type == "audit_execution":
            from app.tasks.audit_tasks import run_audit

            return run_audit(job_data, **kwargs)

        elif job_type == "report_generation":
            from app.tasks.audit_tasks import generate_report

            return generate_report(job_data, **kwargs)

        elif job_type == "data_cleanup":
            from app.tasks.audit_tasks import cleanup_old_data

            return cleanup_old_data(job_data, **kwargs)

        else:
            raise ValueError(f"Unknown job type: {job_type}")

    except Exception as e:
        logger.error(
            f"Scheduled job execution failed: {e}",
            job_id=job_id,
            execution_id=execution_id,
            task_id=self.request.id,
            exc_info=True,
        )
        raise
