"""
Core scheduler engine with integrated execution management.

Provides the main orchestration layer for job scheduling, execution tracking,
and coordination between all scheduling system components.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.models.scheduling import (
    JobExecutionStatus,
    ScheduledJob,
    ScheduledJobStatus,
    TriggerType,
)
from app.utils.logger import get_logger

from .execution_manager import ExecutionContext, ExecutionManager
from .repository import SchedulingRepository
from .triggers.factory import TriggerFactory, get_trigger_factory

logger = get_logger(__name__)


class SchedulerStatus(str, Enum):
    """Scheduler operational status"""

    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class JobDefinition:
    """Definition for creating a new scheduled job"""

    name: str
    job_type: str
    trigger_config: Dict[str, Any]
    job_data: Dict[str, Any]
    description: Optional[str] = None
    priority: int = 5  # 1-10 scale
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: int = 60
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Result of job execution"""

    execution_id: str
    job_id: str
    status: JobExecutionStatus
    started_at: datetime
    finished_at: Optional[datetime]
    runtime_seconds: Optional[float]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]


class SchedulerEngine:
    """
    Core scheduler engine.

    Orchestrates job scheduling, execution, and monitoring with proper
    integration between all system components.
    """

    def __init__(
        self,
        repository: Optional[SchedulingRepository] = None,
        execution_manager: Optional[ExecutionManager] = None,
        trigger_factory: Optional[TriggerFactory] = None,
        poll_interval: int = 30,
    ):
        """Initialize scheduler engine"""
        self.repository = repository or SchedulingRepository()
        self.execution_manager = execution_manager or ExecutionManager(self.repository)
        self.trigger_factory = trigger_factory or get_trigger_factory()
        self.poll_interval = poll_interval

        # Engine state
        self.status = SchedulerStatus.STOPPED
        self.scheduler_id = str(uuid.uuid4())
        self.started_at: Optional[datetime] = None

        # Runtime tracking
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._job_handlers: Dict[str, Callable] = {}

        # Statistics
        self._stats = {
            "jobs_scheduled": 0,
            "jobs_executed": 0,
            "jobs_failed": 0,
            "last_poll": None,
            "uptime_seconds": 0,
        }

        # Setup execution callbacks
        self.execution_manager.add_execution_callback(
            "on_success", self._on_execution_success
        )
        self.execution_manager.add_execution_callback(
            "on_failure", self._on_execution_failure
        )
        self.execution_manager.add_execution_callback(
            "on_complete", self._on_execution_complete
        )

    async def start(self) -> None:
        """Start the scheduler engine"""
        if self.status != SchedulerStatus.STOPPED:
            raise RuntimeError(f"Scheduler already running (status: {self.status})")

        try:
            self.status = SchedulerStatus.STARTING
            self.started_at = datetime.now(timezone.utc)

            logger.info(
                "Starting scheduler engine",
                scheduler_id=self.scheduler_id,
                poll_interval=self.poll_interval,
            )

            # Acquire scheduler lock to prevent multiple instances
            lock_id = self.repository.acquire_scheduler_lock(
                "main_scheduler", timeout_seconds=600
            )
            if not lock_id:
                raise RuntimeError(
                    "Could not acquire scheduler lock - another instance may be running"
                )

            # Start main scheduler loop
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.status = SchedulerStatus.RUNNING

            logger.info(
                "Scheduler engine started successfully", scheduler_id=self.scheduler_id
            )

        except Exception as e:
            self.status = SchedulerStatus.ERROR
            logger.error(f"Failed to start scheduler engine: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop the scheduler engine gracefully"""
        if self.status == SchedulerStatus.STOPPED:
            return

        logger.info("Stopping scheduler engine", scheduler_id=self.scheduler_id)
        self.status = SchedulerStatus.STOPPING

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for scheduler loop to finish
        if self._scheduler_task:
            task_ref = self._scheduler_task
            try:
                if asyncio.isfuture(task_ref) or asyncio.iscoroutine(task_ref):
                    await asyncio.wait_for(task_ref, timeout=30)
                elif callable(task_ref):
                    maybe_coro = task_ref()
                    if asyncio.isfuture(maybe_coro) or asyncio.iscoroutine(maybe_coro):
                        await asyncio.wait_for(maybe_coro, timeout=30)
                # If the reference is a synchronous sentinel (already completed),
                # there's nothing to wait on.
            except asyncio.TimeoutError:
                logger.warning("Scheduler loop didn't stop gracefully, cancelling")
                if hasattr(self._scheduler_task, "cancel"):
                    self._scheduler_task.cancel()
                    try:
                        await self._scheduler_task
                    except asyncio.CancelledError:
                        pass

        # Release scheduler lock
        try:
            self.repository.release_scheduler_lock("main_scheduler", self.scheduler_id)
        except Exception as e:
            logger.warning(f"Failed to release scheduler lock: {e}")

        self.status = SchedulerStatus.STOPPED
        logger.info("Scheduler engine stopped")

    async def pause(self) -> None:
        """Pause job scheduling (but allow running jobs to complete)"""
        if self.status == SchedulerStatus.RUNNING:
            self.status = SchedulerStatus.PAUSED
            logger.info("Scheduler engine paused")

    async def resume(self) -> None:
        """Resume job scheduling"""
        if self.status == SchedulerStatus.PAUSED:
            self.status = SchedulerStatus.RUNNING
            logger.info("Scheduler engine resumed")

    async def schedule_job(self, job_def: JobDefinition) -> str:
        """
        Schedule a new job.

        Args:
            job_def: Job definition containing all necessary configuration

        Returns:
            Job ID of the scheduled job
        """
        try:
            # Validate trigger configuration
            trigger = self.trigger_factory.create_trigger(job_def.trigger_config)

            # Create job record
            job_data = {
                "job_id": str(uuid.uuid4()),
                "name": job_def.name,
                "description": job_def.description,
                "job_type": job_def.job_type,
                "trigger_type": TriggerType(job_def.trigger_config["trigger_type"]),
                "trigger_config": job_def.trigger_config,
                "job_data": job_def.job_data,
                "status": ScheduledJobStatus.ACTIVE,
                "priority": job_def.priority,
                "timeout_seconds": job_def.timeout_seconds,
                "max_retries": job_def.max_retries,
                "retry_delay_seconds": job_def.retry_delay_seconds,
                "tags": job_def.tags or [],
                "metadata": job_def.metadata or {},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            # Calculate initial next run time
            next_run_time = await trigger.get_next_run_time(None)
            job_data["next_run_time"] = next_run_time

            # Create job in database
            job = self.repository.create_job(job_data)

            self._stats["jobs_scheduled"] += 1

            logger.info(
                "Job scheduled successfully",
                job_id=job.job_id,
                job_name=job.name,
                trigger_type=job.trigger_type.value,
                next_run_time=next_run_time.isoformat() if next_run_time else None,
            )

            return job.job_id

        except Exception as e:
            logger.error(f"Failed to schedule job: {e}", exc_info=True)
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        try:
            job = self.repository.get_job(job_id)
            if not job:
                return False

            # Update job status
            self.repository.update_job(
                job_id,
                {
                    "status": ScheduledJobStatus.CANCELLED,
                    "next_run_time": None,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            logger.info("Job cancelled", job_id=job_id, job_name=job.name)
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
            raise

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job status information"""
        try:
            job = self.repository.get_job_with_executions(job_id)
            if not job:
                return None

            # Get recent executions
            recent_executions = self.repository.get_job_executions(job_id, limit=5)

            # Get active execution if any
            active_execution = None
            for context in self.execution_manager.get_active_executions():
                if context.job_id == job_id:
                    active_execution = {
                        "execution_id": context.execution_id,
                        "started_at": context.started_at.isoformat(),
                        "runtime_seconds": (
                            datetime.now(timezone.utc) - context.started_at
                        ).total_seconds(),
                    }
                    break

            # Calculate trigger info
            trigger_info = None
            try:
                trigger = self.trigger_factory.create_trigger(job.trigger_config)
                trigger_info = trigger.get_trigger_info()
            except Exception as e:
                logger.warning(f"Failed to get trigger info for job {job_id}: {e}")

            return {
                "job_id": job.job_id,
                "name": job.name,
                "description": job.description,
                "status": job.status.value,
                "trigger_type": job.trigger_type.value,
                "next_run_time": job.next_run_time.isoformat()
                if job.next_run_time
                else None,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "priority": job.priority,
                "timeout_seconds": job.timeout_seconds,
                "max_retries": job.max_retries,
                "tags": job.tags,
                "trigger_info": trigger_info,
                "active_execution": active_execution,
                "recent_executions": [
                    {
                        "execution_id": exec.execution_id,
                        "status": exec.status.value,
                        "started_at": exec.started_at.isoformat(),
                        "finished_at": exec.finished_at.isoformat()
                        if exec.finished_at
                        else None,
                        "runtime_seconds": exec.runtime_seconds,
                    }
                    for exec in recent_executions[:3]
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}", exc_info=True)
            raise

    async def list_jobs(
        self,
        status: Optional[ScheduledJobStatus] = None,
        trigger_type: Optional[TriggerType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering"""
        try:
            # Get jobs based on filters
            if status:
                jobs = self.repository.get_jobs_by_status(status)
            elif trigger_type:
                jobs = self.repository.get_jobs_by_trigger_type(trigger_type)
            else:
                # Would need a general list method in repository
                jobs = []

            # Apply additional filtering
            if tags:
                jobs = [
                    job for job in jobs if any(tag in (job.tags or []) for tag in tags)
                ]

            # Convert to response format
            job_list = []
            for job in jobs[:limit]:
                # Get latest execution
                recent_executions = self.repository.get_job_executions(
                    job.job_id, limit=1
                )
                latest_execution = recent_executions[0] if recent_executions else None

                job_info = {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status.value,
                    "trigger_type": job.trigger_type.value,
                    "next_run_time": job.next_run_time.isoformat()
                    if job.next_run_time
                    else None,
                    "priority": job.priority,
                    "tags": job.tags,
                    "latest_execution": {
                        "status": latest_execution.status.value,
                        "started_at": latest_execution.started_at.isoformat(),
                        "runtime_seconds": latest_execution.runtime_seconds,
                    }
                    if latest_execution
                    else None,
                }
                job_list.append(job_info)

            return job_list

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}", exc_info=True)
            raise

    def register_job_handler(self, job_type: str, handler: Callable) -> None:
        """Register a handler function for a specific job type"""
        self._job_handlers[job_type] = handler
        logger.info(f"Registered job handler for type: {job_type}")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that polls for due jobs and executes them"""
        logger.info("Starting scheduler polling loop")

        while not self._shutdown_event.is_set():
            try:
                if self.status == SchedulerStatus.RUNNING:
                    await self._poll_and_execute_jobs()
                    self._stats["last_poll"] = datetime.now(timezone.utc)

                # Update uptime
                if self.started_at:
                    self._stats["uptime_seconds"] = (
                        datetime.now(timezone.utc) - self.started_at
                    ).total_seconds()

                # Wait for next poll or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.poll_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue polling

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        logger.info("Scheduler polling loop stopped")

    async def _poll_and_execute_jobs(self) -> None:
        """Poll for due jobs and execute them"""
        try:
            # Get jobs due for execution
            due_jobs = self.repository.get_jobs_due_for_execution()

            if not due_jobs:
                logger.debug("No jobs due for execution")
                return

            logger.info(f"Found {len(due_jobs)} jobs due for execution")

            # Execute each due job
            for job in due_jobs:
                try:
                    await self._execute_job(job)
                except Exception as e:
                    logger.error(
                        f"Failed to execute job {job.job_id}",
                        job_id=job.job_id,
                        job_name=job.name,
                        error=str(e),
                        exc_info=True,
                    )

        except Exception as e:
            logger.error(f"Failed to poll for due jobs: {e}", exc_info=True)

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a single job"""
        logger.info(
            "Executing job", job_id=job.job_id, job_name=job.name, job_type=job.job_type
        )

        # Check if we have a handler for this job type
        handler = self._job_handlers.get(job.job_type)
        if not handler:
            logger.error(
                f"No handler registered for job type: {job.job_type}",
                job_id=job.job_id,
                job_name=job.name,
            )
            self.repository.update_job(
                job.job_id,
                {
                    "status": ScheduledJobStatus.DISABLED,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            return

        # Calculate next run time before execution
        next_run_time = await self._calculate_next_run_time(job)

        # Update job's next run time
        self.repository.update_job(
            job.job_id,
            {
                "next_run_time": next_run_time,
                "last_run_time": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        )

        # Execute job with tracking
        try:
            async with self.execution_manager.track_execution(job) as context:
                result = await handler(job.job_data, context)
                logger.debug(
                    "Job execution completed",
                    job_id=job.job_id,
                    result_keys=list(result.keys())
                    if isinstance(result, dict)
                    else None,
                )

        except Exception as e:
            logger.error(
                "Job execution failed",
                job_id=job.job_id,
                job_name=job.name,
                error=str(e),
                exc_info=True,
            )
            # Exception is handled by execution manager

    async def _calculate_next_run_time(self, job: ScheduledJob) -> Optional[datetime]:
        """Calculate when job should run next"""
        try:
            trigger = self.trigger_factory.create_trigger(job.trigger_config)
            return await trigger.get_next_run_time(job.last_run_time)
        except Exception as e:
            logger.error(
                f"Failed to calculate next run time for job {job.job_id}: {e}",
                exc_info=True,
            )
            return None

    async def _on_execution_success(
        self, context: ExecutionContext, result: Any
    ) -> None:
        """Handle successful job execution"""
        self._stats["jobs_executed"] += 1
        logger.debug(
            "Job execution succeeded",
            job_id=context.job_id,
            execution_id=context.execution_id,
        )

    async def _on_execution_failure(
        self,
        context: ExecutionContext,
        error_message: str,
        error_details: Dict[str, Any],
    ) -> None:
        """Handle failed job execution"""
        self._stats["jobs_failed"] += 1
        logger.warning(
            "Job execution failed",
            job_id=context.job_id,
            execution_id=context.execution_id,
            error_message=error_message,
        )

    async def _on_execution_complete(
        self,
        context: ExecutionContext,
        status: JobExecutionStatus,
        runtime_seconds: float,
    ) -> None:
        """Handle job execution completion"""
        logger.debug(
            "Job execution completed",
            job_id=context.job_id,
            execution_id=context.execution_id,
            status=status.value,
            runtime_seconds=runtime_seconds,
        )

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        return {
            "scheduler_id": self.scheduler_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": self._stats["uptime_seconds"],
            "poll_interval": self.poll_interval,
            "statistics": dict(self._stats),
            "active_executions": len(self.execution_manager.get_active_executions()),
            "registered_job_types": list(self._job_handlers.keys()),
            "last_poll": self._stats["last_poll"].isoformat()
            if self._stats["last_poll"]
            else None,
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        scheduler_health = {
            "scheduler_status": self.status.value,
            "is_healthy": self.status == SchedulerStatus.RUNNING,
            "uptime_seconds": self._stats["uptime_seconds"],
        }

        # Get execution manager health
        execution_health = self.execution_manager.get_health_status()

        # Get repository health
        repository_health = self.repository.health_check()

        return {
            "overall_healthy": (
                scheduler_health["is_healthy"]
                and execution_health["is_healthy"]
                and repository_health["scheduler_healthy"]
            ),
            "scheduler": scheduler_health,
            "executions": execution_health,
            "repository": repository_health,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
