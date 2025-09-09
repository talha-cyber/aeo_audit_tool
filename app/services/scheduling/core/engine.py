"""
Core Scheduler Engine

The main orchestrator for the scheduling system. Handles job scheduling,
execution coordination, leader election, and system lifecycle management.
Built for production use with comprehensive error handling and monitoring.
"""

import asyncio
import signal
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from app.models.scheduling import (
    JobExecution,
    JobExecutionStatus,
    ScheduledJob,
    ScheduledJobStatus,
)
from app.utils.logger import get_logger

from ..executors.celery_executor import CeleryJobExecutor
from ..monitoring.metrics import SchedulerMetrics
from ..persistence.repository import SchedulingRepository
from ..policies.concurrency import ConcurrencyManager
from ..policies.retry import RetryPolicyManager
from ..triggers.factory import TriggerFactory
from .models import (
    ExecutionContext,
    JobDefinition,
    JobSchedule,
    SchedulerConfig,
    SchedulerEvent,
    SchedulerMode,
    SchedulerStatus,
)

logger = get_logger(__name__)


class SchedulerEngineError(Exception):
    """Base exception for scheduler engine errors"""

    pass


class LeaderElectionError(SchedulerEngineError):
    """Leader election related errors"""

    pass


class JobExecutionError(SchedulerEngineError):
    """Job execution related errors"""

    pass


class SchedulerEngine:
    """
    Core scheduler engine for managing persistent job scheduling.

    Features:
    - Persistent job storage with database backing
    - Distributed operation with leader election
    - Multiple trigger types (cron, interval, one-time, dependency)
    - Comprehensive retry policies and error handling
    - Job dependency management
    - Performance monitoring and health checks
    - Graceful shutdown and recovery
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.config.validate()

        # Core components
        self.repository: Optional[SchedulingRepository] = None
        self.trigger_factory: Optional[TriggerFactory] = None
        self.executor: Optional[CeleryJobExecutor] = None
        self.retry_manager: Optional[RetryPolicyManager] = None
        self.concurrency_manager: Optional[ConcurrencyManager] = None
        self.metrics: Optional[SchedulerMetrics] = None

        # Runtime state
        self.is_running = False
        self.is_leader = False
        self.started_at: Optional[datetime] = None
        self.last_poll_at: Optional[datetime] = None
        self.shutdown_event = asyncio.Event()

        # Job tracking
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_cache: Dict[str, ScheduledJob] = {}
        self.next_jobs_cache: List[JobSchedule] = []
        self.cache_last_refreshed: Optional[datetime] = None

        # Event handlers
        self.event_handlers: Dict[SchedulerEvent, List[Callable]] = defaultdict(list)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

        logger.info(
            "Scheduler engine initialized", instance_id=self.config.instance_id
        )

    async def initialize(self) -> None:
        """Initialize the scheduler engine and all components"""
        try:
            logger.info(
                "Initializing scheduler engine", instance_id=self.config.instance_id
            )

            # Initialize components
            self.repository = SchedulingRepository()
            await self.repository.initialize()

            self.trigger_factory = TriggerFactory()
            self.executor = CeleryJobExecutor()
            await self.executor.initialize()

            self.retry_manager = RetryPolicyManager()
            self.concurrency_manager = ConcurrencyManager(
                self.config.max_concurrent_jobs
            )

            if self.config.enable_metrics:
                self.metrics = SchedulerMetrics(self.config.instance_id)
                await self.metrics.initialize()

            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()

            logger.info("Scheduler engine components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler engine: {e}", exc_info=True)
            raise SchedulerEngineError(f"Initialization failed: {e}")

    async def start(self) -> None:
        """Start the scheduler engine"""
        if self.is_running:
            logger.warning("Scheduler engine is already running")
            return

        try:
            async with self._lock:
                logger.info("Starting scheduler engine", mode=self.config.mode.value)

                self.is_running = True
                self.started_at = datetime.now(timezone.utc)
                self.last_poll_at = None

                # Start leader election if enabled
                if self.config.enable_leader_election:
                    await self._start_leader_election()
                else:
                    self.is_leader = True

                # Start background tasks
                if self.is_leader or self.config.mode == SchedulerMode.STANDALONE:
                    await self._start_background_tasks()

                # Emit started event
                await self._emit_event(
                    SchedulerEvent.STARTED,
                    {
                        "instance_id": self.config.instance_id,
                        "mode": self.config.mode.value,
                        "is_leader": self.is_leader,
                    },
                )

                logger.info(
                    "Scheduler engine started successfully", is_leader=self.is_leader
                )

        except Exception as e:
            self.is_running = False
            logger.error(f"Failed to start scheduler engine: {e}", exc_info=True)
            raise SchedulerEngineError(f"Start failed: {e}")

    async def stop(self) -> None:
        """Stop the scheduler engine gracefully"""
        if not self.is_running:
            return

        try:
            logger.info("Stopping scheduler engine gracefully")

            async with self._lock:
                self.is_running = False
                self.shutdown_event.set()

                # Cancel all background tasks
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for background tasks to complete
                if self._background_tasks:
                    await asyncio.gather(
                        *self._background_tasks, return_exceptions=True
                    )
                    self._background_tasks.clear()

                # Wait for running jobs to complete (with timeout)
                if self.running_jobs:
                    logger.info(
                        f"Waiting for {len(self.running_jobs)} jobs to complete"
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                *self.running_jobs.values(), return_exceptions=True
                            ),
                            timeout=30.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timeout waiting for jobs to complete, cancelling"
                        )
                        for task in self.running_jobs.values():
                            if not task.done():
                                task.cancel()

                # Release leadership
                if self.is_leader and self.config.enable_leader_election:
                    await self._release_leadership()

                # Clean up components
                if self.executor:
                    await self.executor.cleanup()
                if self.repository:
                    await self.repository.cleanup()
                if self.metrics:
                    await self.metrics.cleanup()

                # Emit stopped event
                await self._emit_event(
                    SchedulerEvent.STOPPED, {"instance_id": self.config.instance_id}
                )

                logger.info("Scheduler engine stopped successfully")

        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}", exc_info=True)
            raise SchedulerEngineError(f"Shutdown failed: {e}")

    async def schedule_job(self, job_definition: JobDefinition) -> str:
        """Schedule a new job"""
        try:
            logger.info(f"Scheduling new job: {job_definition.name}")

            # Validate job definition
            await self._validate_job_definition(job_definition)

            # Create scheduled job record
            job_id = str(uuid.uuid4())

            scheduled_job = ScheduledJob(
                id=job_id,
                name=job_definition.name,
                description=job_definition.description,
                job_type=job_definition.job_type,
                job_config=job_definition.job_config,
                trigger_type=job_definition.trigger_config.trigger_type,
                trigger_config=job_definition.trigger_config.dict(),
                priority=job_definition.priority.value,
                max_retries=job_definition.max_retries
                or self.config.default_max_retries,
                retry_delay=job_definition.retry_delay
                or self.config.default_retry_delay,
                execution_timeout=job_definition.execution_timeout
                or self.config.execution_timeout,
                max_concurrent=job_definition.max_concurrent,
                start_date=job_definition.start_date,
                end_date=job_definition.end_date,
                max_executions=job_definition.max_executions,
                client_id=job_definition.client_id,
                created_by=job_definition.created_by,
            )

            # Calculate next run time
            trigger = self.trigger_factory.create_trigger(job_definition.trigger_config)
            next_run = await trigger.get_next_run_time(None)
            if next_run:
                scheduled_job.next_run_at = next_run

            # Save to database
            await self.repository.create_job(scheduled_job)

            # Invalidate cache
            await self._invalidate_cache()

            # Emit event
            await self._emit_event(
                SchedulerEvent.JOB_SCHEDULED,
                {
                    "job_id": job_id,
                    "job_name": job_definition.name,
                    "job_type": job_definition.job_type.value,
                    "next_run_at": next_run.isoformat() if next_run else None,
                },
            )

            logger.info("Job scheduled successfully", job_id=job_id, next_run=next_run)
            return job_id

        except Exception as e:
            logger.error(f"Failed to schedule job: {e}", exc_info=True)
            raise SchedulerEngineError(f"Job scheduling failed: {e}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        try:
            logger.info(f"Cancelling job {job_id}")

            # Update job status
            success = await self.repository.update_job_status(
                job_id, ScheduledJobStatus.DISABLED
            )

            if success:
                # Cancel running execution if any
                if job_id in self.running_jobs:
                    task = self.running_jobs[job_id]
                    if not task.done():
                        task.cancel()

                # Invalidate cache
                await self._invalidate_cache()

                # Emit event
                await self._emit_event(SchedulerEvent.JOB_CANCELLED, {"job_id": job_id})

                logger.info("Job cancelled successfully", job_id=job_id)
            else:
                logger.warning("Job not found or already cancelled", job_id=job_id)

            return success

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
            raise SchedulerEngineError(f"Job cancellation failed: {e}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a job"""
        try:
            job = await self.repository.get_job(job_id)
            if not job:
                return None

            # Get recent executions
            recent_executions = await self.repository.get_job_executions(
                job_id, limit=5
            )

            return {
                "job": job.to_dict(),
                "recent_executions": [exec.to_dict() for exec in recent_executions],
                "is_running": job_id in self.running_jobs,
            }

        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}", exc_info=True)
            raise SchedulerEngineError(f"Get job status failed: {e}")

    async def get_scheduler_status(self) -> SchedulerStatus:
        """Get current scheduler status"""
        try:
            # Get basic stats
            stats = await self.repository.get_job_statistics()

            # Calculate uptime
            uptime_seconds = None
            if self.started_at:
                uptime_seconds = int(
                    (datetime.now(timezone.utc) - self.started_at).total_seconds()
                )

            return SchedulerStatus(
                instance_id=self.config.instance_id,
                mode=SchedulerMode.LEADER if self.is_leader else SchedulerMode.FOLLOWER,
                is_running=self.is_running,
                total_jobs=stats.get("total_jobs", 0),
                active_jobs=stats.get("active_jobs", 0),
                running_executions=len(self.running_jobs),
                last_poll_at=self.last_poll_at,
                uptime_seconds=uptime_seconds,
                is_leader=self.is_leader,
            )

        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}", exc_info=True)
            raise SchedulerEngineError(f"Get status failed: {e}")

    # Background Tasks

    async def _start_background_tasks(self) -> None:
        """Start background tasks for job processing"""
        logger.info("Starting background tasks")

        # Main job processing loop
        task = asyncio.create_task(self._job_processing_loop())
        task.set_name("job_processing_loop")
        self._background_tasks.add(task)

        # Heartbeat task (for leader election)
        if self.config.enable_leader_election:
            task = asyncio.create_task(self._heartbeat_loop())
            task.set_name("heartbeat_loop")
            self._background_tasks.add(task)

        # Cache refresh task
        if self.config.enable_job_caching:
            task = asyncio.create_task(self._cache_refresh_loop())
            task.set_name("cache_refresh_loop")
            self._background_tasks.add(task)

        # Cleanup task
        task = asyncio.create_task(self._cleanup_loop())
        task.set_name("cleanup_loop")
        self._background_tasks.add(task)

        # Metrics collection task
        if self.metrics:
            task = asyncio.create_task(self._metrics_collection_loop())
            task.set_name("metrics_collection_loop")
            self._background_tasks.add(task)

        logger.info(f"Started {len(self._background_tasks)} background tasks")

    async def _job_processing_loop(self) -> None:
        """Main job processing loop"""
        logger.info("Starting job processing loop")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                if self.is_leader or self.config.mode == SchedulerMode.STANDALONE:
                    await self._process_scheduled_jobs()

                self.last_poll_at = datetime.now(timezone.utc)

                # Wait for next poll interval or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), timeout=self.config.poll_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal poll interval

            except Exception as e:
                logger.error(f"Error in job processing loop: {e}", exc_info=True)
                await self._emit_event(
                    SchedulerEvent.ERROR_OCCURRED,
                    {"error": str(e), "component": "job_processing_loop"},
                )

                # Wait before retrying
                await asyncio.sleep(min(self.config.poll_interval, 60))

        logger.info("Job processing loop stopped")

    async def _process_scheduled_jobs(self) -> None:
        """Process jobs that are due to run"""
        try:
            # Refresh job cache if needed
            await self._refresh_job_cache_if_needed()

            now = datetime.now(timezone.utc)
            jobs_to_run = []

            # Find jobs that are due
            for job_schedule in self.next_jobs_cache:
                if job_schedule.next_run_time <= now:
                    jobs_to_run.append(job_schedule)
                else:
                    break  # Jobs are sorted by next_run_time

            if not jobs_to_run:
                return

            logger.info(f"Found {len(jobs_to_run)} jobs to execute")

            # Process each job
            for job_schedule in jobs_to_run:
                try:
                    await self._execute_job(job_schedule)
                except Exception as e:
                    logger.error(
                        f"Failed to execute job {job_schedule.job_id}: {e}",
                        exc_info=True,
                    )

            # Remove processed jobs from cache and refresh
            if jobs_to_run:
                await self._invalidate_cache()

        except Exception as e:
            logger.error(f"Error processing scheduled jobs: {e}", exc_info=True)

    async def _execute_job(self, job_schedule: JobSchedule) -> None:
        """Execute a single job"""
        job_id = job_schedule.job_id

        try:
            # Check concurrency limits
            if not await self.concurrency_manager.can_execute(job_id):
                logger.info(f"Job {job_id} skipped due to concurrency limits")
                return

            # Get job details
            job = await self.repository.get_job(job_id)
            if not job or not job.is_active:
                logger.warning(f"Job {job_id} is no longer active, skipping")
                return

            # Create execution record
            execution_id = str(uuid.uuid4())
            execution = JobExecution(
                id=str(uuid.uuid4()),
                job_id=job_id,
                execution_id=execution_id,
                scheduled_time=job_schedule.next_run_time,
                status=JobExecutionStatus.PENDING,
                triggered_by=f"scheduler_{self.config.instance_id}",
                execution_context=job_schedule.execution_context.to_dict(),
            )

            await self.repository.create_execution(execution)

            # Start execution task
            task = asyncio.create_task(self._run_job_execution(job, execution))
            task.set_name(f"job_execution_{job_id}")
            self.running_jobs[job_id] = task

            # Emit event
            await self._emit_event(
                SchedulerEvent.JOB_STARTED,
                {"job_id": job_id, "execution_id": execution_id, "job_name": job.name},
            )

            logger.info(
                "Job execution started", job_id=job_id, execution_id=execution_id
            )

        except Exception as e:
            logger.error(f"Failed to start job execution {job_id}: {e}", exc_info=True)

    async def _run_job_execution(
        self, job: ScheduledJob, execution: JobExecution
    ) -> None:
        """Run a job execution"""
        job_id = job.id
        execution_id = execution.execution_id

        try:
            # Update execution status to running
            execution.status = JobExecutionStatus.RUNNING
            execution.started_at = datetime.now(timezone.utc)
            await self.repository.update_execution(execution)

            # Create execution context
            context = ExecutionContext(
                job_id=job_id,
                execution_id=execution_id,
                scheduled_time=execution.scheduled_time,
                instance_id=self.config.instance_id,
            )

            # Execute the job
            result = await self.executor.execute_job(job, context)

            # Update execution with results
            execution.status = result.status
            execution.completed_at = datetime.now(timezone.utc)
            execution.exit_code = result.exit_code
            execution.output_log = result.output
            execution.error_message = result.error_message
            execution.error_details = result.error_details
            execution.duration_seconds = result.calculate_duration()
            execution.resource_usage = result.resource_usage
            execution.result_data = result.artifacts
            execution.audit_run_id = result.audit_run_id

            await self.repository.update_execution(execution)

            # Update job statistics
            if result.was_successful:
                await self.repository.increment_job_success(job_id)
                await self._emit_event(
                    SchedulerEvent.JOB_COMPLETED,
                    {
                        "job_id": job_id,
                        "execution_id": execution_id,
                        "duration_seconds": execution.duration_seconds,
                    },
                )
            else:
                await self.repository.increment_job_failure(job_id)
                await self._emit_event(
                    SchedulerEvent.JOB_FAILED,
                    {
                        "job_id": job_id,
                        "execution_id": execution_id,
                        "error": result.error_message,
                    },
                )

                # Check if we should retry
                if result.should_retry and execution.retry_count < job.max_retries:
                    await self._schedule_retry(job, execution)

            # Calculate next run time
            await self._update_next_run_time(job)

        except Exception as e:
            logger.error(f"Job execution failed {job_id}: {e}", exc_info=True)

            # Update execution with error
            execution.status = JobExecutionStatus.FAILED
            execution.completed_at = datetime.now(timezone.utc)
            execution.error_message = str(e)
            await self.repository.update_execution(execution)
            await self.repository.increment_job_failure(job_id)

            await self._emit_event(
                SchedulerEvent.JOB_FAILED,
                {"job_id": job_id, "execution_id": execution_id, "error": str(e)},
            )

        finally:
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            # Release concurrency slot
            await self.concurrency_manager.release(job_id)

    # Helper methods (continuing in next part due to length...)

    async def _validate_job_definition(self, job_definition: JobDefinition) -> None:
        """Validate job definition"""
        # Basic validation is handled by Pydantic
        # Add custom business logic validation here
        pass

    async def _invalidate_cache(self) -> None:
        """Invalidate job cache"""
        self.job_cache.clear()
        self.next_jobs_cache.clear()
        self.cache_last_refreshed = None

    async def _refresh_job_cache_if_needed(self) -> None:
        """Refresh job cache if needed"""
        now = datetime.now(timezone.utc)

        if (
            self.cache_last_refreshed is None
            or (now - self.cache_last_refreshed).total_seconds()
            > self.config.cache_refresh_interval
        ):
            await self._refresh_job_cache()

    async def _refresh_job_cache(self) -> None:
        """Refresh job cache with active jobs"""
        try:
            # Get active jobs
            jobs = await self.repository.get_active_jobs()

            # Update cache
            self.job_cache = {job.id: job for job in jobs}

            # Build next jobs schedule
            self.next_jobs_cache = []
            now = datetime.now(timezone.utc)

            for job in jobs:
                if job.next_run_at and job.next_run_at <= now + timedelta(
                    hours=1
                ):  # Next hour
                    context = ExecutionContext(
                        job_id=job.id,
                        execution_id=str(uuid.uuid4()),
                        scheduled_time=job.next_run_at,
                    )

                    schedule = JobSchedule(
                        job_id=job.id,
                        next_run_time=job.next_run_at,
                        priority=job.priority,
                        execution_context=context,
                    )

                    self.next_jobs_cache.append(schedule)

            # Sort by next run time and priority
            self.next_jobs_cache.sort()

            self.cache_last_refreshed = now
            logger.debug(f"Refreshed job cache with {len(jobs)} jobs")

        except Exception as e:
            logger.error(f"Failed to refresh job cache: {e}", exc_info=True)

    async def _schedule_retry(self, job: ScheduledJob, execution: JobExecution) -> None:
        """Schedule a job retry"""
        try:
            retry_time = datetime.now(timezone.utc) + timedelta(seconds=job.retry_delay)

            # Create retry execution
            retry_execution = JobExecution(
                id=str(uuid.uuid4()),
                job_id=job.id,
                execution_id=str(uuid.uuid4()),
                scheduled_time=retry_time,
                status=JobExecutionStatus.PENDING,
                retry_count=execution.retry_count + 1,
                triggered_by=f"retry_{execution.execution_id}",
            )

            await self.repository.create_execution(retry_execution)

            await self._emit_event(
                SchedulerEvent.JOB_RETRYING,
                {
                    "job_id": job.id,
                    "original_execution_id": execution.execution_id,
                    "retry_execution_id": retry_execution.execution_id,
                    "retry_count": retry_execution.retry_count,
                    "retry_time": retry_time.isoformat(),
                },
            )

            logger.info(
                "Scheduled job retry",
                job_id=job.id,
                retry_count=retry_execution.retry_count,
            )

        except Exception as e:
            logger.error(
                f"Failed to schedule retry for job {job.id}: {e}", exc_info=True
            )

    async def _update_next_run_time(self, job: ScheduledJob) -> None:
        """Update next run time for a job"""
        try:
            trigger_config = job.trigger_config
            trigger = self.trigger_factory.create_trigger(trigger_config)

            next_run = await trigger.get_next_run_time(
                job.last_run_at or datetime.now(timezone.utc)
            )

            if next_run:
                await self.repository.update_job_next_run(job.id, next_run)
            else:
                # Job has no more runs, mark as expired
                await self.repository.update_job_status(
                    job.id, ScheduledJobStatus.EXPIRED
                )

        except Exception as e:
            logger.error(
                f"Failed to update next run time for job {job.id}: {e}", exc_info=True
            )

    # Event system

    async def _emit_event(self, event: SchedulerEvent, data: Dict[str, Any]) -> None:
        """Emit scheduler event"""
        try:
            handlers = self.event_handlers.get(event, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event, data)
                    else:
                        handler(event, data)
                except Exception as e:
                    logger.error(
                        f"Event handler failed for {event}: {e}", exc_info=True
                    )

            # Record metrics
            if self.metrics:
                await self.metrics.record_event(event.value, data)

        except Exception as e:
            logger.error(f"Failed to emit event {event}: {e}", exc_info=True)

    def add_event_handler(self, event: SchedulerEvent, handler: Callable) -> None:
        """Add event handler"""
        self.event_handlers[event].append(handler)

    def remove_event_handler(self, event: SchedulerEvent, handler: Callable) -> None:
        """Remove event handler"""
        if handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)

    # Signal handlers and lifecycle

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown"""
        if sys.platform != "win32":
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.stop())

    # Leader election (stub - will be implemented in next part)

    async def _start_leader_election(self) -> None:
        """Start leader election process"""
        # Implementation for distributed leader election
        # For now, assume single instance
        self.is_leader = True

    async def _release_leadership(self) -> None:
        """Release leadership"""
        self.is_leader = False

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for leader election"""
        # Placeholder for heartbeat implementation
        pass

    async def _cache_refresh_loop(self) -> None:
        """Cache refresh loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                await self._refresh_job_cache()
                await asyncio.sleep(self.config.cache_refresh_interval)
            except Exception as e:
                logger.error(f"Error in cache refresh loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for old data"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Clean up old execution records
                cutoff_date = datetime.now(timezone.utc) - timedelta(
                    days=self.config.cleanup_retention_days
                )
                await self.repository.cleanup_old_executions(cutoff_date)

                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if self.metrics:
                    await self.metrics.collect_scheduler_metrics(self)
                await asyncio.sleep(60)  # Collect metrics every minute
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}", exc_info=True)
                await asyncio.sleep(60)
