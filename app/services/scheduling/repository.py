"""
Scheduling system database repository.

Provides comprehensive data access layer for all scheduling operations
with proper transaction handling, concurrency control, and performance optimization.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, or_, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.core.database import get_db
from app.models.scheduling import (
    JobDependency,
    JobExecution,
    JobExecutionStatus,
    ScheduledJob,
    ScheduledJobStatus,
    SchedulerLock,
    SchedulerMetrics,
    TriggerType,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SchedulingRepositoryError(Exception):
    """Base exception for scheduling repository operations"""

    pass


class JobNotFoundError(SchedulingRepositoryError):
    """Raised when a requested job is not found"""

    pass


class ConcurrencyError(SchedulingRepositoryError):
    """Raised when concurrency conflicts occur"""

    pass


class SchedulingRepository:
    """
    Comprehensive repository for scheduling system data operations.

    Handles all database interactions with proper error handling,
    transaction management, and performance optimization.
    """

    def __init__(self, db: Optional[Session] = None):
        """Initialize repository with optional database session"""
        self._db = db
        self._auto_commit = db is None

    @property
    def db(self) -> Session:
        """Get database session"""
        if self._db is None:
            self._db = next(get_db())
        return self._db

    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        db = self.db
        try:
            yield db
            if self._auto_commit:
                db.commit()
        except Exception as e:
            if self._auto_commit:
                db.rollback()
            logger.error(f"Transaction failed: {e}", exc_info=True)
            raise
        finally:
            if self._auto_commit:
                db.close()
                self._db = None

    # Job Management Methods

    def create_job(self, job_data: Dict[str, Any]) -> ScheduledJob:
        """Create a new scheduled job"""
        try:
            with self.transaction() as db:
                job = ScheduledJob(**job_data)
                db.add(job)
                db.flush()  # Get ID without committing

                logger.info(
                    "Created scheduled job",
                    job_id=job.job_id,
                    name=job.name,
                    trigger_type=job.trigger_type.value,
                )

                return job

        except IntegrityError as e:
            logger.error(f"Job creation failed - integrity error: {e}")
            raise SchedulingRepositoryError(f"Job creation failed: {e}")
        except Exception as e:
            logger.error(f"Job creation failed: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Job creation failed: {e}")

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID"""
        try:
            return (
                self.db.query(ScheduledJob)
                .filter(ScheduledJob.job_id == job_id)
                .first()
            )
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get job: {e}")

    def get_job_with_executions(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job with execution history loaded"""
        try:
            return (
                self.db.query(ScheduledJob)
                .options(selectinload(ScheduledJob.executions))
                .filter(ScheduledJob.job_id == job_id)
                .first()
            )
        except Exception as e:
            logger.error(
                f"Failed to get job with executions {job_id}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to get job with executions: {e}")

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> ScheduledJob:
        """Update job with given changes"""
        try:
            with self.transaction() as db:
                job = (
                    db.query(ScheduledJob).filter(ScheduledJob.job_id == job_id).first()
                )
                if not job:
                    raise JobNotFoundError(f"Job not found: {job_id}")

                for key, value in updates.items():
                    if hasattr(job, key):
                        setattr(job, key, value)

                job.updated_at = datetime.now(timezone.utc)

                logger.info("Updated job", job_id=job_id, updates=list(updates.keys()))

                return job

        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to update job: {e}")

    def delete_job(self, job_id: str) -> bool:
        """Delete job and all related data"""
        try:
            with self.transaction() as db:
                job = (
                    db.query(ScheduledJob).filter(ScheduledJob.job_id == job_id).first()
                )
                if not job:
                    return False

                # Delete executions first (due to foreign key)
                db.query(JobExecution).filter(JobExecution.job_id == job_id).delete()

                # Delete dependencies
                db.query(JobDependency).filter(
                    or_(
                        JobDependency.job_id == job_id,
                        JobDependency.depends_on_job_id == job_id,
                    )
                ).delete()

                # Delete job
                db.delete(job)

                logger.info("Deleted job and related data", job_id=job_id)
                return True

        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to delete job: {e}")

    def get_jobs_by_status(self, status: ScheduledJobStatus) -> List[ScheduledJob]:
        """Get all jobs with specified status"""
        try:
            return (
                self.db.query(ScheduledJob)
                .filter(ScheduledJob.status == status)
                .order_by(ScheduledJob.created_at)
                .all()
            )
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get jobs by status: {e}")

    def get_jobs_due_for_execution(
        self, current_time: Optional[datetime] = None
    ) -> List[ScheduledJob]:
        """Get jobs that are due for execution"""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        try:
            return (
                self.db.query(ScheduledJob)
                .filter(
                    ScheduledJob.status == ScheduledJobStatus.ACTIVE,
                    ScheduledJob.next_run_time <= current_time,
                    ScheduledJob.next_run_time.isnot(None),
                )
                .order_by(ScheduledJob.next_run_time)
                .all()
            )
        except Exception as e:
            logger.error(f"Failed to get due jobs: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get due jobs: {e}")

    def get_jobs_by_trigger_type(self, trigger_type: TriggerType) -> List[ScheduledJob]:
        """Get jobs by trigger type"""
        try:
            return (
                self.db.query(ScheduledJob)
                .filter(ScheduledJob.trigger_type == trigger_type)
                .order_by(ScheduledJob.created_at)
                .all()
            )
        except Exception as e:
            logger.error(
                f"Failed to get jobs by trigger type {trigger_type}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to get jobs by trigger type: {e}")

    # Execution Management Methods

    def create_execution(self, execution_data: Dict[str, Any]) -> JobExecution:
        """Create new job execution record"""
        try:
            with self.transaction() as db:
                execution = JobExecution(**execution_data)
                db.add(execution)
                db.flush()

                logger.debug(
                    "Created job execution",
                    execution_id=execution.execution_id,
                    job_id=execution.job_id,
                )

                return execution

        except Exception as e:
            logger.error(f"Failed to create execution: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to create execution: {e}")

    def update_execution(
        self, execution_id: str, updates: Dict[str, Any]
    ) -> JobExecution:
        """Update execution with given changes"""
        try:
            with self.transaction() as db:
                execution = (
                    db.query(JobExecution)
                    .filter(JobExecution.execution_id == execution_id)
                    .first()
                )

                if not execution:
                    raise JobNotFoundError(f"Execution not found: {execution_id}")

                for key, value in updates.items():
                    if hasattr(execution, key):
                        setattr(execution, key, value)

                logger.debug(
                    "Updated execution",
                    execution_id=execution_id,
                    updates=list(updates.keys()),
                )

                return execution

        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update execution {execution_id}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to update execution: {e}")

    def get_execution(self, execution_id: str) -> Optional[JobExecution]:
        """Get execution by ID"""
        try:
            return (
                self.db.query(JobExecution)
                .filter(JobExecution.execution_id == execution_id)
                .first()
            )
        except Exception as e:
            logger.error(f"Failed to get execution {execution_id}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get execution: {e}")

    def get_job_executions(
        self, job_id: str, limit: int = 100, status: Optional[JobExecutionStatus] = None
    ) -> List[JobExecution]:
        """Get executions for a job"""
        try:
            query = (
                self.db.query(JobExecution)
                .filter(JobExecution.job_id == job_id)
                .order_by(desc(JobExecution.started_at))
            )

            if status:
                query = query.filter(JobExecution.status == status)

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(
                f"Failed to get executions for job {job_id}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to get job executions: {e}")

    def get_running_executions(self) -> List[JobExecution]:
        """Get all currently running executions"""
        try:
            return (
                self.db.query(JobExecution)
                .filter(JobExecution.status == JobExecutionStatus.RUNNING)
                .order_by(JobExecution.started_at)
                .all()
            )
        except Exception as e:
            logger.error(f"Failed to get running executions: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get running executions: {e}")

    def cleanup_old_executions(self, older_than: datetime) -> int:
        """Remove execution records older than specified date"""
        try:
            with self.transaction() as db:
                deleted_count = (
                    db.query(JobExecution)
                    .filter(JobExecution.started_at < older_than)
                    .delete()
                )

                logger.info(
                    "Cleaned up old executions",
                    deleted_count=deleted_count,
                    older_than=older_than.isoformat(),
                )

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to cleanup executions: {e}")

    # Dependency Management Methods

    def create_dependency(
        self, job_id: str, depends_on_job_id: str, dependency_type: str = "success"
    ) -> JobDependency:
        """Create job dependency"""
        try:
            with self.transaction() as db:
                dependency = JobDependency(
                    job_id=job_id,
                    depends_on_job_id=depends_on_job_id,
                    dependency_type=dependency_type,
                )
                db.add(dependency)
                db.flush()

                logger.info(
                    "Created job dependency",
                    job_id=job_id,
                    depends_on=depends_on_job_id,
                    dependency_type=dependency_type,
                )

                return dependency

        except IntegrityError as e:
            logger.error(f"Dependency creation failed - integrity error: {e}")
            raise SchedulingRepositoryError(f"Dependency creation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to create dependency: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to create dependency: {e}")

    def get_job_dependencies(self, job_id: str) -> List[JobDependency]:
        """Get all dependencies for a job"""
        try:
            return (
                self.db.query(JobDependency)
                .filter(JobDependency.job_id == job_id)
                .all()
            )
        except Exception as e:
            logger.error(
                f"Failed to get dependencies for job {job_id}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to get dependencies: {e}")

    def get_dependent_jobs(self, job_id: str) -> List[JobDependency]:
        """Get jobs that depend on this job"""
        try:
            return (
                self.db.query(JobDependency)
                .filter(JobDependency.depends_on_job_id == job_id)
                .all()
            )
        except Exception as e:
            logger.error(
                f"Failed to get dependent jobs for {job_id}: {e}", exc_info=True
            )
            raise SchedulingRepositoryError(f"Failed to get dependent jobs: {e}")

    def remove_dependency(self, job_id: str, depends_on_job_id: str) -> bool:
        """Remove specific dependency"""
        try:
            with self.transaction() as db:
                deleted_count = (
                    db.query(JobDependency)
                    .filter(
                        JobDependency.job_id == job_id,
                        JobDependency.depends_on_job_id == depends_on_job_id,
                    )
                    .delete()
                )

                logger.info(
                    "Removed job dependency",
                    job_id=job_id,
                    depends_on=depends_on_job_id,
                    deleted=deleted_count > 0,
                )

                return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to remove dependency: {e}")

    def check_dependency_satisfaction(self, job_id: str) -> Dict[str, bool]:
        """Check if all dependencies for a job are satisfied"""
        try:
            dependencies = self.get_job_dependencies(job_id)
            satisfaction_status = {}

            for dep in dependencies:
                # Get latest successful execution of dependency
                latest_execution = (
                    self.db.query(JobExecution)
                    .filter(
                        JobExecution.job_id == dep.depends_on_job_id,
                        JobExecution.status == JobExecutionStatus.SUCCESS,
                    )
                    .order_by(desc(JobExecution.finished_at))
                    .first()
                )

                if dep.dependency_type == "success":
                    satisfaction_status[dep.depends_on_job_id] = (
                        latest_execution is not None
                    )
                elif dep.dependency_type == "completion":
                    # Check for any completed execution (success or failure)
                    completed_execution = (
                        self.db.query(JobExecution)
                        .filter(
                            JobExecution.job_id == dep.depends_on_job_id,
                            JobExecution.status.in_(
                                [JobExecutionStatus.SUCCESS, JobExecutionStatus.FAILURE]
                            ),
                        )
                        .order_by(desc(JobExecution.finished_at))
                        .first()
                    )
                    satisfaction_status[dep.depends_on_job_id] = (
                        completed_execution is not None
                    )
                elif dep.dependency_type == "failure":
                    failed_execution = (
                        self.db.query(JobExecution)
                        .filter(
                            JobExecution.job_id == dep.depends_on_job_id,
                            JobExecution.status == JobExecutionStatus.FAILURE,
                        )
                        .order_by(desc(JobExecution.finished_at))
                        .first()
                    )
                    satisfaction_status[dep.depends_on_job_id] = (
                        failed_execution is not None
                    )

            return satisfaction_status

        except Exception as e:
            logger.error(
                f"Failed to check dependency satisfaction for {job_id}: {e}",
                exc_info=True,
            )
            raise SchedulingRepositoryError(f"Failed to check dependencies: {e}")

    # Locking Methods

    def acquire_scheduler_lock(
        self, lock_name: str, timeout_seconds: int = 300
    ) -> Optional[str]:
        """Acquire scheduler lock for coordination"""
        lock_id = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

        try:
            with self.transaction() as db:
                # Try to acquire new lock
                lock = SchedulerLock(
                    lock_name=lock_name, lock_id=lock_id, expires_at=expires_at
                )
                db.add(lock)

                logger.debug(
                    "Acquired scheduler lock",
                    lock_name=lock_name,
                    lock_id=lock_id,
                    expires_at=expires_at.isoformat(),
                )

                return lock_id

        except IntegrityError:
            # Lock already exists, check if expired
            try:
                with self.transaction() as db:
                    now = datetime.now(timezone.utc)
                    expired_locks = (
                        db.query(SchedulerLock)
                        .filter(
                            SchedulerLock.lock_name == lock_name,
                            SchedulerLock.expires_at < now,
                        )
                        .delete()
                    )

                    if expired_locks > 0:
                        # Try to acquire again after cleanup
                        lock = SchedulerLock(
                            lock_name=lock_name, lock_id=lock_id, expires_at=expires_at
                        )
                        db.add(lock)

                        logger.debug(
                            "Acquired scheduler lock after cleanup",
                            lock_name=lock_name,
                            lock_id=lock_id,
                            cleaned_locks=expired_locks,
                        )

                        return lock_id

            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup expired locks: {cleanup_error}")

            logger.debug(f"Failed to acquire lock {lock_name} - already held")
            return None

        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_name}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to acquire lock: {e}")

    def release_scheduler_lock(self, lock_name: str, lock_id: str) -> bool:
        """Release scheduler lock"""
        try:
            with self.transaction() as db:
                deleted_count = (
                    db.query(SchedulerLock)
                    .filter(
                        SchedulerLock.lock_name == lock_name,
                        SchedulerLock.lock_id == lock_id,
                    )
                    .delete()
                )

                success = deleted_count > 0

                logger.debug(
                    "Released scheduler lock",
                    lock_name=lock_name,
                    lock_id=lock_id,
                    success=success,
                )

                return success

        except Exception as e:
            logger.error(f"Failed to release lock {lock_name}: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to release lock: {e}")

    def cleanup_expired_locks(self) -> int:
        """Remove expired locks"""
        try:
            with self.transaction() as db:
                now = datetime.now(timezone.utc)
                deleted_count = (
                    db.query(SchedulerLock)
                    .filter(SchedulerLock.expires_at < now)
                    .delete()
                )

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired scheduler locks")

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired locks: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to cleanup locks: {e}")

    # Metrics and Monitoring Methods

    def record_scheduler_metrics(
        self, metrics_data: Dict[str, Any]
    ) -> SchedulerMetrics:
        """Record scheduler performance metrics"""
        try:
            with self.transaction() as db:
                metrics = SchedulerMetrics(**metrics_data)
                db.add(metrics)
                db.flush()

                logger.debug("Recorded scheduler metrics", metrics_id=metrics.id)
                return metrics

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to record metrics: {e}")

    def get_scheduler_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SchedulerMetrics]:
        """Get scheduler metrics within time range"""
        try:
            query = self.db.query(SchedulerMetrics)

            if start_time:
                query = query.filter(SchedulerMetrics.timestamp >= start_time)
            if end_time:
                query = query.filter(SchedulerMetrics.timestamp <= end_time)

            return query.order_by(desc(SchedulerMetrics.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get scheduler metrics: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get metrics: {e}")

    def get_job_statistics(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive job execution statistics"""
        try:
            query = self.db.query(JobExecution)
            if job_id:
                query = query.filter(JobExecution.job_id == job_id)

            # Basic counts
            total_executions = query.count()

            success_count = query.filter(
                JobExecution.status == JobExecutionStatus.SUCCESS
            ).count()
            failure_count = query.filter(
                JobExecution.status == JobExecutionStatus.FAILURE
            ).count()
            running_count = query.filter(
                JobExecution.status == JobExecutionStatus.RUNNING
            ).count()

            # Average runtime for completed executions
            completed_query = query.filter(
                JobExecution.status.in_(
                    [JobExecutionStatus.SUCCESS, JobExecutionStatus.FAILURE]
                ),
                JobExecution.finished_at.isnot(None),
            )

            avg_runtime = None
            if completed_query.count() > 0:
                # Calculate average runtime in seconds
                runtime_query = self.db.query(
                    func.avg(
                        func.extract(
                            "epoch", JobExecution.finished_at - JobExecution.started_at
                        )
                    ).label("avg_runtime_seconds")
                ).select_from(completed_query.subquery())

                result = runtime_query.first()
                avg_runtime = (
                    result.avg_runtime_seconds
                    if result and result.avg_runtime_seconds
                    else None
                )

            # Recent execution trend (last 24 hours)
            last_24h = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_executions = query.filter(
                JobExecution.started_at >= last_24h
            ).count()

            return {
                "total_executions": total_executions,
                "success_count": success_count,
                "failure_count": failure_count,
                "running_count": running_count,
                "success_rate": (success_count / total_executions * 100)
                if total_executions > 0
                else 0,
                "average_runtime_seconds": avg_runtime,
                "recent_executions_24h": recent_executions,
            }

        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}", exc_info=True)
            raise SchedulingRepositoryError(f"Failed to get statistics: {e}")

    # Health Check Methods

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of scheduling system"""
        try:
            health_status = {
                "database_connection": False,
                "active_jobs": 0,
                "running_executions": 0,
                "expired_locks": 0,
                "recent_failures": 0,
                "last_execution": None,
                "scheduler_healthy": True,
                "issues": [],
            }

            # Test database connection
            try:
                self.db.execute(text("SELECT 1")).fetchone()
                health_status["database_connection"] = True
            except Exception as db_error:
                health_status["issues"].append(
                    f"Database connection failed: {db_error}"
                )
                health_status["scheduler_healthy"] = False

            if health_status["database_connection"]:
                # Count active jobs
                health_status["active_jobs"] = (
                    self.db.query(ScheduledJob)
                    .filter(ScheduledJob.status == ScheduledJobStatus.ACTIVE)
                    .count()
                )

                # Count running executions
                health_status["running_executions"] = (
                    self.db.query(JobExecution)
                    .filter(JobExecution.status == JobExecutionStatus.RUNNING)
                    .count()
                )

                # Count expired locks
                now = datetime.now(timezone.utc)
                health_status["expired_locks"] = (
                    self.db.query(SchedulerLock)
                    .filter(SchedulerLock.expires_at < now)
                    .count()
                )

                # Count recent failures (last hour)
                last_hour = now - timedelta(hours=1)
                health_status["recent_failures"] = (
                    self.db.query(JobExecution)
                    .filter(
                        JobExecution.status == JobExecutionStatus.FAILURE,
                        JobExecution.started_at >= last_hour,
                    )
                    .count()
                )

                # Get last execution
                last_execution = (
                    self.db.query(JobExecution)
                    .order_by(desc(JobExecution.started_at))
                    .first()
                )
                if last_execution:
                    health_status[
                        "last_execution"
                    ] = last_execution.started_at.isoformat()

                # Check for concerning patterns
                if health_status["recent_failures"] > 10:
                    health_status["issues"].append("High failure rate in last hour")
                    health_status["scheduler_healthy"] = False

                if health_status["expired_locks"] > 5:
                    health_status["issues"].append("Many expired locks detected")

                # Check for stuck executions (running > 1 hour)
                stuck_threshold = now - timedelta(hours=1)
                stuck_executions = (
                    self.db.query(JobExecution)
                    .filter(
                        JobExecution.status == JobExecutionStatus.RUNNING,
                        JobExecution.started_at < stuck_threshold,
                    )
                    .count()
                )

                if stuck_executions > 0:
                    health_status["issues"].append(
                        f"{stuck_executions} potentially stuck executions"
                    )
                    health_status["scheduler_healthy"] = False

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "database_connection": False,
                "scheduler_healthy": False,
                "issues": [f"Health check failed: {e}"],
            }
