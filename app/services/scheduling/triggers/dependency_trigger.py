"""
Dependency-based trigger implementation.

Provides scheduling for jobs that depend on the completion of other jobs.
Supports various dependency types and complex workflow scenarios.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

from app.utils.logger import get_logger

from .base import BaseTrigger, TriggerCalculationError, TriggerValidationError

logger = get_logger(__name__)


class DependencyType(str, Enum):
    """Types of job dependencies"""

    SUCCESS = "success"  # Depends on successful completion
    COMPLETION = "completion"  # Depends on completion (success or failure)
    FAILURE = "failure"  # Depends on failure (for error handling workflows)


class DependencyTrigger(BaseTrigger):
    """
    Dependency-based trigger.

    Schedules jobs to run based on the completion status of other jobs.
    Supports complex dependency graphs and workflow orchestration.

    Examples:
    - Run after job succeeds: {"depends_on": ["job1"], "dependency_type": "success"}
    - Run after multiple jobs: {"depends_on": ["job1", "job2"], "dependency_type": "success"}
    - Run after job fails: {"depends_on": ["job1"], "dependency_type": "failure"}
    - With delay: {"depends_on": ["job1"], "dependency_type": "success", "delay_seconds": 300}
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize dependency trigger"""
        super().__init__(config)

        # Parse dependency configuration
        self.depends_on_jobs = self.get_config_value("depends_on", required=True)
        self.dependency_type = DependencyType(
            self.get_config_value("dependency_type", default="success")
        )
        self.delay_seconds = self.get_config_value("delay_seconds", default=0)

        # Advanced dependency options
        self.require_all = self.get_config_value(
            "require_all", default=True
        )  # AND vs OR logic
        self.timeout_seconds = self.get_config_value(
            "timeout_seconds", default=86400
        )  # 24 hours

        # Internal state
        self._dependency_status: Dict[str, Optional[bool]] = {}
        self._last_check: Optional[datetime] = None

    def validate_config(self) -> None:
        """Validate dependency trigger configuration"""
        # Validate depends_on jobs list
        depends_on = self.get_config_value("depends_on", required=True)
        if not isinstance(depends_on, list):
            raise TriggerValidationError("'depends_on' must be a list of job IDs")
        if len(depends_on) == 0:
            raise TriggerValidationError("'depends_on' cannot be empty")

        for job_id in depends_on:
            if not isinstance(job_id, str) or not job_id.strip():
                raise TriggerValidationError(
                    f"Job ID must be non-empty string, got: {job_id}"
                )

        # Validate dependency type
        dependency_type = self.get_config_value("dependency_type", default="success")
        if dependency_type not in [dt.value for dt in DependencyType]:
            valid_types = [dt.value for dt in DependencyType]
            raise TriggerValidationError(
                f"dependency_type must be one of {valid_types}, got: {dependency_type}"
            )

        # Validate delay
        delay = self.get_config_value("delay_seconds", default=0)
        if not isinstance(delay, (int, float)) or delay < 0:
            raise TriggerValidationError("delay_seconds must be a non-negative number")

        # Validate require_all flag
        require_all = self.get_config_value("require_all", default=True)
        if not isinstance(require_all, bool):
            raise TriggerValidationError("require_all must be a boolean")

        # Validate timeout
        timeout = self.get_config_value("timeout_seconds", default=86400)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise TriggerValidationError("timeout_seconds must be a positive number")

    async def get_next_run_time(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        """
        Calculate next run time based on dependency status.

        Args:
            previous_run_time: When job was last run

        Returns:
            Next run time if dependencies are met, None otherwise
        """
        try:
            # If job has already run and this is a one-time dependency, don't run again
            if previous_run_time is not None:
                logger.debug(
                    "Dependency trigger already executed",
                    depends_on=self.depends_on_jobs,
                    previous_run=previous_run_time.isoformat(),
                )
                return None

            # Check dependency status
            dependency_satisfied = await self._check_dependencies()

            if not dependency_satisfied:
                # Dependencies not yet satisfied
                logger.debug(
                    "Dependency trigger waiting for dependencies",
                    depends_on=self.depends_on_jobs,
                    dependency_type=self.dependency_type.value,
                    require_all=self.require_all,
                )

                # Check for timeout
                now = datetime.now(timezone.utc)
                if (
                    self._last_check
                    and (now - self._last_check).total_seconds() > self.timeout_seconds
                ):
                    logger.warning(
                        "Dependency trigger timed out waiting for dependencies",
                        depends_on=self.depends_on_jobs,
                        timeout_seconds=self.timeout_seconds,
                    )
                    return None

                # Return far future time to indicate not ready yet
                # Scheduler will check again on next poll
                return now + timedelta(minutes=5)  # Check again in 5 minutes

            # Dependencies satisfied - calculate run time with delay
            earliest_completion_time = (
                await self._get_earliest_dependency_completion_time()
            )
            if earliest_completion_time is None:
                earliest_completion_time = datetime.now(timezone.utc)

            run_time = earliest_completion_time + timedelta(seconds=self.delay_seconds)

            logger.info(
                "Dependency trigger ready for execution",
                depends_on=self.depends_on_jobs,
                dependency_type=self.dependency_type.value,
                earliest_completion=earliest_completion_time.isoformat(),
                scheduled_run=run_time.isoformat(),
                delay_seconds=self.delay_seconds,
            )

            return run_time

        except Exception as e:
            logger.error(
                "Failed to calculate next run time for dependency trigger",
                depends_on=self.depends_on_jobs,
                dependency_type=self.dependency_type.value,
                error=str(e),
                exc_info=True,
            )
            raise TriggerCalculationError(f"Dependency calculation failed: {e}") from e

    async def _check_dependencies(self) -> bool:
        """
        Check if all required dependencies are satisfied.

        Returns:
            True if dependencies are satisfied based on require_all setting
        """
        # This is a simplified implementation
        # In a real system, this would query the job execution history

        self._last_check = datetime.now(timezone.utc)

        # TODO: Implement actual dependency checking by querying job execution records
        # For now, return False to indicate dependencies are not yet implemented

        logger.debug(
            "Checking job dependencies",
            depends_on=self.depends_on_jobs,
            dependency_type=self.dependency_type.value,
            require_all=self.require_all,
        )

        # Placeholder implementation - in real system would check database
        satisfied_count = 0

        for job_id in self.depends_on_jobs:
            # Query job execution status from database
            # For now, assume not satisfied
            job_satisfied = False  # TODO: Implement actual check

            if job_satisfied:
                satisfied_count += 1
                self._dependency_status[job_id] = True
            else:
                self._dependency_status[job_id] = False

        if self.require_all:
            # All dependencies must be satisfied
            return satisfied_count == len(self.depends_on_jobs)
        else:
            # At least one dependency must be satisfied
            return satisfied_count > 0

    async def _get_earliest_dependency_completion_time(self) -> Optional[datetime]:
        """
        Get the earliest completion time among the satisfied dependencies.

        Returns:
            Earliest completion time or None if no dependencies completed
        """
        # TODO: Implement by querying job execution completion times
        # For now, return current time as placeholder
        return datetime.now(timezone.utc)

    def get_trigger_info(self) -> Dict[str, Any]:
        """Get human-readable trigger information"""
        try:
            # Build description
            if len(self.depends_on_jobs) == 1:
                job_desc = f"job {self.depends_on_jobs[0]}"
            else:
                logic = "all" if self.require_all else "any"
                job_desc = f"{logic} of jobs: {', '.join(self.depends_on_jobs)}"

            action_desc = {
                DependencyType.SUCCESS: "succeeds",
                DependencyType.COMPLETION: "completes",
                DependencyType.FAILURE: "fails",
            }[self.dependency_type]

            description = f"Run when {job_desc} {action_desc}"

            if self.delay_seconds > 0:
                if self.delay_seconds < 60:
                    delay_desc = f"{self.delay_seconds} seconds"
                elif self.delay_seconds < 3600:
                    delay_desc = f"{self.delay_seconds // 60} minutes"
                else:
                    delay_desc = f"{self.delay_seconds // 3600} hours"
                description += f" (after {delay_desc} delay)"

            # Get current dependency status
            status_summary = {"waiting": 0, "satisfied": 0, "failed": 0}

            for job_id in self.depends_on_jobs:
                job_status = self._dependency_status.get(job_id)
                if job_status is None:
                    status_summary["waiting"] += 1
                elif job_status:
                    status_summary["satisfied"] += 1
                else:
                    status_summary["failed"] += 1

            return {
                "type": "dependency",
                "depends_on": self.depends_on_jobs,
                "dependency_type": self.dependency_type.value,
                "require_all": self.require_all,
                "delay_seconds": self.delay_seconds,
                "timeout_seconds": self.timeout_seconds,
                "description": description,
                "status_summary": status_summary,
                "dependency_status": dict(self._dependency_status),
                "last_check": self._last_check.isoformat()
                if self._last_check
                else None,
            }

        except Exception as e:
            logger.error(
                f"Failed to get trigger info for dependency: {e}", exc_info=True
            )
            return {
                "type": "dependency",
                "depends_on": self.depends_on_jobs,
                "dependency_type": self.dependency_type.value,
                "description": f"Depends on {len(self.depends_on_jobs)} jobs",
                "error": str(e),
            }

    def should_skip_run(
        self, scheduled_time: datetime, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if run should be skipped for dependency trigger.

        Dependency triggers have their own timeout logic.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check overall timeout
        if self._last_check:
            time_waiting = (current_time - self._last_check).total_seconds()
            if time_waiting > self.timeout_seconds:
                logger.warning(
                    "Skipping dependency trigger run - overall timeout exceeded",
                    depends_on=self.depends_on_jobs,
                    time_waiting=time_waiting,
                    timeout_seconds=self.timeout_seconds,
                )
                return True

        # Use shorter grace time for dependencies since timing is more flexible
        grace_time = self.get_config_value(
            "misfire_grace_time", default=1800
        )  # 30 minutes
        delay_seconds = (current_time - scheduled_time).total_seconds()

        if delay_seconds > grace_time:
            logger.info(
                "Skipping dependency trigger run due to excessive delay",
                depends_on=self.depends_on_jobs,
                scheduled_time=scheduled_time.isoformat(),
                current_time=current_time.isoformat(),
                delay_seconds=delay_seconds,
                grace_time=grace_time,
            )
            return True

        return False

    def add_dependency(self, job_id: str) -> None:
        """Add a new job dependency"""
        if job_id not in self.depends_on_jobs:
            self.depends_on_jobs.append(job_id)
            self.config["depends_on"] = self.depends_on_jobs
            self._dependency_status[job_id] = None

    def remove_dependency(self, job_id: str) -> None:
        """Remove a job dependency"""
        if job_id in self.depends_on_jobs:
            self.depends_on_jobs.remove(job_id)
            self.config["depends_on"] = self.depends_on_jobs
            if job_id in self._dependency_status:
                del self._dependency_status[job_id]
