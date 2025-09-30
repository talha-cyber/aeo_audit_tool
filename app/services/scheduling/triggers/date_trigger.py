"""
One-time date-based trigger implementation.

Provides scheduling for jobs that should run exactly once at a specific date/time.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.utils.logger import get_logger

from .base import BaseTrigger, TriggerCalculationError, TriggerValidationError

logger = get_logger(__name__)


class DateTrigger(BaseTrigger):
    """
    One-time date-based trigger.

    Schedules a job to run exactly once at a specified date and time.
    After execution, the job will not be scheduled again.

    Examples:
    - Run at specific datetime: {"run_date": "2024-12-25T09:00:00Z"}
    - Run with timezone: {"run_date": "2024-12-25T09:00:00", "timezone": "America/New_York"}
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize date trigger"""
        super().__init__(config)

        self.run_date = self.validate_datetime_config("run_date", required=True)
        self.past_date_grace_seconds = int(
            self.get_config_value("past_date_grace_seconds", default=300)
        )
        self.config["past_date_grace_seconds"] = self.past_date_grace_seconds
        self.has_executed = bool(self.get_config_value("has_executed", default=False))
        self.config["has_executed"] = self.has_executed

    def now(self) -> datetime:  # type: ignore[override]
        return datetime.now(timezone.utc)

    def validate_config(self) -> None:
        """Validate date trigger configuration"""
        run_date = self.validate_datetime_config("run_date", required=True)

        now = self.now()
        grace_seconds = int(
            self.get_config_value("past_date_grace_seconds", default=300)
        )
        if run_date < now - timedelta(seconds=grace_seconds):
            logger.warning(
                "Date trigger run_date is in the past",
                run_date=run_date.isoformat(),
                current_time=now.isoformat(),
                grace_seconds=grace_seconds,
            )

        # Validate has_executed flag
        has_executed = self.get_config_value("has_executed", default=False)
        if not isinstance(has_executed, bool):
            raise TriggerValidationError("'has_executed' must be a boolean")

    async def get_next_run_time(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        """
        Calculate next run time for date trigger.

        Args:
            previous_run_time: When job was last run (used to determine if already executed)

        Returns:
            Run date if not yet executed, None if already executed
        """
        try:
            # If job has already run, no more executions
            if previous_run_time is not None or self.has_executed:
                logger.debug(
                    "Date trigger already executed",
                    run_date=self.run_date.isoformat(),
                    previous_run=previous_run_time.isoformat()
                    if previous_run_time
                    else None,
                    has_executed_flag=self.has_executed,
                )
                return None

            # Check if run date is still in the future
            now = self.now()
            if self.run_date <= now:
                # Run date is in the past - check grace period
                if (now - self.run_date).total_seconds() > self.past_date_grace_seconds:
                    logger.warning(
                        "Date trigger run date is too far in the past, skipping",
                        run_date=self.run_date.isoformat(),
                        current_time=now.isoformat(),
                        grace_seconds=self.past_date_grace_seconds,
                    )
                    return None

                # Within grace period - schedule for immediate execution
                logger.info(
                    "Date trigger run date is slightly past, scheduling for immediate execution",
                    run_date=self.run_date.isoformat(),
                    current_time=now.isoformat(),
                )
                return now

            logger.debug(
                "Date trigger scheduled for future execution",
                run_date=self.run_date.isoformat(),
                current_time=now.isoformat(),
            )

            return self.run_date

        except Exception as e:
            logger.error(
                "Failed to calculate next run time for date trigger",
                run_date=self.run_date.isoformat() if self.run_date else None,
                error=str(e),
                exc_info=True,
            )
            raise TriggerCalculationError(f"Date calculation failed: {e}") from e

    def get_trigger_info(self) -> Dict[str, Any]:
        """Get human-readable trigger information"""
        try:
            now = self.now()

            # Determine status
            if self.has_executed:
                status = "completed"
            elif self.run_date <= now:
                status = "overdue"
            else:
                status = "scheduled"

            # Calculate time until execution
            time_until = None
            if not self.has_executed and self.run_date > now:
                time_delta = self.run_date - now
                total_seconds = int(time_delta.total_seconds())

                if total_seconds < 60:
                    time_until = f"{total_seconds} seconds"
                elif total_seconds < 3600:
                    minutes = total_seconds // 60
                    time_until = f"{minutes} minute{'s' if minutes != 1 else ''}"
                elif total_seconds < 86400:
                    hours = total_seconds // 3600
                    time_until = f"{hours} hour{'s' if hours != 1 else ''}"
                else:
                    days = total_seconds // 86400
                    time_until = f"{days} day{'s' if days != 1 else ''}"

            description = (
                f"Run once on {self.run_date.strftime('%Y-%m-%d at %H:%M:%S UTC')}"
            )
            if time_until:
                description += f" (in {time_until})"

            return {
                "type": "date",
                "run_date": self.run_date.isoformat(),
                "status": status,
                "description": description,
                "time_until": time_until,
                "has_executed": self.has_executed,
            }

        except Exception as e:
            logger.error(f"Failed to get trigger info for date: {e}", exc_info=True)
            return {
                "type": "date",
                "run_date": self.run_date.isoformat() if self.run_date else None,
                "description": f"Run once at {self.run_date}",
                "error": str(e),
            }

    def should_skip_run(
        self, scheduled_time: datetime, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if run should be skipped for date trigger.

        Date triggers are more strict about timing since they only run once.
        """
        if current_time is None:
            current_time = self.now()

        # If already executed, always skip
        if self.has_executed:
            logger.debug("Skipping date trigger run - already executed")
            return True

        # Check if we're significantly past the scheduled time
        delay_seconds = (current_time - scheduled_time).total_seconds()

        if delay_seconds > self.past_date_grace_seconds:
            logger.warning(
                "Skipping date trigger run - too far past scheduled time",
                scheduled_time=scheduled_time.isoformat(),
                current_time=current_time.isoformat(),
                delay_seconds=delay_seconds,
                grace_seconds=self.past_date_grace_seconds,
            )
            return True

        return False

    def mark_executed(self) -> None:
        """Mark the trigger as executed"""
        self.has_executed = True
        self.config["has_executed"] = True

        logger.debug(
            "Date trigger marked as executed", run_date=self.run_date.isoformat()
        )
