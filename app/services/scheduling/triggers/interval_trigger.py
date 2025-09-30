"""
Interval-based trigger implementation.

Provides fixed-interval job scheduling with proper handling of misfire scenarios,
system downtime recovery, and flexible interval specifications.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.utils.logger import get_logger

from .base import BaseTrigger, TriggerCalculationError, TriggerValidationError

logger = get_logger(__name__)


class IntervalTrigger(BaseTrigger):
    """
    Fixed-interval based trigger.

    Schedules jobs to run at regular intervals. Supports various time units
    and handles edge cases like system downtime and misfire scenarios.

    Examples:
    - Every 30 seconds: {"seconds": 30}
    - Every 5 minutes: {"minutes": 5}
    - Every 2 hours: {"hours": 2}
    - Every day: {"days": 1}
    - Complex: {"hours": 1, "minutes": 30}  # Every 1.5 hours
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize interval trigger"""
        super().__init__(config)

        self.seconds = self._get_unit_value("seconds")
        self.minutes = self._get_unit_value("minutes")
        self.hours = self._get_unit_value("hours")
        self.days = self._get_unit_value("days")

        self.total_seconds = self._calculate_total_seconds()

        # Start time configuration
        self.start_date = self.validate_datetime_config("start_date", required=False)
        self.end_date = self.validate_datetime_config("end_date", required=False)

        # Misfire handling
        self.misfire_grace_time = int(
            self.get_config_value("misfire_grace_time", default=self.misfire_grace_time)
        )
        self.config["misfire_grace_time"] = self.misfire_grace_time
        self.catch_up_missed_runs = bool(
            self.get_config_value("catch_up_missed_runs", default=False)
        )

    def now(self) -> datetime:  # type: ignore[override]
        return datetime.now(timezone.utc)

    def validate_config(self) -> None:
        """Validate interval trigger configuration"""
        # Validate time units
        for unit in ["seconds", "minutes", "hours", "days"]:
            value = self.get_config_value(unit, default=0)
            if not isinstance(value, (int, float)):
                raise TriggerValidationError(
                    f"'{unit}' must be a number, got {type(value)}"
                )
            if value < 0:
                raise TriggerValidationError(
                    f"'{unit}' must be non-negative, got {value}"
                )

        # Ensure at least one time unit is specified
        total = self._calculate_total_seconds()
        if total <= 0:
            raise TriggerValidationError(
                "At least one time unit (seconds, minutes, hours, days) must be positive"
            )

        # Validate minimum interval (prevent excessive load)
        min_interval = self.get_config_value(
            "min_interval", default=1
        )  # 1 second minimum
        if total < min_interval:
            raise TriggerValidationError(
                f"Interval must be at least {min_interval} seconds"
            )

        # Validate maximum interval (prevent overflow issues)
        max_interval = self.get_config_value(
            "max_interval", default=365 * 24 * 3600
        )  # 1 year
        if total > max_interval:
            raise TriggerValidationError(
                f"Interval must be at most {max_interval} seconds"
            )

        # Validate date range
        start_date = self.validate_datetime_config("start_date", required=False)
        end_date = self.validate_datetime_config("end_date", required=False)

        if start_date and end_date and end_date <= start_date:
            raise TriggerValidationError("end_date must be after start_date")

        # Validate misfire settings
        misfire_grace = self.get_config_value("misfire_grace_time", default=300)
        if not isinstance(misfire_grace, (int, float)) or misfire_grace < 0:
            raise TriggerValidationError(
                "misfire_grace_time must be a non-negative number"
            )

        catch_up = self.get_config_value("catch_up_missed_runs", default=False)
        if not isinstance(catch_up, bool):
            raise TriggerValidationError("catch_up_missed_runs must be a boolean")

    def _calculate_total_seconds(self) -> float:
        """Calculate total interval in seconds"""
        total = (
            self._get_unit_value("seconds")
            + self._get_unit_value("minutes") * 60
            + self._get_unit_value("hours") * 3600
            + self._get_unit_value("days") * 86400
        )
        return total

    def _get_unit_value(self, key: str) -> float:
        value = self.get_config_value(key, default=0)
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TriggerValidationError(
                f"'{key}' must be a number, got {value!r}"
            ) from exc

    def _calculate_next_run(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        try:
            now = self.now()

            # Check if we're past the end date
            if self.end_date and now >= self.end_date:
                logger.debug(
                    "Interval trigger past end date",
                    end_date=self.end_date.isoformat(),
                    current_time=now.isoformat(),
                )
                return None

            # Determine base time for calculation
            if previous_run_time is None:
                # First run - use start_date if specified, otherwise current time
                if self.start_date:
                    base_time = max(self.start_date, now)
                else:
                    base_time = now
            else:
                base_time = previous_run_time + timedelta(seconds=self.total_seconds)

                if self.catch_up_missed_runs and base_time < now:
                    late_seconds = (now - base_time).total_seconds()
                    intervals_missed = int(late_seconds // self.total_seconds)
                    if late_seconds % self.total_seconds:
                        intervals_missed += 1
                    if intervals_missed > 0:
                        logger.info(
                            "Interval trigger catching up missed runs",
                            missed_intervals=intervals_missed,
                            interval_seconds=self.total_seconds,
                            previous_run=previous_run_time.isoformat(),
                            current_time=now.isoformat(),
                        )
                        base_time = base_time + timedelta(
                            seconds=intervals_missed * self.total_seconds
                        )

            # Ensure we're not before start_date
            if self.start_date and base_time < self.start_date:
                base_time = self.start_date

            # Ensure we're not after end_date
            if self.end_date and base_time >= self.end_date:
                logger.debug(
                    "Next run time would be past end date",
                    calculated_time=base_time.isoformat(),
                    end_date=self.end_date.isoformat(),
                )
                return None

            logger.debug(
                "Calculated next interval run time",
                previous_run=previous_run_time.isoformat()
                if previous_run_time
                else None,
                next_run=base_time.isoformat(),
                interval_seconds=self.total_seconds,
            )

            return base_time

        except Exception as e:
            logger.error(
                "Failed to calculate next run time for interval trigger",
                interval_seconds=self.total_seconds,
                error=str(e),
                exc_info=True,
            )
            raise TriggerCalculationError(f"Interval calculation failed: {e}") from e

    async def get_next_run_time(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        """Async wrapper around synchronous interval calculation."""
        return self._calculate_next_run(previous_run_time)

    def get_trigger_info(self) -> Dict[str, Any]:
        """Get human-readable trigger information"""
        try:
            # Build human-readable interval description
            parts = []

            def _describe(value: float, unit: str) -> str:
                if value.is_integer():
                    value_display = int(value)
                else:
                    value_display = value
                suffix = "s" if value_display != 1 else ""
                return f"{value_display} {unit}{suffix}"

            if self.days > 0:
                parts.append(_describe(self.days, "day"))
            if self.hours > 0:
                parts.append(_describe(self.hours, "hour"))
            if self.minutes > 0:
                parts.append(_describe(self.minutes, "minute"))
            if self.seconds > 0:
                parts.append(_describe(self.seconds, "second"))

            if len(parts) == 0:
                description = f"Every {self.total_seconds} seconds"
            elif len(parts) == 1:
                description = f"Every {parts[0]}"
            else:
                description = f"Every {', '.join(parts[:-1])} and {parts[-1]}"

            # Add date constraints if specified
            constraints = []
            if self.start_date:
                constraints.append(
                    f"starting {self.start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
            if self.end_date:
                constraints.append(
                    f"ending {self.end_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )

            if constraints:
                description += f" ({', '.join(constraints)})"

            # Calculate next few run times for display
            next_runs = []
            current_time = None

            try:
                for _ in range(5):  # Get next 5 runs
                    next_time = self._calculate_next_run(current_time)
                    if next_time is None:
                        break
                    next_runs.append(next_time.isoformat())
                    current_time = next_time
            except Exception as e:
                logger.warning(f"Failed to calculate sample run times: {e}")

            return {
                "type": "interval",
                "interval_seconds": self.total_seconds,
                "description": description,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "catch_up_missed_runs": self.catch_up_missed_runs,
                "next_runs": next_runs,
            }

        except Exception as e:
            logger.error(f"Failed to get trigger info for interval: {e}", exc_info=True)
            return {
                "type": "interval",
                "interval_seconds": self.total_seconds,
                "description": f"Every {self.total_seconds} seconds",
                "error": str(e),
            }

    def should_skip_run(
        self, scheduled_time: datetime, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if run should be skipped based on interval-specific logic.

        Intervals are more forgiving than cron expressions for missed runs.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check if we're past the end date
        if self.end_date and current_time >= self.end_date:
            logger.debug(
                "Skipping interval run - past end date",
                scheduled_time=scheduled_time.isoformat(),
                end_date=self.end_date.isoformat(),
                current_time=current_time.isoformat(),
            )
            return True

        # Use longer grace time for intervals vs base implementation
        interval_grace_time = max(self.misfire_grace_time, self.total_seconds * 0.1)
        max_delay_seconds = self.config.get("max_misfire_delay", interval_grace_time)

        delay_seconds = (current_time - scheduled_time).total_seconds()

        if delay_seconds > max_delay_seconds:
            logger.info(
                "Skipping interval run due to excessive delay",
                scheduled_time=scheduled_time.isoformat(),
                current_time=current_time.isoformat(),
                delay_seconds=delay_seconds,
                max_delay=max_delay_seconds,
                interval_seconds=self.total_seconds,
            )
            return True

        return False

    def get_interval_description(self) -> str:
        """Get a concise description of the interval"""
        total = self.total_seconds

        def _format(value: float, suffix: str) -> str:
            return f"{int(value)}{suffix}" if value.is_integer() else f"{value:.1f}{suffix}"

        if total < 60:
            return _format(total, "s")
        if total < 3600:
            return _format(total / 60, "m")
        if total < 86400:
            return _format(total / 3600, "h")
        return _format(total / 86400, "d")
