"""
Cron-based trigger implementation.

Provides cron expression based job scheduling with proper timezone handling,
DST transitions, and comprehensive validation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytz
from croniter import croniter

from app.utils.logger import get_logger

from .base import BaseTrigger, TriggerCalculationError, TriggerValidationError

logger = get_logger(__name__)


class CronTrigger(BaseTrigger):
    """
    Cron expression based trigger.

    Supports standard 5-field cron expressions and extended 6-field expressions
    with seconds. Handles timezone conversions and DST transitions properly.

    Examples:
    - "0 9 * * *" - Daily at 9:00 AM
    - "0 0 * * 1" - Weekly on Mondays at midnight
    - "*/5 * * * *" - Every 5 minutes
    - "0 0 1 * *" - Monthly on the 1st at midnight
    """

    # Valid cron field ranges
    FIELD_RANGES = {
        "second": (0, 59),
        "minute": (0, 59),
        "hour": (0, 23),
        "day": (1, 31),
        "month": (1, 12),
        "dow": (0, 6),  # 0 = Sunday
    }

    # Field names for 5 and 6 field cron expressions
    FIELD_NAMES_5 = ["minute", "hour", "day", "month", "dow"]
    FIELD_NAMES_6 = ["second", "minute", "hour", "day", "month", "dow"]

    def __init__(self, config: Dict[str, Any]):
        """Initialize cron trigger"""
        super().__init__(config)

        # Parse and validate cron expression
        self.expression = self.get_config_value("expression", required=True)
        self.timezone_name = self.get_config_value("timezone", default="UTC")

        # Get timezone object
        try:
            if self.timezone_name == "UTC":
                self.tz = timezone.utc
            else:
                self.tz = pytz.timezone(self.timezone_name)
        except Exception as e:
            raise TriggerValidationError(
                f"Invalid timezone: {self.timezone_name}"
            ) from e

        # Initialize croniter
        self._validate_and_prepare_croniter()

    def validate_config(self) -> None:
        """Validate cron trigger configuration"""
        expression = self.get_config_value("expression", required=True)

        if not isinstance(expression, str):
            raise TriggerValidationError("Cron expression must be a string")

        expression = expression.strip()
        if not expression:
            raise TriggerValidationError("Cron expression cannot be empty")

        # Basic format validation
        fields = expression.split()
        if len(fields) not in [5, 6]:
            raise TriggerValidationError(
                f"Cron expression must have 5 or 6 fields, got {len(fields)}: '{expression}'"
            )

        # Validate each field
        field_names = self.FIELD_NAMES_6 if len(fields) == 6 else self.FIELD_NAMES_5

        for i, (field_value, field_name) in enumerate(zip(fields, field_names)):
            try:
                self._validate_cron_field(field_value, field_name)
            except ValueError as e:
                raise TriggerValidationError(
                    f"Invalid cron field {i+1} ({field_name}): {field_value} - {e}"
                ) from e

    def _validate_cron_field(self, field_value: str, field_name: str) -> None:
        """Validate individual cron field"""
        if field_value == "*":
            return  # Wildcard is always valid

        min_val, max_val = self.FIELD_RANGES[field_name]

        # Handle lists (comma-separated values)
        if "," in field_value:
            for part in field_value.split(","):
                self._validate_cron_field_part(
                    part.strip(), field_name, min_val, max_val
                )
            return

        self._validate_cron_field_part(field_value, field_name, min_val, max_val)

    def _validate_cron_field_part(
        self, part: str, field_name: str, min_val: int, max_val: int
    ) -> None:
        """Validate a single part of a cron field"""
        # Handle step values (e.g., */5, 1-10/2)
        if "/" in part:
            range_part, step_part = part.split("/", 1)
            try:
                step = int(step_part)
                if step <= 0:
                    raise ValueError(f"Step value must be positive, got {step}")
            except ValueError as e:
                raise ValueError(f"Invalid step value: {step_part}") from e

            part = range_part

        # Handle ranges (e.g., 1-5)
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)

                if start < min_val or start > max_val:
                    raise ValueError(
                        f"Range start {start} outside valid range [{min_val}, {max_val}]"
                    )
                if end < min_val or end > max_val:
                    raise ValueError(
                        f"Range end {end} outside valid range [{min_val}, {max_val}]"
                    )
                if start > end:
                    raise ValueError(f"Range start {start} greater than end {end}")

            except ValueError as e:
                if "outside valid range" in str(e) or "greater than end" in str(e):
                    raise
                raise ValueError(f"Invalid range format: {part}") from e

            return

        # Handle single values
        if part != "*":
            try:
                value = int(part)
                if value < min_val or value > max_val:
                    raise ValueError(
                        f"Value {value} outside valid range [{min_val}, {max_val}]"
                    )
            except ValueError as e:
                if "outside valid range" in str(e):
                    raise
                raise ValueError(f"Invalid numeric value: {part}") from e

    def _validate_and_prepare_croniter(self) -> None:
        """Validate cron expression using croniter and prepare for use"""
        try:
            # Test croniter with current time
            now = datetime.now(self.tz)
            test_cron = croniter(self.expression, now)

            # Try to get next run to validate expression works
            next_run = test_cron.get_next(datetime)
            if next_run <= now:
                # This shouldn't happen, but let's be safe
                logger.warning(
                    "Cron expression returned past time for next run",
                    expression=self.expression,
                    now=now.isoformat(),
                    next_run=next_run.isoformat(),
                )

            logger.debug(
                "Cron expression validated successfully",
                expression=self.expression,
                timezone=self.timezone_name,
                next_run=next_run.isoformat(),
            )

        except Exception as e:
            raise TriggerValidationError(
                f"Invalid cron expression '{self.expression}': {e}"
            ) from e

    async def get_next_run_time(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        """
        Calculate next run time based on cron expression.

        Args:
            previous_run_time: When job was last run

        Returns:
            Next run time in UTC timezone
        """
        try:
            # Determine base time for calculation
            if previous_run_time:
                # Start from previous run time to ensure we don't miss runs
                base_time = previous_run_time
            else:
                # For new jobs, start from current time
                base_time = datetime.now(timezone.utc)

            # Convert to target timezone for cron calculation
            if self.tz != timezone.utc:
                if base_time.tzinfo != self.tz:
                    base_time = base_time.astimezone(self.tz)

            # Create croniter instance
            cron = croniter(self.expression, base_time)

            # Get next run time
            next_run = cron.get_next(datetime)

            # Convert back to UTC
            if next_run.tzinfo != timezone.utc:
                next_run = next_run.astimezone(timezone.utc)

            # Validate the calculated time is in the future
            now_utc = datetime.now(timezone.utc)
            if next_run <= now_utc:
                # Try once more from current time
                logger.warning(
                    "Calculated next run time is in the past, recalculating from current time",
                    expression=self.expression,
                    calculated_time=next_run.isoformat(),
                    current_time=now_utc.isoformat(),
                )

                current_in_tz = (
                    now_utc.astimezone(self.tz) if self.tz != timezone.utc else now_utc
                )
                cron = croniter(self.expression, current_in_tz)
                next_run = cron.get_next(datetime)

                if next_run.tzinfo != timezone.utc:
                    next_run = next_run.astimezone(timezone.utc)

                if next_run <= now_utc:
                    logger.error(
                        "Cron expression produces no future run times",
                        expression=self.expression,
                        timezone=self.timezone_name,
                    )
                    return None

            logger.debug(
                "Calculated next cron run time",
                expression=self.expression,
                timezone=self.timezone_name,
                previous_run=previous_run_time.isoformat()
                if previous_run_time
                else None,
                next_run=next_run.isoformat(),
            )

            return next_run

        except Exception as e:
            logger.error(
                "Failed to calculate next run time for cron expression",
                expression=self.expression,
                error=str(e),
                exc_info=True,
            )
            raise TriggerCalculationError(f"Cron calculation failed: {e}") from e

    def get_trigger_info(self) -> Dict[str, Any]:
        """Get human-readable trigger information"""
        try:
            # Try to get next few run times for display
            now = datetime.now(self.tz if self.tz != timezone.utc else timezone.utc)
            cron = croniter(self.expression, now)

            next_runs = []
            for _ in range(5):  # Get next 5 runs
                next_run = cron.get_next(datetime)
                if self.tz != timezone.utc:
                    next_run = next_run.astimezone(timezone.utc)
                next_runs.append(next_run.isoformat())

            # Try to get human-readable description
            description = self._get_cron_description()

            return {
                "type": "cron",
                "expression": self.expression,
                "timezone": self.timezone_name,
                "description": description,
                "next_runs": next_runs,
            }

        except Exception as e:
            logger.error(f"Failed to get trigger info for cron: {e}", exc_info=True)
            return {
                "type": "cron",
                "expression": self.expression,
                "timezone": self.timezone_name,
                "description": f"Cron: {self.expression}",
                "error": str(e),
            }

    def _get_cron_description(self) -> str:
        """Generate human-readable description of cron expression"""
        fields = self.expression.split()

        if len(fields) == 5:
            minute, hour, day, month, dow = fields
            second = "0"
        else:
            second, minute, hour, day, month, dow = fields

        # Simple description logic (can be enhanced with more sophisticated parsing)
        parts = []

        # Time part
        if minute == "*" and hour == "*":
            if second == "0":
                parts.append("every minute")
            else:
                parts.append("every second")
        elif minute.startswith("*/") and hour == "*":
            interval = minute[2:]
            parts.append(f"every {interval} minutes")
        elif hour.startswith("*/") and minute == "0":
            interval = hour[2:]
            parts.append(f"every {interval} hours")
        else:
            time_str = f"{hour}:{minute}"
            if second != "0":
                time_str += f":{second}"
            parts.append(f"at {time_str}")

        # Day part
        if day != "*":
            parts.append(f"on day {day}")

        # Month part
        if month != "*":
            parts.append(f"in month {month}")

        # Day of week part
        if dow != "*":
            dow_names = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
            if dow.isdigit():
                parts.append(f"on {dow_names[int(dow)]}")
            else:
                parts.append(f"on dow {dow}")

        description = " ".join(parts)
        if self.timezone_name != "UTC":
            description += f" ({self.timezone_name})"

        return description.capitalize()

    def should_skip_run(
        self, scheduled_time: datetime, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if run should be skipped due to DST or other timezone issues.

        Cron expressions can be particularly sensitive to DST transitions.
        """
        # Use base implementation first
        if super().should_skip_run(scheduled_time, current_time):
            return True

        # Additional cron-specific checks
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check for DST transition issues
        if self.tz != timezone.utc:
            try:
                # Convert times to local timezone
                scheduled_local = scheduled_time.astimezone(self.tz)
                current_local = current_time.astimezone(self.tz)

                # Check if scheduled time falls in a DST gap (doesn't exist)
                # This is a simplified check - real implementation might need more sophistication
                if hasattr(self.tz, "localize"):
                    try:
                        # Try to localize the naive datetime
                        naive_scheduled = scheduled_local.replace(tzinfo=None)
                        self.tz.localize(naive_scheduled, is_dst=None)
                    except Exception as e:
                        logger.warning(
                            "Skipping run due to DST transition",
                            scheduled_time=scheduled_time.isoformat(),
                            timezone=self.timezone_name,
                            error=str(e),
                        )
                        return True

            except Exception as e:
                logger.warning(
                    f"Error checking DST transition for scheduled run: {e}",
                    scheduled_time=scheduled_time.isoformat(),
                    timezone=self.timezone_name,
                )

        return False
