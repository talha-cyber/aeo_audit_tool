"""
Base trigger interface for job scheduling.

Defines the common interface that all trigger types must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytz

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TriggerError(Exception):
    """Base exception for trigger-related errors"""

    pass


class TriggerValidationError(TriggerError):
    """Raised when trigger configuration is invalid"""

    pass


class TriggerCalculationError(TriggerError):
    """Raised when trigger calculation fails"""

    pass


class BaseTrigger(ABC):
    """
    Base class for all trigger types.

    Defines the interface for calculating when jobs should run next,
    with proper error handling and timezone support.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trigger with configuration.

        Args:
            config: Trigger-specific configuration dictionary
        """
        # Work with a shallow copy so we can add defaults without mutating caller input
        self.config: Dict[str, Any] = dict(config)

        tz_name = self.config.get("timezone", "UTC")
        try:
            timezone_obj = pytz.timezone(tz_name)
        except Exception as exc:  # pragma: no cover - configuration guardrail
            raise TriggerValidationError(f"Invalid timezone: {tz_name}") from exc

        self.timezone = timezone_obj
        self.timezone_name = getattr(timezone_obj, "zone", tz_name)

        # Common scheduling tolerances
        self.misfire_grace_time = int(self.config.get("misfire_grace_time", 3600))
        self.config.setdefault("timezone", self.timezone_name)
        self.config.setdefault("misfire_grace_time", self.misfire_grace_time)

        # Validate configuration during initialization
        self.validate_config()

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate trigger configuration.

        Should raise TriggerValidationError if configuration is invalid.
        """
        pass

    @abstractmethod
    async def get_next_run_time(
        self, previous_run_time: Optional[datetime]
    ) -> Optional[datetime]:
        """
        Calculate the next run time for this trigger.

        Args:
            previous_run_time: When the job was last run (None for new jobs)

        Returns:
            Next run time as timezone-aware datetime, or None if no more runs

        Raises:
            TriggerCalculationError: If calculation fails
        """
        pass

    def get_trigger_info(self) -> Dict[str, Any]:
        """Default trigger metadata implementation."""
        return {
            "type": self.config.get("trigger_type", "unknown"),
            "config": self.config,
        }

    def should_skip_run(
        self, scheduled_time: datetime, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if a scheduled run should be skipped.

        Used for cases like:
        - Job was scheduled too long ago (misfire)
        - System was down during scheduled time
        - Manual override conditions

        Args:
            scheduled_time: When the job was supposed to run
            current_time: Current time (defaults to now)

        Returns:
            True if this run should be skipped
        """
        if current_time is None:
            current_time = self.now()

        max_delay_seconds = self.config.get(
            "max_misfire_delay", self.misfire_grace_time
        )
        delay_seconds = (current_time - scheduled_time).total_seconds()

        if delay_seconds > max_delay_seconds:
            logger.warning(
                "Skipping job run due to excessive delay",
                scheduled_time=scheduled_time.isoformat(),
                current_time=current_time.isoformat(),
                delay_seconds=delay_seconds,
                max_delay=max_delay_seconds,
            )
            return True

        return False

    def normalize_datetime(self, dt: datetime) -> datetime:
        """
        Normalize datetime to UTC timezone.

        Args:
            dt: Datetime to normalize

        Returns:
            Timezone-aware UTC datetime
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt

    def get_config_value(
        self, key: str, default: Any = None, required: bool = False
    ) -> Any:
        """
        Get configuration value with validation.

        Args:
            key: Configuration key
            default: Default value if key not found
            required: Whether key is required

        Returns:
            Configuration value

        Raises:
            TriggerValidationError: If required key is missing
        """
        if key not in self.config:
            if required:
                raise TriggerValidationError(
                    f"Required configuration key '{key}' is missing"
                )
            return default

        return self.config[key]

    def validate_datetime_config(
        self, key: str, required: bool = False
    ) -> Optional[datetime]:
        """
        Validate and normalize datetime configuration value.

        Args:
            key: Configuration key
            required: Whether the datetime is required

        Returns:
            Normalized datetime or None

        Raises:
            TriggerValidationError: If validation fails
        """
        value = self.get_config_value(key, required=required)

        if value is None:
            return None

        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as e:
                raise TriggerValidationError(
                    f"Invalid datetime format for '{key}': {value}"
                ) from e
        elif not isinstance(value, datetime):
            raise TriggerValidationError(
                f"Configuration '{key}' must be datetime or ISO string, got {type(value)}"
            )

        return self.normalize_datetime(value)

    def now(self) -> datetime:
        """Return the current UTC time (safe for monkeypatching in tests)."""
        return datetime.now(timezone.utc)

    def validate_positive_integer(
        self, key: str, required: bool = False, min_value: int = 1
    ) -> Optional[int]:
        """
        Validate positive integer configuration value.

        Args:
            key: Configuration key
            required: Whether the value is required
            min_value: Minimum allowed value

        Returns:
            Integer value or None

        Raises:
            TriggerValidationError: If validation fails
        """
        value = self.get_config_value(key, required=required)

        if value is None:
            return None

        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError) as e:
                raise TriggerValidationError(
                    f"Configuration '{key}' must be an integer, got {type(value)}"
                ) from e

        if value < min_value:
            raise TriggerValidationError(
                f"Configuration '{key}' must be >= {min_value}, got {value}"
            )

        return value

    def __repr__(self) -> str:
        """String representation of trigger"""
        return f"{self.__class__.__name__}({self.config})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary representation"""
        return {
            "type": self.__class__.__name__,
            "config": self.config,
            "info": self.get_trigger_info(),
        }
