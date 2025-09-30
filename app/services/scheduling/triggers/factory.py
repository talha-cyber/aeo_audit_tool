"""
Trigger factory for creating appropriate trigger instances.

Provides a centralized factory for creating trigger instances based on
configuration, with proper validation and error handling.
"""

from typing import Any, Dict, Type, Union

from app.models.scheduling import TriggerType
from app.utils.logger import get_logger

from .base import BaseTrigger, TriggerValidationError
from .cron_trigger import CronTrigger
from .date_trigger import DateTrigger
from .dependency_trigger import DependencyTrigger
from .interval_trigger import IntervalTrigger

logger = get_logger(__name__)


class TriggerFactory:
    """
    Factory for creating trigger instances.

    Provides a centralized way to create trigger instances based on configuration,
    with proper validation, error handling, and extensibility for custom triggers.
    """

    # Registry of trigger types to classes
    TRIGGER_REGISTRY: Dict[TriggerType, Type[BaseTrigger]] = {
        TriggerType.CRON: CronTrigger,
        TriggerType.INTERVAL: IntervalTrigger,
        TriggerType.DATE: DateTrigger,
        TriggerType.DEPENDENCY: DependencyTrigger,
        # TriggerType.MANUAL: Manual triggers don't need trigger instances
    }

    def __init__(self):
        """Initialize trigger factory"""
        self._custom_triggers: Dict[str, Type[BaseTrigger]] = {}

    def create_trigger(self, trigger_config: Union[Dict[str, Any], Any]) -> BaseTrigger:
        """
        Create a trigger instance based on configuration.

        Args:
            trigger_config: Trigger configuration (dict or pydantic model)

        Returns:
            Configured trigger instance

        Raises:
            TriggerValidationError: If trigger type is invalid or configuration is bad
        """
        try:
            # Handle pydantic models
            if hasattr(trigger_config, "dict"):
                config_dict = trigger_config.dict()
            elif isinstance(trigger_config, dict):
                config_dict = trigger_config.copy()
            else:
                raise TriggerValidationError(
                    f"Invalid trigger config type: {type(trigger_config)}"
                )

            # Extract trigger type
            trigger_type_value = config_dict.get("trigger_type")
            if not trigger_type_value:
                raise TriggerValidationError(
                    "Missing 'trigger_type' in trigger configuration"
                )

            # Convert to TriggerType enum if needed
            if isinstance(trigger_type_value, str):
                try:
                    trigger_type = TriggerType(trigger_type_value.lower())
                except ValueError:
                    raise TriggerValidationError(
                        f"Invalid trigger type: {trigger_type_value}"
                    )
            elif isinstance(trigger_type_value, TriggerType):
                trigger_type = trigger_type_value
            else:
                raise TriggerValidationError(
                    f"Invalid trigger type format: {type(trigger_type_value)}"
                )

            # Handle manual triggers (no trigger instance needed)
            if trigger_type == TriggerType.MANUAL:
                raise TriggerValidationError(
                    "Manual triggers do not use trigger instances"
                )

            # Get trigger class
            trigger_class = self._get_trigger_class(trigger_type)

            # Create and return trigger instance
            logger.debug(f"Creating {trigger_type.value} trigger", config=config_dict)

            trigger_instance = trigger_class(config_dict)

            logger.debug(
                f"Successfully created {trigger_type.value} trigger",
                trigger_info=trigger_instance.get_trigger_info(),
            )

            return trigger_instance

        except TriggerValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to create trigger: {e}", exc_info=True)
            raise TriggerValidationError(f"Trigger creation failed: {e}") from e

    def _get_trigger_class(self, trigger_type: TriggerType) -> Type[BaseTrigger]:
        """
        Get trigger class for the given trigger type.

        Args:
            trigger_type: Type of trigger to create

        Returns:
            Trigger class

        Raises:
            TriggerValidationError: If trigger type is not supported
        """
        # Check built-in triggers
        if trigger_type in self.TRIGGER_REGISTRY:
            return self.TRIGGER_REGISTRY[trigger_type]

        # Check custom triggers
        trigger_name = trigger_type.value.lower()
        if trigger_name in self._custom_triggers:
            return self._custom_triggers[trigger_name]

        # Trigger type not found
        supported_types = list(self.TRIGGER_REGISTRY.keys()) + list(
            self._custom_triggers.keys()
        )
        raise TriggerValidationError(
            f"Unsupported trigger type: {trigger_type}. "
            f"Supported types: {[t.value for t in supported_types if hasattr(t, 'value')]}"
        )

    def register_custom_trigger(
        self, trigger_name: str, trigger_class: Type[BaseTrigger]
    ) -> None:
        """
        Register a custom trigger type.

        Args:
            trigger_name: Name of the custom trigger type
            trigger_class: Trigger class that extends BaseTrigger

        Raises:
            TriggerValidationError: If trigger class is invalid
        """
        if not isinstance(trigger_name, str) or not trigger_name.strip():
            raise TriggerValidationError("Trigger name must be a non-empty string")

        if not issubclass(trigger_class, BaseTrigger):
            raise TriggerValidationError("Trigger class must extend BaseTrigger")

        trigger_name = trigger_name.lower().strip()

        if trigger_name in self._custom_triggers:
            logger.warning(f"Overriding existing custom trigger: {trigger_name}")

        self._custom_triggers[trigger_name] = trigger_class

        logger.info(f"Registered custom trigger type: {trigger_name}")

    def unregister_custom_trigger(self, trigger_name: str) -> bool:
        """
        Unregister a custom trigger type.

        Args:
            trigger_name: Name of the custom trigger to remove

        Returns:
            True if trigger was removed, False if not found
        """
        trigger_name = trigger_name.lower().strip()

        if trigger_name in self._custom_triggers:
            del self._custom_triggers[trigger_name]
            logger.info(f"Unregistered custom trigger type: {trigger_name}")
            return True

        return False

    def get_supported_trigger_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported trigger types.

        Returns:
            Dictionary mapping trigger names to their information
        """
        supported = {}

        # Built-in triggers
        for trigger_type, trigger_class in self.TRIGGER_REGISTRY.items():
            try:
                # Create a sample instance to get info (with minimal config)
                sample_config = self._get_sample_config(trigger_type)
                sample_trigger = trigger_class(sample_config)

                supported[trigger_type.name] = {
                    "type": "built-in",
                    "class": trigger_class.__name__,
                    "description": trigger_class.__doc__ or "No description available",
                    "sample_config": sample_config,
                }
            except Exception as e:
                supported[trigger_type.name] = {
                    "type": "built-in",
                    "class": trigger_class.__name__,
                    "description": "Configuration error",
                    "error": str(e),
                }

        # Custom triggers
        for trigger_name, trigger_class in self._custom_triggers.items():
            supported[trigger_name.upper()] = {
                "type": "custom",
                "class": trigger_class.__name__,
                "description": trigger_class.__doc__
                or "Custom trigger - no description available",
            }

        return supported

    def _get_sample_config(self, trigger_type: TriggerType) -> Dict[str, Any]:
        """Get sample configuration for a trigger type"""
        base_config = {"trigger_type": trigger_type.value}

        if trigger_type == TriggerType.CRON:
            return {**base_config, "expression": "0 9 * * *"}  # Daily at 9 AM
        elif trigger_type == TriggerType.INTERVAL:
            return {**base_config, "minutes": 30}  # Every 30 minutes
        elif trigger_type == TriggerType.DATE:
            from datetime import datetime, timedelta, timezone

            future_date = datetime.now(timezone.utc) + timedelta(hours=1)
            return {**base_config, "run_date": future_date.isoformat()}
        elif trigger_type == TriggerType.DEPENDENCY:
            return {**base_config, "depends_on": ["sample-job-id"]}
        else:
            return base_config

    def validate_trigger_config(
        self, trigger_config: Union[Dict[str, Any], Any]
    ) -> Dict[str, Any]:
        """
        Validate trigger configuration without creating instance.

        Args:
            trigger_config: Trigger configuration to validate

        Returns:
            Validation results with errors/warnings

        Raises:
            TriggerValidationError: If configuration is invalid
        """
        try:
            # Try to create trigger instance (validation happens in constructor)
            trigger = self.create_trigger(trigger_config)

            return {
                "valid": True,
                "trigger_type": trigger.config.get("trigger_type"),
                "info": trigger.get_trigger_info(),
                "warnings": [],
            }

        except TriggerValidationError as e:
            return {
                "valid": False,
                "error": str(e),
                "trigger_type": trigger_config.get("trigger_type")
                if isinstance(trigger_config, dict)
                else None,
            }
        except Exception as e:
            logger.error(
                f"Unexpected error validating trigger config: {e}", exc_info=True
            )
            return {
                "valid": False,
                "error": f"Unexpected validation error: {e}",
                "trigger_type": trigger_config.get("trigger_type")
                if isinstance(trigger_config, dict)
                else None,
            }

    def create_trigger_from_job_config(self, job_config: Dict[str, Any]) -> BaseTrigger:
        """
        Create trigger from job configuration.

        Convenience method for extracting trigger config from job definition.

        Args:
            job_config: Full job configuration containing trigger_config

        Returns:
            Configured trigger instance
        """
        if "trigger_config" not in job_config:
            raise TriggerValidationError("Job configuration missing 'trigger_config'")

        return self.create_trigger(job_config["trigger_config"])

    def __repr__(self) -> str:
        """String representation of factory"""
        builtin_count = len(self.TRIGGER_REGISTRY)
        custom_count = len(self._custom_triggers)
        return f"TriggerFactory(builtin={builtin_count}, custom={custom_count})"


# Global factory instance
_global_factory: TriggerFactory = None


def get_trigger_factory() -> TriggerFactory:
    """Get global trigger factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = TriggerFactory()
    return _global_factory
