"""
Decorators for non-invasive organic intelligence integration.

These decorators allow adding organic features to existing functions
without modifying the original code, and automatically bypass
organic features when the system is disabled.
"""

import functools
import inspect
import time
from typing import Any, Callable, Optional, TypeVar

from app.utils.logger import get_logger

from .master_switch import FeatureCategory, get_organic_control

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def organic_enhancement(
    feature: str,
    category: FeatureCategory = FeatureCategory.MONITORING,
    graceful_fallback: bool = True,
    performance_tracking: bool = True,
    error_handling: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to add organic intelligence enhancement to any function.

    This decorator wraps existing functions with organic intelligence features
    while ensuring zero impact when the system is disabled.

    Args:
        feature: Name of the organic feature
        category: Category of the feature
        graceful_fallback: Whether to gracefully fallback on errors
        performance_tracking: Whether to track performance metrics
        error_handling: Whether to handle errors organically

    Returns:
        Decorated function that works with or without organic intelligence
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            control = get_organic_control()

            # Quick check - if disabled, direct passthrough
            if not control.is_enabled(feature):
                return await func(*args, **kwargs)

            # Organic enhancement is enabled
            start_time = time.time() if performance_tracking else None

            try:
                # Pre-processing organic enhancement
                enhanced_args, enhanced_kwargs = await _pre_process_organic(
                    feature, args, kwargs
                )

                # Execute original function
                result = await func(*enhanced_args, **enhanced_kwargs)

                # Post-processing organic enhancement
                enhanced_result = await _post_process_organic(
                    feature, result, args, kwargs
                )

                # Track performance if enabled
                if performance_tracking and start_time:
                    duration = time.time() - start_time
                    await _track_performance(feature, duration, True)

                return enhanced_result

            except Exception as e:
                if error_handling:
                    await _handle_organic_error(feature, e, args, kwargs)

                if graceful_fallback:
                    logger.warning(
                        f"Organic feature {feature} failed, falling back to original"
                    )
                    return await func(*args, **kwargs)
                else:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            control = get_organic_control()

            # Quick check - if disabled, direct passthrough
            if not control.is_enabled(feature):
                return func(*args, **kwargs)

            # Organic enhancement is enabled
            start_time = time.time() if performance_tracking else None

            try:
                # Pre-processing (synchronous version)
                enhanced_args, enhanced_kwargs = _pre_process_organic_sync(
                    feature, args, kwargs
                )

                # Execute original function
                result = func(*enhanced_args, **enhanced_kwargs)

                # Post-processing (synchronous version)
                enhanced_result = _post_process_organic_sync(
                    feature, result, args, kwargs
                )

                # Track performance if enabled
                if performance_tracking and start_time:
                    duration = time.time() - start_time
                    _track_performance_sync(feature, duration, True)

                return enhanced_result

            except Exception as e:
                if error_handling:
                    _handle_organic_error_sync(feature, e, args, kwargs)

                if graceful_fallback:
                    logger.warning(
                        f"Organic feature {feature} failed, falling back to original"
                    )
                    return func(*args, **kwargs)
                else:
                    raise

        # Register the feature
        control = get_organic_control()
        control.register_component(feature, category)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def organic_wrapper(
    enabled_check: bool = True,
    performance_tracking: bool = False,
    feature_name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Simple wrapper that adds organic intelligence to existing functions.

    This is a lighter version of organic_enhancement for simple use cases.

    Args:
        enabled_check: Whether to check if organic intelligence is enabled
        performance_tracking: Whether to track performance
        feature_name: Name of the feature (auto-generated if None)

    Returns:
        Wrapped function
    """

    def decorator(func: F) -> F:
        feature = feature_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enabled_check:
                control = get_organic_control()
                if not control.is_enabled(feature):
                    # Direct passthrough - zero overhead
                    return func(*args, **kwargs)

            # Add minimal organic intelligence
            start_time = time.time() if performance_tracking else None

            try:
                result = func(*args, **kwargs)

                if performance_tracking and start_time:
                    duration = time.time() - start_time
                    logger.debug(f"Function {feature} took {duration:.4f}s")

                return result

            except Exception as e:
                logger.debug(f"Function {feature} failed: {e}")
                raise

        return wrapper

    return decorator


def register_organic_feature(
    feature_name: str,
    category: FeatureCategory = FeatureCategory.MONITORING,
    auto_register: bool = True,
) -> Callable[[type], type]:
    """
    Class decorator to register organic intelligence features.

    Args:
        feature_name: Name of the organic feature
        category: Category of the feature
        auto_register: Whether to auto-register the feature

    Returns:
        Decorated class
    """

    def decorator(cls: type) -> type:
        if auto_register:
            # Add enabled check to the class
            original_init = cls.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                control = get_organic_control()

                # Check if organic intelligence is enabled
                if not control.is_enabled(feature_name):
                    self._organic_enabled = False
                    # Minimal initialization
                    return
                else:
                    self._organic_enabled = True

                # Register the feature
                control.register_component(feature_name, category, self)

                # Call original init
                original_init(self, *args, **kwargs)

            cls.__init__ = new_init

            # Add convenience method to check if enabled
            def is_organic_enabled(self) -> bool:
                return getattr(self, "_organic_enabled", False)

            cls.is_organic_enabled = is_organic_enabled

        return cls

    return decorator


# Helper functions for organic processing


async def _pre_process_organic(feature: str, args: tuple, kwargs: dict) -> tuple:
    """Pre-process function arguments with organic intelligence"""
    # This will be enhanced as organic components are added
    return args, kwargs


async def _post_process_organic(
    feature: str, result: Any, args: tuple, kwargs: dict
) -> Any:
    """Post-process function result with organic intelligence"""
    # This will be enhanced as organic components are added
    return result


async def _handle_organic_error(
    feature: str, error: Exception, args: tuple, kwargs: dict
):
    """Handle errors with organic intelligence"""
    # This will be enhanced with learning and adaptation
    logger.debug(f"Organic error in {feature}: {error}")


async def _track_performance(feature: str, duration: float, success: bool):
    """Track performance metrics for organic intelligence"""
    # This will be enhanced with performance analytics
    pass


# Synchronous versions of helper functions


def _pre_process_organic_sync(feature: str, args: tuple, kwargs: dict) -> tuple:
    """Synchronous version of pre-processing"""
    return args, kwargs


def _post_process_organic_sync(
    feature: str, result: Any, args: tuple, kwargs: dict
) -> Any:
    """Synchronous version of post-processing"""
    return result


def _handle_organic_error_sync(
    feature: str, error: Exception, args: tuple, kwargs: dict
):
    """Synchronous version of error handling"""
    logger.debug(f"Organic error in {feature}: {error}")


def _track_performance_sync(feature: str, duration: float, success: bool):
    """Synchronous version of performance tracking"""
    pass


# Utility functions


def is_organic_enabled(feature: Optional[str] = None) -> bool:
    """Check if organic intelligence is enabled for a feature"""
    control = get_organic_control()
    return control.is_enabled(feature)


def get_organic_status() -> dict:
    """Get current organic intelligence status"""
    control = get_organic_control()
    status = control.get_status()
    return {
        "enabled": status.enabled,
        "mode": status.mode.value,
        "active_features": list(status.active_features),
        "performance_impact": status.performance_impact,
        "uptime": status.uptime,
    }
