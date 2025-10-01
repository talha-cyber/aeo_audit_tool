"""
Bypass Controller for Organic Intelligence Features.

Routes traffic around organic features when disabled, ensuring
zero performance impact and identical behavior to original system.
"""

import functools
import inspect
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from app.utils.logger import get_logger

from .master_switch import get_organic_control

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class BypassController:
    """
    Controls traffic routing around organic intelligence features.

    When organic intelligence is disabled, this controller ensures
    that all traffic bypasses organic features with zero overhead.
    """

    def __init__(self):
        self._bypass_cache: Dict[str, Callable] = {}
        self._original_functions: Dict[str, Callable] = {}
        self._lock = threading.RLock()

    def create_bypass_wrapper(
        self,
        feature_name: str,
        original_func: Callable,
        bypass_func: Optional[Callable] = None,
        cache_result: bool = False,
    ) -> Callable:
        """
        Create a bypass wrapper for a function.

        Args:
            feature_name: Name of the organic feature
            original_func: Original function to wrap
            bypass_func: Custom bypass function (optional)
            cache_result: Whether to cache bypass results

        Returns:
            Wrapper function that can bypass organic features
        """
        with self._lock:
            # Store original function
            self._original_functions[feature_name] = original_func

            if inspect.iscoroutinefunction(original_func):
                return self._create_async_bypass_wrapper(
                    feature_name, original_func, bypass_func, cache_result
                )
            else:
                return self._create_sync_bypass_wrapper(
                    feature_name, original_func, bypass_func, cache_result
                )

    def _create_async_bypass_wrapper(
        self,
        feature_name: str,
        original_func: Callable,
        bypass_func: Optional[Callable],
        cache_result: bool,
    ) -> Callable:
        """Create async bypass wrapper"""

        @functools.wraps(original_func)
        async def async_bypass_wrapper(*args, **kwargs):
            control = get_organic_control()

            # Fast path: if organic intelligence is disabled, direct call
            if not control.is_enabled(feature_name):
                if bypass_func:
                    if inspect.iscoroutinefunction(bypass_func):
                        return await bypass_func(*args, **kwargs)
                    else:
                        return bypass_func(*args, **kwargs)
                else:
                    return await original_func(*args, **kwargs)

            # Organic intelligence is enabled, proceed with enhanced version
            try:
                if bypass_func:
                    # Use bypass function as enhanced version
                    if inspect.iscoroutinefunction(bypass_func):
                        result = await bypass_func(*args, **kwargs)
                    else:
                        result = bypass_func(*args, **kwargs)
                else:
                    result = await original_func(*args, **kwargs)

                # Cache result if requested
                if cache_result:
                    self._cache_result(feature_name, args, kwargs, result)

                return result

            except Exception as e:
                logger.warning(f"Organic feature {feature_name} failed: {e}")
                # Fallback to original function
                return await original_func(*args, **kwargs)

        return async_bypass_wrapper

    def _create_sync_bypass_wrapper(
        self,
        feature_name: str,
        original_func: Callable,
        bypass_func: Optional[Callable],
        cache_result: bool,
    ) -> Callable:
        """Create synchronous bypass wrapper"""

        @functools.wraps(original_func)
        def sync_bypass_wrapper(*args, **kwargs):
            control = get_organic_control()

            # Fast path: if organic intelligence is disabled, direct call
            if not control.is_enabled(feature_name):
                if bypass_func:
                    return bypass_func(*args, **kwargs)
                else:
                    return original_func(*args, **kwargs)

            # Organic intelligence is enabled, proceed with enhanced version
            try:
                if bypass_func:
                    result = bypass_func(*args, **kwargs)
                else:
                    result = original_func(*args, **kwargs)

                # Cache result if requested
                if cache_result:
                    self._cache_result(feature_name, args, kwargs, result)

                return result

            except Exception as e:
                logger.warning(f"Organic feature {feature_name} failed: {e}")
                # Fallback to original function
                return original_func(*args, **kwargs)

        return sync_bypass_wrapper

    def _cache_result(self, feature_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache result for bypass optimization"""
        # Simple caching implementation - can be enhanced
        cache_key = (
            f"{feature_name}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
        )
        self._bypass_cache[cache_key] = result

        # Limit cache size
        if len(self._bypass_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._bypass_cache.keys())[:100]
            for key in oldest_keys:
                del self._bypass_cache[key]

    def get_cached_result(
        self, feature_name: str, args: tuple, kwargs: dict
    ) -> Optional[Any]:
        """Get cached result if available"""
        cache_key = (
            f"{feature_name}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
        )
        return self._bypass_cache.get(cache_key)

    def clear_cache(self, feature_name: Optional[str] = None):
        """Clear bypass cache"""
        with self._lock:
            if feature_name:
                # Clear cache for specific feature
                keys_to_remove = [
                    k for k in self._bypass_cache.keys() if k.startswith(feature_name)
                ]
                for key in keys_to_remove:
                    del self._bypass_cache[key]
            else:
                # Clear all cache
                self._bypass_cache.clear()

    def get_original_function(self, feature_name: str) -> Optional[Callable]:
        """Get original function for a feature"""
        return self._original_functions.get(feature_name)

    def is_bypassed(self, feature_name: str) -> bool:
        """Check if a feature is currently bypassed"""
        control = get_organic_control()
        return not control.is_enabled(feature_name)

    def get_bypass_stats(self) -> Dict[str, Any]:
        """Get bypass statistics"""
        return {
            "cached_results": len(self._bypass_cache),
            "registered_functions": len(self._original_functions),
            "memory_usage_bytes": self._estimate_cache_size(),
        }

    def _estimate_cache_size(self) -> int:
        """Estimate cache memory usage"""
        # Simple estimation - can be enhanced
        return len(self._bypass_cache) * 100  # Rough estimate


class TrafficRouter:
    """
    Routes traffic between original and organic implementations.

    Provides intelligent routing based on system state, performance,
    and feature availability.
    """

    def __init__(self):
        self.bypass_controller = BypassController()
        self._routing_stats: Dict[str, Dict] = {}

    def route_call(
        self,
        feature_name: str,
        original_func: Callable,
        enhanced_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Route function call based on organic intelligence state.

        Args:
            feature_name: Name of the feature
            original_func: Original function
            enhanced_func: Enhanced function with organic intelligence
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        control = get_organic_control()
        start_time = time.time()

        try:
            if not control.is_enabled(feature_name):
                # Route to original function
                result = original_func(*args, **kwargs)
                self._record_routing("original", feature_name, time.time() - start_time)
                return result
            else:
                # Route to enhanced function
                if enhanced_func:
                    result = enhanced_func(*args, **kwargs)
                    self._record_routing(
                        "enhanced", feature_name, time.time() - start_time
                    )
                    return result
                else:
                    result = original_func(*args, **kwargs)
                    self._record_routing(
                        "original", feature_name, time.time() - start_time
                    )
                    return result

        except Exception as e:
            # Always fallback to original on error
            logger.warning(
                f"Enhanced function failed for {feature_name}, falling back: {e}"
            )
            result = original_func(*args, **kwargs)
            self._record_routing("fallback", feature_name, time.time() - start_time)
            return result

    def _record_routing(self, route_type: str, feature_name: str, duration: float):
        """Record routing statistics"""
        if feature_name not in self._routing_stats:
            self._routing_stats[feature_name] = {
                "original": {"count": 0, "total_time": 0.0},
                "enhanced": {"count": 0, "total_time": 0.0},
                "fallback": {"count": 0, "total_time": 0.0},
            }

        stats = self._routing_stats[feature_name][route_type]
        stats["count"] += 1
        stats["total_time"] += duration

    def get_routing_stats(self) -> Dict[str, Dict]:
        """Get routing statistics"""
        return self._routing_stats.copy()

    def reset_stats(self):
        """Reset routing statistics"""
        self._routing_stats.clear()


# Global instances
_bypass_controller: Optional[BypassController] = None
_traffic_router: Optional[TrafficRouter] = None
_controller_lock = threading.Lock()


def get_bypass_controller() -> BypassController:
    """Get global bypass controller instance"""
    global _bypass_controller

    if _bypass_controller is None:
        with _controller_lock:
            if _bypass_controller is None:
                _bypass_controller = BypassController()

    return _bypass_controller


def get_traffic_router() -> TrafficRouter:
    """Get global traffic router instance"""
    global _traffic_router

    if _traffic_router is None:
        with _controller_lock:
            if _traffic_router is None:
                _traffic_router = TrafficRouter()

    return _traffic_router


# Convenience functions


def create_bypass_wrapper(feature_name: str, original_func: Callable) -> Callable:
    """Create bypass wrapper for a function"""
    controller = get_bypass_controller()
    return controller.create_bypass_wrapper(feature_name, original_func)


def route_function_call(
    feature_name: str, original_func: Callable, *args, **kwargs
) -> Any:
    """Route function call through traffic router"""
    router = get_traffic_router()
    return router.route_call(feature_name, original_func, None, *args, **kwargs)
