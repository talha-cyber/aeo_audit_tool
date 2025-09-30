"""
Organic Feature Registry and Bypass Layer.

Manages registration of all organic intelligence features and provides
bypass mechanisms when features are disabled.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from app.utils.logger import get_logger

from .master_switch import FeatureCategory

logger = get_logger(__name__)


@dataclass
class FeatureInfo:
    """Information about a registered organic feature"""
    name: str
    category: FeatureCategory
    instance: Optional[Any] = None
    enabled: bool = True
    registered_at: float = field(default_factory=time.time)
    usage_count: int = 0
    last_used: float = 0.0
    performance_metrics: Dict = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class BypassRoute:
    """Bypass route for when organic features are disabled"""
    original_function: Callable
    bypass_function: Optional[Callable] = None
    fallback_behavior: str = "passthrough"  # passthrough, cached, default


class OrganicFeatureRegistry:
    """
    Registry for all organic intelligence features with bypass capabilities.

    Manages feature registration, dependency tracking, and bypass routing
    when organic intelligence is disabled.
    """

    def __init__(self):
        self._features: Dict[str, FeatureInfo] = {}
        self._categories: Dict[FeatureCategory, Set[str]] = defaultdict(set)
        self._bypass_routes: Dict[str, BypassRoute] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._performance_baseline: Dict[str, float] = {}
        self._performance_current: Dict[str, float] = {}

    def register_feature(
        self,
        name: str,
        category: FeatureCategory,
        instance: Optional[Any] = None,
        dependencies: Optional[Set[str]] = None,
        bypass_function: Optional[Callable] = None,
        enabled: bool = True
    ) -> bool:
        """
        Register an organic intelligence feature.

        Args:
            name: Unique name of the feature
            category: Category of the feature
            instance: Instance of the feature component
            dependencies: Set of feature names this feature depends on
            bypass_function: Function to use when feature is disabled
            enabled: Whether the feature is initially enabled

        Returns:
            True if registration successful
        """
        with self._lock:
            try:
                if name in self._features:
                    logger.warning(f"Feature {name} already registered, updating")

                # Create feature info
                feature_info = FeatureInfo(
                    name=name,
                    category=category,
                    instance=instance,
                    enabled=enabled,
                    dependencies=dependencies or set()
                )

                # Register feature
                self._features[name] = feature_info
                self._categories[category].add(name)

                # Handle dependencies
                if dependencies:
                    self._dependency_graph[name] = dependencies
                    for dep in dependencies:
                        if dep in self._features:
                            self._features[dep].dependents.add(name)

                # Create bypass route if provided
                if bypass_function:
                    self._bypass_routes[name] = BypassRoute(
                        original_function=instance,
                        bypass_function=bypass_function
                    )

                logger.debug(f"Registered organic feature: {name} ({category.value})")
                return True

            except Exception as e:
                logger.error(f"Failed to register feature {name}: {e}")
                return False

    def unregister_feature(self, name: str) -> bool:
        """
        Unregister an organic intelligence feature.

        Args:
            name: Name of the feature to unregister

        Returns:
            True if unregistration successful
        """
        with self._lock:
            try:
                if name not in self._features:
                    logger.warning(f"Feature {name} not found for unregistration")
                    return False

                feature_info = self._features[name]

                # Remove from category
                self._categories[feature_info.category].discard(name)

                # Handle dependencies
                for dep in feature_info.dependencies:
                    if dep in self._features:
                        self._features[dep].dependents.discard(name)

                for dependent in feature_info.dependents:
                    if dependent in self._features:
                        self._features[dependent].dependencies.discard(name)

                # Remove from dependency graph
                if name in self._dependency_graph:
                    del self._dependency_graph[name]

                # Remove bypass route
                if name in self._bypass_routes:
                    del self._bypass_routes[name]

                # Remove feature
                del self._features[name]

                logger.debug(f"Unregistered organic feature: {name}")
                return True

            except Exception as e:
                logger.error(f"Failed to unregister feature {name}: {e}")
                return False

    def is_feature_enabled(self, name: str) -> bool:
        """Check if a specific feature is enabled"""
        with self._lock:
            if name not in self._features:
                return False
            return self._features[name].enabled

    def enable_feature(self, name: str) -> bool:
        """Enable a specific feature"""
        with self._lock:
            if name not in self._features:
                logger.warning(f"Cannot enable unknown feature: {name}")
                return False

            # Check dependencies
            for dep in self._features[name].dependencies:
                if not self.is_feature_enabled(dep):
                    logger.warning(f"Cannot enable {name}: dependency {dep} is disabled")
                    return False

            self._features[name].enabled = True
            logger.info(f"Enabled organic feature: {name}")
            return True

    def disable_feature(self, name: str, cascade: bool = True) -> bool:
        """
        Disable a specific feature.

        Args:
            name: Name of the feature to disable
            cascade: Whether to disable dependent features

        Returns:
            True if disable successful
        """
        with self._lock:
            if name not in self._features:
                logger.warning(f"Cannot disable unknown feature: {name}")
                return False

            # Disable dependents if cascade is enabled
            if cascade:
                for dependent in self._features[name].dependents:
                    self.disable_feature(dependent, cascade=True)

            self._features[name].enabled = False
            logger.info(f"Disabled organic feature: {name}")
            return True

    def enable_category(self, category: FeatureCategory) -> int:
        """
        Enable all features in a category.

        Args:
            category: Category to enable

        Returns:
            Number of features enabled
        """
        enabled_count = 0
        for feature_name in self._categories[category]:
            if self.enable_feature(feature_name):
                enabled_count += 1
        return enabled_count

    def disable_category(self, category: FeatureCategory) -> int:
        """
        Disable all features in a category.

        Args:
            category: Category to disable

        Returns:
            Number of features disabled
        """
        disabled_count = 0
        for feature_name in self._categories[category]:
            if self.disable_feature(feature_name):
                disabled_count += 1
        return disabled_count

    def get_feature_info(self, name: str) -> Optional[FeatureInfo]:
        """Get information about a specific feature"""
        with self._lock:
            return self._features.get(name)

    def get_features_by_category(self, category: FeatureCategory) -> List[FeatureInfo]:
        """Get all features in a category"""
        with self._lock:
            return [
                self._features[name]
                for name in self._categories[category]
                if name in self._features
            ]

    def get_all_features(self) -> Dict[str, FeatureInfo]:
        """Get all registered features"""
        with self._lock:
            return self._features.copy()

    def get_enabled_features(self) -> List[str]:
        """Get list of enabled feature names"""
        with self._lock:
            return [
                name for name, info in self._features.items()
                if info.enabled
            ]

    def get_dependency_chain(self, name: str) -> List[str]:
        """Get dependency chain for a feature"""
        with self._lock:
            visited = set()
            chain = []

            def _build_chain(feature_name: str):
                if feature_name in visited:
                    return
                visited.add(feature_name)

                if feature_name in self._features:
                    for dep in self._features[feature_name].dependencies:
                        _build_chain(dep)
                    chain.append(feature_name)

            _build_chain(name)
            return chain

    def validate_dependencies(self) -> List[str]:
        """
        Validate all feature dependencies.

        Returns:
            List of validation errors
        """
        errors = []

        with self._lock:
            for name, info in self._features.items():
                for dep in info.dependencies:
                    if dep not in self._features:
                        errors.append(f"Feature {name} depends on unknown feature {dep}")
                    elif not self._features[dep].enabled and info.enabled:
                        errors.append(f"Feature {name} is enabled but dependency {dep} is disabled")

        return errors

    def record_usage(self, name: str, performance_metric: Optional[float] = None):
        """Record usage of a feature"""
        with self._lock:
            if name in self._features:
                self._features[name].usage_count += 1
                self._features[name].last_used = time.time()

                if performance_metric:
                    metrics = self._features[name].performance_metrics
                    metrics.setdefault('total_time', 0.0)
                    metrics.setdefault('call_count', 0)
                    metrics['total_time'] += performance_metric
                    metrics['call_count'] += 1
                    metrics['avg_time'] = metrics['total_time'] / metrics['call_count']

    def get_bypass_route(self, name: str) -> Optional[BypassRoute]:
        """Get bypass route for a feature"""
        return self._bypass_routes.get(name)

    def create_bypass_function(self, original_func: Callable, fallback_value: Any = None) -> Callable:
        """
        Create a bypass function that returns fallback value or passes through.

        Args:
            original_func: Original function
            fallback_value: Value to return when bypassed

        Returns:
            Bypass function
        """
        def bypass_func(*args, **kwargs):
            if fallback_value is not None:
                return fallback_value
            # Simple passthrough
            return original_func(*args, **kwargs)

        return bypass_func

    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all features"""
        with self._lock:
            summary = {}
            for name, info in self._features.items():
                if info.performance_metrics:
                    summary[name] = {
                        'usage_count': info.usage_count,
                        'last_used': info.last_used,
                        'avg_time': info.performance_metrics.get('avg_time', 0.0),
                        'total_time': info.performance_metrics.get('total_time', 0.0),
                        'enabled': info.enabled
                    }
            return summary

    def cleanup_unused_features(self, min_age_hours: float = 24.0) -> List[str]:
        """
        Clean up features that haven't been used recently.

        Args:
            min_age_hours: Minimum age in hours before cleanup

        Returns:
            List of cleaned up feature names
        """
        current_time = time.time()
        min_age_seconds = min_age_hours * 3600
        cleaned_up = []

        with self._lock:
            for name, info in list(self._features.items()):
                if (current_time - info.last_used > min_age_seconds and
                    info.usage_count == 0):

                    if self.unregister_feature(name):
                        cleaned_up.append(name)

        if cleaned_up:
            logger.info(f"Cleaned up unused organic features: {cleaned_up}")

        return cleaned_up


# Global registry instance
_feature_registry: Optional[OrganicFeatureRegistry] = None
_registry_lock = threading.Lock()


def get_feature_registry() -> OrganicFeatureRegistry:
    """Get the global feature registry instance"""
    global _feature_registry

    if _feature_registry is None:
        with _registry_lock:
            if _feature_registry is None:
                _feature_registry = OrganicFeatureRegistry()

    return _feature_registry