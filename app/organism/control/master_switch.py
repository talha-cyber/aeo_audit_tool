"""
Master Control System for Organic Intelligence.

Single point of control to enable/disable all organic intelligence features
across the entire AEO Audit Tool while preserving normal operations.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.utils.logger import get_logger

logger = get_logger(__name__)


class OrganicMode(str, Enum):
    """Operating modes for organic intelligence system"""

    DISABLED = "disabled"  # Completely disabled
    MONITORING_ONLY = "monitoring"  # Monitor but don't act
    LEARNING_ONLY = "learning"  # Learn but don't heal
    HEALING_ONLY = "healing"  # Heal but don't learn
    FULL = "full"  # All features enabled


class FeatureCategory(str, Enum):
    """Categories of organic features"""

    MONITORING = "monitoring"
    LEARNING = "learning"
    HEALING = "healing"
    EVOLUTION = "evolution"
    PREDICTION = "prediction"
    COLLABORATION = "collaboration"


@dataclass
class ControlAction:
    """Record of control actions for audit trail"""

    timestamp: datetime
    action: str
    user: str
    reason: str
    feature: Optional[str] = None
    previous_state: Optional[Dict] = None
    new_state: Optional[Dict] = None


@dataclass
class OrganicStatus:
    """Current status of the organic intelligence system"""

    enabled: bool
    mode: OrganicMode
    active_features: Set[str]
    performance_impact: float
    uptime: float
    last_action: Optional[ControlAction]
    safe_to_disable: bool


class OrganicMasterControl:
    """
    Master control system for all organic intelligence features.

    Provides single-point control to enable/disable all organic features
    while ensuring zero impact on normal operations when disabled.
    """

    _instance: Optional["OrganicMasterControl"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OrganicMasterControl":
        """Singleton pattern for global control"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._enabled = self._load_initial_state()
        self._mode = OrganicMode.FULL if self._enabled else OrganicMode.DISABLED
        self._feature_registry: Dict[str, Any] = {}
        self._active_components: Set[str] = set()
        self._control_history: List[ControlAction] = []
        self._performance_monitor = PerformanceMonitor()
        self._config_path = Path("organism_config.json")
        self._startup_time = time.time()

        # Thread locks for safety
        self._registry_lock = threading.RLock()
        self._control_lock = threading.RLock()

        logger.info(f"Organic Master Control initialized - Enabled: {self._enabled}")

    def _load_initial_state(self) -> bool:
        """Load initial enabled state from various sources"""
        # Priority order: Environment -> Config file -> Default

        # 1. Environment variable (highest priority)
        env_value = os.getenv("ORGANIC_INTELLIGENCE_ENABLED", "").lower()
        if env_value in ("false", "0", "no", "off", "disabled"):
            logger.info("Organic intelligence disabled via environment variable")
            return False
        elif env_value in ("true", "1", "yes", "on", "enabled"):
            logger.info("Organic intelligence enabled via environment variable")
            return True

        # 2. Config file
        config_path = Path("organism_config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                enabled = config.get("organic", {}).get("enabled", True)
                logger.info(f"Organic intelligence state from config: {enabled}")
                return enabled
            except Exception as e:
                logger.warning(f"Failed to load organic config: {e}")

        # 3. Default (enabled)
        logger.info("Organic intelligence using default state: enabled")
        return True

    def _save_state(self):
        """Save current state to config file"""
        try:
            config = {
                "organic": {
                    "enabled": self._enabled,
                    "mode": self._mode.value,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                    "active_features": list(self._active_components),
                }
            }

            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save organic state: {e}")

    def is_enabled(self, feature: Optional[str] = None) -> bool:
        """
        Check if organic features are enabled globally or for a specific feature.

        Args:
            feature: Specific feature to check (None for global check)

        Returns:
            True if enabled, False otherwise
        """
        if not self._enabled:
            return False

        if feature is None:
            return True

        # Check if specific feature is enabled
        with self._registry_lock:
            feature_info = self._feature_registry.get(feature, {})
            return feature_info.get("enabled", True)

    def get_mode(self) -> OrganicMode:
        """Get current operating mode"""
        return self._mode

    def set_mode(
        self, mode: OrganicMode, user: str = "system", reason: str = "Mode change"
    ):
        """Set operating mode"""
        with self._control_lock:
            previous_mode = self._mode
            self._mode = mode

            # Adjust enabled state based on mode
            if mode == OrganicMode.DISABLED:
                self._enabled = False
            else:
                self._enabled = True

            self._log_action("set_mode", user, reason, new_state={"mode": mode.value})
            self._save_state()

            logger.info(
                f"Organic mode changed from {previous_mode.value} to {mode.value}"
            )

    async def disable_all(
        self, user: str = "system", reason: str = "Manual disable"
    ) -> bool:
        """
        Instantly disable all organic intelligence features.

        Args:
            user: User performing the action
            reason: Reason for disabling

        Returns:
            True if successfully disabled
        """
        with self._control_lock:
            if not self._enabled:
                logger.info("Organic intelligence already disabled")
                return True

            try:
                logger.info(
                    f"Disabling all organic intelligence features - Reason: {reason}"
                )

                previous_state = self.get_status()

                # 1. Stop all learning processes
                await self._stop_learning_processes()

                # 2. Stop all healing processes
                await self._stop_healing_processes()

                # 3. Stop monitoring (except performance monitoring)
                await self._stop_monitoring_processes()

                # 4. Clear active components
                self._active_components.clear()

                # 5. Set disabled state
                self._enabled = False
                self._mode = OrganicMode.DISABLED

                # 6. Log action
                self._log_action(
                    "disable_all", user, reason, previous_state=asdict(previous_state)
                )

                # 7. Save state
                self._save_state()

                logger.info("All organic intelligence features disabled successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to disable organic intelligence: {e}")
                return False

    async def enable_all(
        self, user: str = "system", reason: str = "Manual enable"
    ) -> bool:
        """
        Enable all organic intelligence features.

        Args:
            user: User performing the action
            reason: Reason for enabling

        Returns:
            True if successfully enabled
        """
        with self._control_lock:
            if self._enabled:
                logger.info("Organic intelligence already enabled")
                return True

            try:
                logger.info(
                    f"Enabling all organic intelligence features - Reason: {reason}"
                )

                # 1. Set enabled state
                self._enabled = True
                self._mode = OrganicMode.FULL

                # 2. Start monitoring processes
                await self._start_monitoring_processes()

                # 3. Start learning processes
                await self._start_learning_processes()

                # 4. Start healing processes
                await self._start_healing_processes()

                # 5. Log action
                self._log_action("enable_all", user, reason)

                # 6. Save state
                self._save_state()

                logger.info("All organic intelligence features enabled successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to enable organic intelligence: {e}")
                return False

    async def emergency_shutdown(
        self, user: str = "system", reason: str = "Emergency"
    ) -> bool:
        """
        Emergency shutdown - immediate stop without graceful cleanup.

        Args:
            user: User performing emergency shutdown
            reason: Emergency reason

        Returns:
            True if shutdown successful
        """
        logger.warning(f"EMERGENCY SHUTDOWN initiated by {user} - Reason: {reason}")

        try:
            # Immediate disable - no graceful cleanup
            self._enabled = False
            self._mode = OrganicMode.DISABLED
            self._active_components.clear()

            # Log emergency action
            self._log_action("emergency_shutdown", user, reason)

            # Force save state
            self._save_state()

            logger.warning("Emergency shutdown completed")
            return True

        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
            return False

    def register_component(
        self,
        component_name: str,
        category: FeatureCategory,
        component_instance: Any = None,
    ) -> bool:
        """
        Register an organic component.

        Args:
            component_name: Name of the component
            category: Category of the feature
            component_instance: Instance of the component

        Returns:
            True if registered successfully
        """
        with self._registry_lock:
            try:
                self._feature_registry[component_name] = {
                    "category": category.value,
                    "instance": component_instance,
                    "enabled": True,
                    "registered_at": time.time(),
                }

                if self._enabled:
                    self._active_components.add(component_name)

                logger.debug(f"Registered organic component: {component_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to register component {component_name}: {e}")
                return False

    def unregister_component(self, component_name: str) -> bool:
        """Unregister an organic component"""
        with self._registry_lock:
            try:
                if component_name in self._feature_registry:
                    del self._feature_registry[component_name]

                self._active_components.discard(component_name)

                logger.debug(f"Unregistered organic component: {component_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to unregister component {component_name}: {e}")
                return False

    def get_status(self) -> OrganicStatus:
        """Get comprehensive status of organic intelligence system"""
        uptime = time.time() - self._startup_time
        performance_impact = self._performance_monitor.get_current_overhead()

        return OrganicStatus(
            enabled=self._enabled,
            mode=self._mode,
            active_features=self._active_components.copy(),
            performance_impact=performance_impact,
            uptime=uptime,
            last_action=self._control_history[-1] if self._control_history else None,
            safe_to_disable=self._is_safe_to_disable(),
        )

    def get_registered_features(self) -> Dict[str, Dict]:
        """Get all registered features and their status"""
        with self._registry_lock:
            return self._feature_registry.copy()

    def get_control_history(self, limit: int = 100) -> List[ControlAction]:
        """Get control action history"""
        return self._control_history[-limit:]

    def _log_action(
        self,
        action: str,
        user: str,
        reason: str,
        feature: Optional[str] = None,
        previous_state: Optional[Dict] = None,
        new_state: Optional[Dict] = None,
    ):
        """Log control action for audit trail"""
        control_action = ControlAction(
            timestamp=datetime.now(timezone.utc),
            action=action,
            user=user,
            reason=reason,
            feature=feature,
            previous_state=previous_state,
            new_state=new_state,
        )

        self._control_history.append(control_action)

        # Keep only last 1000 actions
        if len(self._control_history) > 1000:
            self._control_history = self._control_history[-1000:]

    def _is_safe_to_disable(self) -> bool:
        """Check if it's safe to disable organic features"""
        # Always safe to disable - organic features are non-critical
        return True

    async def _stop_learning_processes(self):
        """Stop all learning processes"""
        # Implementation will be added when learning components are created
        pass

    async def _stop_healing_processes(self):
        """Stop all healing processes"""
        # Implementation will be added when healing components are created
        pass

    async def _stop_monitoring_processes(self):
        """Stop monitoring processes (except performance monitoring)"""
        # Implementation will be added when monitoring components are created
        pass

    async def _start_monitoring_processes(self):
        """Start monitoring processes"""
        # Implementation will be added when monitoring components are created
        pass

    async def _start_learning_processes(self):
        """Start learning processes"""
        # Implementation will be added when learning components are created
        pass

    async def _start_healing_processes(self):
        """Start healing processes"""
        # Implementation will be added when healing components are created
        pass


class PerformanceMonitor:
    """Monitor performance impact of organic intelligence features"""

    def __init__(self):
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.overhead_threshold = 10.0  # 10% overhead limit

    def get_current_overhead(self) -> float:
        """Get current performance overhead percentage"""
        # This will be implemented with actual performance metrics
        return 0.0

    def check_overhead_threshold(self) -> bool:
        """Check if overhead exceeds threshold"""
        return self.get_current_overhead() > self.overhead_threshold


# Global instance
_organic_control_instance: Optional[OrganicMasterControl] = None
_control_lock = threading.Lock()


def get_organic_control() -> OrganicMasterControl:
    """Get the global organic control instance"""
    global _organic_control_instance

    if _organic_control_instance is None:
        with _control_lock:
            if _organic_control_instance is None:
                _organic_control_instance = OrganicMasterControl()

    return _organic_control_instance
