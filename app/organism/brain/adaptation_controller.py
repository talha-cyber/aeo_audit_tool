"""
Adaptation Controller for Organic Intelligence.

Executes system adaptations and healing actions based on decisions
made by the central intelligence system.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.organism.control.decorators import register_organic_feature
from app.organism.control.master_switch import FeatureCategory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptationStatus(str, Enum):
    """Status of adaptation execution"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HealingType(str, Enum):
    """Types of healing actions"""

    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_CLEANUP = "resource_cleanup"
    SYSTEM_RESTART = "system_restart"
    CONFIGURATION_REPAIR = "configuration_repair"


@dataclass
class AdaptationExecution:
    """Execution context for an adaptation"""

    id: str
    plan_id: str
    status: AdaptationStatus
    start_time: datetime
    end_time: Optional[datetime]
    steps_completed: List[str]
    current_step: Optional[str]
    error_message: Optional[str]
    rollback_executed: bool = False
    metrics: Dict[str, Any] = None


@dataclass
class HealingAction:
    """A healing action to be executed"""

    id: str
    healing_type: HealingType
    target_component: str
    action_function: Callable
    parameters: Dict[str, Any]
    priority: int
    estimated_duration: float
    prerequisites: List[str]
    success_criteria: Dict[str, Any]


@register_organic_feature("adaptation_controller", FeatureCategory.LEARNING)
class AdaptationController:
    """
    Controls execution of system adaptations and healing actions.

    Manages the execution of adaptation plans created by the decision engine
    and performs healing actions to maintain system health.
    """

    def __init__(self):
        if not self.is_organic_enabled():
            return

        # Execution tracking
        self._active_executions: Dict[str, AdaptationExecution] = {}
        self._execution_history: List[AdaptationExecution] = []
        self._healing_actions: Dict[str, HealingAction] = {}

        # Execution control
        self._max_concurrent_adaptations = 3
        self._execution_timeout = 300  # 5 minutes default

        # Threading
        self._execution_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()

        # Metrics
        self._metrics = {
            "adaptations_executed": 0,
            "adaptations_successful": 0,
            "adaptations_failed": 0,
            "adaptations_rolled_back": 0,
            "healing_actions_performed": 0,
            "average_execution_time": 0.0,
        }

        # Built-in healing actions
        self._register_default_healing_actions()

        logger.info("Adaptation Controller initialized")

    async def initialize(self):
        """Initialize adaptation controller"""
        if not self.is_organic_enabled():
            return

        try:
            # Start execution thread
            self._running = True
            self._execution_thread = threading.Thread(
                target=self._execution_loop, daemon=True, name="AdaptationController"
            )
            self._execution_thread.start()

            logger.info("Adaptation Controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Adaptation Controller: {e}")
            raise

    def _register_default_healing_actions(self):
        """Register default healing actions"""
        try:
            # Error recovery action
            self._healing_actions["error_recovery"] = HealingAction(
                id="error_recovery",
                healing_type=HealingType.ERROR_RECOVERY,
                target_component="system",
                action_function=self._perform_error_recovery,
                parameters={},
                priority=1,
                estimated_duration=30.0,
                prerequisites=[],
                success_criteria={"error_rate_reduction": 0.5},
            )

            # Performance optimization action
            self._healing_actions["performance_optimization"] = HealingAction(
                id="performance_optimization",
                healing_type=HealingType.PERFORMANCE_OPTIMIZATION,
                target_component="system",
                action_function=self._perform_performance_optimization,
                parameters={},
                priority=2,
                estimated_duration=60.0,
                prerequisites=[],
                success_criteria={"performance_improvement": 0.2},
            )

            # Resource cleanup action
            self._healing_actions["resource_cleanup"] = HealingAction(
                id="resource_cleanup",
                healing_type=HealingType.RESOURCE_CLEANUP,
                target_component="memory",
                action_function=self._perform_resource_cleanup,
                parameters={},
                priority=3,
                estimated_duration=15.0,
                prerequisites=[],
                success_criteria={"memory_freed": 0.1},
            )

        except Exception as e:
            logger.error(f"Failed to register default healing actions: {e}")

    def _execution_loop(self):
        """Main execution processing loop"""
        logger.info("Adaptation execution loop started")

        while self._running:
            try:
                time.sleep(5)  # Check every 5 seconds

                if not self._running:
                    break

                # Process pending executions
                asyncio.run(self._process_pending_executions())

                # Check for timeout executions
                asyncio.run(self._check_execution_timeouts())

            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(10)

        logger.info("Adaptation execution loop stopped")

    async def _process_pending_executions(self):
        """Process pending adaptation executions"""
        try:
            # Count active executions
            active_count = sum(
                1
                for exec in self._active_executions.values()
                if exec.status == AdaptationStatus.IN_PROGRESS
            )

            if active_count >= self._max_concurrent_adaptations:
                return

            # Find pending executions
            pending_executions = [
                exec
                for exec in self._active_executions.values()
                if exec.status == AdaptationStatus.PENDING
            ]

            # Start new executions up to limit
            for execution in pending_executions[
                : self._max_concurrent_adaptations - active_count
            ]:
                await self._start_execution(execution)

        except Exception as e:
            logger.error(f"Error processing pending executions: {e}")

    async def _start_execution(self, execution: AdaptationExecution):
        """Start an adaptation execution"""
        try:
            with self._lock:
                execution.status = AdaptationStatus.IN_PROGRESS
                execution.start_time = datetime.now(timezone.utc)

            logger.info(f"Starting adaptation execution: {execution.id}")

            # Execute in background
            asyncio.create_task(self._execute_adaptation(execution))

        except Exception as e:
            logger.error(f"Failed to start execution {execution.id}: {e}")
            execution.status = AdaptationStatus.FAILED
            execution.error_message = str(e)

    async def _execute_adaptation(self, execution: AdaptationExecution):
        """Execute an adaptation plan"""
        try:
            # Get the adaptation plan (this would need to be retrieved from storage)
            # For now, we'll simulate the execution

            steps = [
                "prepare_environment",
                "backup_current_state",
                "apply_changes",
                "validate_changes",
                "cleanup",
            ]

            for step in steps:
                if not self._running:
                    break

                execution.current_step = step
                logger.debug(f"Executing step '{step}' for adaptation {execution.id}")

                # Simulate step execution
                await asyncio.sleep(2)  # Simulate work

                execution.steps_completed.append(step)

            # Mark as completed
            with self._lock:
                execution.status = AdaptationStatus.COMPLETED
                execution.end_time = datetime.now(timezone.utc)
                execution.current_step = None

            # Update metrics
            self._metrics["adaptations_executed"] += 1
            self._metrics["adaptations_successful"] += 1

            # Calculate execution time
            if execution.end_time and execution.start_time:
                exec_time = (execution.end_time - execution.start_time).total_seconds()
                current_avg = self._metrics["average_execution_time"]
                total_executions = self._metrics["adaptations_executed"]
                self._metrics["average_execution_time"] = (
                    current_avg * (total_executions - 1) + exec_time
                ) / total_executions

            logger.info(f"Adaptation execution completed successfully: {execution.id}")

        except Exception as e:
            logger.error(f"Adaptation execution failed: {execution.id} - {e}")

            with self._lock:
                execution.status = AdaptationStatus.FAILED
                execution.error_message = str(e)
                execution.end_time = datetime.now(timezone.utc)

            # Update metrics
            self._metrics["adaptations_failed"] += 1

            # Attempt rollback
            await self._rollback_adaptation(execution)

    async def _rollback_adaptation(self, execution: AdaptationExecution):
        """Rollback a failed adaptation"""
        try:
            logger.info(f"Rolling back adaptation: {execution.id}")

            # Simulate rollback steps
            rollback_steps = ["restore_backup", "revert_changes", "validate_rollback"]

            for step in rollback_steps:
                logger.debug(f"Rollback step '{step}' for adaptation {execution.id}")
                await asyncio.sleep(1)  # Simulate rollback work

            with self._lock:
                execution.status = AdaptationStatus.ROLLED_BACK
                execution.rollback_executed = True

            self._metrics["adaptations_rolled_back"] += 1
            logger.info(f"Adaptation rollback completed: {execution.id}")

        except Exception as e:
            logger.error(f"Rollback failed for adaptation {execution.id}: {e}")

    async def _check_execution_timeouts(self):
        """Check for and handle execution timeouts"""
        try:
            current_time = datetime.now(timezone.utc)

            for execution in list(self._active_executions.values()):
                if execution.status == AdaptationStatus.IN_PROGRESS:
                    elapsed_time = (current_time - execution.start_time).total_seconds()

                    if elapsed_time > self._execution_timeout:
                        logger.warning(
                            f"Adaptation execution timed out: {execution.id}"
                        )

                        with self._lock:
                            execution.status = AdaptationStatus.FAILED
                            execution.error_message = "Execution timeout"
                            execution.end_time = current_time

                        self._metrics["adaptations_failed"] += 1
                        await self._rollback_adaptation(execution)

        except Exception as e:
            logger.error(f"Error checking execution timeouts: {e}")

    async def execute_adaptation_plan(self, adaptation_plan) -> str:
        """
        Execute an adaptation plan.

        Args:
            adaptation_plan: The adaptation plan to execute

        Returns:
            Execution ID
        """
        try:
            execution_id = f"exec_{int(time.time())}"

            execution = AdaptationExecution(
                id=execution_id,
                plan_id=adaptation_plan.id,
                status=AdaptationStatus.PENDING,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                steps_completed=[],
                current_step=None,
                error_message=None,
                metrics={},
            )

            with self._lock:
                self._active_executions[execution_id] = execution

            logger.info(f"Queued adaptation plan for execution: {adaptation_plan.name}")
            return execution_id

        except Exception as e:
            logger.error(f"Failed to execute adaptation plan: {e}")
            raise

    async def execute_healing_actions(self):
        """Execute available healing actions"""
        try:
            # Sort healing actions by priority
            healing_actions = sorted(
                self._healing_actions.values(), key=lambda x: x.priority
            )

            for action in healing_actions:
                try:
                    logger.info(f"Executing healing action: {action.id}")

                    # Check prerequisites
                    if not self._check_healing_prerequisites(action):
                        continue

                    # Execute the healing action
                    start_time = time.time()
                    result = await action.action_function(**action.parameters)
                    execution_time = time.time() - start_time

                    # Validate success criteria
                    if self._validate_healing_success(action, result):
                        logger.info(
                            f"Healing action completed successfully: {action.id}"
                        )
                        self._metrics["healing_actions_performed"] += 1
                    else:
                        logger.warning(
                            f"Healing action did not meet success criteria: {action.id}"
                        )

                except Exception as e:
                    logger.error(f"Healing action failed: {action.id} - {e}")

        except Exception as e:
            logger.error(f"Error executing healing actions: {e}")

    def _check_healing_prerequisites(self, action: HealingAction) -> bool:
        """Check if healing action prerequisites are met"""
        try:
            # For now, assume all prerequisites are met
            # In a real implementation, this would check system state
            return True

        except Exception as e:
            logger.error(f"Error checking healing prerequisites: {e}")
            return False

    def _validate_healing_success(self, action: HealingAction, result: Any) -> bool:
        """Validate if healing action met success criteria"""
        try:
            # For now, assume success if no exception was raised
            # In a real implementation, this would check actual metrics
            return result is not None

        except Exception as e:
            logger.error(f"Error validating healing success: {e}")
            return False

    async def _perform_error_recovery(self, **kwargs) -> Dict[str, Any]:
        """Perform error recovery healing action"""
        try:
            logger.info("Performing error recovery")

            # Simulate error recovery actions
            await asyncio.sleep(2)

            # Return recovery metrics
            return {"errors_recovered": 5, "recovery_time": 2.0, "success": True}

        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_performance_optimization(self, **kwargs) -> Dict[str, Any]:
        """Perform performance optimization healing action"""
        try:
            logger.info("Performing performance optimization")

            # Simulate performance optimization
            await asyncio.sleep(3)

            return {
                "performance_improvement": 0.25,
                "optimization_time": 3.0,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_resource_cleanup(self, **kwargs) -> Dict[str, Any]:
        """Perform resource cleanup healing action"""
        try:
            logger.info("Performing resource cleanup")

            # Simulate resource cleanup
            await asyncio.sleep(1)

            return {"memory_freed_mb": 128, "cleanup_time": 1.0, "success": True}

        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    def register_healing_action(self, action: HealingAction) -> bool:
        """Register a new healing action"""
        try:
            with self._lock:
                self._healing_actions[action.id] = action

            logger.info(f"Registered healing action: {action.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register healing action {action.id}: {e}")
            return False

    def get_execution_status(self, execution_id: str) -> Optional[AdaptationExecution]:
        """Get status of an adaptation execution"""
        return self._active_executions.get(execution_id)

    def get_active_executions(self) -> List[AdaptationExecution]:
        """Get all active executions"""
        return [
            exec
            for exec in self._active_executions.values()
            if exec.status in [AdaptationStatus.PENDING, AdaptationStatus.IN_PROGRESS]
        ]

    def get_execution_history(
        self, limit: Optional[int] = None
    ) -> List[AdaptationExecution]:
        """Get execution history"""
        if limit:
            return self._execution_history[-limit:]
        return self._execution_history.copy()

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a pending or in-progress execution"""
        try:
            if execution_id not in self._active_executions:
                return False

            execution = self._active_executions[execution_id]

            if execution.status == AdaptationStatus.PENDING:
                with self._lock:
                    execution.status = AdaptationStatus.FAILED
                    execution.error_message = "Cancelled by user"
                    execution.end_time = datetime.now(timezone.utc)

                logger.info(f"Cancelled pending execution: {execution_id}")
                return True

            elif execution.status == AdaptationStatus.IN_PROGRESS:
                # Set a flag to stop execution (would need coordination with execution logic)
                logger.info(
                    f"Cancellation requested for in-progress execution: {execution_id}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get adaptation controller performance metrics"""
        metrics = self._metrics.copy()

        # Add current state metrics
        metrics["active_executions"] = len(self.get_active_executions())
        metrics["registered_healing_actions"] = len(self._healing_actions)

        # Calculate success rate
        total_attempts = self._metrics["adaptations_executed"]
        if total_attempts > 0:
            metrics["success_rate"] = (
                self._metrics["adaptations_successful"] / total_attempts
            )
        else:
            metrics["success_rate"] = 0.0

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get adaptation controller status"""
        return {
            "running": self._running,
            "active_executions": len(self.get_active_executions()),
            "total_executions": len(self._execution_history),
            "healing_actions_available": len(self._healing_actions),
            "performance_metrics": self.get_performance_metrics(),
        }

    async def shutdown(self):
        """Shutdown adaptation controller"""
        try:
            logger.info("Shutting down Adaptation Controller...")

            # Stop execution thread
            self._running = False
            if self._execution_thread and self._execution_thread.is_alive():
                self._execution_thread.join(timeout=5.0)

            # Cancel any pending executions
            for execution_id, execution in self._active_executions.items():
                if execution.status == AdaptationStatus.PENDING:
                    execution.status = AdaptationStatus.FAILED
                    execution.error_message = "System shutdown"

            logger.info("Adaptation Controller shutdown complete")

        except Exception as e:
            logger.error(f"Error during Adaptation Controller shutdown: {e}")
