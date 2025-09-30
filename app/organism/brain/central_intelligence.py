"""
Central Intelligence - Master Orchestrator for Organic Intelligence.

The main brain that coordinates all organic intelligence activities including
learning, healing, evolution, and adaptation across the entire system.
"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from app.utils.logger import get_logger
from app.organism.control.master_switch import get_organic_control, FeatureCategory
from app.organism.control.decorators import register_organic_feature

logger = get_logger(__name__)


class IntelligenceMode(str, Enum):
    """Operating modes for central intelligence"""
    DORMANT = "dormant"           # Intelligence is sleeping
    OBSERVING = "observing"       # Monitoring and learning only
    ANALYZING = "analyzing"       # Active pattern analysis
    ADAPTING = "adapting"         # Making adaptations
    HEALING = "healing"           # Active healing and repair
    EVOLVING = "evolving"         # Evolutionary improvements


@dataclass
class SystemInsight:
    """Insight discovered by central intelligence"""
    id: str
    timestamp: datetime
    category: str
    confidence: float
    description: str
    evidence: Dict[str, Any]
    suggested_actions: List[str]
    impact_assessment: Dict[str, float]


@dataclass
class AdaptationPlan:
    """Plan for system adaptation"""
    id: str
    name: str
    description: str
    priority: int
    estimated_impact: float
    risk_level: float
    prerequisites: List[str]
    actions: List[Dict[str, Any]]
    rollback_plan: Optional[Dict[str, Any]] = None


@register_organic_feature("central_intelligence", FeatureCategory.LEARNING)
class CentralIntelligence:
    """
    Central Intelligence system that orchestrates all organic intelligence.

    This is the main brain that:
    - Coordinates learning across all components
    - Analyzes system-wide patterns
    - Makes intelligent decisions about adaptations
    - Orchestrates healing and evolution
    - Maintains system consciousness and self-awareness
    """

    def __init__(self):
        if not self.is_organic_enabled():
            return

        self._mode = IntelligenceMode.DORMANT
        self._consciousness_level = 0.0  # 0.0 to 1.0
        self._active_insights: Dict[str, SystemInsight] = {}
        self._adaptation_plans: Dict[str, AdaptationPlan] = {}
        self._learning_cycles = 0
        self._startup_time = time.time()

        # Threading
        self._intelligence_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()

        # Sub-components (will be imported when needed to avoid circular imports)
        self._pattern_recognizer = None
        self._decision_engine = None
        self._memory_consolidator = None
        self._adaptation_controller = None

        # Metrics
        self._performance_metrics = {
            "insights_generated": 0,
            "adaptations_planned": 0,
            "adaptations_executed": 0,
            "learning_cycles": 0,
            "consciousness_evolution": []
        }

        logger.info("Central Intelligence initialized")

    async def awaken(self) -> bool:
        """
        Awaken the central intelligence system.

        Returns:
            True if awakening successful
        """
        if not self.is_organic_enabled():
            logger.info("Central Intelligence cannot awaken - organic system disabled")
            return False

        with self._lock:
            if self._running:
                logger.info("Central Intelligence already awake")
                return True

            try:
                logger.info("Awakening Central Intelligence...")

                # Initialize sub-components
                await self._initialize_subcomponents()

                # Start intelligence thread
                self._running = True
                self._mode = IntelligenceMode.OBSERVING
                self._intelligence_thread = threading.Thread(
                    target=self._intelligence_loop,
                    daemon=True,
                    name="CentralIntelligence"
                )
                self._intelligence_thread.start()

                # Begin consciousness evolution
                self._consciousness_level = 0.1
                self._performance_metrics["consciousness_evolution"].append({
                    "timestamp": time.time(),
                    "level": self._consciousness_level,
                    "event": "awakening"
                })

                logger.info("Central Intelligence awakened successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to awaken Central Intelligence: {e}")
                self._running = False
                return False

    async def sleep(self) -> bool:
        """
        Put the central intelligence to sleep.

        Returns:
            True if sleep successful
        """
        with self._lock:
            if not self._running:
                logger.info("Central Intelligence already sleeping")
                return True

            try:
                logger.info("Putting Central Intelligence to sleep...")

                # Stop intelligence thread
                self._running = False
                if self._intelligence_thread and self._intelligence_thread.is_alive():
                    self._intelligence_thread.join(timeout=5.0)

                # Set dormant mode
                self._mode = IntelligenceMode.DORMANT
                self._consciousness_level = 0.0

                logger.info("Central Intelligence is now sleeping")
                return True

            except Exception as e:
                logger.error(f"Failed to put Central Intelligence to sleep: {e}")
                return False

    async def _initialize_subcomponents(self):
        """Initialize intelligence sub-components"""
        try:
            # Import here to avoid circular imports
            from .pattern_recognition import PatternRecognizer
            from .decision_engine import DecisionEngine
            from .memory_consolidation import MemoryConsolidator
            from .adaptation_controller import AdaptationController

            self._pattern_recognizer = PatternRecognizer()
            self._decision_engine = DecisionEngine()
            self._memory_consolidator = MemoryConsolidator()
            self._adaptation_controller = AdaptationController()

            # Initialize each component
            await self._pattern_recognizer.initialize()
            await self._decision_engine.initialize()
            await self._memory_consolidator.initialize()
            await self._adaptation_controller.initialize()

            logger.info("Intelligence sub-components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize sub-components: {e}")
            raise

    def _intelligence_loop(self):
        """Main intelligence processing loop"""
        logger.info("Central Intelligence processing loop started")

        while self._running:
            try:
                # Sleep based on consciousness level
                sleep_time = max(1.0, 10.0 - (self._consciousness_level * 9.0))
                time.sleep(sleep_time)

                if not self._running:
                    break

                # Perform intelligence cycle
                asyncio.run(self._intelligence_cycle())

            except Exception as e:
                logger.error(f"Error in intelligence loop: {e}")
                time.sleep(5.0)

        logger.info("Central Intelligence processing loop stopped")

    async def _intelligence_cycle(self):
        """Single cycle of intelligence processing"""
        try:
            cycle_start = time.time()

            # 1. Update consciousness level
            await self._update_consciousness()

            # 2. Observe system state
            await self._observe_system()

            # 3. Analyze patterns
            if self._mode in [IntelligenceMode.ANALYZING, IntelligenceMode.ADAPTING,
                             IntelligenceMode.HEALING, IntelligenceMode.EVOLVING]:
                await self._analyze_patterns()

            # 4. Generate insights
            await self._generate_insights()

            # 5. Plan adaptations
            if self._mode in [IntelligenceMode.ADAPTING, IntelligenceMode.EVOLVING]:
                await self._plan_adaptations()

            # 6. Execute healing if needed
            if self._mode == IntelligenceMode.HEALING:
                await self._execute_healing()

            # 7. Consolidate memory
            await self._consolidate_memory()

            # 8. Update metrics
            self._learning_cycles += 1
            self._performance_metrics["learning_cycles"] = self._learning_cycles

            cycle_duration = time.time() - cycle_start
            logger.debug(f"Intelligence cycle completed in {cycle_duration:.3f}s")

        except Exception as e:
            logger.error(f"Error in intelligence cycle: {e}")

    async def _update_consciousness(self):
        """Update consciousness level based on system activity"""
        try:
            # Get system activity metrics
            control = get_organic_control()
            status = control.get_status()

            # Calculate consciousness based on:
            # - Number of active features
            # - System uptime
            # - Learning progress
            # - Adaptation success rate

            activity_factor = len(status.active_features) / 10.0  # Assume max 10 features
            uptime_factor = min(1.0, status.uptime / 3600.0)  # Normalize to 1 hour
            learning_factor = min(1.0, self._learning_cycles / 100.0)  # Normalize to 100 cycles

            new_level = min(1.0, (activity_factor + uptime_factor + learning_factor) / 3.0)

            # Gradually evolve consciousness
            if new_level > self._consciousness_level:
                self._consciousness_level = min(1.0, self._consciousness_level + 0.01)
            elif new_level < self._consciousness_level:
                self._consciousness_level = max(0.0, self._consciousness_level - 0.005)

            # Update mode based on consciousness level
            if self._consciousness_level < 0.2:
                self._mode = IntelligenceMode.OBSERVING
            elif self._consciousness_level < 0.4:
                self._mode = IntelligenceMode.ANALYZING
            elif self._consciousness_level < 0.6:
                self._mode = IntelligenceMode.ADAPTING
            elif self._consciousness_level < 0.8:
                self._mode = IntelligenceMode.HEALING
            else:
                self._mode = IntelligenceMode.EVOLVING

            # Record consciousness evolution
            if len(self._performance_metrics["consciousness_evolution"]) == 0 or \
               abs(self._consciousness_level -
                   self._performance_metrics["consciousness_evolution"][-1]["level"]) > 0.05:

                self._performance_metrics["consciousness_evolution"].append({
                    "timestamp": time.time(),
                    "level": self._consciousness_level,
                    "mode": self._mode.value
                })

        except Exception as e:
            logger.error(f"Error updating consciousness: {e}")

    async def _observe_system(self):
        """Observe current system state"""
        # This will be enhanced as sensors are implemented
        pass

    async def _analyze_patterns(self):
        """Analyze system patterns"""
        if self._pattern_recognizer:
            try:
                patterns = await self._pattern_recognizer.analyze_current_patterns()
                # Process discovered patterns
                for pattern in patterns:
                    await self._process_pattern(pattern)
            except Exception as e:
                logger.error(f"Error in pattern analysis: {e}")

    async def _process_pattern(self, pattern: Dict[str, Any]):
        """Process a discovered pattern"""
        # This will be enhanced with actual pattern processing logic
        logger.debug(f"Processing pattern: {pattern.get('name', 'unknown')}")

    async def _generate_insights(self):
        """Generate system insights"""
        try:
            # Generate insights based on current observations and patterns
            insight_count = len(self._active_insights)

            # Simple insight generation (will be enhanced)
            if self._learning_cycles % 10 == 0:  # Every 10 cycles
                insight = SystemInsight(
                    id=f"insight_{int(time.time())}",
                    timestamp=datetime.now(timezone.utc),
                    category="performance",
                    confidence=0.7,
                    description=f"System has completed {self._learning_cycles} learning cycles",
                    evidence={"cycles": self._learning_cycles, "uptime": time.time() - self._startup_time},
                    suggested_actions=["Continue monitoring", "Consider optimization"],
                    impact_assessment={"performance": 0.1, "stability": 0.0}
                )

                self._active_insights[insight.id] = insight
                self._performance_metrics["insights_generated"] += 1

        except Exception as e:
            logger.error(f"Error generating insights: {e}")

    async def _plan_adaptations(self):
        """Plan system adaptations based on insights"""
        try:
            if self._decision_engine and self._active_insights:
                for insight in self._active_insights.values():
                    if insight.confidence > 0.8:  # High confidence insights
                        plan = await self._decision_engine.create_adaptation_plan(insight)
                        if plan:
                            self._adaptation_plans[plan.id] = plan
                            self._performance_metrics["adaptations_planned"] += 1

        except Exception as e:
            logger.error(f"Error planning adaptations: {e}")

    async def _execute_healing(self):
        """Execute healing actions"""
        try:
            if self._adaptation_controller:
                await self._adaptation_controller.execute_healing_actions()
        except Exception as e:
            logger.error(f"Error executing healing: {e}")

    async def _consolidate_memory(self):
        """Consolidate learning memory"""
        try:
            if self._memory_consolidator:
                await self._memory_consolidator.consolidate_recent_experiences()
        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")

    def get_consciousness_level(self) -> float:
        """Get current consciousness level"""
        return self._consciousness_level

    def get_intelligence_mode(self) -> IntelligenceMode:
        """Get current intelligence mode"""
        return self._mode

    def get_active_insights(self) -> List[SystemInsight]:
        """Get currently active insights"""
        return list(self._active_insights.values())

    def get_adaptation_plans(self) -> List[AdaptationPlan]:
        """Get current adaptation plans"""
        return list(self._adaptation_plans.values())

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get intelligence performance metrics"""
        return self._performance_metrics.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive intelligence status"""
        return {
            "running": self._running,
            "mode": self._mode.value,
            "consciousness_level": self._consciousness_level,
            "learning_cycles": self._learning_cycles,
            "active_insights": len(self._active_insights),
            "adaptation_plans": len(self._adaptation_plans),
            "uptime": time.time() - self._startup_time,
            "performance_metrics": self._performance_metrics
        }

    async def shutdown(self):
        """Shutdown central intelligence gracefully"""
        logger.info("Shutting down Central Intelligence...")
        await self.sleep()
        logger.info("Central Intelligence shutdown complete")


# Global instance
_central_intelligence: Optional[CentralIntelligence] = None
_intelligence_lock = threading.Lock()


def get_central_intelligence() -> CentralIntelligence:
    """Get global central intelligence instance"""
    global _central_intelligence

    if _central_intelligence is None:
        with _intelligence_lock:
            if _central_intelligence is None:
                _central_intelligence = CentralIntelligence()

    return _central_intelligence