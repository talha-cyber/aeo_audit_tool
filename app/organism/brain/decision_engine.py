"""
Decision Engine for Organic Intelligence.

Makes intelligent decisions about system adaptations, optimizations,
and responses based on insights and patterns.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import get_logger
from app.organism.control.master_switch import get_organic_control, FeatureCategory
from app.organism.control.decorators import register_organic_feature

logger = get_logger(__name__)


class DecisionType(str, Enum):
    """Types of decisions the engine can make"""
    OPTIMIZATION = "optimization"
    ADAPTATION = "adaptation"
    HEALING = "healing"
    PREVENTION = "prevention"
    ENHANCEMENT = "enhancement"


class DecisionPriority(str, Enum):
    """Priority levels for decisions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DecisionContext:
    """Context for making a decision"""
    trigger_event: str
    system_state: Dict[str, Any]
    available_resources: Dict[str, float]
    constraints: List[str]
    time_pressure: float  # 0.0 to 1.0
    risk_tolerance: float  # 0.0 to 1.0


@dataclass
class DecisionOption:
    """A potential decision option"""
    id: str
    name: str
    description: str
    decision_type: DecisionType
    estimated_impact: float
    confidence: float
    resource_cost: Dict[str, float]
    prerequisites: List[str]
    risks: List[str]
    benefits: List[str]
    execution_time: float


@dataclass
class Decision:
    """A made decision with execution plan"""
    id: str
    option: DecisionOption
    priority: DecisionPriority
    reasoning: str
    execution_plan: List[Dict[str, Any]]
    rollback_plan: List[Dict[str, Any]]
    monitoring_criteria: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None


@register_organic_feature("decision_engine", FeatureCategory.LEARNING)
class DecisionEngine:
    """
    Intelligent decision-making system for organic intelligence.

    Makes strategic decisions about system adaptations, optimizations,
    and responses based on insights, patterns, and current context.
    """

    def __init__(self):
        if not self.is_organic_enabled():
            return

        self._decisions_made: Dict[str, Decision] = {}
        self._decision_history: List[Decision] = []
        self._decision_templates: Dict[DecisionType, List[DecisionOption]] = {}

        # Learning parameters
        self._success_rate: Dict[DecisionType, float] = {}
        self._average_impact: Dict[DecisionType, float] = {}

        # Performance metrics
        self._metrics = {
            "decisions_made": 0,
            "successful_decisions": 0,
            "average_decision_time": 0.0,
            "total_impact": 0.0
        }

        logger.info("Decision Engine initialized")

    async def initialize(self):
        """Initialize decision engine"""
        if not self.is_organic_enabled():
            return

        try:
            # Load decision templates
            await self._load_decision_templates()

            # Initialize learning parameters
            for decision_type in DecisionType:
                self._success_rate[decision_type] = 0.5  # Start neutral
                self._average_impact[decision_type] = 0.0

            logger.info("Decision Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Decision Engine: {e}")
            raise

    async def _load_decision_templates(self):
        """Load predefined decision templates"""
        try:
            # Define template decision options for different scenarios
            self._decision_templates = {
                DecisionType.OPTIMIZATION: [
                    DecisionOption(
                        id="cache_optimization",
                        name="Optimize Caching Strategy",
                        description="Improve system performance through intelligent caching",
                        decision_type=DecisionType.OPTIMIZATION,
                        estimated_impact=0.3,
                        confidence=0.8,
                        resource_cost={"cpu": 0.1, "memory": 0.2},
                        prerequisites=["performance_data_available"],
                        risks=["temporary_slowdown"],
                        benefits=["improved_response_times", "reduced_load"],
                        execution_time=30.0
                    ),
                    DecisionOption(
                        id="query_optimization",
                        name="Optimize Database Queries",
                        description="Improve database query performance",
                        decision_type=DecisionType.OPTIMIZATION,
                        estimated_impact=0.4,
                        confidence=0.7,
                        resource_cost={"cpu": 0.05, "io": 0.1},
                        prerequisites=["query_analysis_available"],
                        risks=["query_changes"],
                        benefits=["faster_queries", "reduced_db_load"],
                        execution_time=60.0
                    )
                ],
                DecisionType.ADAPTATION: [
                    DecisionOption(
                        id="scaling_adaptation",
                        name="Adaptive Resource Scaling",
                        description="Dynamically adjust system resources based on load",
                        decision_type=DecisionType.ADAPTATION,
                        estimated_impact=0.5,
                        confidence=0.9,
                        resource_cost={"cpu": 0.0, "memory": 0.0},
                        prerequisites=["load_monitoring"],
                        risks=["over_provisioning"],
                        benefits=["improved_performance", "cost_optimization"],
                        execution_time=5.0
                    )
                ],
                DecisionType.HEALING: [
                    DecisionOption(
                        id="error_recovery",
                        name="Automatic Error Recovery",
                        description="Implement automatic recovery from detected errors",
                        decision_type=DecisionType.HEALING,
                        estimated_impact=0.8,
                        confidence=0.9,
                        resource_cost={"cpu": 0.2, "memory": 0.1},
                        prerequisites=["error_detection"],
                        risks=["false_positives"],
                        benefits=["improved_reliability", "reduced_downtime"],
                        execution_time=10.0
                    )
                ],
                DecisionType.PREVENTION: [
                    DecisionOption(
                        id="proactive_monitoring",
                        name="Enhanced Proactive Monitoring",
                        description="Increase monitoring to prevent issues",
                        decision_type=DecisionType.PREVENTION,
                        estimated_impact=0.3,
                        confidence=0.8,
                        resource_cost={"cpu": 0.05, "memory": 0.05},
                        prerequisites=["monitoring_system"],
                        risks=["increased_overhead"],
                        benefits=["early_detection", "prevention"],
                        execution_time=15.0
                    )
                ]
            }

        except Exception as e:
            logger.error(f"Failed to load decision templates: {e}")
            raise

    async def analyze_situation(self, context: DecisionContext) -> List[DecisionOption]:
        """
        Analyze current situation and generate decision options.

        Args:
            context: Current decision context

        Returns:
            List of available decision options
        """
        try:
            options = []

            # Analyze context to determine relevant decision types
            relevant_types = self._determine_relevant_decision_types(context)

            for decision_type in relevant_types:
                template_options = self._decision_templates.get(decision_type, [])

                for template in template_options:
                    # Check if option is applicable given context
                    if self._is_option_applicable(template, context):
                        # Customize option based on context
                        customized_option = await self._customize_option(template, context)
                        options.append(customized_option)

            # Sort options by estimated impact and confidence
            options.sort(key=lambda x: x.estimated_impact * x.confidence, reverse=True)

            return options

        except Exception as e:
            logger.error(f"Error analyzing situation: {e}")
            return []

    def _determine_relevant_decision_types(self, context: DecisionContext) -> List[DecisionType]:
        """Determine which decision types are relevant for the context"""
        relevant_types = []

        # Analyze trigger event and system state
        if "error" in context.trigger_event.lower():
            relevant_types.extend([DecisionType.HEALING, DecisionType.PREVENTION])

        if "performance" in context.trigger_event.lower():
            relevant_types.extend([DecisionType.OPTIMIZATION, DecisionType.ADAPTATION])

        if "load" in context.trigger_event.lower():
            relevant_types.extend([DecisionType.ADAPTATION, DecisionType.OPTIMIZATION])

        # Check system state for indicators
        if context.system_state.get("error_rate", 0) > 0.1:
            relevant_types.append(DecisionType.HEALING)

        if context.system_state.get("cpu_usage", 0) > 0.8:
            relevant_types.append(DecisionType.OPTIMIZATION)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(relevant_types))

    def _is_option_applicable(self, option: DecisionOption, context: DecisionContext) -> bool:
        """Check if a decision option is applicable given the context"""
        try:
            # Check prerequisites
            for prereq in option.prerequisites:
                if prereq not in context.system_state:
                    return False

            # Check resource availability
            for resource, cost in option.resource_cost.items():
                available = context.available_resources.get(resource, 0)
                if available < cost:
                    return False

            # Check constraints
            for constraint in context.constraints:
                if any(risk in constraint for risk in option.risks):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking option applicability: {e}")
            return False

    async def _customize_option(self, template: DecisionOption, context: DecisionContext) -> DecisionOption:
        """Customize a template option based on context"""
        try:
            # Adjust confidence based on historical success rate
            decision_type = template.decision_type
            historical_success = self._success_rate.get(decision_type, 0.5)
            adjusted_confidence = (template.confidence + historical_success) / 2

            # Adjust impact based on time pressure and risk tolerance
            impact_multiplier = 1.0
            if context.time_pressure > 0.7:
                impact_multiplier *= 1.2  # Increase impact weight under time pressure
            if context.risk_tolerance < 0.3:
                impact_multiplier *= 0.8  # Reduce impact for low risk tolerance

            adjusted_impact = template.estimated_impact * impact_multiplier

            # Create customized option
            return DecisionOption(
                id=f"{template.id}_{int(time.time())}",
                name=template.name,
                description=template.description,
                decision_type=template.decision_type,
                estimated_impact=min(1.0, adjusted_impact),
                confidence=min(1.0, adjusted_confidence),
                resource_cost=template.resource_cost,
                prerequisites=template.prerequisites,
                risks=template.risks,
                benefits=template.benefits,
                execution_time=template.execution_time
            )

        except Exception as e:
            logger.error(f"Error customizing option: {e}")
            return template

    async def make_decision(
        self,
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Optional[Decision]:
        """
        Make a decision from available options.

        Args:
            options: Available decision options
            context: Decision context

        Returns:
            Made decision or None if no suitable option
        """
        if not options:
            return None

        try:
            start_time = time.time()

            # Score and rank options
            scored_options = await self._score_options(options, context)

            if not scored_options:
                return None

            # Select best option
            best_option = scored_options[0]

            # Determine priority
            priority = self._determine_priority(best_option, context)

            # Create execution plan
            execution_plan = await self._create_execution_plan(best_option, context)
            rollback_plan = await self._create_rollback_plan(best_option, context)

            # Create monitoring criteria
            monitoring_criteria = self._create_monitoring_criteria(best_option)

            # Generate reasoning
            reasoning = self._generate_reasoning(best_option, context, scored_options)

            # Create decision
            decision = Decision(
                id=f"decision_{int(time.time())}",
                option=best_option,
                priority=priority,
                reasoning=reasoning,
                execution_plan=execution_plan,
                rollback_plan=rollback_plan,
                monitoring_criteria=monitoring_criteria,
                created_at=datetime.now(timezone.utc)
            )

            # Record decision
            self._decisions_made[decision.id] = decision
            self._decision_history.append(decision)

            # Update metrics
            decision_time = time.time() - start_time
            self._metrics["decisions_made"] += 1
            self._metrics["average_decision_time"] = (
                (self._metrics["average_decision_time"] * (self._metrics["decisions_made"] - 1) + decision_time) /
                self._metrics["decisions_made"]
            )

            logger.info(f"Decision made: {decision.option.name} (Priority: {priority.value})")
            return decision

        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return None

    async def _score_options(
        self,
        options: List[DecisionOption],
        context: DecisionContext
    ) -> List[DecisionOption]:
        """Score and rank decision options"""
        try:
            scored_options = []

            for option in options:
                # Calculate composite score
                impact_score = option.estimated_impact * 0.4
                confidence_score = option.confidence * 0.3

                # Resource efficiency score
                total_cost = sum(option.resource_cost.values())
                efficiency_score = (1.0 - min(1.0, total_cost)) * 0.2

                # Historical performance score
                historical_success = self._success_rate.get(option.decision_type, 0.5)
                history_score = historical_success * 0.1

                # Composite score
                score = impact_score + confidence_score + efficiency_score + history_score

                # Apply context adjustments
                if context.time_pressure > 0.8 and option.execution_time > 60:
                    score *= 0.8  # Penalize slow options under time pressure

                if context.risk_tolerance < 0.3 and len(option.risks) > 2:
                    score *= 0.7  # Penalize risky options for low risk tolerance

                # Store score in option (temporarily)
                option._score = score
                scored_options.append(option)

            # Sort by score
            scored_options.sort(key=lambda x: x._score, reverse=True)

            # Remove temporary score attribute
            for option in scored_options:
                delattr(option, '_score')

            return scored_options

        except Exception as e:
            logger.error(f"Error scoring options: {e}")
            return options

    def _determine_priority(self, option: DecisionOption, context: DecisionContext) -> DecisionPriority:
        """Determine priority for a decision"""
        try:
            # Base priority on decision type
            base_priority = {
                DecisionType.HEALING: DecisionPriority.HIGH,
                DecisionType.PREVENTION: DecisionPriority.MEDIUM,
                DecisionType.OPTIMIZATION: DecisionPriority.MEDIUM,
                DecisionType.ADAPTATION: DecisionPriority.MEDIUM,
                DecisionType.ENHANCEMENT: DecisionPriority.LOW
            }.get(option.decision_type, DecisionPriority.MEDIUM)

            # Adjust based on context
            if context.time_pressure > 0.8:
                if base_priority == DecisionPriority.HIGH:
                    return DecisionPriority.CRITICAL
                elif base_priority == DecisionPriority.MEDIUM:
                    return DecisionPriority.HIGH

            if option.estimated_impact > 0.8:
                if base_priority == DecisionPriority.MEDIUM:
                    return DecisionPriority.HIGH
                elif base_priority == DecisionPriority.LOW:
                    return DecisionPriority.MEDIUM

            return base_priority

        except Exception as e:
            logger.error(f"Error determining priority: {e}")
            return DecisionPriority.MEDIUM

    async def _create_execution_plan(
        self,
        option: DecisionOption,
        context: DecisionContext
    ) -> List[Dict[str, Any]]:
        """Create execution plan for a decision"""
        try:
            plan = []

            # Basic execution steps based on decision type
            if option.decision_type == DecisionType.OPTIMIZATION:
                plan = [
                    {"step": "prepare_optimization", "timeout": 10},
                    {"step": "apply_optimization", "timeout": option.execution_time},
                    {"step": "validate_optimization", "timeout": 20}
                ]
            elif option.decision_type == DecisionType.HEALING:
                plan = [
                    {"step": "diagnose_issue", "timeout": 5},
                    {"step": "apply_healing", "timeout": option.execution_time},
                    {"step": "verify_healing", "timeout": 10}
                ]
            elif option.decision_type == DecisionType.ADAPTATION:
                plan = [
                    {"step": "prepare_adaptation", "timeout": 5},
                    {"step": "implement_adaptation", "timeout": option.execution_time},
                    {"step": "monitor_adaptation", "timeout": 30}
                ]
            else:
                plan = [
                    {"step": "execute_decision", "timeout": option.execution_time},
                    {"step": "validate_execution", "timeout": 10}
                ]

            return plan

        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return [{"step": "execute_decision", "timeout": 60}]

    async def _create_rollback_plan(
        self,
        option: DecisionOption,
        context: DecisionContext
    ) -> List[Dict[str, Any]]:
        """Create rollback plan for a decision"""
        try:
            rollback_plan = [
                {"step": "detect_failure", "timeout": 5},
                {"step": "stop_execution", "timeout": 5},
                {"step": "restore_previous_state", "timeout": 30},
                {"step": "verify_rollback", "timeout": 10}
            ]

            return rollback_plan

        except Exception as e:
            logger.error(f"Error creating rollback plan: {e}")
            return []

    def _create_monitoring_criteria(self, option: DecisionOption) -> Dict[str, Any]:
        """Create monitoring criteria for a decision"""
        try:
            criteria = {
                "success_indicators": [],
                "failure_indicators": [],
                "performance_metrics": [],
                "timeout": option.execution_time * 2
            }

            # Add criteria based on decision type
            if option.decision_type == DecisionType.OPTIMIZATION:
                criteria["success_indicators"] = ["improved_performance", "reduced_resource_usage"]
                criteria["failure_indicators"] = ["performance_degradation", "increased_errors"]
                criteria["performance_metrics"] = ["response_time", "cpu_usage", "memory_usage"]

            elif option.decision_type == DecisionType.HEALING:
                criteria["success_indicators"] = ["error_reduction", "system_stability"]
                criteria["failure_indicators"] = ["continued_errors", "system_instability"]
                criteria["performance_metrics"] = ["error_rate", "uptime", "recovery_time"]

            return criteria

        except Exception as e:
            logger.error(f"Error creating monitoring criteria: {e}")
            return {"timeout": 60}

    def _generate_reasoning(
        self,
        selected_option: DecisionOption,
        context: DecisionContext,
        all_options: List[DecisionOption]
    ) -> str:
        """Generate human-readable reasoning for the decision"""
        try:
            reasoning_parts = [
                f"Selected '{selected_option.name}' based on analysis of {len(all_options)} options.",
                f"Estimated impact: {selected_option.estimated_impact:.2f}",
                f"Confidence level: {selected_option.confidence:.2f}",
                f"Decision type: {selected_option.decision_type.value}"
            ]

            if context.time_pressure > 0.7:
                reasoning_parts.append("High time pressure influenced rapid decision-making.")

            if context.risk_tolerance < 0.3:
                reasoning_parts.append("Low risk tolerance favored conservative options.")

            if selected_option.benefits:
                reasoning_parts.append(f"Expected benefits: {', '.join(selected_option.benefits)}")

            return " ".join(reasoning_parts)

        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Selected {selected_option.name} as best available option."

    async def create_adaptation_plan(self, insight) -> Optional[Any]:
        """Create adaptation plan from system insight"""
        try:
            # Create context from insight
            context = DecisionContext(
                trigger_event=f"insight_{insight.category}",
                system_state=insight.evidence,
                available_resources={"cpu": 0.8, "memory": 0.8, "io": 0.8},
                constraints=[],
                time_pressure=0.3,
                risk_tolerance=0.7
            )

            # Analyze situation and make decision
            options = await self.analyze_situation(context)
            decision = await self.make_decision(options, context)

            if decision:
                # Convert decision to adaptation plan format expected by central intelligence
                from app.organism.brain.central_intelligence import AdaptationPlan

                plan = AdaptationPlan(
                    id=decision.id,
                    name=decision.option.name,
                    description=decision.option.description,
                    priority=1 if decision.priority == DecisionPriority.CRITICAL else
                            2 if decision.priority == DecisionPriority.HIGH else
                            3 if decision.priority == DecisionPriority.MEDIUM else 4,
                    estimated_impact=decision.option.estimated_impact,
                    risk_level=len(decision.option.risks) / 5.0,  # Normalize to 0-1
                    prerequisites=decision.option.prerequisites,
                    actions=[{"step": step["step"], "timeout": step["timeout"]}
                           for step in decision.execution_plan],
                    rollback_plan={"steps": decision.rollback_plan} if decision.rollback_plan else None
                )

                return plan

            return None

        except Exception as e:
            logger.error(f"Error creating adaptation plan: {e}")
            return None

    def record_decision_outcome(self, decision_id: str, success: bool, actual_impact: float):
        """Record the outcome of a decision for learning"""
        try:
            if decision_id not in self._decisions_made:
                return

            decision = self._decisions_made[decision_id]
            decision_type = decision.option.decision_type

            # Update success rate
            current_success = self._success_rate.get(decision_type, 0.5)
            total_decisions = sum(1 for d in self._decision_history if d.option.decision_type == decision_type)

            if total_decisions > 0:
                self._success_rate[decision_type] = (
                    (current_success * (total_decisions - 1) + (1.0 if success else 0.0)) / total_decisions
                )

            # Update average impact
            current_impact = self._average_impact.get(decision_type, 0.0)
            self._average_impact[decision_type] = (
                (current_impact * (total_decisions - 1) + actual_impact) / total_decisions
            )

            # Update global metrics
            if success:
                self._metrics["successful_decisions"] += 1
            self._metrics["total_impact"] += actual_impact

            logger.debug(f"Decision outcome recorded: {decision_id} - Success: {success}, Impact: {actual_impact}")

        except Exception as e:
            logger.error(f"Error recording decision outcome: {e}")

    def get_active_decisions(self) -> List[Decision]:
        """Get currently active decisions"""
        current_time = datetime.now(timezone.utc)
        return [
            decision for decision in self._decisions_made.values()
            if decision.expires_at is None or decision.expires_at > current_time
        ]

    def get_decision_history(self, limit: Optional[int] = None) -> List[Decision]:
        """Get decision history"""
        if limit:
            return self._decision_history[-limit:]
        return self._decision_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get decision engine performance metrics"""
        metrics = self._metrics.copy()
        metrics["success_rates"] = self._success_rate.copy()
        metrics["average_impacts"] = self._average_impact.copy()
        metrics["decision_types_used"] = len(self._success_rate)

        if self._metrics["decisions_made"] > 0:
            metrics["overall_success_rate"] = (
                self._metrics["successful_decisions"] / self._metrics["decisions_made"]
            )
        else:
            metrics["overall_success_rate"] = 0.0

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get decision engine status"""
        return {
            "active_decisions": len(self.get_active_decisions()),
            "total_decisions": len(self._decision_history),
            "performance_metrics": self.get_performance_metrics(),
            "decision_templates": {dt.value: len(opts) for dt, opts in self._decision_templates.items()}
        }