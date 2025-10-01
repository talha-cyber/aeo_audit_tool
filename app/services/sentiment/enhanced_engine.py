"""
Enhanced Sentiment Engine with Cost-Effective ML Capabilities.

Integrates all new ML capabilities including efficient transformers, domain adaptation,
model optimization, and cost monitoring into the existing sentiment analysis framework.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from app.utils.logger import get_logger

from .core.engine import SentimentEngine
from .core.models import (
    AnalysisContext,
    BatchSentimentResult,
    SentimentConfig,
    SentimentResult,
)
from .cost_management.cost_monitor import CostMonitor
from .optimization.model_manager import ModelManager

# New ML capabilities
from .providers.efficient_transformer_provider import (
    create_cost_effective_sentiment_provider,
)
from .training.domain_adapter import DomainAdapter

logger = get_logger(__name__)


class EnhancedSentimentEngine(SentimentEngine):
    """
    Enhanced sentiment engine with advanced ML capabilities.

    New Features:
    - Cost-effective transformer models with quantization
    - Domain adaptation and fine-tuning
    - Advanced model caching and optimization
    - Comprehensive cost monitoring
    - Resource management and optimization
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        enable_cost_monitoring: bool = True,
        budget_limit: float = 10.0,  # $10/month budget
        enable_model_optimization: bool = True,
        cache_dir: str = "enhanced_sentiment_cache",
    ):
        super().__init__(config)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Enhanced components
        self.cost_monitor = None
        self.model_manager = None
        self.domain_adapter = None

        # Configuration
        self.enable_cost_monitoring = enable_cost_monitoring
        self.budget_limit = budget_limit
        self.enable_model_optimization = enable_model_optimization

        # Enhanced provider registry
        self.enhanced_providers = {}

    async def initialize(self):
        """Initialize enhanced engine with all new capabilities"""
        logger.info("Initializing Enhanced Sentiment Engine")

        # Initialize base engine
        await super().initialize()

        # Initialize cost monitoring
        if self.enable_cost_monitoring:
            self.cost_monitor = CostMonitor(
                storage_dir=str(self.cache_dir / "cost_monitoring"),
                budget_limit=self.budget_limit,
            )
            self.cost_monitor.start_monitoring()
            logger.info("Cost monitoring enabled")

        # Initialize model manager
        if self.enable_model_optimization:
            self.model_manager = ModelManager(
                cache_dir=str(self.cache_dir / "model_cache")
            )
            self.model_manager.start_monitoring()
            logger.info("Model optimization enabled")

        # Initialize domain adapter
        self.domain_adapter = DomainAdapter(
            cache_dir=str(self.cache_dir / "domain_models")
        )
        await self.domain_adapter.initialize()
        logger.info("Domain adaptation capabilities enabled")

        # Initialize enhanced providers
        await self._initialize_enhanced_providers()

        logger.info("Enhanced Sentiment Engine fully initialized")

    async def _initialize_enhanced_providers(self):
        """Initialize enhanced transformer providers"""
        try:
            # Add efficient transformer providers for different cost preferences
            cost_preferences = ["ultra_low", "low", "balanced", "high_accuracy"]

            for preference in cost_preferences:
                provider = create_cost_effective_sentiment_provider(
                    cost_preference=preference
                )

                # Initialize provider
                await provider.initialize()

                # Add to enhanced providers
                method_name = f"EFFICIENT_TRANSFORMER_{preference.upper()}"
                self.enhanced_providers[method_name] = provider

                logger.info(f"Initialized efficient transformer: {preference}")

        except Exception as e:
            logger.warning(f"Failed to initialize enhanced providers: {e}")

    async def analyze_cost_effective(
        self,
        text: str,
        cost_preference: str = "balanced",  # ultra_low, low, balanced, high_accuracy
        brand: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
        **kwargs,
    ) -> SentimentResult:
        """
        Analyze sentiment with cost optimization.

        Args:
            text: Text to analyze
            cost_preference: Cost vs accuracy preference
            brand: Brand context
            context: Analysis context
            **kwargs: Additional arguments

        Returns:
            SentimentResult with cost tracking
        """
        start_time = time.time()

        try:
            # Prepare context
            if context is None:
                context = AnalysisContext()
            if brand:
                context.brand = brand

            # Select provider based on cost preference
            provider_key = f"EFFICIENT_TRANSFORMER_{cost_preference.upper()}"

            if provider_key in self.enhanced_providers:
                provider = self.enhanced_providers[provider_key]
                result = await provider.analyze(text, context)

                # Record operation for cost monitoring
                if self.cost_monitor:
                    duration = time.time() - start_time
                    self.cost_monitor.record_operation(
                        operation_type="inference",
                        duration=duration,
                        model_name=provider.model_name,
                        provider_name=provider.name,
                        input_tokens=len(text.split()),
                        success=True,
                    )

                # Add cost metadata
                if hasattr(provider, "get_cost_metrics"):
                    cost_metrics = provider.get_cost_metrics()
                    result.metadata.update(
                        {
                            "cost_metrics": cost_metrics,
                            "cost_preference": cost_preference,
                            "enhanced_analysis": True,
                        }
                    )

                return result

            else:
                # Fallback to regular analysis
                logger.warning(
                    f"Enhanced provider {provider_key} not available, using fallback"
                )
                return await self.analyze(text, brand, context, **kwargs)

        except Exception as e:
            # Record failed operation
            if self.cost_monitor:
                duration = time.time() - start_time
                self.cost_monitor.record_operation(
                    operation_type="inference",
                    duration=duration,
                    model_name="unknown",
                    provider_name="enhanced_engine",
                    success=False,
                    error_message=str(e),
                )

            logger.error(f"Cost-effective analysis failed: {e}")
            # Fallback to regular analysis
            return await self.analyze(text, brand, context, **kwargs)

    async def train_domain_model(
        self,
        domain_name: str,
        training_texts: List[str],
        training_labels: List[str],
        validation_split: float = 0.2,
        cost_limit: float = 1.0,  # Max $1 for training
        base_model: str = "small",
    ) -> Dict:
        """
        Train a domain-specific sentiment model.

        Args:
            domain_name: Name for the domain model
            training_texts: List of training texts
            training_labels: List of sentiment labels
            validation_split: Validation data split
            cost_limit: Maximum training cost in hours
            base_model: Base model size to use

        Returns:
            Training results and metrics
        """
        start_time = time.time()

        try:
            logger.info(f"Starting domain training for: {domain_name}")

            # Estimate training cost
            cost_estimate = self.domain_adapter.estimate_training_cost(
                num_samples=len(training_texts), num_epochs=3, batch_size=8
            )

            logger.info(
                f"Estimated training cost: ${cost_estimate['estimated_gpu_cost']:.4f}"
            )

            # Check budget
            if self.cost_monitor:
                month_summary = self.cost_monitor.get_cost_summary("month")
                if (
                    month_summary["total_cost"] + cost_estimate["estimated_gpu_cost"]
                    > self.budget_limit
                ):
                    logger.warning("Training would exceed monthly budget")
                    return {
                        "error": "Training would exceed monthly budget",
                        "estimated_cost": cost_estimate,
                        "current_monthly_cost": month_summary["total_cost"],
                        "budget_limit": self.budget_limit,
                    }

            # Prepare training data
            train_dataset, val_dataset = self.domain_adapter.prepare_training_data(
                training_texts, training_labels, validation_split
            )

            # Fine-tune model
            results = await self.domain_adapter.fine_tune_domain(
                domain_name=domain_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                cost_limit=cost_limit,
                save_model=True,
            )

            # Record training operation
            if self.cost_monitor:
                duration = time.time() - start_time
                self.cost_monitor.record_operation(
                    operation_type="training",
                    duration=duration,
                    model_name=f"domain_{domain_name}",
                    provider_name="domain_adapter",
                    input_tokens=len(training_texts),
                    success=True,
                )

            logger.info(f"Domain training completed for: {domain_name}")
            return results

        except Exception as e:
            # Record failed training
            if self.cost_monitor:
                duration = time.time() - start_time
                self.cost_monitor.record_operation(
                    operation_type="training",
                    duration=duration,
                    model_name=f"domain_{domain_name}",
                    provider_name="domain_adapter",
                    input_tokens=len(training_texts),
                    success=False,
                    error_message=str(e),
                )

            logger.error(f"Domain training failed for {domain_name}: {e}")
            return {"error": str(e)}

    async def analyze_with_domain_model(
        self,
        text: str,
        domain_name: str,
        brand: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment using a domain-specific model.

        Args:
            text: Text to analyze
            domain_name: Name of the domain model to use
            brand: Brand context
            context: Analysis context

        Returns:
            SentimentResult from domain model
        """
        start_time = time.time()

        try:
            # Create domain provider
            domain_provider = self.domain_adapter.create_domain_provider(domain_name)

            if not domain_provider:
                raise ValueError(f"Domain model not found: {domain_name}")

            # Prepare context
            if context is None:
                context = AnalysisContext()
            if brand:
                context.brand = brand

            # Analyze with domain model
            result = await domain_provider.analyze(text, context)

            # Add domain metadata
            result.metadata.update(
                {
                    "domain_model": domain_name,
                    "domain_analysis": True,
                    "enhanced_analysis": True,
                }
            )

            # Record operation
            if self.cost_monitor:
                duration = time.time() - start_time
                self.cost_monitor.record_operation(
                    operation_type="inference",
                    duration=duration,
                    model_name=f"domain_{domain_name}",
                    provider_name="domain_provider",
                    input_tokens=len(text.split()),
                    success=True,
                )

            return result

        except Exception as e:
            # Record failed operation
            if self.cost_monitor:
                duration = time.time() - start_time
                self.cost_monitor.record_operation(
                    operation_type="inference",
                    duration=duration,
                    model_name=f"domain_{domain_name}",
                    provider_name="domain_provider",
                    success=False,
                    error_message=str(e),
                )

            logger.error(f"Domain analysis failed for {domain_name}: {e}")
            # Fallback to regular analysis
            return await self.analyze_cost_effective(text, "balanced", brand, context)

    async def batch_analyze_optimized(
        self,
        texts: List[str],
        cost_preference: str = "balanced",
        brand: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
        **kwargs,
    ) -> BatchSentimentResult:
        """
        Optimized batch sentiment analysis with cost tracking.

        Args:
            texts: List of texts to analyze
            cost_preference: Cost optimization preference
            brand: Brand context
            context: Analysis context
            **kwargs: Additional arguments

        Returns:
            BatchSentimentResult with cost tracking
        """
        start_time = time.time()

        try:
            # Select efficient provider
            provider_key = f"EFFICIENT_TRANSFORMER_{cost_preference.upper()}"

            if provider_key in self.enhanced_providers:
                provider = self.enhanced_providers[provider_key]

                # Prepare context
                if context is None:
                    context = AnalysisContext()
                if brand:
                    context.brand = brand

                # Batch analyze
                results = await provider.analyze_batch(texts, context)

                # Create batch result
                batch_result = BatchSentimentResult(
                    results=results, config_used=self.config
                )

                # Record batch operation
                if self.cost_monitor:
                    duration = time.time() - start_time
                    total_tokens = sum(len(text.split()) for text in texts)

                    self.cost_monitor.record_operation(
                        operation_type="batch_inference",
                        duration=duration,
                        model_name=provider.model_name,
                        provider_name=provider.name,
                        input_tokens=total_tokens,
                        success=True,
                    )

                # Add cost metadata
                if hasattr(provider, "get_cost_metrics"):
                    cost_metrics = provider.get_cost_metrics()
                    batch_result.config_used.metadata = {
                        "cost_metrics": cost_metrics,
                        "cost_preference": cost_preference,
                        "enhanced_batch_analysis": True,
                        "batch_size": len(texts),
                    }

                return batch_result

            else:
                # Fallback to regular batch analysis
                return await self.analyze_batch(texts, brand, context, **kwargs)

        except Exception as e:
            logger.error(f"Optimized batch analysis failed: {e}")
            # Fallback to regular batch analysis
            return await self.analyze_batch(texts, brand, context, **kwargs)

    def get_cost_report(self, period: str = "month") -> Dict:
        """Get comprehensive cost report"""
        if not self.cost_monitor:
            return {"error": "Cost monitoring not enabled"}

        report = {
            "cost_summary": self.cost_monitor.get_cost_summary(period),
            "usage_analytics": self.cost_monitor.get_usage_analytics(),
            "optimization_recommendations": self.cost_monitor.get_optimization_recommendations(),
            "system_stats": {},
        }

        # Add system stats if model manager is available
        if self.model_manager:
            report["system_stats"] = self.model_manager.get_system_stats()

        return report

    def get_available_capabilities(self) -> Dict:
        """Get list of available enhanced capabilities"""
        capabilities = {
            "cost_effective_models": list(self.enhanced_providers.keys()),
            "domain_models": self.domain_adapter.get_available_domains()
            if self.domain_adapter
            else [],
            "cost_monitoring": self.cost_monitor is not None,
            "model_optimization": self.model_manager is not None,
            "domain_adaptation": self.domain_adapter is not None,
            "budget_limit": self.budget_limit,
            "cache_directory": str(self.cache_dir),
        }

        # Add provider capabilities
        capabilities["provider_details"] = {}
        for name, provider in self.enhanced_providers.items():
            capabilities["provider_details"][name] = provider.get_capabilities()

        return capabilities

    async def optimize_for_cost(self, target_budget: float) -> Dict:
        """
        Optimize engine configuration for a specific budget.

        Args:
            target_budget: Target monthly budget in USD

        Returns:
            Optimization recommendations and new configuration
        """
        if not self.cost_monitor:
            return {"error": "Cost monitoring not enabled"}

        current_usage = self.cost_monitor.get_cost_summary("month")
        recommendations = []

        # If over budget, suggest cost reduction strategies
        if current_usage["total_cost"] > target_budget:
            overage = current_usage["total_cost"] - target_budget
            recommendations.extend(
                [
                    f"Current monthly cost (${current_usage['total_cost']:.2f}) exceeds target (${target_budget:.2f})",
                    f"Need to reduce costs by ${overage:.2f} ({overage/current_usage['total_cost']*100:.1f}%)",
                    "Recommendations:",
                    "- Use 'ultra_low' cost preference for non-critical analyses",
                    "- Enable more aggressive model caching",
                    "- Use smaller models for bulk processing",
                    "- Consider domain-specific models for repeated use cases",
                ]
            )

            # Update budget limit
            self.budget_limit = target_budget
            if self.cost_monitor:
                self.cost_monitor.budget_limit = target_budget

        else:
            recommendations.extend(
                [
                    f"Current usage (${current_usage['total_cost']:.2f}) is within target (${target_budget:.2f})",
                    "System is operating efficiently within budget",
                ]
            )

        return {
            "target_budget": target_budget,
            "current_cost": current_usage["total_cost"],
            "recommendations": recommendations,
            "optimization_applied": True,
        }

    async def cleanup(self):
        """Clean up all enhanced components"""
        logger.info("Cleaning up Enhanced Sentiment Engine")

        # Cleanup enhanced providers
        for provider in self.enhanced_providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup enhanced provider: {e}")

        # Cleanup components
        if self.cost_monitor:
            await self.cost_monitor.cleanup()

        if self.model_manager:
            await self.model_manager.cleanup()

        if self.domain_adapter:
            await self.domain_adapter.cleanup()

        # Cleanup base engine
        await super().cleanup()

        logger.info("Enhanced Sentiment Engine cleanup completed")


# Factory function for easy creation
async def create_enhanced_sentiment_engine(
    cost_preference: str = "balanced",
    enable_domain_adaptation: bool = True,
    budget_limit: float = 10.0,
    **kwargs,
) -> EnhancedSentimentEngine:
    """
    Factory function to create and initialize enhanced sentiment engine.

    Args:
        cost_preference: Default cost preference for models
        enable_domain_adaptation: Enable domain adaptation capabilities
        budget_limit: Monthly budget limit in USD
        **kwargs: Additional configuration options

    Returns:
        Initialized EnhancedSentimentEngine
    """
    engine = EnhancedSentimentEngine(budget_limit=budget_limit, **kwargs)

    await engine.initialize()

    # Set default cost preference
    engine.default_cost_preference = cost_preference

    logger.info(
        f"Enhanced sentiment engine created with {cost_preference} cost preference"
    )

    return engine
