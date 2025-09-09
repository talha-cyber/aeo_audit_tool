"""
Core sentiment analysis engine.

The main orchestrator for sentiment analysis, managing multiple providers,
ensemble methods, caching, and performance optimization. Built for production
use with comprehensive error handling and monitoring.
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from typing import Dict, List, Optional

from app.utils.cache import get_cache_client
from app.utils.logger import get_logger

from ..providers.base import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    SentimentProvider,
)
from .models import (
    AnalysisContext,
    BatchSentimentResult,
    ContextType,
    EnsembleStrategy,
    SentimentConfig,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)

logger = get_logger(__name__)


class SentimentEngineError(Exception):
    """Base exception for sentiment engine errors"""

    pass


class NoProvidersAvailableError(SentimentEngineError):
    """Raised when no sentiment providers are available"""

    pass


class SentimentEngine:
    """
    Advanced sentiment analysis engine with multiple providers and ensemble methods.

    Features:
    - Multiple sentiment providers (VADER, Transformers, API-based)
    - Ensemble methods for improved accuracy
    - Intelligent provider fallback and circuit breaking
    - Performance monitoring and adaptive routing
    - Caching for improved performance
    - Batch processing optimization
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self.providers: Dict[SentimentMethod, SentimentProvider] = {}
        self.provider_health: Dict[SentimentMethod, bool] = {}
        self.cache = None
        self._initialized = False
        self._provider_lock = asyncio.Lock()

        # Performance tracking
        self.engine_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "ensemble_analyses": 0,
            "fallback_uses": 0,
            "provider_failures": defaultdict(int),
        }

    async def initialize(self):
        """Initialize the sentiment engine and its providers"""
        if self._initialized:
            return

        logger.info("Initializing sentiment engine", config=self.config.__dict__)

        # Initialize cache if enabled
        if self.config.enable_caching:
            self.cache = await get_cache_client()

        # Initialize providers
        await self._initialize_providers()

        self._initialized = True
        logger.info(
            "Sentiment engine initialized",
            providers=list(self.providers.keys()),
            primary=self.config.primary_provider.value,
        )

    async def analyze(
        self,
        text: str,
        brand: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
        method: Optional[SentimentMethod] = None,
        **kwargs,
    ) -> SentimentResult:
        """
        Analyze sentiment for a single text.

        Args:
            text: Text to analyze
            brand: Brand context for business-aware analysis
            context: Additional analysis context
            method: Specific method to use (None for default strategy)
            **kwargs: Additional arguments passed to providers

        Returns:
            SentimentResult with analysis results
        """
        if not self._initialized:
            await self.initialize()

        self.engine_stats["total_requests"] += 1

        # Prepare context
        if context is None:
            context = AnalysisContext()
        if brand:
            context.brand = brand

        # Check cache first
        if self.config.enable_caching and self.cache:
            cache_key = self._generate_cache_key(text, context)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.engine_stats["cache_hits"] += 1
                return cached_result

        # Determine analysis strategy
        if method:
            # Use specific method
            result = await self._analyze_with_method(text, context, method)
        elif len(self.config.enabled_providers) == 1:
            # Single provider mode
            result = await self._analyze_with_method(
                text, context, self.config.enabled_providers[0]
            )
        else:
            # Ensemble mode
            result = await self._analyze_with_ensemble(text, context)
            self.engine_stats["ensemble_analyses"] += 1

        # Cache result
        if self.config.enable_caching and self.cache and result.confidence > 0.5:
            await self._cache_result(cache_key, result)

        return result

    async def analyze_batch(
        self,
        texts: List[str],
        brand: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
        method: Optional[SentimentMethod] = None,
        **kwargs,
    ) -> BatchSentimentResult:
        """
        Analyze sentiment for multiple texts with batch optimization.

        Args:
            texts: List of texts to analyze
            brand: Brand context for business-aware analysis
            context: Additional analysis context
            method: Specific method to use
            **kwargs: Additional arguments

        Returns:
            BatchSentimentResult with all analysis results
        """
        if not texts:
            return BatchSentimentResult(results=[], config_used=self.config)

        start_time = time.time()

        # Prepare context for all texts
        if context is None:
            context = AnalysisContext()
        if brand:
            context.brand = brand

        # Check for native batch processing support
        target_method = method or self.config.primary_provider
        if target_method in self.providers:
            provider = self.providers[target_method]
            if provider.supports_batch_processing():
                try:
                    results = await provider.analyze_batch(texts, context)
                    return BatchSentimentResult(
                        results=results, config_used=self.config
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch processing failed, falling back to individual: {e}"
                    )

        # Process individually with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def analyze_single(text: str) -> SentimentResult:
            async with semaphore:
                return await self.analyze(text, brand, context, method, **kwargs)

        # Execute batch with concurrency limit
        tasks = [analyze_single(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to analyze text {i}: {result}")
                final_results.append(
                    SentimentResult(
                        polarity=SentimentPolarity.NEUTRAL,
                        score=0.0,
                        confidence=0.0,
                        method=target_method,
                        context_type=ContextType.ERROR,
                        metadata={"error": str(result), "text_index": i},
                    )
                )
            else:
                final_results.append(result)

        return BatchSentimentResult(results=final_results, config_used=self.config)

    async def _initialize_providers(self):
        """Initialize all configured providers"""
        from ..providers.vader_provider import VaderSentimentProvider

        # Always ensure VADER is available as fallback
        if SentimentMethod.VADER not in self.providers:
            vader_provider = VaderSentimentProvider()
            try:
                await vader_provider.initialize()
                self.providers[SentimentMethod.VADER] = vader_provider
                self.provider_health[SentimentMethod.VADER] = True
                logger.info("VADER provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize VADER provider: {e}")
                self.provider_health[SentimentMethod.VADER] = False

        # Initialize other providers lazily as needed
        for method in self.config.enabled_providers:
            if method not in self.providers:
                await self._initialize_provider(method)

    async def _initialize_provider(self, method: SentimentMethod):
        """Initialize a specific provider"""
        async with self._provider_lock:
            if method in self.providers:
                return  # Already initialized

            try:
                if method == SentimentMethod.VADER:
                    from ..providers.vader_provider import VaderSentimentProvider

                    provider = VaderSentimentProvider()
                elif method == SentimentMethod.TRANSFORMER:
                    from ..providers.transformer_provider import (
                        TransformerSentimentProvider,
                    )

                    provider = TransformerSentimentProvider(
                        model_name=self.config.transformer_model,
                        device=self.config.transformer_device,
                    )
                elif method == SentimentMethod.BUSINESS_CONTEXT:
                    from ..providers.business_provider import BusinessContextProvider

                    provider = BusinessContextProvider()
                else:
                    logger.warning(f"Unknown provider method: {method}")
                    return

                await provider.initialize()
                self.providers[method] = provider
                self.provider_health[method] = True
                logger.info(f"Provider {method.value} initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize provider {method.value}: {e}")
                self.provider_health[method] = False

    async def _analyze_with_method(
        self, text: str, context: AnalysisContext, method: SentimentMethod
    ) -> SentimentResult:
        """Analyze text using a specific method with fallback"""

        # Ensure provider is available
        if method not in self.providers or not self.provider_health.get(method, False):
            await self._initialize_provider(method)

        if method not in self.providers or not self.provider_health.get(method, False):
            # Primary method failed, use fallback
            if method != self.config.fallback_provider:
                logger.warning(f"Provider {method.value} unavailable, using fallback")
                self.engine_stats["fallback_uses"] += 1
                return await self._analyze_with_method(
                    text, context, self.config.fallback_provider
                )
            else:
                raise NoProvidersAvailableError(
                    "No providers available, including fallback"
                )

        provider = self.providers[method]

        try:
            result = await provider.analyze(text, context)
            self.provider_health[method] = True  # Mark as healthy on success
            return result

        except (ProviderUnavailableError, ProviderTimeoutError) as e:
            logger.warning(f"Provider {method.value} failed: {e}")
            self.provider_health[method] = False
            self.engine_stats["provider_failures"][method.value] += 1

            # Try fallback if this isn't already the fallback
            if method != self.config.fallback_provider:
                self.engine_stats["fallback_uses"] += 1
                return await self._analyze_with_method(
                    text, context, self.config.fallback_provider
                )
            else:
                # Fallback also failed, return error result
                return SentimentResult(
                    polarity=SentimentPolarity.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    method=method,
                    context_type=ContextType.ERROR,
                    metadata={"error": str(e), "provider": method.value},
                )

    async def _analyze_with_ensemble(
        self, text: str, context: AnalysisContext
    ) -> SentimentResult:
        """Analyze text using ensemble of multiple providers"""

        # Get available providers
        available_methods = [
            method
            for method in self.config.enabled_providers
            if method in self.providers and self.provider_health.get(method, False)
        ]

        if not available_methods:
            # No providers available, try to initialize fallback
            await self._initialize_provider(self.config.fallback_provider)
            if self.config.fallback_provider in self.providers:
                return await self._analyze_with_method(
                    text, context, self.config.fallback_provider
                )
            else:
                raise NoProvidersAvailableError(
                    "No providers available for ensemble analysis"
                )

        # Run analyses in parallel
        tasks = [
            self._analyze_with_method(text, context, method)
            for method in available_methods
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                method = available_methods[i]
                logger.warning(f"Provider {method.value} failed in ensemble: {result}")
                self.engine_stats["provider_failures"][method.value] += 1
            elif isinstance(result, SentimentResult) and result.confidence > 0:
                successful_results.append(result)

        if not successful_results:
            # All providers failed
            return SentimentResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.ENSEMBLE,
                context_type=ContextType.ERROR,
                metadata={"error": "All ensemble providers failed"},
            )

        # Combine results using configured strategy
        return self._combine_results(successful_results, self.config.ensemble_strategy)

    def _combine_results(
        self, results: List[SentimentResult], strategy: EnsembleStrategy
    ) -> SentimentResult:
        """Combine multiple sentiment results using the specified strategy"""

        if len(results) == 1:
            result = results[0]
            result.method = SentimentMethod.ENSEMBLE
            return result

        if strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_combination(results)
        elif strategy == EnsembleStrategy.MAJORITY_VOTE:
            return self._majority_vote_combination(results)
        elif strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_combination(results)
        elif strategy == EnsembleStrategy.ADAPTIVE:
            return self._adaptive_combination(results)
        else:
            logger.warning(
                f"Unknown ensemble strategy: {strategy}, using confidence_weighted"
            )
            return self._confidence_weighted_combination(results)

    def _confidence_weighted_combination(
        self, results: List[SentimentResult]
    ) -> SentimentResult:
        """Combine results weighted by confidence scores"""

        total_weight = sum(r.confidence for r in results)
        if total_weight == 0:
            total_weight = len(results)
            weights = [1.0] * len(results)
        else:
            weights = [r.confidence for r in results]

        # Weighted average of scores
        weighted_score = (
            sum(r.score * w for r, w in zip(results, weights)) / total_weight
        )

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Determine polarity from weighted score
        if weighted_score >= 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif weighted_score <= -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL

        # Determine context type (prefer business context if available)
        context_types = [r.context_type for r in results]
        if (
            ContextType.POSITIVE_BUSINESS in context_types
            or ContextType.NEGATIVE_BUSINESS in context_types
        ):
            context_type = ContextType.MIXED_BUSINESS
        elif ContextType.BUSINESS in context_types:
            context_type = ContextType.BUSINESS
        else:
            context_type = ContextType.GENERAL

        return SentimentResult(
            polarity=polarity,
            score=weighted_score,
            confidence=min(1.0, avg_confidence + 0.1),  # Ensemble bonus
            method=SentimentMethod.ENSEMBLE,
            context_type=context_type,
            metadata={
                "ensemble_strategy": EnsembleStrategy.CONFIDENCE_WEIGHTED.value,
                "individual_results": [r.to_dict() for r in results],
                "weights": weights,
                "total_weight": total_weight,
            },
        )

    def _weighted_average_combination(
        self, results: List[SentimentResult]
    ) -> SentimentResult:
        """Combine results using predefined weights"""

        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0

        for result in results:
            weight = self.config.ensemble_weights.get(result.method, 1.0)
            total_weight += weight
            weighted_score += result.score * weight
            weighted_confidence += result.confidence * weight

        if total_weight == 0:
            total_weight = len(results)
            weighted_score = sum(r.score for r in results)
            weighted_confidence = sum(r.confidence for r in results)

        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / total_weight

        # Determine polarity
        if final_score >= 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif final_score <= -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL

        return SentimentResult(
            polarity=polarity,
            score=final_score,
            confidence=final_confidence,
            method=SentimentMethod.ENSEMBLE,
            context_type=ContextType.GENERAL,
            metadata={
                "ensemble_strategy": EnsembleStrategy.WEIGHTED_AVERAGE.value,
                "individual_results": [r.to_dict() for r in results],
                "ensemble_weights": dict(self.config.ensemble_weights),
            },
        )

    def _majority_vote_combination(
        self, results: List[SentimentResult]
    ) -> SentimentResult:
        """Combine results using majority voting"""

        # Count votes for each polarity
        votes = defaultdict(int)
        total_score = 0
        total_confidence = 0

        for result in results:
            votes[result.polarity] += 1
            total_score += result.score
            total_confidence += result.confidence

        # Find majority polarity
        majority_polarity = max(votes.keys(), key=lambda k: votes[k])

        # Average score and confidence
        avg_score = total_score / len(results)
        avg_confidence = total_confidence / len(results)

        return SentimentResult(
            polarity=majority_polarity,
            score=avg_score,
            confidence=avg_confidence,
            method=SentimentMethod.ENSEMBLE,
            context_type=ContextType.GENERAL,
            metadata={
                "ensemble_strategy": EnsembleStrategy.MAJORITY_VOTE.value,
                "individual_results": [r.to_dict() for r in results],
                "votes": {k.value: v for k, v in votes.items()},
            },
        )

    def _adaptive_combination(self, results: List[SentimentResult]) -> SentimentResult:
        """Adaptive combination based on provider performance"""

        # Weight by provider performance (inverse of error rate)
        weights = []
        for result in results:
            provider_perf = self.providers[result.method].get_performance()
            # Use inverse error rate as weight (add small epsilon to avoid division by zero)
            weight = 1.0 / (provider_perf.error_rate + 0.01)
            # Also factor in confidence
            weight *= result.confidence
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            return self._confidence_weighted_combination(results)

        # Weighted combination
        weighted_score = (
            sum(r.score * w for r, w in zip(results, weights)) / total_weight
        )
        weighted_confidence = (
            sum(r.confidence * w for r, w in zip(results, weights)) / total_weight
        )

        # Determine polarity
        if weighted_score >= 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif weighted_score <= -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL

        return SentimentResult(
            polarity=polarity,
            score=weighted_score,
            confidence=weighted_confidence,
            method=SentimentMethod.ENSEMBLE,
            context_type=ContextType.GENERAL,
            metadata={
                "ensemble_strategy": EnsembleStrategy.ADAPTIVE.value,
                "individual_results": [r.to_dict() for r in results],
                "adaptive_weights": weights,
                "provider_performances": {
                    r.method.value: self.providers[r.method].get_performance().__dict__
                    for r in results
                },
            },
        )

    def _generate_cache_key(self, text: str, context: AnalysisContext) -> str:
        """Generate cache key for text and context"""
        # Create deterministic key from text and relevant context
        key_data = {
            "text": text,
            "brand": context.brand,
            "language": context.language,
            "source": context.source,
            "competitive_context": context.competitive_context,
            "providers": sorted([p.value for p in self.config.enabled_providers]),
            "strategy": self.config.ensemble_strategy.value,
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _get_cached_result(self, cache_key: str) -> Optional[SentimentResult]:
        """Retrieve result from cache"""
        if not self.cache:
            return None

        try:
            cached_data = await self.cache.get(f"sentiment:{cache_key}")
            if cached_data:
                # Reconstruct SentimentResult from cached data
                # This would need proper serialization/deserialization
                # For now, just return None to skip caching
                return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: SentimentResult):
        """Store result in cache"""
        if not self.cache:
            return

        try:
            # Cache the result dictionary
            await self.cache.setex(
                f"sentiment:{cache_key}", self.config.cache_ttl, result.to_dict()
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def get_status(self) -> Dict:
        """Get engine status and performance statistics"""
        return {
            "initialized": self._initialized,
            "config": {
                "enabled_providers": [p.value for p in self.config.enabled_providers],
                "primary_provider": self.config.primary_provider.value,
                "ensemble_strategy": self.config.ensemble_strategy.value,
                "cache_enabled": self.config.enable_caching,
            },
            "providers": {
                method.value: {
                    "available": self.provider_health.get(method, False),
                    "capabilities": provider.get_capabilities() if provider else None,
                }
                for method, provider in self.providers.items()
            },
            "performance": self.engine_stats,
            "provider_performances": {
                method.value: provider.get_performance().__dict__
                for method, provider in self.providers.items()
            },
        }

    async def health_check(self) -> Dict:
        """Perform health check on all providers"""
        health_status = {}

        for method, provider in self.providers.items():
            try:
                if provider.is_available():
                    # Quick test analysis
                    test_result = await provider.analyze("Test sentiment analysis.")
                    health_status[method.value] = {
                        "status": "healthy",
                        "response_time": test_result.processing_time,
                        "confidence": test_result.confidence,
                    }
                    self.provider_health[method] = True
                else:
                    health_status[method.value] = {
                        "status": "unavailable",
                        "error": "Provider not initialized",
                    }
                    self.provider_health[method] = False
            except Exception as e:
                health_status[method.value] = {"status": "error", "error": str(e)}
                self.provider_health[method] = False

        return {
            "engine_status": "healthy"
            if any(self.provider_health.values())
            else "unhealthy",
            "providers": health_status,
            "timestamp": time.time(),
        }

    async def cleanup(self):
        """Clean up engine resources"""
        if not self._initialized:
            return

        logger.info("Cleaning up sentiment engine")

        # Clean up all providers
        cleanup_tasks = []
        for provider in self.providers.values():
            cleanup_tasks.append(provider.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.providers.clear()
        self.provider_health.clear()
        self._initialized = False

        logger.info("Sentiment engine cleanup completed")
