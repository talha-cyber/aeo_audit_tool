"""
Backward Compatibility Layer for Sentiment Analysis.

Provides compatibility with the existing brand detection sentiment system
while enabling gradual migration to the new modular system.
"""

import asyncio
from typing import Dict, Optional

# Import existing classes for backward compatibility
from app.services.brand_detection.core.sentiment import (
    BusinessContextAnalyzer,
    SentimentAnalysisResult,
)
from app.services.brand_detection.core.sentiment import (
    SentimentMethod as LegacySentimentMethod,
)
from app.utils.logger import get_logger

from .core.config import get_sentiment_settings
from .core.engine import SentimentEngine
from .core.models import AnalysisContext, SentimentConfig

logger = get_logger(__name__)

# Global compatibility engine instance
_compat_engine: Optional[SentimentEngine] = None


async def get_compat_engine() -> SentimentEngine:
    """Get or create the compatibility sentiment engine"""
    global _compat_engine

    if _compat_engine is None:
        # Create engine with settings from new system
        sentiment_settings = get_sentiment_settings()

        # Convert to SentimentConfig
        config = SentimentConfig(
            enabled_providers=sentiment_settings.enabled_providers,
            primary_provider=sentiment_settings.primary_provider,
            fallback_provider=sentiment_settings.fallback_provider,
            ensemble_strategy=sentiment_settings.ensemble_strategy,
            confidence_threshold=sentiment_settings.confidence_threshold,
            enable_caching=sentiment_settings.enable_caching,
            cache_ttl=sentiment_settings.cache_ttl,
            batch_size=sentiment_settings.batch_size,
            max_concurrent=sentiment_settings.max_concurrent,
            request_timeout=sentiment_settings.request_timeout,
            transformer_model=sentiment_settings.transformer_model,
            transformer_device=sentiment_settings.transformer_device,
            business_weight=sentiment_settings.business_weight,
            linguistic_weight=sentiment_settings.linguistic_weight,
            context_boost=sentiment_settings.context_boost,
            min_text_length=sentiment_settings.min_text_length,
            max_text_length=sentiment_settings.max_text_length,
            language_detection=sentiment_settings.enable_language_detection,
            supported_languages=sentiment_settings.supported_languages,
        )

        _compat_engine = SentimentEngine(config)
        await _compat_engine.initialize()

        logger.info("Backward compatibility sentiment engine initialized")

    return _compat_engine


class CompatibilitySentimentAnalyzer:
    """
    Drop-in replacement for the legacy SentimentAnalyzer class.

    Provides the same interface as the original class while using
    the new modular sentiment system underneath.
    """

    def __init__(self):
        """Initialize compatibility analyzer"""
        self.business_analyzer = (
            BusinessContextAnalyzer()
        )  # Keep original for full compatibility
        self._engine: Optional[SentimentEngine] = None

    async def _get_engine(self) -> SentimentEngine:
        """Get sentiment engine, initializing if necessary"""
        if self._engine is None:
            self._engine = await get_compat_engine()
        return self._engine

    def analyze_sentiment_vader(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment using VADER (synchronous compatibility method)"""
        # Run async method synchronously for backward compatibility
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self._analyze_vader_async(text))
                    )
                    return future.result()
            else:
                return asyncio.run(self._analyze_vader_async(text))
        except Exception as e:
            logger.error(f"VADER compatibility analysis failed: {e}")
            # Return default result
            from app.services.brand_detection.models.brand_mention import (
                SentimentPolarity,
            )

            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=LegacySentimentMethod.VADER,
                context_type="error",
                metadata={"error": str(e)},
            )

    async def _analyze_vader_async(self, text: str) -> SentimentAnalysisResult:
        """Internal async VADER analysis"""
        from .core.models import SentimentMethod

        engine = await self._get_engine()
        context = AnalysisContext()

        result = await engine.analyze(
            text, context=context, method=SentimentMethod.VADER
        )
        return self._convert_to_legacy_result(result)

    def analyze_business_sentiment(
        self, text: str, brand: str, language: str = "en"
    ) -> SentimentAnalysisResult:
        """Analyze business sentiment (synchronous compatibility method)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(
                            self._analyze_business_async(text, brand, language)
                        )
                    )
                    return future.result()
            else:
                return asyncio.run(self._analyze_business_async(text, brand, language))
        except Exception as e:
            logger.error(f"Business sentiment compatibility analysis failed: {e}")
            # Return default result
            from app.services.brand_detection.models.brand_mention import (
                SentimentPolarity,
            )

            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=LegacySentimentMethod.CUSTOM_BUSINESS,
                context_type="error",
                metadata={"error": str(e)},
            )

    async def _analyze_business_async(
        self, text: str, brand: str, language: str
    ) -> SentimentAnalysisResult:
        """Internal async business sentiment analysis"""
        from .core.models import SentimentMethod

        engine = await self._get_engine()
        context = AnalysisContext(brand=brand, language=language)

        result = await engine.analyze(
            text, context=context, method=SentimentMethod.BUSINESS_CONTEXT
        )
        return self._convert_to_legacy_result(result)

    async def analyze_sentiment_hybrid(
        self, text: str, brand: str, language: str = "en"
    ) -> SentimentAnalysisResult:
        """Hybrid sentiment analysis using ensemble methods"""

        engine = await self._get_engine()
        context = AnalysisContext(brand=brand, language=language)

        # Use ensemble method if multiple providers are enabled
        config = engine.config
        if len(config.enabled_providers) > 1:
            result = await engine.analyze(text, context=context)  # Uses ensemble
        else:
            # Fall back to primary provider
            result = await engine.analyze(
                text, context=context, method=config.primary_provider
            )

        legacy_result = self._convert_to_legacy_result(result)
        legacy_result.method = LegacySentimentMethod.HYBRID
        return legacy_result

    def _convert_to_legacy_result(self, result) -> SentimentAnalysisResult:
        """Convert new SentimentResult to legacy SentimentAnalysisResult"""
        from .core.models import SentimentResult

        if not isinstance(result, SentimentResult):
            raise ValueError(f"Expected SentimentResult, got {type(result)}")

        # Map new method to legacy method
        method_mapping = {
            "vader": LegacySentimentMethod.VADER,
            "business_context": LegacySentimentMethod.CUSTOM_BUSINESS,
            "ensemble": LegacySentimentMethod.HYBRID,
            "transformer": LegacySentimentMethod.VADER,  # Map to VADER for compatibility
        }

        legacy_method = method_mapping.get(
            result.method.value, LegacySentimentMethod.VADER
        )

        # Map context type
        context_mapping = {
            "general": "general",
            "business": "business",
            "positive_business": "positive_business",
            "negative_business": "negative_business",
            "mixed_business": "mixed_business",
            "comparative": "comparative",
            "error": "error",
        }

        context_type = context_mapping.get(result.context_type.value, "general")

        return SentimentAnalysisResult(
            polarity=result.polarity,
            score=result.score,
            confidence=result.confidence,
            method=legacy_method,
            context_type=context_type,
            metadata=result.metadata,
        )

    def _combine_sentiment_results(
        self, analyses: Dict[str, SentimentAnalysisResult]
    ) -> SentimentAnalysisResult:
        """Legacy method for combining results (kept for compatibility)"""
        if not analyses:
            from app.services.brand_detection.models.brand_mention import (
                SentimentPolarity,
            )

            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=LegacySentimentMethod.HYBRID,
                context_type="error",
                metadata={"error": "No analyses to combine"},
            )

        # Use the first available result as base
        result = list(analyses.values())[0]
        result.method = LegacySentimentMethod.HYBRID
        result.metadata["combined_from"] = list(analyses.keys())
        return result


# Factory function for backward compatibility
def create_sentiment_analyzer() -> CompatibilitySentimentAnalyzer:
    """
    Create a sentiment analyzer compatible with legacy code.

    Returns:
        CompatibilitySentimentAnalyzer instance
    """
    return CompatibilitySentimentAnalyzer()


# Monkey-patch replacement for gradual migration
def replace_legacy_sentiment_analyzer():
    """
    Replace the legacy SentimentAnalyzer with the compatibility version.

    This allows existing code to use the new system without modification.
    Call this during application initialization for seamless migration.
    """
    try:
        import app.services.brand_detection.core.sentiment as legacy_module

        # Replace the class
        legacy_module.SentimentAnalyzer = CompatibilitySentimentAnalyzer

        logger.info("Legacy SentimentAnalyzer replaced with compatibility version")

    except Exception as e:
        logger.error(f"Failed to replace legacy sentiment analyzer: {e}")


# Initialization function for the compatibility layer
async def initialize_compatibility_layer():
    """Initialize the backward compatibility layer"""
    try:
        # Ensure the engine is initialized
        await get_compat_engine()

        # Replace legacy analyzer if configured
        settings = get_sentiment_settings()
        if settings.enable_legacy_api:
            replace_legacy_sentiment_analyzer()

        logger.info("Sentiment compatibility layer initialized")

    except Exception as e:
        logger.error(f"Failed to initialize sentiment compatibility layer: {e}")
        raise
