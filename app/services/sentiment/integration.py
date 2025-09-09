"""
Sentiment Analysis System Integration.

Main integration point for the new modular sentiment analysis system.
Provides easy initialization and integration with the existing AEO audit system.
"""

import asyncio
from typing import Any, Dict, Optional

from app.utils.logger import get_logger

from .compat import initialize_compatibility_layer
from .core.config import apply_config_profile, get_sentiment_settings
from .core.engine import SentimentEngine
from .core.models import AnalysisContext, SentimentConfig

logger = get_logger(__name__)

# Global sentiment engine instance
_global_engine: Optional[SentimentEngine] = None
_initialization_lock = asyncio.Lock()


async def initialize_sentiment_system(
    config_profile: Optional[str] = None, enable_compatibility: bool = True
) -> SentimentEngine:
    """
    Initialize the global sentiment analysis system.

    Args:
        config_profile: Configuration profile to apply ('development', 'production', 'high_accuracy')
        enable_compatibility: Whether to enable backward compatibility layer

    Returns:
        Initialized SentimentEngine instance
    """
    global _global_engine

    async with _initialization_lock:
        if _global_engine is not None:
            return _global_engine

        logger.info(
            "Initializing sentiment analysis system", config_profile=config_profile
        )

        try:
            # Load configuration
            if config_profile:
                settings = apply_config_profile(config_profile)
                config = SentimentConfig(
                    enabled_providers=settings.enabled_providers,
                    primary_provider=settings.primary_provider,
                    fallback_provider=settings.fallback_provider,
                    ensemble_strategy=settings.ensemble_strategy,
                    ensemble_weights={
                        provider.value: weight
                        for provider, weight in settings.ensemble_weights.items()
                        if hasattr(provider, "value")
                    }
                    if isinstance(settings.ensemble_weights, dict)
                    else {},
                    confidence_threshold=settings.confidence_threshold,
                    enable_caching=settings.enable_caching,
                    cache_ttl=settings.cache_ttl,
                    batch_size=settings.batch_size,
                    max_concurrent=settings.max_concurrent,
                    request_timeout=settings.request_timeout,
                    transformer_model=settings.transformer_model,
                    transformer_device=settings.transformer_device,
                    business_weight=settings.business_weight,
                    linguistic_weight=settings.linguistic_weight,
                    context_boost=settings.context_boost,
                    min_text_length=settings.min_text_length,
                    max_text_length=settings.max_text_length,
                    language_detection=settings.enable_language_detection,
                    supported_languages=settings.supported_languages,
                )
            else:
                # Use default settings
                settings = get_sentiment_settings()
                config = SentimentConfig(
                    enabled_providers=settings.enabled_providers,
                    primary_provider=settings.primary_provider,
                    fallback_provider=settings.fallback_provider,
                    ensemble_strategy=settings.ensemble_strategy,
                    confidence_threshold=settings.confidence_threshold,
                    enable_caching=settings.enable_caching,
                    cache_ttl=settings.cache_ttl,
                    batch_size=settings.batch_size,
                    max_concurrent=settings.max_concurrent,
                    request_timeout=settings.request_timeout,
                    transformer_model=settings.transformer_model,
                    transformer_device=settings.transformer_device,
                    business_weight=settings.business_weight,
                    linguistic_weight=settings.linguistic_weight,
                    context_boost=settings.context_boost,
                    min_text_length=settings.min_text_length,
                    max_text_length=settings.max_text_length,
                    language_detection=settings.enable_language_detection,
                    supported_languages=settings.supported_languages,
                )

            # Create and initialize engine
            _global_engine = SentimentEngine(config)
            await _global_engine.initialize()

            # Initialize compatibility layer if requested
            if enable_compatibility and settings.enable_legacy_api:
                await initialize_compatibility_layer()

            logger.info(
                "Sentiment analysis system initialized successfully",
                providers=list(_global_engine.providers.keys()),
                config_profile=config_profile,
            )

            return _global_engine

        except Exception as e:
            logger.error(f"Failed to initialize sentiment system: {e}", exc_info=True)
            raise


def get_sentiment_engine() -> Optional[SentimentEngine]:
    """
    Get the global sentiment engine instance.

    Returns:
        SentimentEngine instance if initialized, None otherwise
    """
    return _global_engine


async def get_or_initialize_sentiment_engine() -> SentimentEngine:
    """
    Get the global sentiment engine, initializing it if necessary.

    Returns:
        SentimentEngine instance
    """
    if _global_engine is None:
        return await initialize_sentiment_system()
    return _global_engine


async def analyze_sentiment(
    text: str,
    brand: Optional[str] = None,
    context: Optional[AnalysisContext] = None,
    **kwargs,
):
    """
    Convenience function for sentiment analysis using the global engine.

    Args:
        text: Text to analyze
        brand: Brand context for business analysis
        context: Additional analysis context
        **kwargs: Additional arguments passed to engine

    Returns:
        SentimentResult with analysis results
    """
    engine = await get_or_initialize_sentiment_engine()

    if context is None and brand:
        context = AnalysisContext(brand=brand)

    return await engine.analyze(text, context=context, **kwargs)


async def analyze_sentiment_batch(
    texts: list,
    brand: Optional[str] = None,
    context: Optional[AnalysisContext] = None,
    **kwargs,
):
    """
    Convenience function for batch sentiment analysis using the global engine.

    Args:
        texts: List of texts to analyze
        brand: Brand context for business analysis
        context: Additional analysis context
        **kwargs: Additional arguments passed to engine

    Returns:
        BatchSentimentResult with analysis results
    """
    engine = await get_or_initialize_sentiment_engine()

    if context is None and brand:
        context = AnalysisContext(brand=brand)

    return await engine.analyze_batch(texts, context=context, **kwargs)


async def get_sentiment_system_status() -> Dict[str, Any]:
    """
    Get status of the sentiment analysis system.

    Returns:
        Dictionary with system status information
    """
    if _global_engine is None:
        return {
            "initialized": False,
            "status": "not_initialized",
            "message": "Sentiment system not initialized",
        }

    try:
        engine_status = _global_engine.get_status()
        health_status = await _global_engine.health_check()

        return {
            "initialized": True,
            "status": "healthy"
            if health_status["engine_status"] == "healthy"
            else "degraded",
            "engine": engine_status,
            "health": health_status,
            "message": "Sentiment system operational",
        }
    except Exception as e:
        logger.error(f"Failed to get sentiment system status: {e}")
        return {
            "initialized": True,
            "status": "error",
            "error": str(e),
            "message": "Error getting system status",
        }


async def cleanup_sentiment_system():
    """Clean up the global sentiment system"""
    global _global_engine

    if _global_engine:
        logger.info("Cleaning up sentiment analysis system")
        await _global_engine.cleanup()
        _global_engine = None
        logger.info("Sentiment analysis system cleaned up")


# Context manager for sentiment system
class SentimentSystemContext:
    """Context manager for sentiment system lifecycle"""

    def __init__(
        self, config_profile: Optional[str] = None, enable_compatibility: bool = True
    ):
        self.config_profile = config_profile
        self.enable_compatibility = enable_compatibility
        self.engine: Optional[SentimentEngine] = None

    async def __aenter__(self) -> SentimentEngine:
        self.engine = await initialize_sentiment_system(
            self.config_profile, self.enable_compatibility
        )
        return self.engine

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            await cleanup_sentiment_system()


# FastAPI integration helpers
def create_sentiment_status_endpoint():
    """Create a FastAPI endpoint for sentiment system status"""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter()

    @router.get("/sentiment/status")
    async def sentiment_status():
        """Get sentiment analysis system status"""
        try:
            status = await get_sentiment_system_status()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error(f"Status endpoint error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "initialized": False,
                    "status": "error",
                    "error": str(e),
                    "message": "Failed to get system status",
                },
            )

    @router.post("/sentiment/health-check")
    async def sentiment_health_check():
        """Perform sentiment system health check"""
        try:
            engine = get_sentiment_engine()
            if not engine:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unavailable",
                        "message": "Sentiment system not initialized",
                    },
                )

            health = await engine.health_check()
            status_code = 200 if health["engine_status"] == "healthy" else 503

            return JSONResponse(status_code=status_code, content=health)

        except Exception as e:
            logger.error(f"Health check endpoint error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": str(e),
                    "message": "Health check failed",
                },
            )

    return router


# Celery task integration helper
def create_sentiment_celery_tasks(celery_app):
    """Create Celery tasks for sentiment analysis"""

    @celery_app.task(bind=True, name="sentiment.analyze")
    def analyze_sentiment_task(self, text: str, brand: str = None, **kwargs):
        """Celery task for sentiment analysis"""
        try:
            # Run async analysis in sync context
            result = asyncio.run(analyze_sentiment(text, brand=brand, **kwargs))
            return result.to_dict()
        except Exception as e:
            logger.error(f"Sentiment analysis task failed: {e}")
            self.retry(countdown=60, max_retries=3)

    @celery_app.task(bind=True, name="sentiment.analyze_batch")
    def analyze_batch_task(self, texts: list, brand: str = None, **kwargs):
        """Celery task for batch sentiment analysis"""
        try:
            # Run async analysis in sync context
            result = asyncio.run(analyze_sentiment_batch(texts, brand=brand, **kwargs))
            return result.to_dict()
        except Exception as e:
            logger.error(f"Batch sentiment analysis task failed: {e}")
            self.retry(countdown=60, max_retries=3)

    return {
        "analyze_sentiment": analyze_sentiment_task,
        "analyze_batch": analyze_batch_task,
    }


# Application lifecycle integration
async def startup_sentiment_system(app=None):
    """Startup handler for sentiment system"""
    try:
        # Determine configuration based on environment
        import os

        env = os.getenv("APP_ENV", "development")

        config_profile = {
            "development": "development",
            "production": "production",
            "staging": "production",
        }.get(env, "development")

        await initialize_sentiment_system(config_profile=config_profile)
        logger.info("Sentiment system startup completed")

    except Exception as e:
        logger.error(f"Sentiment system startup failed: {e}")
        # Don't fail the entire application startup
        pass


async def shutdown_sentiment_system(app=None):
    """Shutdown handler for sentiment system"""
    try:
        await cleanup_sentiment_system()
        logger.info("Sentiment system shutdown completed")
    except Exception as e:
        logger.error(f"Sentiment system shutdown error: {e}")


# Export main integration functions
__all__ = [
    "initialize_sentiment_system",
    "get_sentiment_engine",
    "get_or_initialize_sentiment_engine",
    "analyze_sentiment",
    "analyze_sentiment_batch",
    "get_sentiment_system_status",
    "cleanup_sentiment_system",
    "SentimentSystemContext",
    "create_sentiment_status_endpoint",
    "create_sentiment_celery_tasks",
    "startup_sentiment_system",
    "shutdown_sentiment_system",
]
