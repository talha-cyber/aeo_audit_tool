"""
Advanced Sentiment Analysis System

This module provides a modular, extensible sentiment analysis system with multiple
providers, ensemble methods, and performance optimizations. Designed for production
use with comprehensive error handling, caching, and monitoring.

Key Features:
- Multiple sentiment analysis providers (VADER, Transformers, API-based)
- Ensemble methods for improved accuracy
- Async processing with batching support
- Performance monitoring and caching
- Backward compatibility with existing brand detection system
"""

from .core.engine import SentimentEngine
from .core.models import (
    BatchSentimentResult,
    EnsembleStrategy,
    SentimentConfig,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from .providers.base import SentimentProvider
from .providers.transformer_provider import TransformerSentimentProvider
from .providers.vader_provider import VaderSentimentProvider

# Main engine instance for easy import
sentiment_engine: SentimentEngine = None


def get_sentiment_engine() -> SentimentEngine:
    """Get or create the global sentiment engine instance"""
    global sentiment_engine
    if sentiment_engine is None:
        sentiment_engine = SentimentEngine()
    return sentiment_engine


# Convenience functions for backward compatibility
async def analyze_sentiment(text: str, brand: str = None, **kwargs) -> SentimentResult:
    """Convenience function for single sentiment analysis"""
    engine = get_sentiment_engine()
    return await engine.analyze(text, brand=brand, **kwargs)


async def analyze_batch(
    texts: list, brand: str = None, **kwargs
) -> BatchSentimentResult:
    """Convenience function for batch sentiment analysis"""
    engine = get_sentiment_engine()
    return await engine.analyze_batch(texts, brand=brand, **kwargs)


__all__ = [
    "SentimentEngine",
    "SentimentMethod",
    "SentimentResult",
    "SentimentPolarity",
    "BatchSentimentResult",
    "SentimentConfig",
    "EnsembleStrategy",
    "SentimentProvider",
    "VaderSentimentProvider",
    "TransformerSentimentProvider",
    "get_sentiment_engine",
    "analyze_sentiment",
    "analyze_batch",
]
