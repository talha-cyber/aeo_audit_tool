"""
Brand Detection Engine for AEO/GEO Audit Tool

This module provides comprehensive brand detection capabilities optimized for
German and US markets, featuring multi-modal detection methods, advanced
sentiment analysis, and robust error handling.
"""

from .core.detector import (
    BrandDetectionEngine,
    BrandDetectionOrchestrator,
    DetectionConfig,
)
from .market_adapters.base import MarketAdapterFactory
from .market_adapters.german_adapter import GermanMarketAdapter
from .models.brand_mention import (
    BrandContext,
    BrandMention,
    DetectionMethod,
    DetectionResult,
    SentimentPolarity,
)
from .utils.cache_manager import get_global_cache, initialize_global_cache
from .utils.performance import get_global_monitor

__version__ = "1.0.0"
__author__ = "AEO Audit Tool Team"

# Package-level configuration
DEFAULT_CONFIG = DetectionConfig(
    confidence_threshold=0.7,
    similarity_threshold=0.8,
    market_code="DE",
    language_code="de",
    enable_caching=True,
    cache_ttl=1800,
)


# Initialize global components
def initialize_brand_detection(**kwargs):
    """Initialize brand detection system with global components"""

    # Initialize caching
    initialize_global_cache()

    # Register market adapters
    MarketAdapterFactory.register_adapter("DE", GermanMarketAdapter)

    return True


# Convenience functions
def create_detection_engine(
    openai_api_key: str, config: DetectionConfig = None
) -> BrandDetectionEngine:
    """Create a configured brand detection engine"""
    return BrandDetectionEngine(openai_api_key, config or DEFAULT_CONFIG)


def create_orchestrator(openai_api_key: str) -> BrandDetectionOrchestrator:
    """Create a brand detection orchestrator"""
    return BrandDetectionOrchestrator(openai_api_key)


# Export main classes and functions
__all__ = [
    # Core classes
    "BrandDetectionEngine",
    "BrandDetectionOrchestrator",
    "DetectionConfig",
    # Data models
    "BrandMention",
    "BrandContext",
    "DetectionResult",
    "SentimentPolarity",
    "DetectionMethod",
    # Market adapters
    "MarketAdapterFactory",
    "GermanMarketAdapter",
    # Utilities
    "initialize_brand_detection",
    "create_detection_engine",
    "create_orchestrator",
    "get_global_cache",
    "get_global_monitor",
    # Configuration
    "DEFAULT_CONFIG",
]
