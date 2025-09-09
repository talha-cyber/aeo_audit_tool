"""
Sentiment analysis data models and types.

Comprehensive type definitions for sentiment analysis results, configurations,
and internal data structures. Designed for type safety and extensibility.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Import existing polarity for backward compatibility
from app.services.brand_detection.models.brand_mention import SentimentPolarity


class SentimentMethod(str, Enum):
    """Sentiment analysis methods/providers"""

    VADER = "vader"
    TRANSFORMER = "transformer"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BUSINESS_CONTEXT = "business_context"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"  # Keep for backward compatibility


class EnsembleStrategy(str, Enum):
    """Strategies for combining multiple sentiment results"""

    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE = "adaptive"


class ContextType(str, Enum):
    """Types of context detected in text"""

    GENERAL = "general"
    BUSINESS = "business"
    POSITIVE_BUSINESS = "positive_business"
    NEGATIVE_BUSINESS = "negative_business"
    MIXED_BUSINESS = "mixed_business"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"
    ERROR = "error"


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""

    # Provider settings
    enabled_providers: List[SentimentMethod] = field(
        default_factory=lambda: [
            SentimentMethod.VADER,
            SentimentMethod.BUSINESS_CONTEXT,
        ]
    )
    primary_provider: SentimentMethod = SentimentMethod.VADER
    fallback_provider: SentimentMethod = SentimentMethod.VADER

    # Ensemble settings
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED
    ensemble_weights: Dict[SentimentMethod, float] = field(default_factory=dict)
    confidence_threshold: float = 0.7

    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 32
    max_concurrent: int = 10
    request_timeout: float = 30.0

    # Provider-specific configs
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    transformer_device: str = "auto"  # "auto", "cpu", "cuda"

    # Business context settings
    business_weight: float = 0.3
    linguistic_weight: float = 0.7
    context_boost: float = 0.2

    # Quality settings
    min_text_length: int = 3
    max_text_length: int = 5000
    language_detection: bool = True
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "de", "fr", "es"]
    )

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.enabled_providers:
            self.enabled_providers = [SentimentMethod.VADER]

        if self.primary_provider not in self.enabled_providers:
            self.primary_provider = self.enabled_providers[0]

        # Set default ensemble weights if not provided
        if not self.ensemble_weights and len(self.enabled_providers) > 1:
            weight = 1.0 / len(self.enabled_providers)
            self.ensemble_weights = {
                provider: weight for provider in self.enabled_providers
            }


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single text"""

    # Core sentiment data
    polarity: SentimentPolarity
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 to 1.0

    # Method and context
    method: SentimentMethod
    context_type: ContextType = ContextType.GENERAL

    # Analysis metadata
    text_length: int = 0
    language: Optional[str] = None
    processing_time: float = 0.0

    # Provider-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking data
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "polarity": self.polarity.value,
            "score": self.score,
            "confidence": self.confidence,
            "method": self.method.value,
            "context_type": self.context_type.value,
            "text_length": self.text_length,
            "language": self.language,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_legacy_result(cls, legacy_result: Any) -> SentimentResult:
        """Create SentimentResult from legacy SentimentAnalysisResult"""
        from app.services.brand_detection.core.sentiment import SentimentAnalysisResult

        if not isinstance(legacy_result, SentimentAnalysisResult):
            raise ValueError(
                f"Expected SentimentAnalysisResult, got {type(legacy_result)}"
            )

        # Map legacy method to new method
        method_mapping = {
            "vader": SentimentMethod.VADER,
            "custom_business": SentimentMethod.BUSINESS_CONTEXT,
            "hybrid": SentimentMethod.ENSEMBLE,
        }

        method = method_mapping.get(legacy_result.method.value, SentimentMethod.VADER)

        # Map context type
        context_mapping = {
            "general": ContextType.GENERAL,
            "positive_business": ContextType.POSITIVE_BUSINESS,
            "negative_business": ContextType.NEGATIVE_BUSINESS,
            "mixed_business": ContextType.MIXED_BUSINESS,
            "error": ContextType.ERROR,
        }

        context_type = context_mapping.get(
            legacy_result.context_type, ContextType.GENERAL
        )

        return cls(
            polarity=legacy_result.polarity,
            score=legacy_result.score,
            confidence=legacy_result.confidence,
            method=method,
            context_type=context_type,
            metadata=legacy_result.metadata,
        )


@dataclass
class BatchSentimentResult:
    """Result of batch sentiment analysis"""

    results: List[SentimentResult]

    # Batch statistics
    total_processed: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    average_confidence: float = 0.0
    total_processing_time: float = 0.0

    # Batch metadata
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_used: Optional[SentimentConfig] = None

    def __post_init__(self):
        """Calculate statistics after initialization"""
        self.total_processed = len(self.results)
        self.successful_analyses = len([r for r in self.results if r.confidence > 0])
        self.failed_analyses = self.total_processed - self.successful_analyses

        if self.successful_analyses > 0:
            self.average_confidence = sum(r.confidence for r in self.results) / len(
                self.results
            )

        self.total_processing_time = sum(r.processing_time for r in self.results)

    def get_sentiment_distribution(self) -> Dict[SentimentPolarity, int]:
        """Get distribution of sentiment polarities"""
        distribution = {polarity: 0 for polarity in SentimentPolarity}
        for result in self.results:
            distribution[result.polarity] += 1
        return distribution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "total_processed": self.total_processed,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "average_confidence": self.average_confidence,
            "total_processing_time": self.total_processing_time,
            "sentiment_distribution": {
                k.value: v for k, v in self.get_sentiment_distribution().items()
            },
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ProviderPerformance:
    """Performance metrics for a sentiment provider"""

    provider: SentimentMethod
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    average_confidence: float = 0.0
    last_used: Optional[datetime] = None
    error_rate: float = 0.0

    def update_stats(
        self, response_time: float, success: bool, confidence: float = 0.0
    ):
        """Update performance statistics"""
        self.total_requests += 1
        self.last_used = datetime.now(timezone.utc)

        if success:
            self.successful_requests += 1
            # Running average calculation
            n = self.successful_requests
            self.average_response_time = (
                (n - 1) * self.average_response_time + response_time
            ) / n
            self.average_confidence = (
                (n - 1) * self.average_confidence + confidence
            ) / n
        else:
            self.failed_requests += 1

        self.error_rate = (
            self.failed_requests / self.total_requests
            if self.total_requests > 0
            else 0.0
        )


@dataclass
class AnalysisContext:
    """Context information for sentiment analysis"""

    brand: Optional[str] = None
    industry: Optional[str] = None
    language: Optional[str] = None
    source: Optional[str] = None  # "web", "social", "review", etc.
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Additional context for business analysis
    competitive_context: bool = False
    product_context: Optional[str] = None

    # Processing hints
    priority: str = "normal"  # "low", "normal", "high"
    require_explanation: bool = False
    custom_patterns: Optional[List[str]] = None
