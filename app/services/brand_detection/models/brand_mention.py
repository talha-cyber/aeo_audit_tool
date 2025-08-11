import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List


class SentimentPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class DetectionMethod(Enum):
    NER = "named_entity_recognition"
    FUZZY = "fuzzy_matching"
    SEMANTIC = "semantic_similarity"
    REGEX = "regex_pattern"
    HYBRID = "hybrid_approach"


@dataclass
class BrandContext:
    """Context surrounding a brand mention"""

    text: str
    start_position: int
    end_position: int
    sentence: str
    surrounding_entities: List[str] = field(default_factory=list)
    competitive_mentions: List[str] = field(default_factory=list)


class BrandMention:
    """Enhanced brand mention with comprehensive metadata"""

    def __init__(
        self,
        brand: str,
        original_text: str,
        confidence: float,
        detection_method: DetectionMethod,
        language: str = "en",
    ):
        self.id = str(uuid.uuid4())
        self.brand = brand
        self.original_text = original_text
        self.confidence = confidence
        self.detection_method = detection_method
        self.language = language
        self.created_at = datetime.utcnow()

        # Analytics data
        self.contexts: List[BrandContext] = []
        self.sentiment_score: float = 0.0
        self.sentiment_polarity: SentimentPolarity = SentimentPolarity.NEUTRAL
        self.mention_count: int = 0
        self.competitive_context: Dict[str, int] = {}
        self.market_specific_data: Dict[str, any] = {}

    def add_context(self, context: BrandContext):
        """Add context with validation"""
        if not isinstance(context, BrandContext):
            raise ValueError("Context must be BrandContext instance")
        self.contexts.append(context)
        self.mention_count = len(self.contexts)

    def calculate_relevance_score(self) -> float:
        """Calculate overall relevance based on multiple factors"""
        base_score = self.confidence

        # Boost for multiple mentions
        mention_boost = min(0.2, len(self.contexts) * 0.05)

        # Boost for competitive context
        competitive_boost = 0.1 if self.competitive_context else 0.0

        # Sentiment influence
        sentiment_influence = abs(self.sentiment_score) * 0.1

        return min(
            1.0, base_score + mention_boost + competitive_boost + sentiment_influence
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "brand": self.brand,
            "mention_count": self.mention_count,
            "confidence": self.confidence,
            "sentiment_score": self.sentiment_score,
            "sentiment_polarity": self.sentiment_polarity.value,
            "detection_method": self.detection_method.value,
            "language": self.language,
            "relevance_score": self.calculate_relevance_score(),
            "contexts": [
                {
                    "text": ctx.text,
                    "sentence": ctx.sentence,
                    "competitive_mentions": ctx.competitive_mentions,
                }
                for ctx in self.contexts[:3]  # Limit for API response size
            ],
            "competitive_context": self.competitive_context,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DetectionResult:
    """Result container for brand detection operations"""

    text_analyzed: str
    language: str
    processing_time_ms: float
    mentions: List[BrandMention]
    total_brands_found: int
    confidence_threshold: float
    market_adapter_used: str

    def __post_init__(self):
        self.total_brands_found = len(self.mentions)

    def get_top_mentions(self, limit: int = 10) -> List[BrandMention]:
        """Get top mentions by relevance score"""
        return sorted(
            self.mentions, key=lambda x: x.calculate_relevance_score(), reverse=True
        )[:limit]

    def get_mentions_by_brand(self, brand_name: str) -> List[BrandMention]:
        """Get all mentions for specific brand"""
        return [m for m in self.mentions if m.brand.lower() == brand_name.lower()]

    def to_summary_dict(self) -> Dict:
        """Summary for logging and monitoring"""
        return {
            "total_mentions": self.total_brands_found,
            "processing_time_ms": self.processing_time_ms,
            "language": self.language,
            "market_adapter": self.market_adapter_used,
            "top_brands": [m.brand for m in self.get_top_mentions(5)],
            "avg_confidence": sum(m.confidence for m in self.mentions)
            / len(self.mentions)
            if self.mentions
            else 0,
        }
