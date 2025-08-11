import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..models.brand_mention import SentimentPolarity

logger = logging.getLogger(__name__)


class SentimentMethod(Enum):
    VADER = "vader"
    CUSTOM_BUSINESS = "custom_business"
    HYBRID = "hybrid"


@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis"""

    polarity: SentimentPolarity
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    method: SentimentMethod
    context_type: str
    metadata: Dict[str, Any]


class BusinessContextAnalyzer:
    """Analyze business-specific context around brand mentions"""

    def __init__(self):
        # Business context patterns
        self.positive_business_patterns = [
            # Recommendation patterns
            r"\b(recommend|suggests?|advise|endorse)\b.*?{brand}",
            r"{brand}.*?\b(excellent|outstanding|superior|leading|top|best)\b",
            r"\b(choose|select|opt for|go with)\b.*?{brand}",
            r"{brand}.*?\b(solution|winner|leader|choice)\b",
            # Quality patterns
            r"{brand}.*?\b(high[- ]quality|reliable|trusted|proven)\b",
            r"\b(award[- ]winning|certified|approved)\b.*?{brand}",
            r"{brand}.*?\b(industry[- ]standard|market[- ]leader)\b",
        ]

        self.negative_business_patterns = [
            # Problem patterns
            r"{brand}.*?\b(problem|issue|bug|error|fault|fail)\b",
            r"\b(avoid|skip|ignore)\b.*?{brand}",
            r"{brand}.*?\b(disappointing|terrible|awful|poor)\b",
            r"\b(regret|mistake)\b.*?{brand}",
            # Comparison disadvantages
            r"{brand}.*?\b(worse|inferior|behind|lacking)\b",
            r"\b(better|superior)\b.*?than.*?{brand}",
            r"{brand}.*?\b(outdated|obsolete|deprecated)\b",
        ]

    def analyze_business_context(self, text: str, brand: str) -> Dict[str, Any]:
        """Analyze business context around brand mention"""

        # Escape brand name for regex
        brand_escaped = re.escape(brand)

        context_analysis = {
            "positive_signals": 0,
            "negative_signals": 0,
            "context_type": "neutral",
            "specific_patterns": [],
        }

        # Check positive patterns
        for pattern in self.positive_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            matches = re.finditer(regex_pattern, text, re.IGNORECASE)
            for match in matches:
                context_analysis["positive_signals"] += 1
                context_analysis["specific_patterns"].append(
                    {
                        "type": "positive",
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                    }
                )

        # Check negative patterns
        for pattern in self.negative_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            matches = re.finditer(regex_pattern, text, re.IGNORECASE)
            for match in matches:
                context_analysis["negative_signals"] += 1
                context_analysis["specific_patterns"].append(
                    {
                        "type": "negative",
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                    }
                )

        # Determine overall context type
        total_signals = (
            context_analysis["positive_signals"] + context_analysis["negative_signals"]
        )
        if total_signals > 0:
            positive_ratio = context_analysis["positive_signals"] / total_signals
            if positive_ratio > 0.6:
                context_analysis["context_type"] = "positive_business"
            elif positive_ratio < 0.4:
                context_analysis["context_type"] = "negative_business"
            else:
                context_analysis["context_type"] = "mixed_business"

        return context_analysis


class SentimentAnalyzer:
    """Multi-modal sentiment analysis for brand mentions"""

    def __init__(self):
        self.business_analyzer = BusinessContextAnalyzer()

        # Initialize VADER (lightweight, always available)
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment_vader(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound_score = scores["compound"]

            # Convert to our polarity enum
            if compound_score >= 0.05:
                polarity = SentimentPolarity.POSITIVE
            elif compound_score <= -0.05:
                polarity = SentimentPolarity.NEGATIVE
            else:
                polarity = SentimentPolarity.NEUTRAL

            # Calculate confidence based on score magnitude
            confidence = min(1.0, abs(compound_score) * 2)

            return SentimentAnalysisResult(
                polarity=polarity,
                score=compound_score,
                confidence=confidence,
                method=SentimentMethod.VADER,
                context_type="general",
                metadata={
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                    "compound": scores["compound"],
                },
            )

        except Exception as e:
            logger.error(f"VADER sentiment analysis failed: {e}")
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.VADER,
                context_type="error",
                metadata={"error": str(e)},
            )

    def analyze_business_sentiment(
        self, text: str, brand: str, language: str = "en"
    ) -> SentimentAnalysisResult:
        """Analyze sentiment with business context awareness"""

        # Get business context analysis
        business_context = self.business_analyzer.analyze_business_context(text, brand)

        # Get base sentiment analysis
        base_sentiment = self.analyze_sentiment_vader(text)

        # Adjust sentiment based on business context
        adjusted_score = base_sentiment.score
        adjusted_confidence = base_sentiment.confidence
        linguistic_weight = 1.0  # Default weight
        business_weight = 0.0  # Default weight

        # Business context adjustments
        positive_signals = business_context["positive_signals"]
        negative_signals = business_context["negative_signals"]
        total_signals = positive_signals + negative_signals

        if total_signals > 0:
            business_sentiment_bias = (
                positive_signals - negative_signals
            ) / total_signals

            # Weighted combination of linguistic and business sentiment
            linguistic_weight = 0.7
            business_weight = 0.3

            adjusted_score = (
                linguistic_weight * base_sentiment.score
                + business_weight * business_sentiment_bias
            )

            # Increase confidence if business signals align with linguistic sentiment
            if (business_sentiment_bias > 0 and base_sentiment.score > 0) or (
                business_sentiment_bias < 0 and base_sentiment.score < 0
            ):
                adjusted_confidence = min(1.0, adjusted_confidence + 0.2)

        # Determine final polarity
        if adjusted_score >= 0.1:
            final_polarity = SentimentPolarity.POSITIVE
        elif adjusted_score <= -0.1:
            final_polarity = SentimentPolarity.NEGATIVE
        else:
            final_polarity = SentimentPolarity.NEUTRAL

        # Handle mixed signals
        if total_signals > 2 and abs(positive_signals - negative_signals) <= 1:
            final_polarity = SentimentPolarity.MIXED

        return SentimentAnalysisResult(
            polarity=final_polarity,
            score=adjusted_score,
            confidence=adjusted_confidence,
            method=SentimentMethod.CUSTOM_BUSINESS,
            context_type=business_context["context_type"],
            metadata={
                "base_sentiment": base_sentiment.__dict__,
                "business_context": business_context,
                "adjustment_applied": abs(adjusted_score - base_sentiment.score) > 0.05,
                "linguistic_weight": linguistic_weight,
                "business_weight": business_weight,
            },
        )

    async def analyze_sentiment_hybrid(
        self, text: str, brand: str, language: str = "en"
    ) -> SentimentAnalysisResult:
        """Hybrid sentiment analysis using multiple methods"""

        # Run multiple analyses
        analyses = {}

        # VADER (fast, baseline)
        try:
            analyses["vader"] = self.analyze_sentiment_vader(text)
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")

        # Business context (domain-specific)
        try:
            analyses["business"] = self.analyze_business_sentiment(
                text, brand, language
            )
        except Exception as e:
            logger.error(f"Business sentiment analysis failed: {e}")

        if not analyses:
            # All methods failed
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.HYBRID,
                context_type="error",
                metadata={"error": "All sentiment analysis methods failed"},
            )

        # Combine results intelligently
        combined_result = self._combine_sentiment_results(analyses)
        combined_result.method = SentimentMethod.HYBRID

        return combined_result

    def _combine_sentiment_results(
        self, analyses: Dict[str, SentimentAnalysisResult]
    ) -> SentimentAnalysisResult:
        """Combine multiple sentiment analysis results"""

        if not analyses:
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.HYBRID,
                context_type="error",
                metadata={"error": "No analyses to combine"},
            )

        # Weight different methods
        method_weights = {"vader": 0.4, "business": 0.6}

        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        polarity_votes = {
            SentimentPolarity.POSITIVE: 0,
            SentimentPolarity.NEGATIVE: 0,
            SentimentPolarity.NEUTRAL: 0,
            SentimentPolarity.MIXED: 0,
        }

        for method, result in analyses.items():
            weight = method_weights.get(method, 0.1)
            total_weight += weight

            weighted_score += result.score * weight
            weighted_confidence += result.confidence * weight
            polarity_votes[result.polarity] += weight

        # Normalize weights
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Determine final polarity by voting
        final_polarity = max(polarity_votes.keys(), key=lambda k: polarity_votes[k])

        # Override polarity if score suggests otherwise
        if final_score >= 0.15:
            final_polarity = SentimentPolarity.POSITIVE
        elif final_score <= -0.15:
            final_polarity = SentimentPolarity.NEGATIVE
        elif abs(final_score) < 0.05:
            final_polarity = SentimentPolarity.NEUTRAL

        # Determine context type
        context_types = [result.context_type for result in analyses.values()]
        if "positive_business" in context_types:
            context_type = "positive_business"
        elif "negative_business" in context_types:
            context_type = "negative_business"
        else:
            context_type = "general"

        return SentimentAnalysisResult(
            polarity=final_polarity,
            score=final_score,
            confidence=final_confidence,
            method=SentimentMethod.HYBRID,
            context_type=context_type,
            metadata={
                "individual_results": {k: v.__dict__ for k, v in analyses.items()},
                "method_weights": method_weights,
                "polarity_votes": {k.value: v for k, v in polarity_votes.items()},
                "total_methods": len(analyses),
            },
        )
