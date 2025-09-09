"""
Business Context Sentiment Provider.

Implements business-context-aware sentiment analysis that combines linguistic
sentiment with business-specific patterns and contexts. This provider is
specifically designed for competitive intelligence and brand analysis.
"""

import re
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

from ..core.models import (
    AnalysisContext,
    ContextType,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from .base import SentimentProvider, SentimentProviderError
from .vader_provider import VaderSentimentProvider

logger = get_logger(__name__)


class BusinessContextProvider(SentimentProvider):
    """
    Business-context-aware sentiment analysis provider.

    This provider combines traditional linguistic sentiment analysis with
    business-specific pattern matching to provide more accurate sentiment
    analysis for competitive intelligence and brand monitoring use cases.

    Features:
    - Brand-specific context analysis
    - Business pattern recognition (recommendations, comparisons, etc.)
    - Weighted combination of linguistic and business signals
    - Support for competitive analysis contexts
    """

    def __init__(self, linguistic_weight: float = 0.7, business_weight: float = 0.3):
        super().__init__(SentimentMethod.BUSINESS_CONTEXT, "BusinessContext")
        self.linguistic_weight = linguistic_weight
        self.business_weight = business_weight
        self.vader_provider: Optional[VaderSentimentProvider] = None

        # Business context patterns (from existing implementation)
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
            # Satisfaction patterns
            r"{brand}.*?\b(satisfied|happy|pleased|impressed)\b",
            r"\b(love|loved|loving)\b.*?{brand}",
            r"{brand}.*?\b(perfect|amazing|incredible|fantastic)\b",
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
            # Dissatisfaction patterns
            r"{brand}.*?\b(frustrated|annoyed|disappointed|upset)\b",
            r"\b(hate|hated|hating)\b.*?{brand}",
            r"{brand}.*?\b(useless|worthless|horrible)\b",
        ]

        self.competitive_patterns = [
            r"{brand}.*?\b(vs|versus|compared to|against)\b",
            r"\b(alternative to|instead of|rather than)\b.*?{brand}",
            r"{brand}.*?\b(competitor|competition|rival)\b",
        ]

    async def initialize(self) -> None:
        """Initialize the business context provider"""
        try:
            logger.info("Initializing Business Context sentiment provider")

            # Initialize underlying VADER provider for linguistic analysis
            self.vader_provider = VaderSentimentProvider()
            await self.vader_provider.initialize()

            logger.info("Business Context provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Business Context provider: {e}")
            raise SentimentProviderError(
                f"Business Context initialization failed: {e}",
                self.name,
                retryable=False,
            )

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self.vader_provider:
            await self.vader_provider.cleanup()
            self.vader_provider = None
        logger.info("Business Context provider cleaned up")

    async def _analyze_text(
        self, text: str, context: Optional[AnalysisContext] = None
    ) -> SentimentResult:
        """
        Analyze sentiment with business context awareness.

        Args:
            text: Text to analyze
            context: Analysis context including brand information

        Returns:
            SentimentResult with business-context-aware analysis
        """
        if not self.vader_provider:
            raise SentimentProviderError(
                "Business Context provider not initialized", self.name, retryable=False
            )

        # Validate text
        cleaned_text = self._validate_text(text)

        # Get base linguistic sentiment
        base_sentiment = await self.vader_provider._analyze_text(cleaned_text, context)

        # Extract brand from context
        brand = context.brand if context else None
        if not brand:
            # If no brand context, return base sentiment with different method tag
            base_sentiment.method = SentimentMethod.BUSINESS_CONTEXT
            base_sentiment.metadata["business_analysis"] = "no_brand_context"
            return base_sentiment

        # Analyze business context
        business_analysis = self._analyze_business_context(cleaned_text, brand)

        # Combine linguistic and business signals
        combined_result = self._combine_linguistic_and_business_sentiment(
            base_sentiment, business_analysis, cleaned_text, brand
        )

        return combined_result

    def _analyze_business_context(self, text: str, brand: str) -> Dict[str, Any]:
        """
        Analyze business context patterns around brand mentions.

        Args:
            text: Text to analyze
            brand: Brand name to look for

        Returns:
            Dictionary with business context analysis
        """
        # Escape brand name for regex
        brand_escaped = re.escape(brand)

        analysis = {
            "positive_signals": 0,
            "negative_signals": 0,
            "competitive_signals": 0,
            "context_type": "neutral",
            "specific_patterns": [],
            "brand_mentions": 0,
        }

        # Count brand mentions
        brand_pattern = re.compile(re.escape(brand), re.IGNORECASE)
        analysis["brand_mentions"] = len(brand_pattern.findall(text))

        # Check positive business patterns
        for pattern in self.positive_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            try:
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                for match in matches:
                    analysis["positive_signals"] += 1
                    analysis["specific_patterns"].append(
                        {
                            "type": "positive_business",
                            "pattern": pattern,
                            "match": match.group(),
                            "position": match.span(),
                        }
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")

        # Check negative business patterns
        for pattern in self.negative_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            try:
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                for match in matches:
                    analysis["negative_signals"] += 1
                    analysis["specific_patterns"].append(
                        {
                            "type": "negative_business",
                            "pattern": pattern,
                            "match": match.group(),
                            "position": match.span(),
                        }
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")

        # Check competitive patterns
        for pattern in self.competitive_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            try:
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                for match in matches:
                    analysis["competitive_signals"] += 1
                    analysis["specific_patterns"].append(
                        {
                            "type": "competitive",
                            "pattern": pattern,
                            "match": match.group(),
                            "position": match.span(),
                        }
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")

        # Determine context type
        total_business_signals = (
            analysis["positive_signals"] + analysis["negative_signals"]
        )

        if total_business_signals > 0:
            positive_ratio = analysis["positive_signals"] / total_business_signals
            if positive_ratio > 0.6:
                analysis["context_type"] = "positive_business"
            elif positive_ratio < 0.4:
                analysis["context_type"] = "negative_business"
            else:
                analysis["context_type"] = "mixed_business"

        if analysis["competitive_signals"] > 0:
            analysis["context_type"] = f"{analysis['context_type']}_competitive"

        return analysis

    def _combine_linguistic_and_business_sentiment(
        self,
        linguistic_result: SentimentResult,
        business_analysis: Dict[str, Any],
        text: str,
        brand: str,
    ) -> SentimentResult:
        """
        Combine linguistic sentiment with business context analysis.

        Args:
            linguistic_result: Base linguistic sentiment from VADER
            business_analysis: Business context analysis results
            text: Original text
            brand: Brand name

        Returns:
            Combined SentimentResult
        """
        # Start with linguistic sentiment
        base_score = linguistic_result.score
        base_confidence = linguistic_result.confidence

        # Calculate business sentiment bias
        positive_signals = business_analysis["positive_signals"]
        negative_signals = business_analysis["negative_signals"]
        total_signals = positive_signals + negative_signals

        business_sentiment_bias = 0.0
        if total_signals > 0:
            business_sentiment_bias = (
                positive_signals - negative_signals
            ) / total_signals

        # Combine linguistic and business sentiment with weights
        if total_signals > 0:
            adjusted_score = (
                self.linguistic_weight * base_score
                + self.business_weight * business_sentiment_bias
            )

            # Boost confidence if business signals align with linguistic sentiment
            confidence_boost = 0.0
            if (business_sentiment_bias > 0 and base_score > 0) or (
                business_sentiment_bias < 0 and base_score < 0
            ):
                confidence_boost = min(0.3, total_signals * 0.1)

            adjusted_confidence = min(1.0, base_confidence + confidence_boost)
        else:
            # No business signals, use linguistic sentiment
            adjusted_score = base_score
            adjusted_confidence = base_confidence

        # Determine final polarity
        final_polarity = self._score_to_polarity(adjusted_score, total_signals)

        # Determine context type
        context_type = self._determine_context_type(business_analysis)

        # Create comprehensive metadata
        metadata = {
            "linguistic_sentiment": linguistic_result.to_dict(),
            "business_analysis": business_analysis,
            "adjustment_applied": abs(adjusted_score - base_score) > 0.05,
            "linguistic_weight": self.linguistic_weight,
            "business_weight": self.business_weight,
            "brand": brand,
            "business_sentiment_bias": business_sentiment_bias,
            "confidence_boost": adjusted_confidence - base_confidence
            if total_signals > 0
            else 0.0,
        }

        return SentimentResult(
            polarity=final_polarity,
            score=adjusted_score,
            confidence=adjusted_confidence,
            method=SentimentMethod.BUSINESS_CONTEXT,
            context_type=context_type,
            text_length=len(text),
            metadata=metadata,
        )

    def _score_to_polarity(
        self, score: float, business_signals: int
    ) -> SentimentPolarity:
        """
        Convert combined score to polarity, considering business signals.

        Business context can create mixed sentiment scenarios.
        """
        # Use slightly wider thresholds when business signals present
        threshold = 0.15 if business_signals > 2 else 0.1

        if score >= threshold:
            return SentimentPolarity.POSITIVE
        elif score <= -threshold:
            return SentimentPolarity.NEGATIVE
        elif business_signals > 2 and abs(score) < 0.05:
            # Strong business signals but neutral linguistic sentiment = mixed
            return SentimentPolarity.MIXED
        else:
            return SentimentPolarity.NEUTRAL

    def _determine_context_type(self, business_analysis: Dict[str, Any]) -> ContextType:
        """Determine context type from business analysis"""
        context_str = business_analysis["context_type"]

        context_mapping = {
            "positive_business": ContextType.POSITIVE_BUSINESS,
            "negative_business": ContextType.NEGATIVE_BUSINESS,
            "mixed_business": ContextType.MIXED_BUSINESS,
            "positive_business_competitive": ContextType.COMPARATIVE,
            "negative_business_competitive": ContextType.COMPARATIVE,
            "mixed_business_competitive": ContextType.COMPARATIVE,
            "neutral_competitive": ContextType.COMPARATIVE,
            "neutral": ContextType.BUSINESS,
        }

        return context_mapping.get(context_str, ContextType.BUSINESS)

    def supports_language(self, language: str) -> bool:
        """Business context patterns are primarily English"""
        return language.lower() in ["en", "english"]

    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return ["en"]

    def supports_batch_processing(self) -> bool:
        """Supports batch processing through VADER optimization"""
        return True

    async def analyze_batch(
        self, texts: List[str], context: Optional[AnalysisContext] = None
    ) -> List[SentimentResult]:
        """
        Batch analyze texts with business context.

        Optimizes by doing batch VADER analysis first, then applying
        business context analysis to each result.
        """
        if not texts:
            return []

        if not self.vader_provider:
            raise SentimentProviderError(
                "Business Context provider not initialized", self.name, retryable=False
            )

        # Get base linguistic sentiments in batch
        linguistic_results = await self.vader_provider.analyze_batch(texts, context)

        # Apply business context analysis to each
        results = []
        brand = context.brand if context else None

        for i, (text, linguistic_result) in enumerate(zip(texts, linguistic_results)):
            try:
                if brand:
                    business_analysis = self._analyze_business_context(text, brand)
                    combined_result = self._combine_linguistic_and_business_sentiment(
                        linguistic_result, business_analysis, text, brand
                    )
                else:
                    # No brand context, just update method
                    combined_result = linguistic_result
                    combined_result.method = SentimentMethod.BUSINESS_CONTEXT
                    combined_result.metadata["business_analysis"] = "no_brand_context"

                results.append(combined_result)

            except Exception as e:
                logger.warning(
                    f"Business context analysis failed for batch item {i}: {e}"
                )
                # Create error result
                error_result = SentimentResult(
                    polarity=SentimentPolarity.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    method=SentimentMethod.BUSINESS_CONTEXT,
                    context_type=ContextType.ERROR,
                    metadata={
                        "error": str(e),
                        "batch_index": i,
                        "provider": "BusinessContext",
                    },
                )
                results.append(error_result)

        return results

    def get_capabilities(self) -> dict:
        """Get business context provider capabilities"""
        base_capabilities = super().get_capabilities()
        base_capabilities.update(
            {
                "model_type": "hybrid_rule_linguistic",
                "model_name": "BusinessContext",
                "supports_brand_analysis": True,
                "supports_competitive_analysis": True,
                "supports_business_patterns": True,
                "context_aware": True,
                "requires_brand_context": True,
                "linguistic_provider": "VADER",
                "business_patterns": {
                    "positive_patterns": len(self.positive_business_patterns),
                    "negative_patterns": len(self.negative_business_patterns),
                    "competitive_patterns": len(self.competitive_patterns),
                },
            }
        )
        return base_capabilities
