"""
VADER Sentiment Analysis Provider.

Implements the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment
analysis tool as a provider in our modular system. VADER is specifically tuned
for social media text and provides fast, lightweight sentiment analysis.
"""

from typing import List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.utils.logger import get_logger

from ..core.models import (
    AnalysisContext,
    ContextType,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from .base import SentimentProvider, SentimentProviderError

logger = get_logger(__name__)


class VaderSentimentProvider(SentimentProvider):
    """
    VADER sentiment analysis provider.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
    rule-based sentiment analysis tool that is specifically attuned to sentiments
    expressed in social media, but works well on other text domains too.

    Advantages:
    - Fast and lightweight
    - Works well on social media text
    - No need for training data
    - Good baseline performance

    Limitations:
    - Limited context understanding
    - May struggle with sarcasm/irony
    - Not domain-specific
    """

    def __init__(self):
        super().__init__(SentimentMethod.VADER, "VADER")
        self.analyzer: Optional[SentimentIntensityAnalyzer] = None

    async def initialize(self) -> None:
        """Initialize VADER analyzer"""
        try:
            logger.info("Initializing VADER sentiment analyzer")
            self.analyzer = SentimentIntensityAnalyzer()

            # Test the analyzer with a simple example
            test_scores = self.analyzer.polarity_scores("This is a good test.")
            if not test_scores or "compound" not in test_scores:
                raise ValueError("VADER analyzer test failed")

            logger.info("VADER sentiment analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VADER analyzer: {e}")
            raise SentimentProviderError(
                f"VADER initialization failed: {e}", self.name, retryable=False
            )

    async def cleanup(self) -> None:
        """Clean up VADER resources (minimal cleanup needed)"""
        self.analyzer = None
        logger.info("VADER provider cleaned up")

    async def _analyze_text(
        self, text: str, context: Optional[AnalysisContext] = None
    ) -> SentimentResult:
        """
        Analyze sentiment using VADER.

        Args:
            text: Text to analyze
            context: Analysis context (not used by VADER but preserved for interface)

        Returns:
            SentimentResult with VADER analysis
        """
        if not self.analyzer:
            raise SentimentProviderError(
                "VADER analyzer not initialized", self.name, retryable=False
            )

        try:
            # Validate and clean text
            cleaned_text = self._validate_text(text)

            # Run VADER analysis
            scores = self.analyzer.polarity_scores(cleaned_text)
            compound_score = scores.get("compound", 0.0)

            # Convert VADER compound score to our polarity system
            polarity = self._score_to_polarity(compound_score)

            # Calculate confidence based on score magnitude
            # VADER compound scores range from -1 to 1
            confidence = min(
                1.0, abs(compound_score) * 1.5
            )  # Scale to increase sensitivity

            # Determine context type (VADER doesn't do context analysis)
            context_type = ContextType.GENERAL

            return SentimentResult(
                polarity=polarity,
                score=compound_score,
                confidence=confidence,
                method=SentimentMethod.VADER,
                context_type=context_type,
                metadata={
                    "positive": scores.get("pos", 0.0),
                    "negative": scores.get("neg", 0.0),
                    "neutral": scores.get("neu", 0.0),
                    "compound": compound_score,
                    "text_length": len(cleaned_text),
                    "provider": "VADER",
                },
            )

        except Exception as e:
            logger.error(f"VADER analysis failed for text length {len(text)}: {e}")
            raise SentimentProviderError(f"VADER analysis failed: {e}", self.name)

    def _score_to_polarity(self, compound_score: float) -> SentimentPolarity:
        """
        Convert VADER compound score to sentiment polarity.

        VADER compound score interpretation:
        - >= 0.05: positive
        - <= -0.05: negative
        - between -0.05 and 0.05: neutral
        """
        if compound_score >= 0.05:
            return SentimentPolarity.POSITIVE
        elif compound_score <= -0.05:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL

    def supports_language(self, language: str) -> bool:
        """VADER primarily supports English"""
        return language.lower() in ["en", "english"]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en"]

    def supports_batch_processing(self) -> bool:
        """VADER can be optimized for batch processing"""
        return True

    async def analyze_batch(
        self, texts: List[str], context: Optional[AnalysisContext] = None
    ) -> List[SentimentResult]:
        """
        Batch analyze texts using VADER.

        VADER is fast enough that we can process sequentially without
        significant performance loss, but we optimize by avoiding
        async overhead per text.
        """
        if not self.analyzer:
            raise SentimentProviderError(
                "VADER analyzer not initialized", self.name, retryable=False
            )

        if not texts:
            return []

        results = []

        try:
            for i, text in enumerate(texts):
                try:
                    # Process synchronously for speed
                    cleaned_text = self._validate_text(text)
                    scores = self.analyzer.polarity_scores(cleaned_text)
                    compound_score = scores.get("compound", 0.0)

                    polarity = self._score_to_polarity(compound_score)
                    confidence = min(1.0, abs(compound_score) * 1.5)

                    result = SentimentResult(
                        polarity=polarity,
                        score=compound_score,
                        confidence=confidence,
                        method=SentimentMethod.VADER,
                        context_type=ContextType.GENERAL,
                        text_length=len(cleaned_text),
                        metadata={
                            "positive": scores.get("pos", 0.0),
                            "negative": scores.get("neg", 0.0),
                            "neutral": scores.get("neu", 0.0),
                            "compound": compound_score,
                            "batch_index": i,
                            "provider": "VADER",
                        },
                    )

                    results.append(result)

                except Exception as e:
                    logger.warning(f"VADER batch analysis failed for text {i}: {e}")
                    # Create error result for this text
                    error_result = SentimentResult(
                        polarity=SentimentPolarity.NEUTRAL,
                        score=0.0,
                        confidence=0.0,
                        method=SentimentMethod.VADER,
                        context_type=ContextType.ERROR,
                        metadata={
                            "error": str(e),
                            "batch_index": i,
                            "provider": "VADER",
                        },
                    )
                    results.append(error_result)

            return results

        except Exception as e:
            logger.error(f"VADER batch analysis failed: {e}")
            raise SentimentProviderError(f"VADER batch analysis failed: {e}", self.name)

    def get_capabilities(self) -> dict:
        """Get VADER provider capabilities"""
        base_capabilities = super().get_capabilities()
        base_capabilities.update(
            {
                "model_type": "rule_based",
                "model_name": "VADER",
                "supports_social_media": True,
                "supports_emoticons": True,
                "supports_slang": True,
                "context_aware": False,
                "training_required": False,
                "resource_requirements": "low",
            }
        )
        return base_capabilities
