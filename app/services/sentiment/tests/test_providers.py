"""
Test suite for sentiment analysis providers.

Tests individual provider implementations including VADER, business context,
and transformer providers.
"""

from unittest.mock import patch

import pytest

from app.services.sentiment.core.models import (
    AnalysisContext,
    ContextType,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from app.services.sentiment.providers.base import SentimentProviderError
from app.services.sentiment.providers.business_provider import BusinessContextProvider
from app.services.sentiment.providers.vader_provider import VaderSentimentProvider


@pytest.mark.asyncio
class TestVaderProvider:
    """Test VADER sentiment analysis provider"""

    async def test_vader_initialization(self):
        """Test VADER provider initialization"""
        provider = VaderSentimentProvider()
        assert provider.method == SentimentMethod.VADER
        assert not provider.is_available()

        await provider.initialize()
        assert provider.is_available()
        assert provider.analyzer is not None

        await provider.cleanup()

    async def test_vader_basic_analysis(self):
        """Test basic VADER sentiment analysis"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        # Test positive sentiment
        result = await provider.analyze("I love this product! It's amazing!")
        assert result.polarity == SentimentPolarity.POSITIVE
        assert result.score > 0
        assert result.confidence > 0
        assert result.method == SentimentMethod.VADER
        assert result.context_type == ContextType.GENERAL

        # Test negative sentiment
        result = await provider.analyze("This is terrible and awful.")
        assert result.polarity == SentimentPolarity.NEGATIVE
        assert result.score < 0

        # Test neutral sentiment
        result = await provider.analyze("The weather is okay today.")
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert abs(result.score) < 0.05

        await provider.cleanup()

    async def test_vader_metadata(self):
        """Test VADER result metadata"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        result = await provider.analyze("Great product!")

        assert "positive" in result.metadata
        assert "negative" in result.metadata
        assert "neutral" in result.metadata
        assert "compound" in result.metadata
        assert "provider" in result.metadata
        assert result.metadata["provider"] == "VADER"

        await provider.cleanup()

    async def test_vader_batch_processing(self):
        """Test VADER batch processing"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay I guess.",
            "Amazing product!",
            "Worst experience ever.",
        ]

        results = await provider.analyze_batch(texts)

        assert len(results) == len(texts)
        assert all(isinstance(r, SentimentResult) for r in results)
        assert all(r.method == SentimentMethod.VADER for r in results)

        # Check that we get varied sentiments
        polarities = [r.polarity for r in results]
        assert SentimentPolarity.POSITIVE in polarities
        assert SentimentPolarity.NEGATIVE in polarities

        await provider.cleanup()

    async def test_vader_empty_text_handling(self):
        """Test VADER handling of empty or invalid text"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        # Empty text should raise error
        with pytest.raises(ValueError):
            await provider.analyze("")

        # Whitespace-only text should raise error
        with pytest.raises(ValueError):
            await provider.analyze("   ")

        await provider.cleanup()

    async def test_vader_language_support(self):
        """Test VADER language support"""
        provider = VaderSentimentProvider()

        assert provider.supports_language("en")
        assert provider.supports_language("english")
        assert not provider.supports_language("es")
        assert not provider.supports_language("fr")

        languages = provider.get_supported_languages()
        assert "en" in languages

    async def test_vader_capabilities(self):
        """Test VADER capabilities reporting"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        capabilities = provider.get_capabilities()

        assert capabilities["method"] == "vader"
        assert capabilities["model_type"] == "rule_based"
        assert capabilities["supports_social_media"] == True
        assert capabilities["supports_emoticons"] == True
        assert capabilities["context_aware"] == False
        assert capabilities["training_required"] == False

        await provider.cleanup()


@pytest.mark.asyncio
class TestBusinessContextProvider:
    """Test business context sentiment analysis provider"""

    async def test_business_provider_initialization(self):
        """Test business context provider initialization"""
        provider = BusinessContextProvider()
        assert provider.method == SentimentMethod.BUSINESS_CONTEXT

        await provider.initialize()
        assert provider.is_available()
        assert provider.vader_provider is not None

        await provider.cleanup()

    async def test_business_analysis_without_brand(self):
        """Test business analysis without brand context"""
        provider = BusinessContextProvider()
        await provider.initialize()

        context = AnalysisContext()  # No brand
        result = await provider.analyze("This is a good product.", context)

        assert result.method == SentimentMethod.BUSINESS_CONTEXT
        assert "no_brand_context" in result.metadata.get("business_analysis", "")

        await provider.cleanup()

    async def test_business_analysis_with_brand(self):
        """Test business analysis with brand context"""
        provider = BusinessContextProvider()
        await provider.initialize()

        context = AnalysisContext(brand="TestBrand")
        text = "I recommend TestBrand for this solution. It's excellent."

        result = await provider.analyze(text, context)

        assert result.method == SentimentMethod.BUSINESS_CONTEXT
        assert result.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.MIXED]

        # Check that business analysis was performed
        assert "business_analysis" in result.metadata
        business_analysis = result.metadata["business_analysis"]
        assert business_analysis["positive_signals"] > 0
        assert business_analysis["brand_mentions"] > 0

        await provider.cleanup()

    async def test_positive_business_patterns(self):
        """Test detection of positive business patterns"""
        provider = BusinessContextProvider()
        await provider.initialize()

        test_cases = [
            "I highly recommend BrandX for this project.",
            "BrandX is an excellent choice for our needs.",
            "We should go with BrandX as the solution.",
            "BrandX is the industry leader in this space.",
            "BrandX provides high-quality service.",
        ]

        context = AnalysisContext(brand="BrandX")

        for text in test_cases:
            result = await provider.analyze(text, context)
            business_analysis = result.metadata["business_analysis"]

            assert business_analysis["positive_signals"] > 0, f"Failed for: {text}"
            assert business_analysis["context_type"] == "positive_business"

        await provider.cleanup()

    async def test_negative_business_patterns(self):
        """Test detection of negative business patterns"""
        provider = BusinessContextProvider()
        await provider.initialize()

        test_cases = [
            "Avoid BrandX at all costs.",
            "BrandX has serious problems with their service.",
            "I regret choosing BrandX for this project.",
            "BrandX is worse than the competition.",
            "BrandX is outdated and obsolete.",
        ]

        context = AnalysisContext(brand="BrandX")

        for text in test_cases:
            result = await provider.analyze(text, context)
            business_analysis = result.metadata["business_analysis"]

            assert business_analysis["negative_signals"] > 0, f"Failed for: {text}"
            assert business_analysis["context_type"] == "negative_business"

        await provider.cleanup()

    async def test_competitive_patterns(self):
        """Test detection of competitive context patterns"""
        provider = BusinessContextProvider()
        await provider.initialize()

        test_cases = [
            "BrandX vs BrandY comparison shows...",
            "We chose BrandY instead of BrandX.",
            "BrandX is a competitor to our solution.",
        ]

        context = AnalysisContext(brand="BrandX")

        for text in test_cases:
            result = await provider.analyze(text, context)
            business_analysis = result.metadata["business_analysis"]

            assert business_analysis["competitive_signals"] > 0, f"Failed for: {text}"

        await provider.cleanup()

    async def test_mixed_business_sentiment(self):
        """Test handling of mixed business signals"""
        provider = BusinessContextProvider()
        await provider.initialize()

        # Text with both positive and negative signals
        text = "I recommend BrandX for some use cases, but avoid it for others due to problems."
        context = AnalysisContext(brand="BrandX")

        result = await provider.analyze(text, context)
        business_analysis = result.metadata["business_analysis"]

        assert business_analysis["positive_signals"] > 0
        assert business_analysis["negative_signals"] > 0
        assert business_analysis["context_type"] == "mixed_business"

        await provider.cleanup()

    async def test_business_batch_processing(self):
        """Test business context batch processing"""
        provider = BusinessContextProvider()
        await provider.initialize()

        texts = [
            "I recommend BrandX highly.",
            "BrandX has some issues.",
            "BrandX is okay for basic needs.",
            "Avoid BrandX completely.",
            "BrandX is excellent quality.",
        ]

        context = AnalysisContext(brand="BrandX")
        results = await provider.analyze_batch(texts, context)

        assert len(results) == len(texts)
        assert all(r.method == SentimentMethod.BUSINESS_CONTEXT for r in results)

        # Should have varied business analysis results
        business_contexts = [
            r.metadata["business_analysis"]["context_type"] for r in results
        ]
        assert "positive_business" in business_contexts
        assert "negative_business" in business_contexts

        await provider.cleanup()

    async def test_business_provider_capabilities(self):
        """Test business context provider capabilities"""
        provider = BusinessContextProvider()
        await provider.initialize()

        capabilities = provider.get_capabilities()

        assert capabilities["method"] == "business_context"
        assert capabilities["supports_brand_analysis"] == True
        assert capabilities["supports_competitive_analysis"] == True
        assert capabilities["supports_business_patterns"] == True
        assert capabilities["context_aware"] == True
        assert capabilities["requires_brand_context"] == True

        # Check pattern counts
        patterns = capabilities["business_patterns"]
        assert patterns["positive_patterns"] > 0
        assert patterns["negative_patterns"] > 0
        assert patterns["competitive_patterns"] > 0

        await provider.cleanup()


@pytest.mark.asyncio
class TestTransformerProvider:
    """Test transformer-based sentiment analysis provider"""

    @pytest.mark.skipif(
        True,  # Skip by default as it requires transformers/torch
        reason="Requires transformers and torch packages",
    )
    async def test_transformer_initialization(self):
        """Test transformer provider initialization"""
        from app.services.sentiment.providers.transformer_provider import (
            TransformerSentimentProvider,
        )

        provider = TransformerSentimentProvider(
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu",  # Force CPU for testing
        )

        assert provider.method == SentimentMethod.TRANSFORMER

        await provider.initialize()
        assert provider.is_available()
        assert provider.model is not None
        assert provider.tokenizer is not None

        await provider.cleanup()

    @pytest.mark.skipif(
        True, reason="Requires transformers and torch packages"  # Skip by default
    )
    async def test_transformer_analysis(self):
        """Test transformer sentiment analysis"""
        from app.services.sentiment.providers.transformer_provider import (
            TransformerSentimentProvider,
        )

        provider = TransformerSentimentProvider(device="cpu")
        await provider.initialize()

        # Test positive sentiment
        result = await provider.analyze("I love this amazing product!")
        assert result.method == SentimentMethod.TRANSFORMER
        assert result.confidence > 0

        # Check metadata
        assert "model_name" in result.metadata
        assert "predicted_label" in result.metadata
        assert "class_scores" in result.metadata

        await provider.cleanup()

    async def test_transformer_import_error(self):
        """Test handling when transformers package is not available"""
        with patch(
            "app.services.sentiment.providers.transformer_provider.TransformerSentimentProvider._setup_device"
        ) as mock_setup:
            mock_setup.side_effect = ImportError("No module named 'transformers'")

            from app.services.sentiment.providers.transformer_provider import (
                TransformerSentimentProvider,
            )

            provider = TransformerSentimentProvider()

            with pytest.raises(SentimentProviderError):
                await provider.initialize()


@pytest.mark.asyncio
class TestProviderBase:
    """Test base provider functionality"""

    async def test_provider_performance_tracking(self):
        """Test that provider performance is tracked correctly"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        # Initial performance stats
        perf = provider.get_performance()
        assert perf.total_requests == 0
        assert perf.successful_requests == 0
        assert perf.failed_requests == 0

        # Successful analysis
        await provider.analyze("Test message")

        perf = provider.get_performance()
        assert perf.total_requests == 1
        assert perf.successful_requests == 1
        assert perf.failed_requests == 0
        assert perf.average_response_time > 0

        await provider.cleanup()

    async def test_provider_error_handling(self):
        """Test provider error handling"""
        provider = VaderSentimentProvider()

        # Try to analyze without initialization
        with pytest.raises(SentimentProviderError):
            await provider.analyze("Test")

    async def test_provider_timeout_handling(self):
        """Test provider timeout handling"""
        provider = VaderSentimentProvider()
        await provider.initialize()

        # Test with very short timeout
        with pytest.raises(SentimentProviderError):
            await provider.analyze("Test message", timeout=0.001)

        await provider.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
