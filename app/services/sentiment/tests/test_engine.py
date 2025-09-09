"""
Test suite for SentimentEngine core functionality.

Tests the main sentiment analysis engine including provider management,
ensemble methods, caching, and error handling.
"""


import pytest

from app.services.sentiment.core.engine import (
    NoProvidersAvailableError,
    SentimentEngine,
)
from app.services.sentiment.core.models import (
    AnalysisContext,
    ContextType,
    EnsembleStrategy,
    SentimentConfig,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from app.services.sentiment.providers.base import (
    SentimentProvider,
    SentimentProviderError,
)


class MockSentimentProvider(SentimentProvider):
    """Mock provider for testing"""

    def __init__(
        self,
        method: SentimentMethod,
        fail: bool = False,
        result: SentimentResult = None,
    ):
        super().__init__(method, f"Mock{method.value}")
        self.fail = fail
        self.mock_result = result or SentimentResult(
            polarity=SentimentPolarity.POSITIVE,
            score=0.8,
            confidence=0.9,
            method=method,
            context_type=ContextType.GENERAL,
        )
        self.analyze_calls = []

    async def initialize(self) -> None:
        if self.fail:
            raise SentimentProviderError("Mock initialization failure", self.name)

    async def cleanup(self) -> None:
        pass

    async def _analyze_text(self, text: str, context=None) -> SentimentResult:
        self.analyze_calls.append((text, context))
        if self.fail:
            raise SentimentProviderError("Mock analysis failure", self.name)
        return self.mock_result


@pytest.fixture
def default_config():
    """Default configuration for testing"""
    return SentimentConfig(
        enabled_providers=[SentimentMethod.VADER],
        primary_provider=SentimentMethod.VADER,
        fallback_provider=SentimentMethod.VADER,
        enable_caching=False,  # Disable caching for simpler testing
    )


@pytest.fixture
def ensemble_config():
    """Configuration with multiple providers for ensemble testing"""
    return SentimentConfig(
        enabled_providers=[SentimentMethod.VADER, SentimentMethod.BUSINESS_CONTEXT],
        primary_provider=SentimentMethod.VADER,
        ensemble_strategy=EnsembleStrategy.CONFIDENCE_WEIGHTED,
        enable_caching=False,
    )


@pytest.mark.asyncio
class TestSentimentEngineInitialization:
    """Test sentiment engine initialization"""

    async def test_engine_initialization(self, default_config):
        """Test basic engine initialization"""
        engine = SentimentEngine(default_config)
        assert not engine._initialized

        await engine.initialize()
        assert engine._initialized

        await engine.cleanup()

    async def test_initialization_with_mock_provider(self, default_config):
        """Test initialization with mock provider"""
        engine = SentimentEngine(default_config)

        # Replace provider initialization with mock
        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()
        assert engine._initialized

        await engine.cleanup()

    async def test_initialization_failure_handling(self, default_config):
        """Test handling of provider initialization failures"""
        engine = SentimentEngine(default_config)

        # Mock a failing provider
        failing_provider = MockSentimentProvider(SentimentMethod.VADER, fail=True)
        engine.providers[SentimentMethod.VADER] = failing_provider

        # Initialization should not fail completely
        await engine.initialize()
        assert engine._initialized
        assert not engine.provider_health.get(SentimentMethod.VADER, True)


@pytest.mark.asyncio
class TestSingleProviderAnalysis:
    """Test single provider sentiment analysis"""

    async def test_basic_analysis(self, default_config):
        """Test basic sentiment analysis"""
        engine = SentimentEngine(default_config)

        # Set up mock provider
        mock_result = SentimentResult(
            polarity=SentimentPolarity.POSITIVE,
            score=0.7,
            confidence=0.8,
            method=SentimentMethod.VADER,
            context_type=ContextType.GENERAL,
        )
        mock_provider = MockSentimentProvider(SentimentMethod.VADER, result=mock_result)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        result = await engine.analyze("This is a positive test message")

        assert result.polarity == SentimentPolarity.POSITIVE
        assert result.score == 0.7
        assert result.confidence == 0.8
        assert result.method == SentimentMethod.VADER

        # Check that the provider was called
        assert len(mock_provider.analyze_calls) == 1
        assert mock_provider.analyze_calls[0][0] == "This is a positive test message"

        await engine.cleanup()

    async def test_analysis_with_context(self, default_config):
        """Test analysis with context"""
        engine = SentimentEngine(default_config)

        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        context = AnalysisContext(brand="TestBrand", language="en")
        await engine.analyze("Test message", context=context)

        # Check that context was passed to provider
        assert len(mock_provider.analyze_calls) == 1
        call_context = mock_provider.analyze_calls[0][1]
        assert call_context.brand == "TestBrand"
        assert call_context.language == "en"

        await engine.cleanup()

    async def test_provider_failure_fallback(self, default_config):
        """Test fallback when primary provider fails"""
        # Configure with different fallback
        config = SentimentConfig(
            enabled_providers=[SentimentMethod.VADER, SentimentMethod.BUSINESS_CONTEXT],
            primary_provider=SentimentMethod.BUSINESS_CONTEXT,
            fallback_provider=SentimentMethod.VADER,
            enable_caching=False,
        )

        engine = SentimentEngine(config)

        # Set up failing primary provider
        failing_provider = MockSentimentProvider(
            SentimentMethod.BUSINESS_CONTEXT, fail=True
        )
        engine.providers[SentimentMethod.BUSINESS_CONTEXT] = failing_provider
        engine.provider_health[SentimentMethod.BUSINESS_CONTEXT] = True

        # Set up working fallback provider
        fallback_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = fallback_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        result = await engine.analyze("Test message")

        # Should get result from fallback provider
        assert result.method == SentimentMethod.VADER
        assert len(fallback_provider.analyze_calls) == 1

        # Check that fallback was used
        assert engine.engine_stats["fallback_uses"] == 1

        await engine.cleanup()


@pytest.mark.asyncio
class TestEnsembleAnalysis:
    """Test ensemble sentiment analysis"""

    async def test_confidence_weighted_ensemble(self, ensemble_config):
        """Test confidence-weighted ensemble combination"""
        engine = SentimentEngine(ensemble_config)

        # Set up providers with different confidence scores
        provider1_result = SentimentResult(
            polarity=SentimentPolarity.POSITIVE,
            score=0.6,
            confidence=0.9,  # High confidence
            method=SentimentMethod.VADER,
            context_type=ContextType.GENERAL,
        )
        provider1 = MockSentimentProvider(
            SentimentMethod.VADER, result=provider1_result
        )

        provider2_result = SentimentResult(
            polarity=SentimentPolarity.POSITIVE,
            score=0.8,
            confidence=0.5,  # Lower confidence
            method=SentimentMethod.BUSINESS_CONTEXT,
            context_type=ContextType.BUSINESS,
        )
        provider2 = MockSentimentProvider(
            SentimentMethod.BUSINESS_CONTEXT, result=provider2_result
        )

        engine.providers[SentimentMethod.VADER] = provider1
        engine.providers[SentimentMethod.BUSINESS_CONTEXT] = provider2
        engine.provider_health[SentimentMethod.VADER] = True
        engine.provider_health[SentimentMethod.BUSINESS_CONTEXT] = True

        await engine.initialize()

        result = await engine.analyze("Test message for ensemble")

        # Result should be ensemble
        assert result.method == SentimentMethod.ENSEMBLE
        assert result.polarity == SentimentPolarity.POSITIVE

        # Score should be weighted toward higher confidence result
        # (0.6 * 0.9 + 0.8 * 0.5) / (0.9 + 0.5) = 0.64
        expected_score = (0.6 * 0.9 + 0.8 * 0.5) / (0.9 + 0.5)
        assert abs(result.score - expected_score) < 0.01

        # Confidence should be average + ensemble bonus
        expected_confidence = (0.9 + 0.5) / 2 + 0.1  # ensemble bonus
        assert abs(result.confidence - expected_confidence) < 0.01

        # Both providers should have been called
        assert len(provider1.analyze_calls) == 1
        assert len(provider2.analyze_calls) == 1

        await engine.cleanup()

    async def test_ensemble_with_partial_failure(self, ensemble_config):
        """Test ensemble when one provider fails"""
        engine = SentimentEngine(ensemble_config)

        # Set up one working and one failing provider
        working_provider = MockSentimentProvider(SentimentMethod.VADER)
        failing_provider = MockSentimentProvider(
            SentimentMethod.BUSINESS_CONTEXT, fail=True
        )

        engine.providers[SentimentMethod.VADER] = working_provider
        engine.providers[SentimentMethod.BUSINESS_CONTEXT] = failing_provider
        engine.provider_health[SentimentMethod.VADER] = True
        engine.provider_health[SentimentMethod.BUSINESS_CONTEXT] = True

        await engine.initialize()

        result = await engine.analyze("Test message")

        # Should get result from working provider only
        assert result.method == SentimentMethod.ENSEMBLE
        assert len(working_provider.analyze_calls) == 1

        # Failure should be recorded
        assert (
            engine.engine_stats["provider_failures"][
                SentimentMethod.BUSINESS_CONTEXT.value
            ]
            == 1
        )

        await engine.cleanup()


@pytest.mark.asyncio
class TestBatchProcessing:
    """Test batch processing functionality"""

    async def test_basic_batch_processing(self, default_config):
        """Test basic batch processing"""
        engine = SentimentEngine(default_config)

        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        texts = ["Positive message", "Negative message", "Neutral message"]
        batch_result = await engine.analyze_batch(texts)

        assert len(batch_result.results) == 3
        assert batch_result.total_processed == 3
        assert batch_result.successful_analyses == 3
        assert batch_result.failed_analyses == 0

        # All should be positive based on mock provider
        for result in batch_result.results:
            assert result.polarity == SentimentPolarity.POSITIVE
            assert result.method == SentimentMethod.VADER

        # Provider should have been called for each text
        assert len(mock_provider.analyze_calls) == 3

        await engine.cleanup()

    async def test_empty_batch(self, default_config):
        """Test handling of empty batch"""
        engine = SentimentEngine(default_config)
        await engine.initialize()

        batch_result = await engine.analyze_batch([])

        assert len(batch_result.results) == 0
        assert batch_result.total_processed == 0

        await engine.cleanup()

    async def test_batch_with_failures(self, default_config):
        """Test batch processing with some failures"""
        config = SentimentConfig(
            enabled_providers=[SentimentMethod.VADER],
            primary_provider=SentimentMethod.VADER,
            enable_caching=False,
            max_concurrent=2,  # Limit concurrency for predictable testing
        )

        engine = SentimentEngine(config)

        # Mock provider that fails on specific texts
        class PartiallyFailingProvider(MockSentimentProvider):
            async def _analyze_text(self, text: str, context=None) -> SentimentResult:
                self.analyze_calls.append((text, context))
                if "fail" in text.lower():
                    raise SentimentProviderError("Mock failure", self.name)
                return self.mock_result

        mock_provider = PartiallyFailingProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        texts = ["Good message", "FAIL message", "Another good message"]
        batch_result = await engine.analyze_batch(texts)

        assert len(batch_result.results) == 3
        assert batch_result.total_processed == 3
        assert batch_result.successful_analyses == 2
        assert batch_result.failed_analyses == 1

        # Check individual results
        assert batch_result.results[0].polarity == SentimentPolarity.POSITIVE  # Success
        assert batch_result.results[1].context_type == ContextType.ERROR  # Failure
        assert batch_result.results[2].polarity == SentimentPolarity.POSITIVE  # Success

        await engine.cleanup()


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases"""

    async def test_no_providers_available(self):
        """Test behavior when no providers are available"""
        config = SentimentConfig(
            enabled_providers=[SentimentMethod.VADER], enable_caching=False
        )

        engine = SentimentEngine(config)
        # Don't set up any providers
        await engine.initialize()

        with pytest.raises(NoProvidersAvailableError):
            await engine.analyze("Test message")

    async def test_engine_stats_tracking(self, default_config):
        """Test that engine statistics are properly tracked"""
        engine = SentimentEngine(default_config)

        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        # Perform multiple analyses
        await engine.analyze("Message 1")
        await engine.analyze("Message 2")

        assert engine.engine_stats["total_requests"] == 2

        await engine.cleanup()

    async def test_health_check(self, default_config):
        """Test health check functionality"""
        engine = SentimentEngine(default_config)

        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        health_status = await engine.health_check()

        assert health_status["engine_status"] == "healthy"
        assert SentimentMethod.VADER.value in health_status["providers"]
        assert (
            health_status["providers"][SentimentMethod.VADER.value]["status"]
            == "healthy"
        )

        await engine.cleanup()

    async def test_get_status(self, default_config):
        """Test status reporting"""
        engine = SentimentEngine(default_config)

        mock_provider = MockSentimentProvider(SentimentMethod.VADER)
        engine.providers[SentimentMethod.VADER] = mock_provider
        engine.provider_health[SentimentMethod.VADER] = True

        await engine.initialize()

        status = engine.get_status()

        assert status["initialized"] == True
        assert status["config"]["primary_provider"] == SentimentMethod.VADER.value
        assert SentimentMethod.VADER.value in status["providers"]

        await engine.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
