"""
Base sentiment analysis provider interface.

Defines the common interface that all sentiment providers must implement.
This ensures consistency and enables easy provider swapping and ensemble methods.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from app.utils.logger import get_logger

from ..core.models import (
    AnalysisContext,
    ContextType,
    ProviderPerformance,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)

logger = get_logger(__name__)


class SentimentProviderError(Exception):
    """Base exception for sentiment provider errors"""

    def __init__(self, message: str, provider: str, retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ProviderTimeoutError(SentimentProviderError):
    """Raised when a provider request times out"""

    def __init__(self, provider: str, timeout: float):
        super().__init__(
            f"Provider {provider} timed out after {timeout}s", provider, retryable=True
        )
        self.timeout = timeout


class ProviderUnavailableError(SentimentProviderError):
    """Raised when a provider is temporarily unavailable"""

    def __init__(self, provider: str, reason: str = "Service unavailable"):
        super().__init__(
            f"Provider {provider} unavailable: {reason}", provider, retryable=True
        )


class ProviderConfigurationError(SentimentProviderError):
    """Raised when a provider is misconfigured"""

    def __init__(self, provider: str, reason: str):
        super().__init__(
            f"Provider {provider} configuration error: {reason}",
            provider,
            retryable=False,
        )


class SentimentProvider(ABC):
    """
    Abstract base class for sentiment analysis providers.

    All sentiment providers must implement this interface to ensure consistency
    and enable provider swapping, ensemble methods, and performance monitoring.
    """

    def __init__(self, method: SentimentMethod, name: str = None):
        self.method = method
        self.name = name or method.value
        self.performance = ProviderPerformance(provider=method)
        self._initialized = False
        self._initialization_error: Optional[Exception] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider (load models, establish connections, etc.).
        Called once before first use.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up provider resources.
        Called when the provider is no longer needed.
        """
        pass

    @abstractmethod
    async def _analyze_text(
        self, text: str, context: Optional[AnalysisContext] = None
    ) -> SentimentResult:
        """
        Analyze sentiment for a single text.

        Args:
            text: The text to analyze
            context: Additional context for analysis

        Returns:
            SentimentResult with analysis results

        Raises:
            SentimentProviderError: If analysis fails
        """
        pass

    async def analyze(
        self,
        text: str,
        context: Optional[AnalysisContext] = None,
        timeout: float = 30.0,
    ) -> SentimentResult:
        """
        Public interface for analyzing sentiment with error handling and monitoring.

        Args:
            text: The text to analyze
            context: Additional context for analysis
            timeout: Timeout in seconds

        Returns:
            SentimentResult with analysis results

        Raises:
            SentimentProviderError: If analysis fails
        """
        if not self._initialized:
            await self._ensure_initialized()

        # Validate input
        if not text or not text.strip():
            return self._create_error_result("Empty text provided")

        if len(text) > 10000:  # Reasonable limit
            logger.warning(
                f"Text length {len(text)} exceeds recommended limit", provider=self.name
            )
            text = text[:10000]  # Truncate

        start_time = time.time()
        success = False
        result = None

        try:
            # Apply timeout
            result = await asyncio.wait_for(
                self._analyze_text(text, context), timeout=timeout
            )

            # Validate result
            if not isinstance(result, SentimentResult):
                raise ValueError(
                    f"Provider returned invalid result type: {type(result)}"
                )

            # Set metadata
            result.text_length = len(text)
            result.processing_time = time.time() - start_time

            success = True
            return result

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(self.name, timeout)

        except SentimentProviderError:
            # Re-raise provider-specific errors
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error in provider {self.name}: {e}", exc_info=True
            )
            raise SentimentProviderError(f"Internal error: {e}", self.name)

        finally:
            # Update performance metrics
            processing_time = time.time() - start_time
            confidence = result.confidence if result and success else 0.0
            self.performance.update_stats(processing_time, success, confidence)

    async def analyze_batch(
        self,
        texts: List[str],
        context: Optional[AnalysisContext] = None,
        timeout: float = 60.0,
    ) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts.

        Default implementation processes texts sequentially. Providers can
        override this for batch optimization.

        Args:
            texts: List of texts to analyze
            context: Additional context for analysis
            timeout: Timeout per text in seconds

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        results = []
        per_text_timeout = min(timeout / len(texts), timeout)

        for text in texts:
            try:
                result = await self.analyze(text, context, per_text_timeout)
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Failed to analyze text in batch: {e}", provider=self.name
                )
                results.append(self._create_error_result(f"Batch analysis failed: {e}"))

        return results

    def is_available(self) -> bool:
        """Check if the provider is available for use"""
        return self._initialized and self._initialization_error is None

    def get_performance(self) -> ProviderPerformance:
        """Get performance statistics for this provider"""
        return self.performance

    def supports_language(self, language: str) -> bool:
        """
        Check if provider supports the given language.
        Default implementation supports English only.
        """
        return language.lower() in ["en", "english"]

    def supports_batch_processing(self) -> bool:
        """Check if provider supports native batch processing"""
        return False

    def get_capabilities(self) -> dict:
        """Get provider capabilities information"""
        return {
            "method": self.method.value,
            "name": self.name,
            "supports_batch": self.supports_batch_processing(),
            "supported_languages": self.get_supported_languages(),
            "is_available": self.is_available(),
            "performance": {
                "total_requests": self.performance.total_requests,
                "error_rate": self.performance.error_rate,
                "average_response_time": self.performance.average_response_time,
                "average_confidence": self.performance.average_confidence,
            },
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en"]

    async def _ensure_initialized(self):
        """Ensure the provider is initialized"""
        if self._initialized:
            return

        if self._initialization_error:
            raise ProviderConfigurationError(
                self.name,
                f"Previous initialization failed: {self._initialization_error}",
            )

        try:
            await self.initialize()
            self._initialized = True
        except Exception as e:
            self._initialization_error = e
            logger.error(
                f"Failed to initialize provider {self.name}: {e}", exc_info=True
            )
            raise ProviderConfigurationError(self.name, str(e))

    def _create_error_result(self, error_message: str) -> SentimentResult:
        """Create a SentimentResult for error cases"""
        return SentimentResult(
            polarity=SentimentPolarity.NEUTRAL,
            score=0.0,
            confidence=0.0,
            method=self.method,
            context_type=ContextType.ERROR,
            metadata={"error": error_message, "provider": self.name},
        )

    def _validate_text(self, text: str) -> str:
        """Validate and clean text input"""
        if not text:
            raise ValueError("Text cannot be empty")

        # Basic text cleaning
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty after cleaning")

        return text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(method={self.method.value}, name={self.name})"
        )
