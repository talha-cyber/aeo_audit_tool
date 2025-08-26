"""
AI Platform Client Module

Provides unified interface for querying multiple AI platforms (OpenAI, Anthropic,
Perplexity, Google AI)
with rate limiting, error handling, and circuit breaker functionality.
"""

from .anthropic_client import AnthropicPlatform
from .base import AIRateLimiter, BasePlatform
from .exceptions import (
    AuthenticationError,
    MalformedResponseError,
    PlatformError,
    QuotaExceededError,
    RateLimitError,
    TransientError,
)
from .google_ai_client import GoogleAIPlatform
from .openai_client import OpenAIPlatform
from .perplexity_client import PerplexityPlatform
from .registry import PlatformRegistry

__all__ = [
    "BasePlatform",
    "AIRateLimiter",
    "PlatformError",
    "TransientError",
    "RateLimitError",
    "AuthenticationError",
    "QuotaExceededError",
    "MalformedResponseError",
    "OpenAIPlatform",
    "AnthropicPlatform",
    "PerplexityPlatform",
    "GoogleAIPlatform",
    "PlatformRegistry",
]

