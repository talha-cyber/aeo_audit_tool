"""
Unit tests for platform-specific implementations.

Tests text extraction, response handling, and configuration for each
AI platform client (OpenAI, Anthropic, Perplexity, Google AI).
"""

import pytest

from app.services.ai_platforms.anthropic_client import AnthropicPlatform
from app.services.ai_platforms.google_ai_client import GoogleAIPlatform
from app.services.ai_platforms.openai_client import OpenAIPlatform
from app.services.ai_platforms.perplexity_client import PerplexityPlatform


class TestOpenAIPlatform:
    """Test cases for OpenAI platform implementation."""

    def test_initialization_defaults(self):
        """Test OpenAI platform initialization with defaults."""
        platform = OpenAIPlatform("test_key")

        assert platform.api_key == "test_key"
        assert platform.base_url == "https://api.openai.com/v1"
        assert platform.default_model == "gpt-4"
        assert platform.max_tokens == 500
        assert platform.temperature == 0.1
        assert platform.platform_name == "openai"

    def test_initialization_custom_config(self):
        """Test OpenAI platform initialization with custom configuration."""
        config = {
            "base_url": "https://custom.openai.com/v1",
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.5,
        }

        platform = OpenAIPlatform("test_key", rate_limit=100, **config)

        assert platform.base_url == "https://custom.openai.com/v1"
        assert platform.default_model == "gpt-3.5-turbo"
        assert platform.max_tokens == 1000
        assert platform.temperature == 0.5

    def test_default_headers(self):
        """Test OpenAI default headers."""
        platform = OpenAIPlatform("test_key")
        headers = platform._get_default_headers()

        expected_headers = {
            "Authorization": "Bearer test_key",
            "Content-Type": "application/json",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

        assert headers == expected_headers

    def test_endpoint_url(self):
        """Test OpenAI endpoint URL construction."""
        platform = OpenAIPlatform("test_key")
        url = platform._get_endpoint_url()

        assert url == "https://api.openai.com/v1/chat/completions"

    def test_request_payload_preparation(self):
        """Test OpenAI request payload preparation."""
        platform = OpenAIPlatform("test_key")
        payload = platform._prepare_request_payload("Test question")

        expected_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test question"}],
            "max_tokens": 500,
            "temperature": 0.1,
            "stream": False,
        }

        assert payload == expected_payload

    def test_request_payload_with_overrides(self):
        """Test OpenAI request payload with parameter overrides."""
        platform = OpenAIPlatform("test_key")
        payload = platform._prepare_request_payload(
            "Test question", model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7
        )

        assert payload["model"] == "gpt-3.5-turbo"
        assert payload["max_tokens"] == 1000
        assert payload["temperature"] == 0.7

    def test_text_extraction_success(self):
        """Test successful text extraction from OpenAI response."""
        platform = OpenAIPlatform("test_key")

        response = {"choices": [{"message": {"content": "This is a test response"}}]}

        text = platform.extract_text_response(response)
        assert text == "This is a test response"

    def test_text_extraction_with_whitespace(self):
        """Test text extraction with whitespace trimming."""
        platform = OpenAIPlatform("test_key")

        response = {
            "choices": [{"message": {"content": "  \n  This is a test response  \n  "}}]
        }

        text = platform.extract_text_response(response)
        assert text == "This is a test response"

    def test_text_extraction_malformed_response(self):
        """Test text extraction with malformed response."""
        platform = OpenAIPlatform("test_key")

        # Missing choices
        with pytest.raises(ValueError, match="Invalid OpenAI response format"):
            platform.extract_text_response({"invalid": "format"})

        # Empty choices
        with pytest.raises(ValueError, match="Invalid OpenAI response format"):
            platform.extract_text_response({"choices": []})

        # Missing message
        with pytest.raises(ValueError, match="Invalid OpenAI response format"):
            platform.extract_text_response({"choices": [{"invalid": "format"}]})

    @pytest.mark.asyncio
    async def test_query_not_implemented(self):
        """Test that direct query method raises NotImplementedError."""
        platform = OpenAIPlatform("test_key")

        with pytest.raises(NotImplementedError, match="Use safe_query\\(\\) instead"):
            await platform.query("Test question")


class TestAnthropicPlatform:
    """Test cases for Anthropic platform implementation."""

    def test_initialization_defaults(self):
        """Test Anthropic platform initialization with defaults."""
        platform = AnthropicPlatform("test_key")

        assert platform.api_key == "test_key"
        assert platform.base_url == "https://api.anthropic.com"
        assert platform.default_model == "claude-3-sonnet-20240229"
        assert platform.max_tokens == 500
        assert platform.temperature == 0.1
        assert platform.platform_name == "anthropic"

    def test_default_headers(self):
        """Test Anthropic default headers."""
        platform = AnthropicPlatform("test_key")
        headers = platform._get_default_headers()

        expected_headers = {
            "x-api-key": "test_key",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

        assert headers == expected_headers

    def test_endpoint_url(self):
        """Test Anthropic endpoint URL construction."""
        platform = AnthropicPlatform("test_key")
        url = platform._get_endpoint_url()

        assert url == "https://api.anthropic.com/v1/messages"

    def test_request_payload_preparation(self):
        """Test Anthropic request payload preparation."""
        platform = AnthropicPlatform("test_key")
        payload = platform._prepare_request_payload("Test question")

        expected_payload = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Test question"}],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        assert payload == expected_payload

    def test_text_extraction_success(self):
        """Test successful text extraction from Anthropic response."""
        platform = AnthropicPlatform("test_key")

        response = {"content": [{"text": "This is a Claude response"}]}

        text = platform.extract_text_response(response)
        assert text == "This is a Claude response"

    def test_text_extraction_malformed_response(self):
        """Test text extraction with malformed Anthropic response."""
        platform = AnthropicPlatform("test_key")

        # Missing content
        with pytest.raises(ValueError, match="Invalid Anthropic response format"):
            platform.extract_text_response({"invalid": "format"})

        # Empty content
        with pytest.raises(ValueError, match="Invalid Anthropic response format"):
            platform.extract_text_response({"content": []})

        # Missing text
        with pytest.raises(ValueError, match="Invalid Anthropic response format"):
            platform.extract_text_response({"content": [{"invalid": "format"}]})


class TestPerplexityPlatform:
    """Test cases for Perplexity platform implementation."""

    def test_initialization_defaults(self):
        """Test Perplexity platform initialization with defaults."""
        platform = PerplexityPlatform("test_key")

        assert platform.api_key == "test_key"
        assert platform.base_url == "https://api.perplexity.ai"
        assert platform.default_model == "llama-3.1-sonar-small-128k-online"
        assert platform.max_tokens == 500
        assert platform.temperature == 0.1
        assert platform.platform_name == "perplexity"

    def test_default_headers(self):
        """Test Perplexity default headers."""
        platform = PerplexityPlatform("test_key")
        headers = platform._get_default_headers()

        expected_headers = {
            "Authorization": "Bearer test_key",
            "Content-Type": "application/json",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

        assert headers == expected_headers

    def test_endpoint_url(self):
        """Test Perplexity endpoint URL construction."""
        platform = PerplexityPlatform("test_key")
        url = platform._get_endpoint_url()

        assert url == "https://api.perplexity.ai/chat/completions"

    def test_request_payload_preparation(self):
        """Test Perplexity request payload preparation."""
        platform = PerplexityPlatform("test_key")
        payload = platform._prepare_request_payload("Test question")

        expected_payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": "Test question"}],
            "max_tokens": 500,
            "temperature": 0.1,
            "stream": False,
        }

        assert payload == expected_payload

    def test_text_extraction_success(self):
        """Test successful text extraction from Perplexity response."""
        platform = PerplexityPlatform("test_key")

        response = {
            "choices": [{"message": {"content": "This is a Perplexity response"}}]
        }

        text = platform.extract_text_response(response)
        assert text == "This is a Perplexity response"

    def test_text_extraction_malformed_response(self):
        """Test text extraction with malformed Perplexity response."""
        platform = PerplexityPlatform("test_key")

        with pytest.raises(ValueError, match="Invalid Perplexity response format"):
            platform.extract_text_response({"invalid": "format"})


class TestGoogleAIPlatform:
    """Test cases for Google AI platform implementation."""

    def test_initialization_defaults(self):
        """Test Google AI platform initialization with defaults."""
        platform = GoogleAIPlatform("test_key")

        assert platform.api_key == "test_key"
        assert platform.base_url == "https://generativelanguage.googleapis.com"
        assert platform.default_model == "gemini-pro"
        assert platform.max_tokens == 500
        assert platform.temperature == 0.1
        assert platform.platform_name == "googleai"

    def test_default_headers(self):
        """Test Google AI default headers."""
        platform = GoogleAIPlatform("test_key")
        headers = platform._get_default_headers()

        expected_headers = {
            "Content-Type": "application/json",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

        assert headers == expected_headers

    def test_endpoint_url(self):
        """Test Google AI endpoint URL construction."""
        platform = GoogleAIPlatform("test_key")
        url = platform._get_endpoint_url()

        expected_url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=test_key"
        assert url == expected_url

    def test_request_payload_preparation(self):
        """Test Google AI request payload preparation."""
        platform = GoogleAIPlatform("test_key")
        payload = platform._prepare_request_payload("Test question")

        expected_payload = {
            "contents": [{"parts": [{"text": "Test question"}]}],
            "generationConfig": {"maxOutputTokens": 500, "temperature": 0.1},
        }

        assert payload == expected_payload

    def test_request_payload_with_overrides(self):
        """Test Google AI request payload with parameter overrides."""
        platform = GoogleAIPlatform("test_key")
        payload = platform._prepare_request_payload(
            "Test question", max_tokens=1000, temperature=0.7
        )

        assert payload["generationConfig"]["maxOutputTokens"] == 1000
        assert payload["generationConfig"]["temperature"] == 0.7

    def test_text_extraction_success(self):
        """Test successful text extraction from Google AI response."""
        platform = GoogleAIPlatform("test_key")

        response = {
            "candidates": [
                {"content": {"parts": [{"text": "This is a Gemini response"}]}}
            ]
        }

        text = platform.extract_text_response(response)
        assert text == "This is a Gemini response"

    def test_text_extraction_malformed_response(self):
        """Test text extraction with malformed Google AI response."""
        platform = GoogleAIPlatform("test_key")

        # Missing candidates
        with pytest.raises(ValueError, match="Invalid Google AI response format"):
            platform.extract_text_response({"invalid": "format"})

        # Empty candidates
        with pytest.raises(ValueError, match="Invalid Google AI response format"):
            platform.extract_text_response({"candidates": []})

        # Missing content
        with pytest.raises(ValueError, match="Invalid Google AI response format"):
            platform.extract_text_response({"candidates": [{"invalid": "format"}]})

        # Missing parts
        with pytest.raises(ValueError, match="Invalid Google AI response format"):
            platform.extract_text_response(
                {"candidates": [{"content": {"invalid": "format"}}]}
            )

        # Empty parts
        with pytest.raises(ValueError, match="Invalid Google AI response format"):
            platform.extract_text_response({"candidates": [{"content": {"parts": []}}]})


class TestPlatformComparison:
    """Test cases comparing platform implementations."""

    def test_all_platforms_inherit_base(self):
        """Test that all platforms properly inherit from BasePlatform."""
        from app.services.ai_platforms.base import BasePlatform

        platforms = [
            OpenAIPlatform("key"),
            AnthropicPlatform("key"),
            PerplexityPlatform("key"),
            GoogleAIPlatform("key"),
        ]

        for platform in platforms:
            assert isinstance(platform, BasePlatform)
            assert hasattr(platform, "safe_query")
            assert hasattr(platform, "extract_text_response")
            assert hasattr(platform, "_get_default_headers")
            assert hasattr(platform, "_prepare_request_payload")

    def test_platform_name_uniqueness(self):
        """Test that all platforms have unique names."""
        platforms = [
            OpenAIPlatform("key"),
            AnthropicPlatform("key"),
            PerplexityPlatform("key"),
            GoogleAIPlatform("key"),
        ]

        names = [p.platform_name for p in platforms]
        assert len(names) == len(set(names))  # All names should be unique

    def test_rate_limit_defaults(self):
        """Test that platforms have appropriate rate limit defaults."""
        # Based on the build plan specifications
        assert OpenAIPlatform("key").rate_limiter.requests_per_minute == 50
        assert AnthropicPlatform("key").rate_limiter.requests_per_minute == 100
        assert PerplexityPlatform("key").rate_limiter.requests_per_minute == 20
        assert GoogleAIPlatform("key").rate_limiter.requests_per_minute == 60
