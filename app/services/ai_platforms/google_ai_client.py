"""
Google AI platform client implementation.

Provides integration with Google's Generative AI API using the standardized
BasePlatform interface with proper error handling and response parsing.
"""

from typing import Any, Dict

from .base import BasePlatform


class GoogleAIPlatform(BasePlatform):
    """
    Google AI platform implementation.

    Supports Google's Generative AI API (Gemini) with configurable models,
    parameters, and proper response text extraction.
    """

    def __init__(self, api_key: str, rate_limit: int = 60, **config):
        """
        Initialize Google AI platform client.

        Args:
            api_key: Google AI API key
            rate_limit: Requests per minute limit (default: 60)
            **config: Additional configuration options:
                - base_url: API base URL (default: https://generativelanguage.googleapis.com)
                - default_model: Default model to use (default: gemini-1.5-flash)
                - max_tokens: Default max tokens (default: 500)
                - temperature: Default temperature (default: 0.1)
        """
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get(
            "base_url", "https://generativelanguage.googleapis.com"
        )
        self.default_model = config.get("default_model", "gemini-1.5-flash")
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Google AI requests."""
        return {"Content-Type": "application/json", "User-Agent": "AEO-Audit-Tool/1.0"}

    def _get_endpoint_url(self) -> str:
        """Get Google AI generate content endpoint URL."""
        return (
            f"{self.base_url}/v1/models/{self.default_model}:"
            f"generateContent?key={self.api_key}"
        )

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare Google AI API request payload.

        Args:
            question: The question/prompt to send
            **kwargs: Optional parameters to override defaults

        Returns:
            Request payload dictionary
        """
        return {
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Google AI-specific query implementation.

        Note: This method should not be called directly.
        Use safe_query() instead which provides error handling and rate limiting.
        """
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract text from Google AI response format.

        Args:
            raw_response: Raw response from Google AI API

        Returns:
            Clean text content from the response

        Raises:
            ValueError: If response format is invalid
        """
        try:
            return raw_response["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Google AI response format: {e}")
