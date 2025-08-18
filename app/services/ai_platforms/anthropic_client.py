"""
Anthropic platform client implementation.

Provides integration with Anthropic's Claude API using the standardized
BasePlatform interface with proper error handling and response parsing.
"""

from typing import Any, Dict

from .base import BasePlatform


class AnthropicPlatform(BasePlatform):
    """
    Anthropic platform implementation.

    Supports Anthropic's Claude messaging API with configurable models,
    parameters, and proper response text extraction.
    """

    def __init__(self, api_key: str, rate_limit: int = 100, **config):
        """
        Initialize Anthropic platform client.

        Args:
            api_key: Anthropic API key
            rate_limit: Requests per minute limit (default: 100)
            **config: Additional configuration options:
                - base_url: API base URL (default: https://api.anthropic.com)
                - default_model: Default model to use
                  (default: claude-3-sonnet-20240229)
                - max_tokens: Default max tokens (default: 500)
                - temperature: Default temperature (default: 0.1)
        """
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.default_model = config.get("default_model", "claude-3-sonnet-20240229")
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Anthropic requests."""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

    def _get_endpoint_url(self) -> str:
        """Get Anthropic messages endpoint URL."""
        return f"{self.base_url}/v1/messages"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare Anthropic API request payload.

        Args:
            question: The question/prompt to send
            **kwargs: Optional parameters to override defaults

        Returns:
            Request payload dictionary
        """
        return {
            "model": kwargs.get("model", self.default_model),
            "messages": [{"role": "user", "content": question}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Anthropic-specific query implementation.

        Note: This method should not be called directly.
        Use safe_query() instead which provides error handling and rate limiting.
        """
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract text from Anthropic response format.

        Args:
            raw_response: Raw response from Anthropic API

        Returns:
            Clean text content from the response

        Raises:
            ValueError: If response format is invalid
        """
        try:
            return raw_response["content"][0]["text"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Anthropic response format: {e}")
