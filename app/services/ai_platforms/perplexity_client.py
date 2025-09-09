"""
Perplexity platform client implementation.

Provides integration with Perplexity's chat completion API using the standardized
BasePlatform interface with proper error handling and response parsing.
"""

from typing import Any, Dict

from .base import BasePlatform


class PerplexityPlatform(BasePlatform):
    """
    Perplexity platform implementation.

    Supports Perplexity's chat completion API with configurable models,
    parameters, and proper response text extraction.
    """

    def __init__(self, api_key: str, rate_limit: int = 20, **config):
        """
        Initialize Perplexity platform client.

        Args:
            api_key: Perplexity API key
            rate_limit: Requests per minute limit (default: 20)
            **config: Additional configuration options:
                - base_url: API base URL (default: https://api.perplexity.ai)
                - default_model: Default model to use
                  (default: llama-3.1-sonar-small-128k-online)
                - max_tokens: Default max tokens (default: 500)
                - temperature: Default temperature (default: 0.1)
        """
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get("base_url", "https://api.perplexity.ai")
        self.default_model = config.get(
            "default_model", "llama-3.1-sonar-small-128k-online"
        )
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Perplexity requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AEO-Audit-Tool/1.0",
        }

    def _get_endpoint_url(self) -> str:
        """Get Perplexity chat completions endpoint URL."""
        return f"{self.base_url}/chat/completions"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare Perplexity API request payload.

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
            "stream": False,
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Perplexity-specific query implementation.

        Note: This method should not be called directly.
        Use safe_query() instead which provides error handling and rate limiting.
        """
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract text from Perplexity response format.

        Args:
            raw_response: Raw response from Perplexity API

        Returns:
            Clean text content from the response

        Raises:
            ValueError: If response format is invalid
        """
        try:
            return raw_response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Perplexity response format: {e}")
