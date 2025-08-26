"""
Platform-specific configurations for AI platforms.

Provides centralized configuration for all AI platform clients including
rate limits, model defaults, and environment variable mappings.
"""

from typing import Any, Dict

# Platform-specific configurations
PLATFORM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 50,  # RPM
        "timeout": 30,
        "max_retries": 3,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-sonnet-20240229",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 100,
        "timeout": 30,
        "max_retries": 3,
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "default_model": "llama-3.1-sonar-small-128k-online",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 20,
        "timeout": 45,  # Perplexity can be slower
        "max_retries": 2,
    },
    "google_ai": {
        "base_url": "https://generativelanguage.googleapis.com",
        "default_model": "gemini-pro",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 60,
        "timeout": 30,
        "max_retries": 3,
    },
}

# Environment variable mapping
REQUIRED_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "google_ai": "GOOGLE_AI_API_KEY",
}


def get_platform_config(platform_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific platform.

    Args:
        platform_name: Name of the platform

    Returns:
        Configuration dictionary for the platform

    Raises:
        KeyError: If platform is not configured
    """
    if platform_name not in PLATFORM_CONFIGS:
        raise KeyError(f"No configuration found for platform: {platform_name}")

    return PLATFORM_CONFIGS[platform_name].copy()


def get_api_key_env_var(platform_name: str) -> str:
    """
    Get the environment variable name for a platform's API key.

    Args:
        platform_name: Name of the platform

    Returns:
        Environment variable name

    Raises:
        KeyError: If platform is not configured
    """
    if platform_name not in REQUIRED_ENV_VARS:
        raise KeyError(
            f"No API key environment variable configured for platform: {platform_name}"
        )

    return REQUIRED_ENV_VARS[platform_name]


def get_all_platform_names() -> list[str]:
    """
    Get list of all configured platform names.

    Returns:
        List of platform names
    """
    return list(PLATFORM_CONFIGS.keys())

