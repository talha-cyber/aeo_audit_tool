"""
Platform registry and factory for AI platform clients.

Provides a centralized registry for all available AI platforms and factory
methods for creating platform instances with proper configuration.
"""

from typing import Dict, List, Type

from .anthropic_client import AnthropicPlatform
from .base import BasePlatform
from .google_ai_client import GoogleAIPlatform
from .openai_client import OpenAIPlatform
from .perplexity_client import PerplexityPlatform


class PlatformRegistry:
    """
    Factory for creating and managing AI platform instances.

    Provides centralized registration and creation of platform clients
    with support for dynamic platform registration and configuration.
    """

    _platforms: Dict[str, Type[BasePlatform]] = {
        "openai": OpenAIPlatform,
        "anthropic": AnthropicPlatform,
        "perplexity": PerplexityPlatform,
        "google_ai": GoogleAIPlatform,
    }

    @classmethod
    def create_platform(
        cls, platform_name: str, api_key: str, config: Dict = None
    ) -> BasePlatform:
        """
        Create a platform instance by name.

        Args:
            platform_name: Name of the platform to create
            api_key: API key for the platform
            config: Optional configuration dictionary

        Returns:
            Configured platform instance

        Raises:
            ValueError: If platform name is not recognized
        """
        if platform_name not in cls._platforms:
            raise ValueError(f"Unknown platform: {platform_name}")

        platform_class = cls._platforms[platform_name]
        platform_config = config or {}

        return platform_class(api_key=api_key, **platform_config)

    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """
        Get list of available platform names.

        Returns:
            List of registered platform names
        """
        return list(cls._platforms.keys())

    @classmethod
    def register_platform(cls, name: str, platform_class: Type[BasePlatform]) -> None:
        """
        Register a new platform implementation.

        Args:
            name: Name to register the platform under
            platform_class: Platform class that inherits from BasePlatform

        Raises:
            TypeError: If platform_class doesn't inherit from BasePlatform
        """
        if not issubclass(platform_class, BasePlatform):
            raise TypeError("Platform class must inherit from BasePlatform")

        cls._platforms[name] = platform_class

    @classmethod
    def is_platform_available(cls, platform_name: str) -> bool:
        """
        Check if a platform is available for use.

        Args:
            platform_name: Name of the platform to check

        Returns:
            True if platform is registered, False otherwise
        """
        return platform_name in cls._platforms

    @classmethod
    def unregister_platform(cls, platform_name: str) -> None:
        """
        Remove a platform from the registry.

        Args:
            platform_name: Name of the platform to remove

        Raises:
            KeyError: If platform is not registered
        """
        if platform_name not in cls._platforms:
            raise KeyError(f"Platform '{platform_name}' is not registered")

        del cls._platforms[platform_name]
