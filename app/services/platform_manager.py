"""
Platform manager for AI platform clients.

Provides centralized management of multiple AI platform instances for the application,
including initialization, health checks, and platform availability management.
"""

from typing import Any, Dict, List

from app.core.config import settings
from app.core.platform_settings import PLATFORM_CONFIGS, REQUIRED_ENV_VARS
from app.services.ai_platforms.base import BasePlatform
from app.services.ai_platforms.registry import PlatformRegistry
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PlatformManager:
    """
    Manages multiple AI platform instances for the application.

    Provides centralized initialization, configuration, and management
    of AI platform clients with health checking capabilities.
    """

    def __init__(self):
        """Initialize the platform manager and load all available platforms."""
        self.platforms: Dict[str, BasePlatform] = {}
        self._initialize_platforms()

    def _initialize_platforms(self) -> None:
        """
        Initialize all configured platforms.

        Loads platform configurations and creates instances for platforms
        that have valid API keys configured.
        """
        for platform_name, config in PLATFORM_CONFIGS.items():
            try:
                api_key_env = REQUIRED_ENV_VARS[platform_name]
                api_key = getattr(settings, api_key_env, None)

                if not api_key or api_key == "dummy_key":
                    logger.warning(
                        "No valid API key found for platform, skipping",
                        platform=platform_name,
                        env_var=api_key_env,
                    )
                    continue

                platform = PlatformRegistry.create_platform(
                    platform_name, api_key, config
                )

                self.platforms[platform_name] = platform
                logger.info(
                    "Successfully initialized platform",
                    platform=platform_name,
                    rate_limit=config.get("rate_limit", "unknown"),
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize platform",
                    platform=platform_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def get_platform(self, name: str) -> BasePlatform:
        """
        Get a specific platform instance.

        Args:
            name: Name of the platform to retrieve

        Returns:
            Platform instance

        Raises:
            ValueError: If platform is not available
        """
        if name not in self.platforms:
            raise ValueError(f"Platform '{name}' not available")
        return self.platforms[name]

    def get_available_platforms(self) -> List[str]:
        """
        Get list of available platform names.

        Returns:
            List of platform names that are properly initialized
        """
        return list(self.platforms.keys())

    def is_platform_available(self, name: str) -> bool:
        """
        Check if a platform is available.

        Args:
            name: Name of the platform to check

        Returns:
            True if platform is available, False otherwise
        """
        return name in self.platforms

    def get_platform_count(self) -> int:
        """
        Get the number of available platforms.

        Returns:
            Number of initialized platforms
        """
        return len(self.platforms)

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all platforms.

        Performs a simple query to each platform to verify it's working correctly.

        Returns:
            Dictionary mapping platform names to health status
        """
        health_status = {}

        for name, platform in self.platforms.items():
            try:
                async with platform:
                    result = await platform.safe_query("Health check")
                health_status[name] = result["success"]

                if result["success"]:
                    logger.info(
                        "Platform health check passed",
                        platform=name,
                        duration=result["metadata"].get("duration", 0),
                    )
                else:
                    logger.warning(
                        "Platform health check failed",
                        platform=name,
                        error=result["error"],
                    )

            except Exception as e:
                logger.error(
                    "Platform health check error",
                    platform=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                health_status[name] = False

        return health_status

    async def health_check_single(self, platform_name: str) -> bool:
        """
        Check health of a single platform.

        Args:
            platform_name: Name of the platform to check

        Returns:
            True if platform is healthy, False otherwise

        Raises:
            ValueError: If platform is not available
        """
        if platform_name not in self.platforms:
            raise ValueError(f"Platform '{platform_name}' not available")

        platform = self.platforms[platform_name]

        try:
            async with platform:
                result = await platform.safe_query("Health check")
            return result["success"]
        except Exception as e:
            logger.error(
                "Platform health check error",
                platform=platform_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def register_platform(self, name: str, platform: BasePlatform) -> None:
        """
        Register additional platform (for testing or custom platforms).

        Args:
            name: Name to register the platform under
            platform: Platform instance to register
        """
        self.platforms[name] = platform
        logger.info(
            "Manually registered platform",
            platform=name,
            class_name=type(platform).__name__,
        )

    def unregister_platform(self, name: str) -> None:
        """
        Remove a platform from the manager.

        Args:
            name: Name of the platform to remove

        Raises:
            KeyError: If platform is not registered
        """
        if name not in self.platforms:
            raise KeyError(f"Platform '{name}' is not registered")

        del self.platforms[name]
        logger.info("Unregistered platform", platform=name)

    def get_platform_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available platforms.

        Returns:
            Dictionary with platform information including configuration
        """
        info = {}

        for name, platform in self.platforms.items():
            config = PLATFORM_CONFIGS.get(name, {})
            info[name] = {
                "platform_name": platform.platform_name,
                "rate_limit": config.get("rate_limit", "unknown"),
                "default_model": config.get("default_model", "unknown"),
                "max_tokens": config.get("max_tokens", "unknown"),
                "base_url": config.get("base_url", "unknown"),
                "circuit_breaker_open": platform.circuit_open,
                "failure_count": platform.failure_count,
            }

        return info
