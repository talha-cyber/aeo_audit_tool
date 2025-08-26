"""
Integration tests for PlatformRegistry and PlatformManager.

Tests the factory pattern, platform registration, and manager functionality
including initialization, health checks, and platform availability.
"""

from unittest.mock import Mock, patch

import pytest

from app.services.ai_platforms.anthropic_client import AnthropicPlatform
from app.services.ai_platforms.base import BasePlatform
from app.services.ai_platforms.google_ai_client import GoogleAIPlatform
from app.services.ai_platforms.openai_client import OpenAIPlatform
from app.services.ai_platforms.perplexity_client import PerplexityPlatform
from app.services.ai_platforms.registry import PlatformRegistry
from app.services.platform_manager import PlatformManager


class TestPlatformRegistry:
    """Test cases for PlatformRegistry factory class."""

    def test_get_available_platforms(self):
        """Test getting list of available platforms."""
        platforms = PlatformRegistry.get_available_platforms()

        expected_platforms = ["openai", "anthropic", "perplexity", "google_ai"]
        assert set(platforms) == set(expected_platforms)
        assert len(platforms) == 4

    def test_create_openai_platform(self):
        """Test creating OpenAI platform instance."""
        platform = PlatformRegistry.create_platform("openai", "test_key")

        assert isinstance(platform, OpenAIPlatform)
        assert platform.api_key == "test_key"
        assert platform.platform_name == "openai"

    def test_create_anthropic_platform(self):
        """Test creating Anthropic platform instance."""
        platform = PlatformRegistry.create_platform("anthropic", "test_key")

        assert isinstance(platform, AnthropicPlatform)
        assert platform.api_key == "test_key"
        assert platform.platform_name == "anthropic"

    def test_create_perplexity_platform(self):
        """Test creating Perplexity platform instance."""
        platform = PlatformRegistry.create_platform("perplexity", "test_key")

        assert isinstance(platform, PerplexityPlatform)
        assert platform.api_key == "test_key"
        assert platform.platform_name == "perplexity"

    def test_create_google_ai_platform(self):
        """Test creating Google AI platform instance."""
        platform = PlatformRegistry.create_platform("google_ai", "test_key")

        assert isinstance(platform, GoogleAIPlatform)
        assert platform.api_key == "test_key"
        assert platform.platform_name == "googleai"

    def test_create_platform_with_config(self):
        """Test creating platform with custom configuration."""
        config = {
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "rate_limit": 100,
        }

        platform = PlatformRegistry.create_platform("openai", "test_key", config)

        assert platform.default_model == "gpt-3.5-turbo"
        assert platform.max_tokens == 1000
        assert platform.rate_limiter.requests_per_minute == 100

    def test_create_unknown_platform(self):
        """Test creating unknown platform raises ValueError."""
        with pytest.raises(ValueError, match="Unknown platform: unknown"):
            PlatformRegistry.create_platform("unknown", "test_key")

    def test_is_platform_available(self):
        """Test checking platform availability."""
        assert PlatformRegistry.is_platform_available("openai") is True
        assert PlatformRegistry.is_platform_available("anthropic") is True
        assert PlatformRegistry.is_platform_available("unknown") is False

    def test_register_custom_platform(self):
        """Test registering a custom platform."""

        class CustomPlatform(BasePlatform):
            def _get_default_headers(self):
                return {}

            def _get_endpoint_url(self):
                return ""

            def _prepare_request_payload(self, question, **kwargs):
                return {}

            async def query(self, question, **kwargs):
                return {}

            def extract_text_response(self, raw_response):
                return ""

        # Register custom platform
        PlatformRegistry.register_platform("custom", CustomPlatform)

        # Should be available now
        assert PlatformRegistry.is_platform_available("custom") is True
        assert "custom" in PlatformRegistry.get_available_platforms()

        # Should be able to create instance
        platform = PlatformRegistry.create_platform("custom", "test_key")
        assert isinstance(platform, CustomPlatform)

        # Clean up
        PlatformRegistry.unregister_platform("custom")

    def test_register_invalid_platform(self):
        """Test registering invalid platform raises TypeError."""

        class NotAPlatform:
            pass

        with pytest.raises(
            TypeError, match="Platform class must inherit from BasePlatform"
        ):
            PlatformRegistry.register_platform("invalid", NotAPlatform)

    def test_unregister_platform(self):
        """Test unregistering a platform."""

        # Register a temporary platform first
        class TempPlatform(BasePlatform):
            def _get_default_headers(self):
                return {}

            def _get_endpoint_url(self):
                return ""

            def _prepare_request_payload(self, question, **kwargs):
                return {}

            async def query(self, question, **kwargs):
                return {}

            def extract_text_response(self, raw_response):
                return ""

        PlatformRegistry.register_platform("temp", TempPlatform)
        assert PlatformRegistry.is_platform_available("temp") is True

        # Unregister it
        PlatformRegistry.unregister_platform("temp")
        assert PlatformRegistry.is_platform_available("temp") is False

    def test_unregister_nonexistent_platform(self):
        """Test unregistering non-existent platform raises KeyError."""
        with pytest.raises(KeyError, match="Platform 'nonexistent' is not registered"):
            PlatformRegistry.unregister_platform("nonexistent")


class TestPlatformManager:
    """Test cases for PlatformManager class."""

    @patch("app.core.config.settings")
    def test_initialization_with_valid_keys(self, mock_settings):
        """Test platform manager initialization with valid API keys."""
        # Mock settings with valid API keys
        mock_settings.OPENAI_API_KEY = "valid_openai_key"
        mock_settings.ANTHROPIC_API_KEY = "valid_anthropic_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"  # Skip this one
        mock_settings.GOOGLE_AI_API_KEY = "valid_google_key"

        manager = PlatformManager()

        # Should initialize platforms with valid keys
        assert "openai" in manager.platforms
        assert "anthropic" in manager.platforms
        assert "google_ai" in manager.platforms
        assert "perplexity" not in manager.platforms  # Skipped due to dummy key

    @patch("app.core.config.settings")
    def test_initialization_with_no_keys(self, mock_settings):
        """Test platform manager initialization with no valid API keys."""
        # Mock settings with dummy keys
        mock_settings.OPENAI_API_KEY = "dummy_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Should not initialize any platforms
        assert len(manager.platforms) == 0

    @patch("app.core.config.settings")
    def test_get_platform_success(self, mock_settings):
        """Test getting an available platform."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        platform = manager.get_platform("openai")
        assert isinstance(platform, OpenAIPlatform)
        assert platform.api_key == "valid_key"

    @patch("app.core.config.settings")
    def test_get_platform_not_available(self, mock_settings):
        """Test getting unavailable platform raises ValueError."""
        mock_settings.OPENAI_API_KEY = "dummy_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        with pytest.raises(ValueError, match="Platform 'openai' not available"):
            manager.get_platform("openai")

    @patch("app.core.config.settings")
    def test_get_available_platforms(self, mock_settings):
        """Test getting list of available platforms."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "valid_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        available = manager.get_available_platforms()
        assert "openai" in available
        assert "anthropic" in available
        assert "perplexity" not in available
        assert "google_ai" not in available

    @patch("app.core.config.settings")
    def test_is_platform_available(self, mock_settings):
        """Test checking if specific platforms are available."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        assert manager.is_platform_available("openai") is True
        assert manager.is_platform_available("anthropic") is False
        assert manager.is_platform_available("nonexistent") is False

    @patch("app.core.config.settings")
    def test_get_platform_count(self, mock_settings):
        """Test getting count of available platforms."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "valid_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        assert manager.get_platform_count() == 2

    @patch("app.core.config.settings")
    def test_register_platform_manually(self, mock_settings):
        """Test manually registering a platform."""
        mock_settings.OPENAI_API_KEY = "dummy_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Create mock platform
        mock_platform = Mock(spec=BasePlatform)
        mock_platform.platform_name = "mock"

        # Register it
        manager.register_platform("mock", mock_platform)

        assert manager.is_platform_available("mock") is True
        assert manager.get_platform("mock") == mock_platform

    @patch("app.core.config.settings")
    def test_unregister_platform(self, mock_settings):
        """Test unregistering a platform."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        assert manager.is_platform_available("openai") is True

        manager.unregister_platform("openai")

        assert manager.is_platform_available("openai") is False

        with pytest.raises(ValueError, match="Platform 'openai' not available"):
            manager.get_platform("openai")

    @patch("app.core.config.settings")
    def test_unregister_nonexistent_platform(self, mock_settings):
        """Test unregistering non-existent platform raises KeyError."""
        mock_settings.OPENAI_API_KEY = "dummy_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        with pytest.raises(KeyError, match="Platform 'nonexistent' is not registered"):
            manager.unregister_platform("nonexistent")

    @patch("app.core.config.settings")
    def test_get_platform_info(self, mock_settings):
        """Test getting platform information."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        info = manager.get_platform_info()

        assert "openai" in info
        assert info["openai"]["platform_name"] == "openai"
        assert info["openai"]["rate_limit"] == 50
        assert info["openai"]["default_model"] == "gpt-4"
        assert info["openai"]["circuit_breaker_open"] is False
        assert info["openai"]["failure_count"] == 0

    @patch("app.core.config.settings")
    @pytest.mark.asyncio
    async def test_health_check_single_success(self, mock_settings):
        """Test health check for single platform success."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Mock successful platform response
        with patch.object(manager.platforms["openai"], "safe_query") as mock_query:
            mock_query.return_value = {"success": True, "metadata": {"duration": 0.5}}

            result = await manager.health_check_single("openai")

            assert result is True
            mock_query.assert_called_once_with("Health check")

    @patch("app.core.config.settings")
    @pytest.mark.asyncio
    async def test_health_check_single_failure(self, mock_settings):
        """Test health check for single platform failure."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Mock failed platform response
        with patch.object(manager.platforms["openai"], "safe_query") as mock_query:
            mock_query.return_value = {"success": False, "error": "API error"}

            result = await manager.health_check_single("openai")

            assert result is False

    @patch("app.core.config.settings")
    @pytest.mark.asyncio
    async def test_health_check_single_exception(self, mock_settings):
        """Test health check for single platform with exception."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Mock platform exception
        with patch.object(manager.platforms["openai"], "safe_query") as mock_query:
            mock_query.side_effect = Exception("Network error")

            result = await manager.health_check_single("openai")

            assert result is False

    @patch("app.core.config.settings")
    @pytest.mark.asyncio
    async def test_health_check_single_unavailable_platform(self, mock_settings):
        """Test health check for unavailable platform."""
        mock_settings.OPENAI_API_KEY = "dummy_key"
        mock_settings.ANTHROPIC_API_KEY = "dummy_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        with pytest.raises(ValueError, match="Platform 'openai' not available"):
            await manager.health_check_single("openai")

    @patch("app.core.config.settings")
    @pytest.mark.asyncio
    async def test_health_check_all_platforms(self, mock_settings):
        """Test health check for all platforms."""
        mock_settings.OPENAI_API_KEY = "valid_key"
        mock_settings.ANTHROPIC_API_KEY = "valid_key"
        mock_settings.PERPLEXITY_API_KEY = "dummy_key"
        mock_settings.GOOGLE_AI_API_KEY = "dummy_key"

        manager = PlatformManager()

        # Mock platform responses
        with patch.object(
            manager.platforms["openai"], "safe_query"
        ) as mock_openai, patch.object(
            manager.platforms["anthropic"], "safe_query"
        ) as mock_anthropic:
            mock_openai.return_value = {"success": True, "metadata": {"duration": 0.5}}
            mock_anthropic.return_value = {"success": False, "error": "API error"}

            health_status = await manager.health_check()

            assert health_status["openai"] is True
            assert health_status["anthropic"] is False

