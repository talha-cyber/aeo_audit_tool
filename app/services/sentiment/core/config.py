"""
Sentiment analysis configuration management.

Centralized configuration for the sentiment analysis system with environment
variable support, validation, and default settings optimized for production use.
"""

import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .models import EnsembleStrategy, SentimentMethod


class SentimentSettings(BaseModel):
    """Sentiment analysis configuration settings"""

    # Provider Configuration
    enabled_providers: List[SentimentMethod] = Field(
        default=[SentimentMethod.VADER, SentimentMethod.BUSINESS_CONTEXT],
        description="List of enabled sentiment providers",
    )

    primary_provider: SentimentMethod = Field(
        default=SentimentMethod.VADER, description="Primary sentiment provider to use"
    )

    fallback_provider: SentimentMethod = Field(
        default=SentimentMethod.VADER,
        description="Fallback provider when primary fails",
    )

    # Ensemble Configuration
    enable_ensemble: bool = Field(
        default=True,
        description="Enable ensemble methods when multiple providers available",
    )

    ensemble_strategy: EnsembleStrategy = Field(
        default=EnsembleStrategy.CONFIDENCE_WEIGHTED,
        description="Strategy for combining multiple provider results",
    )

    ensemble_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "vader": 0.4,
            "business_context": 0.6,
            "transformer": 0.8,
        },
        description="Weights for ensemble combination (by provider name)",
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for high-confidence results",
    )

    # Performance Configuration
    enable_caching: bool = Field(default=True, description="Enable result caching")

    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")

    batch_size: int = Field(
        default=32, ge=1, le=128, description="Default batch size for batch processing"
    )

    max_concurrent: int = Field(
        default=10, ge=1, le=50, description="Maximum concurrent analyses"
    )

    request_timeout: float = Field(
        default=30.0, ge=5.0, description="Request timeout in seconds"
    )

    # Transformer Model Configuration
    transformer_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="HuggingFace model identifier for transformer provider",
    )

    transformer_device: str = Field(
        default="auto",
        description="Device for transformer model (auto, cpu, cuda, mps)",
    )

    transformer_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum token length for transformer models",
    )

    transformer_batch_size: int = Field(
        default=16, ge=1, le=64, description="Batch size for transformer inference"
    )

    # Business Context Configuration
    business_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of business signals in business context provider",
    )

    linguistic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight of linguistic signals in business context provider",
    )

    context_boost: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Confidence boost when business signals align with linguistic sentiment",
    )

    # Quality and Safety Configuration
    min_text_length: int = Field(
        default=3, ge=1, description="Minimum text length for analysis"
    )

    max_text_length: int = Field(
        default=5000, ge=100, description="Maximum text length for analysis"
    )

    enable_language_detection: bool = Field(
        default=True, description="Enable language detection"
    )

    supported_languages: List[str] = Field(
        default=["en"], description="List of supported language codes"
    )

    # Monitoring Configuration
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring and metrics"
    )

    enable_health_checks: bool = Field(
        default=True, description="Enable provider health checks"
    )

    health_check_interval: int = Field(
        default=300, ge=60, description="Health check interval in seconds"
    )

    # Backward Compatibility
    enable_legacy_api: bool = Field(
        default=True,
        description="Enable backward compatibility with legacy sentiment API",
    )

    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v):
        """Ensure ensemble weights are valid"""
        if not v:
            return v

        # Check that weights are positive
        for provider, weight in v.items():
            if weight < 0:
                raise ValueError(f"Ensemble weight for {provider} must be non-negative")

        return v

    @validator("business_weight", "linguistic_weight")
    def validate_weight_sum(cls, v, values):
        """Ensure business and linguistic weights sum to <= 1.0"""
        if "business_weight" in values:
            business_weight = values["business_weight"]
            if v == business_weight:  # This is business_weight being validated
                return v
            # This is linguistic_weight being validated
            if business_weight + v > 1.0:
                raise ValueError(
                    "Business weight + linguistic weight must not exceed 1.0"
                )
        return v

    @validator("primary_provider", "fallback_provider")
    def validate_provider_enabled(cls, v, values):
        """Ensure primary and fallback providers are in enabled list"""
        if "enabled_providers" in values and v not in values["enabled_providers"]:
            raise ValueError(f"Provider {v} must be in enabled_providers list")
        return v

    class Config:
        env_prefix = "SENTIMENT_"
        case_sensitive = False


def load_sentiment_settings() -> SentimentSettings:
    """
    Load sentiment settings from environment variables and configuration.

    Environment variables follow the pattern SENTIMENT_<SETTING_NAME>
    Example: SENTIMENT_ENABLED_PROVIDERS=vader,business_context
    """

    # Create base settings
    sentiment_settings = SentimentSettings()

    # Override with environment variables
    env_overrides = {}

    # Handle list of providers from env
    if os.getenv("SENTIMENT_ENABLED_PROVIDERS"):
        provider_names = os.getenv("SENTIMENT_ENABLED_PROVIDERS").split(",")
        providers = []
        for name in provider_names:
            name = name.strip().lower()
            try:
                provider = SentimentMethod(name)
                providers.append(provider)
            except ValueError:
                continue
        if providers:
            env_overrides["enabled_providers"] = providers

    # Handle ensemble strategy
    if os.getenv("SENTIMENT_ENSEMBLE_STRATEGY"):
        try:
            strategy = EnsembleStrategy(os.getenv("SENTIMENT_ENSEMBLE_STRATEGY"))
            env_overrides["ensemble_strategy"] = strategy
        except ValueError:
            pass

    # Handle primary and fallback providers
    if os.getenv("SENTIMENT_PRIMARY_PROVIDER"):
        try:
            provider = SentimentMethod(os.getenv("SENTIMENT_PRIMARY_PROVIDER"))
            env_overrides["primary_provider"] = provider
        except ValueError:
            pass

    if os.getenv("SENTIMENT_FALLBACK_PROVIDER"):
        try:
            provider = SentimentMethod(os.getenv("SENTIMENT_FALLBACK_PROVIDER"))
            env_overrides["fallback_provider"] = provider
        except ValueError:
            pass

    # Handle boolean settings
    bool_settings = [
        "enable_ensemble",
        "enable_caching",
        "enable_language_detection",
        "enable_performance_monitoring",
        "enable_health_checks",
        "enable_legacy_api",
    ]

    for setting in bool_settings:
        env_key = f"SENTIMENT_{setting.upper()}"
        if os.getenv(env_key):
            env_overrides[setting] = os.getenv(env_key).lower() in (
                "true",
                "1",
                "yes",
                "on",
            )

    # Handle numeric settings
    numeric_settings = [
        ("confidence_threshold", float),
        ("cache_ttl", int),
        ("batch_size", int),
        ("max_concurrent", int),
        ("request_timeout", float),
        ("transformer_max_length", int),
        ("transformer_batch_size", int),
        ("business_weight", float),
        ("linguistic_weight", float),
        ("context_boost", float),
        ("min_text_length", int),
        ("max_text_length", int),
        ("health_check_interval", int),
    ]

    for setting, type_func in numeric_settings:
        env_key = f"SENTIMENT_{setting.upper()}"
        if os.getenv(env_key):
            try:
                env_overrides[setting] = type_func(os.getenv(env_key))
            except (ValueError, TypeError):
                continue

    # Handle string settings
    string_settings = ["transformer_model", "transformer_device"]

    for setting in string_settings:
        env_key = f"SENTIMENT_{setting.upper()}"
        if os.getenv(env_key):
            env_overrides[setting] = os.getenv(env_key)

    # Handle list settings
    if os.getenv("SENTIMENT_SUPPORTED_LANGUAGES"):
        languages = [
            lang.strip()
            for lang in os.getenv("SENTIMENT_SUPPORTED_LANGUAGES").split(",")
        ]
        env_overrides["supported_languages"] = languages

    # Create new settings with overrides
    if env_overrides:
        sentiment_settings = SentimentSettings(
            **{**sentiment_settings.dict(), **env_overrides}
        )

    return sentiment_settings


# Global settings instance
_sentiment_settings: Optional[SentimentSettings] = None


def get_sentiment_settings() -> SentimentSettings:
    """Get global sentiment settings instance"""
    global _sentiment_settings
    if _sentiment_settings is None:
        _sentiment_settings = load_sentiment_settings()
    return _sentiment_settings


def reload_sentiment_settings():
    """Reload sentiment settings from environment"""
    global _sentiment_settings
    _sentiment_settings = load_sentiment_settings()


# Configuration profiles for different environments
DEVELOPMENT_CONFIG = {
    "enabled_providers": [SentimentMethod.VADER],
    "enable_caching": False,
    "enable_performance_monitoring": True,
    "enable_health_checks": False,
    "transformer_device": "cpu",
    "max_concurrent": 5,
}

PRODUCTION_CONFIG = {
    "enabled_providers": [SentimentMethod.VADER, SentimentMethod.BUSINESS_CONTEXT],
    "enable_ensemble": True,
    "enable_caching": True,
    "cache_ttl": 3600,
    "enable_performance_monitoring": True,
    "enable_health_checks": True,
    "health_check_interval": 300,
    "max_concurrent": 20,
    "transformer_device": "auto",
}

HIGH_ACCURACY_CONFIG = {
    "enabled_providers": [
        SentimentMethod.VADER,
        SentimentMethod.BUSINESS_CONTEXT,
        SentimentMethod.TRANSFORMER,
    ],
    "ensemble_strategy": EnsembleStrategy.CONFIDENCE_WEIGHTED,
    "confidence_threshold": 0.8,
    "enable_caching": True,
    "transformer_device": "auto",
}


def apply_config_profile(profile_name: str) -> SentimentSettings:
    """
    Apply a predefined configuration profile.

    Args:
        profile_name: Name of the profile ('development', 'production', 'high_accuracy')

    Returns:
        SentimentSettings with profile applied
    """
    profiles = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "high_accuracy": HIGH_ACCURACY_CONFIG,
    }

    if profile_name not in profiles:
        raise ValueError(
            f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}"
        )

    profile_config = profiles[profile_name]
    base_settings = load_sentiment_settings()

    # Apply profile overrides
    updated_config = {**base_settings.dict(), **profile_config}
    return SentimentSettings(**updated_config)
