"""
Comprehensive audit processor configuration system.
"""

from enum import Enum
from typing import Any, Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AuditBatchStrategy(str, Enum):
    """Strategy for batching audit questions"""

    FIXED_SIZE = "fixed_size"
    ADAPTIVE = "adaptive"
    PLATFORM_OPTIMIZED = "platform_optimized"


class AuditRetryStrategy(str, Enum):
    """Strategy for retrying failed audits"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"


class AuditSettings(BaseSettings):
    """Comprehensive audit processor configuration"""

    # Batch Processing Configuration
    AUDIT_BATCH_SIZE: int = Field(
        default=10, description="Number of questions per batch"
    )
    AUDIT_BATCH_STRATEGY: AuditBatchStrategy = Field(
        default=AuditBatchStrategy.FIXED_SIZE
    )
    AUDIT_MAX_QUESTIONS: int = Field(
        default=200, description="Maximum questions per audit"
    )
    AUDIT_INTER_BATCH_DELAY: float = Field(
        default=2.0, description="Delay between batches in seconds"
    )

    # Platform Configuration
    AUDIT_PLATFORM_TIMEOUT_SECONDS: int = Field(
        default=30, description="Timeout for platform requests"
    )
    AUDIT_MAX_CONCURRENT_PLATFORMS: int = Field(
        default=4, description="Max platforms to query concurrently"
    )
    AUDIT_PLATFORM_RETRY_ATTEMPTS: int = Field(
        default=3, description="Retry attempts per platform"
    )

    # Question Generation Configuration
    AUDIT_QUESTION_GENERATION_TIMEOUT: int = Field(
        default=60, description="Question generation timeout"
    )
    AUDIT_MIN_QUESTIONS_PER_CATEGORY: int = Field(
        default=5, description="Minimum questions per category"
    )
    AUDIT_MAX_QUESTIONS_PER_CATEGORY: int = Field(
        default=50, description="Maximum questions per category"
    )

    # Brand Detection Configuration
    AUDIT_BRAND_DETECTION_TIMEOUT: int = Field(
        default=10, description="Brand detection timeout per response"
    )
    AUDIT_BRAND_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7, description="Minimum confidence for brand detection"
    )
    AUDIT_MAX_BRAND_CONTEXTS: int = Field(
        default=3, description="Maximum contexts to store per brand mention"
    )

    # Progress Tracking Configuration
    AUDIT_PROGRESS_UPDATE_INTERVAL: int = Field(
        default=5, description="Progress update interval in seconds"
    )
    AUDIT_PROGRESS_PERSISTENCE: bool = Field(
        default=True, description="Persist progress to database"
    )

    # Error Handling & Recovery
    AUDIT_RETRY_STRATEGY: AuditRetryStrategy = Field(
        default=AuditRetryStrategy.EXPONENTIAL_BACKOFF
    )
    AUDIT_MAX_RETRY_ATTEMPTS: int = Field(
        default=3, description="Maximum retry attempts for failed audits"
    )
    AUDIT_RETRY_BACKOFF_MULTIPLIER: float = Field(
        default=2.0, description="Backoff multiplier for retries"
    )
    AUDIT_CIRCUIT_BREAKER_THRESHOLD: int = Field(
        default=5, description="Failures before circuit breaker opens"
    )

    # Resource Management
    AUDIT_MAX_MEMORY_MB: int = Field(
        default=1024, description="Maximum memory usage per audit"
    )
    AUDIT_CLEANUP_INTERVAL_HOURS: int = Field(
        default=24, description="Cleanup interval for old data"
    )
    AUDIT_MAX_REPORT_SIZE_MB: int = Field(
        default=50, description="Maximum report file size"
    )

    # Monitoring & Observability
    AUDIT_METRICS_ENABLED: bool = Field(
        default=True, description="Enable metrics collection"
    )
    AUDIT_DETAILED_LOGGING: bool = Field(
        default=True, description="Enable detailed audit logging"
    )
    AUDIT_PERFORMANCE_TRACKING: bool = Field(
        default=True, description="Track performance metrics"
    )

    # Platform Rate Limits (requests per minute)
    AUDIT_OPENAI_RPM: int = Field(default=50, description="OpenAI requests per minute")
    AUDIT_ANTHROPIC_RPM: int = Field(
        default=100, description="Anthropic requests per minute"
    )
    AUDIT_PERPLEXITY_RPM: int = Field(
        default=20, description="Perplexity requests per minute"
    )
    AUDIT_GOOGLE_AI_RPM: int = Field(
        default=60, description="Google AI requests per minute"
    )

    # Development/Testing Configuration
    AUDIT_MOCK_AI_RESPONSES: bool = Field(
        default=False, description="Use mock AI responses for testing"
    )
    AUDIT_SKIP_BRAND_DETECTION: bool = Field(
        default=False, description="Skip brand detection for testing"
    )
    AUDIT_DRY_RUN: bool = Field(
        default=False, description="Run audit without persisting results"
    )

    @field_validator("AUDIT_BATCH_SIZE")
    def validate_batch_size(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("Batch size must be between 1 and 100")
        return v

    @field_validator("AUDIT_BRAND_CONFIDENCE_THRESHOLD")
    def validate_confidence_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

    @field_validator("AUDIT_INTER_BATCH_DELAY")
    def validate_inter_batch_delay(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Inter-batch delay must be non-negative")
        return v

    @field_validator("AUDIT_MAX_QUESTIONS")
    def validate_max_questions(cls, v: int) -> int:
        if v < 1 or v > 1000:
            raise ValueError("Max questions must be between 1 and 1000")
        return v

    @field_validator("AUDIT_PLATFORM_TIMEOUT_SECONDS")
    def validate_platform_timeout(cls, v: int) -> int:
        if v < 1 or v > 300:  # 5 minutes max
            raise ValueError("Platform timeout must be between 1 and 300 seconds")
        return v

    @property
    def platform_rate_limits(self) -> Dict[str, int]:
        """Get platform rate limits as a dictionary"""
        return {
            "openai": self.AUDIT_OPENAI_RPM,
            "anthropic": self.AUDIT_ANTHROPIC_RPM,
            "perplexity": self.AUDIT_PERPLEXITY_RPM,
            "google_ai": self.AUDIT_GOOGLE_AI_RPM,
        }

    @property
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.AUDIT_MOCK_AI_RESPONSES or self.AUDIT_DRY_RUN

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy and attempt number"""
        if self.AUDIT_RETRY_STRATEGY == AuditRetryStrategy.IMMEDIATE:
            return 0.0
        elif self.AUDIT_RETRY_STRATEGY == AuditRetryStrategy.LINEAR_BACKOFF:
            return attempt * self.AUDIT_RETRY_BACKOFF_MULTIPLIER
        else:  # EXPONENTIAL_BACKOFF
            return (self.AUDIT_RETRY_BACKOFF_MULTIPLIER**attempt) - 1

    def get_batch_size_for_platform(self, platform: str) -> int:
        """Get optimized batch size for specific platform"""
        if self.AUDIT_BATCH_STRATEGY == AuditBatchStrategy.PLATFORM_OPTIMIZED:
            # Platform-specific optimizations
            platform_optimizations = {
                "openai": min(self.AUDIT_BATCH_SIZE, 15),
                "anthropic": min(self.AUDIT_BATCH_SIZE, 20),
                "perplexity": min(self.AUDIT_BATCH_SIZE, 5),
                "google_ai": min(self.AUDIT_BATCH_SIZE, 10),
            }
            return platform_optimizations.get(platform, self.AUDIT_BATCH_SIZE)
        return self.AUDIT_BATCH_SIZE

    class Config:
        env_prefix = "AUDIT_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global audit settings instance
audit_settings = AuditSettings()


def get_audit_settings() -> AuditSettings:
    """Get the global audit settings instance"""
    return audit_settings


def update_audit_settings(**kwargs: Any) -> None:
    """Update audit settings at runtime (for testing)"""
    global audit_settings
    for key, value in kwargs.items():
        if hasattr(audit_settings, key):
            setattr(audit_settings, key, value)
        else:
            raise ValueError(f"Unknown audit setting: {key}")
