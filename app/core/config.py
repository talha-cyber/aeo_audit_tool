# app/core/config.py
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables and/or .env file.
    """

    # Environment settings
    APP_NAME: str = "AEO Competitive Intelligence Tool"
    APP_ENV: str = "development"

    # Database settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "aeo_audit"
    POSTGRES_SERVER: str = "db"
    POSTGRES_PORT: int = 5432

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # AI Platform API keys
    OPENAI_API_KEY: str = "dummy_key"
    ANTHROPIC_API_KEY: str = "dummy_key"
    PERPLEXITY_API_KEY: str = "dummy_key"
    GOOGLE_AI_API_KEY: str = "dummy_key"

    # Celery settings
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Observability settings
    SENTRY_DSN: Optional[str] = None

    # Dynamic Question Engine settings
    DYNAMIC_Q_ENABLED: bool = True
    DYNAMIC_Q_MAX: int = 25
    DYNAMIC_Q_CACHE_TTL: int = 3600 * 24  # 24 hours
    LLM_CONCURRENCY: int = 5
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_INPUT_COST_PER_1K: float = 0.0005
    LLM_OUTPUT_COST_PER_1K: float = 0.0015

    # Resilience defaults
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: int = 60
    CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD: int = 2

    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE_SECONDS: float = 0.2
    RETRY_BACKOFF_MULTIPLIER: float = 2.0
    RETRY_BACKOFF_MAX_SECONDS: float = 5.0
    RETRY_USE_JITTER: bool = True

    DLQ_ENABLED: bool = True
    DLQ_MAX_RETRIES: int = 3
    DLQ_RETENTION_SECONDS: int = 7 * 24 * 3600

    BULKHEAD_DEFAULT_MAX_CONCURRENCY: int = 10

    @property
    def database_url(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
