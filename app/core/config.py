# app/core/config.py
from __future__ import annotations

import secrets
import warnings
from typing import Any, List, Optional

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> List[str]:
    """Parse CORS origins from comma-separated string or list."""
    if isinstance(v, str) and v:
        return [i.strip() for i in v.split(",")]
    if isinstance(v, (list, set)):
        return v
    return []


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables and/or .env file.
    """

    # Environment settings
    APP_NAME: str = "AEO Competitive Intelligence Tool"
    APP_ENV: str = "development"

    # Security / auth
    SECRET_KEY: str = Field(default="", repr=False)
    SECRET_KEY_AUTO_GENERATED: bool = False
    ENABLE_DEBUG_ENDPOINTS: bool = False
    CORS_ALLOW_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Comma-delimited list of allowed origins",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Project metadata
    PROJECT_NAME: str = "AEO Audit Tool"
    API_V1_STR: str = "/api/v1"

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
    OPENAI_API_KEY: Optional[str] = Field(default=None, repr=False)
    OPENAI_ORG_ID: Optional[str] = Field(default=None, repr=False)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, repr=False)
    PERPLEXITY_API_KEY: Optional[str] = Field(default=None, repr=False)
    GOOGLE_API_KEY: Optional[str] = Field(default=None, repr=False)

    # Celery settings
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Observability settings
    SENTRY_DSN: Optional[str] = None

    # Runtime environment flags
    RUNNING_IN_DOCKER: bool = False

    # Dynamic Question Engine settings
    DYNAMIC_Q_ENABLED: bool = True
    DYNAMIC_Q_MAX: int = 25
    DYNAMIC_Q_CACHE_TTL: int = 3600 * 24  # 24 hours
    LLM_CONCURRENCY: int = 5
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_INPUT_COST_PER_1K: float = 0.0005
    LLM_OUTPUT_COST_PER_1K: float = 0.0015
    QUESTION_ENGINE_V2: bool = False

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

    # Security settings
    ENABLE_SECURITY_HEADERS: bool = True
    CSP_POLICY: str = "default-src 'self'"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRES_MINUTES: int = 60
    JWT_ISSUER: Optional[str] = None
    SESSION_TTL_SECONDS: int = 3600
    FIELD_ENCRYPTION_KEY: Optional[str] = None
    API_KEY_PEPPER: Optional[str] = None

    # Monitoring/Tracing
    TRACING_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4318/v1/traces"
    TRACING_SAMPLE_RATIO: float = 0.1
    SERVICE_NAME: Optional[str] = None
    ALERT_WEBHOOK_URL: Optional[str] = None

    # Dashboard persona library persistence
    PERSONA_LIBRARY_PATH: Optional[str] = None

    @property
    def database_url(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @model_validator(mode="after")
    def _ensure_secret_key(self) -> "Settings":
        """Guarantee SECRET_KEY is present in non-development environments."""
        secret = (self.SECRET_KEY or "").strip()
        environment = (self.APP_ENV or "development").lower()

        if not secret or secret.lower() == "change-me":
            if environment in {"development", "test", "testing"}:
                # Generate an ephemeral key for local dev/tests and warn loudly.
                generated = secrets.token_urlsafe(48)
                self.SECRET_KEY = generated
                self.SECRET_KEY_AUTO_GENERATED = True
                warnings.warn(
                    (
                        "SECRET_KEY was not provided; generated ephemeral key for "
                        f"{environment} environment. "
                        "Do not use this configuration in production."
                    ),
                    RuntimeWarning,
                )
            else:
                raise ValueError(
                    (
                        "SECRET_KEY must be set for secure operation. "
                        "Set SECRET_KEY in the environment or .env file before "
                        "starting the service."
                    )
                )
        return self

    @model_validator(mode="after")
    def _adjust_service_hosts(self) -> "Settings":
        """Resolve service hostnames when running outside of Docker."""
        if not self.RUNNING_IN_DOCKER:
            if self.POSTGRES_SERVER in {"db", "postgres"}:
                self.POSTGRES_SERVER = "localhost"

            if self.REDIS_HOST == "redis":
                self.REDIS_HOST = "localhost"

            if self.CELERY_BROKER_URL.startswith("redis://redis"):
                self.CELERY_BROKER_URL = self.CELERY_BROKER_URL.replace(
                    "redis://redis", "redis://localhost", 1
                )
            if self.CELERY_RESULT_BACKEND.startswith("redis://redis"):
                self.CELERY_RESULT_BACKEND = self.CELERY_RESULT_BACKEND.replace(
                    "redis://redis", "redis://localhost", 1
                )
        return self

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    def _parse_cors(cls, v: Any) -> List[str]:
        return parse_cors(v)

    @field_validator("SECRET_KEY", mode="before")
    def _validate_secret_key(cls, v: Optional[str], info: ValidationInfo) -> str:
        if not v:
            if info.data.get("SECRET_KEY_AUTO_GENERATED", False):
                import secrets

                return secrets.token_urlsafe(32)
            raise ValueError("SECRET_KEY must be set")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
