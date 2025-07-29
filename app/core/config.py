# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables and/or .env file.
    """

    # Environment settings
    APP_NAME: str = "AEO Competitive Intelligence Tool"
    APP_ENV: str = "development"

    # Database settings
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: int = 5432

    # Redis settings
    REDIS_HOST: str
    REDIS_PORT: int = 6379

    # AI Platform API keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str

    # Celery settings
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    @property
    def database_url(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
