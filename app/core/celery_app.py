import sentry_sdk
from celery import Celery
from celery.signals import setup_logging
from sentry_sdk.integrations.celery import CeleryIntegration

from app.core.config import settings
from app.utils.logger import configure_logging, get_logger


@setup_logging.connect
def setup_celery_logging(**kwargs):
    """Configure structured logging for Celery workers"""
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Celery logging configured")


# Initialize Sentry for Celery if DSN is provided
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[CeleryIntegration()],
        environment=settings.APP_ENV,
        traces_sample_rate=0.1,
    )


celery_app = Celery(
    "aeo_audit_tool",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.audit_tasks", "app.tasks.dlq_tasks"],
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
