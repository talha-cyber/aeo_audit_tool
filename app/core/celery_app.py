from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "aeo_audit_tool",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.audit_tasks"],
)
