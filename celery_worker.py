import os

from celery import Celery

# Redis URL from environment or default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Create Celery app
celery_app = Celery("aeo_audit_tool", broker=REDIS_URL, backend=REDIS_URL)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
)


@celery_app.task
def test_task(message: str = "Hello from Celery!") -> dict:
    """
    No-op test task to verify Celery is working
    """
    print(f"Test task executed: {message}")
    return {"status": "completed", "message": message}


if __name__ == "__main__":
    celery_app.start()
