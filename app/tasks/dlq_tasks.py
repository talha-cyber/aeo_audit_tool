"""
Celery tasks for processing Dead Letter Queue (DLQ) messages.

These tasks allow periodic or ad-hoc recovery of failed audit tasks
captured in the Redis-backed DLQ.
"""

from typing import Any, Dict

from app.core.celery_app import celery_app
from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.resilience.dead_letter import DeadLetterQueue

from .audit_tasks import run_audit_task

logger = get_logger(__name__)


@celery_app.task
def process_audit_dlq(max_messages: int = 100) -> Dict[str, Any]:
    """
    Process audit DLQ and resubmit tasks.

    Pops up to `max_messages` from `dlq:audit:tasks` and for each message
    resubmits `run_audit_task` with the captured `audit_run_id`.
    """
    if not settings.DLQ_ENABLED:
        return {"status": "disabled"}

    dlq = DeadLetterQueue(max_retries=settings.DLQ_MAX_RETRIES)
    queue_name = "audit:tasks"
    submitted = 0

    async def _processor(message: Dict[str, Any]) -> bool:
        nonlocal submitted
        audit_run_id = message.get("audit_run_id")
        if not audit_run_id:
            logger.warning("DLQ message missing audit_run_id; skipping")
            return True  # drop invalid message
        # Resubmit Celery task
        run_audit_task.delay(audit_run_id)
        submitted += 1
        return True

    # Run the async processor from sync Celery task
    import asyncio

    processed = asyncio.run(
        dlq.process_dlq_messages(queue_name, _processor, max_messages=max_messages)
    )

    result = {"status": "completed", "processed": processed, "resubmitted": submitted}
    logger.info("DLQ processing finished", **result)
    return result
