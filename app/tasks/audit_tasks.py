"""
Celery tasks for audit processing using the new AuditProcessor orchestrator.

This module provides the Celery task integration for running audit processes
with comprehensive error handling, retry logic, and monitoring.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from celery.signals import task_failure, task_postrun, task_prerun

from app.core.audit_config import get_audit_settings
from app.core.celery_app import celery_app
from app.core.config import settings as core_settings
from app.db.session import SessionLocal
from app.services.audit_context import add_audit_context, contextual_logger
from app.services.audit_metrics import get_audit_metrics
from app.services.audit_processor import AuditProcessor
from app.services.platform_manager import PlatformManager
from app.services.progress_tracker import create_progress_tracker
from app.services.report_generator import ReportGenerator
from app.utils.logger import get_logger
from app.utils.resilience.dead_letter import DeadLetterQueue

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes max
    max_retries=3,
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=3900,  # 65 minutes hard limit
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_audit_task(self, audit_run_id: str) -> Dict[str, Any]:
    """
    Execute an audit run using the AuditProcessor orchestrator.

    This task coordinates the entire audit process including:
    - Question generation
    - Multi-platform querying
    - Brand detection
    - Progress tracking
    - Error handling and recovery

    Args:
        audit_run_id: The ID of the audit run to execute

    Returns:
        Dict containing task result information

    Raises:
        Exception: Re-raises exceptions to trigger Celery retry mechanism
    """
    return asyncio.run(run_audit_async(self, audit_run_id))


async def run_audit_async(task_instance, audit_run_id: str) -> Dict[str, Any]:
    """
    Async implementation of audit task execution.

    Args:
        task_instance: Celery task instance for accessing task metadata
        audit_run_id: The ID of the audit run to execute

    Returns:
        Dict containing execution results and metadata
    """
    settings = get_audit_settings()
    metrics = get_audit_metrics()

    # Add audit context to all logs in this execution
    with add_audit_context(
        audit_run_id=audit_run_id,
        task_id=task_instance.request.id,
        worker_id=task_instance.request.hostname,
    ):
        contextual_logger.info(
            "Starting audit task execution",
            task_id=task_instance.request.id,
            retry_count=task_instance.request.retries,
        )

        # Initialize database session
        db = SessionLocal()
        progress_tracker = None

        try:
            # Initialize progress tracker
            progress_tracker = create_progress_tracker(db, audit_run_id)
            await progress_tracker.initialize()

            # Initialize platform manager
            contextual_logger.info("Initializing platform manager")
            platform_manager = PlatformManager()

            # Verify at least one platform is available
            available_platforms = platform_manager.get_available_platforms()
            if not available_platforms:
                raise ValueError("No AI platforms are available for audit execution")

            contextual_logger.info(
                "Platform manager initialized", available_platforms=available_platforms
            )

            # Initialize audit processor
            contextual_logger.info("Initializing audit processor")
            audit_processor = AuditProcessor(db, platform_manager)

            # Record task start metrics
            metrics.increment_audit_started()

            # Execute the audit
            contextual_logger.info("Starting audit execution")
            result_audit_run_id = await audit_processor.run_audit(audit_run_id)

            # Update progress tracker to completed
            if progress_tracker:
                await progress_tracker.finalize_tracking(
                    final_stage=progress_tracker.ProgressStage.COMPLETED, success=True
                )

            # Record success metrics
            metrics.increment_audit_completed()

            contextual_logger.info(
                "Audit task completed successfully",
                result_audit_run_id=result_audit_run_id,
            )

            return {
                "status": "completed",
                "audit_run_id": result_audit_run_id,
                "task_id": task_instance.request.id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": task_instance.request.retries,
                "available_platforms": available_platforms,
            }

        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__

            contextual_logger.error(
                "Audit task failed with exception",
                error=error_message,
                error_type=error_type,
                task_id=task_instance.request.id,
                retry_count=task_instance.request.retries,
                exc_info=True,
            )

            # Update progress tracker to failed
            if progress_tracker:
                try:
                    await progress_tracker.finalize_tracking(
                        final_stage=progress_tracker.ProgressStage.FAILED, success=False
                    )
                    await progress_tracker.record_error(error_message=error_message)
                except Exception as tracker_error:
                    contextual_logger.error(
                        "Failed to update progress tracker", error=str(tracker_error)
                    )

            # Record failure metrics
            metrics.increment_audit_failed()

            # Determine if we should retry
            should_retry = (
                task_instance.request.retries < task_instance.max_retries
                and _is_retryable_error(e)
            )

            if should_retry:
                contextual_logger.warning(
                    "Audit task will be retried",
                    retry_count=task_instance.request.retries + 1,
                    max_retries=task_instance.max_retries,
                    error_type=error_type,
                )

                # Calculate retry delay based on settings
                retry_delay = settings.get_retry_delay(
                    task_instance.request.retries + 1
                )

                # Retry with backoff
                raise task_instance.retry(
                    countdown=retry_delay, exc=e, max_retries=task_instance.max_retries
                )
            else:
                contextual_logger.error(
                    "Audit task failed permanently",
                    retry_count=task_instance.request.retries,
                    max_retries=task_instance.max_retries,
                    error_type=error_type,
                )

                # Send to DLQ if enabled
                try:
                    if core_settings.DLQ_ENABLED:
                        dlq = DeadLetterQueue(max_retries=core_settings.DLQ_MAX_RETRIES)
                        payload = {
                            "audit_run_id": audit_run_id,
                            "task_id": task_instance.request.id,
                            "worker_id": task_instance.request.hostname,
                            "error_type": error_type,
                            "error_message": error_message,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        await dlq.send_to_dlq(
                            payload,
                            original_queue="audit:tasks",
                            error=e,
                        )
                        contextual_logger.warning(
                            "Task payload sent to DLQ", dlq_queue="audit:tasks"
                        )
                except Exception as dlq_err:
                    contextual_logger.error(
                        "Failed to send to DLQ",
                        dlq_error=str(dlq_err),
                    )

                # Final failure - re-raise to mark task as failed
                raise

        finally:
            # Cleanup database connection
            try:
                db.close()
                contextual_logger.info("Database connection closed")
            except Exception as cleanup_error:
                contextual_logger.error(
                    "Error during cleanup", error=str(cleanup_error)
                )


@celery_app.task(bind=True)
def generate_report_task(
    self, audit_run_id: str, report_type: str = "summary"
) -> Dict[str, Any]:
    """
    Generate a report for a completed audit run.

    Args:
        audit_run_id: ID of the completed audit run
        report_type: Type of report to generate (summary, detailed, competitive)

    Returns:
        Dict containing report generation results
    """
    return asyncio.run(generate_report_async(self, audit_run_id, report_type))


async def generate_report_async(
    task_instance, audit_run_id: str, report_type: str
) -> Dict[str, Any]:
    """
    Async implementation of report generation.
    """
    with add_audit_context(
        audit_run_id=audit_run_id,
        task_id=task_instance.request.id,
        report_type=report_type,
    ):
        contextual_logger.info("Starting report generation task")

        db = SessionLocal()

        try:
            # Initialize and run the report generator
            report_generator = ReportGenerator(db_session=db)
            report_path = report_generator.generate_audit_report(
                audit_run_id=audit_run_id, report_type=report_type
            )

            contextual_logger.info(
                "Report generation completed successfully", report_path=report_path
            )

            return {
                "status": "completed",
                "audit_run_id": audit_run_id,
                "report_type": report_type,
                "report_path": report_path,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            contextual_logger.error(
                "Report generation failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

        finally:
            db.close()


@celery_app.task
def cleanup_old_audit_runs() -> Dict[str, Any]:
    """
    Cleanup old audit runs and associated data.

    This task should be run periodically to maintain database size
    and remove audit runs older than the configured retention period.

    Returns:
        Dict containing cleanup results
    """
    settings = get_audit_settings()

    logger.info("Starting audit cleanup task")

    db = SessionLocal()

    try:
        from datetime import timedelta

        from app.models.audit import AuditRun
        from app.models.question import Question
        from app.models.response import Response

        # Calculate cutoff date based on retention settings
        retention_days = getattr(settings, "AUDIT_DATA_RETENTION_DAYS", 90)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

        # Find old audit runs
        old_runs = (
            db.query(AuditRun)
            .filter(
                AuditRun.created_at < cutoff_date,
                AuditRun.status.in_(["completed", "failed"]),
            )
            .all()
        )

        cleaned_count = 0

        for run in old_runs:
            try:
                # Delete associated responses and questions
                db.query(Response).filter(Response.audit_run_id == run.id).delete()
                db.query(Question).filter(Question.audit_run_id == run.id).delete()
                db.delete(run)
                cleaned_count += 1

            except Exception as e:
                logger.error(
                    "Failed to delete audit run", audit_run_id=run.id, error=str(e)
                )
                db.rollback()
                continue

        db.commit()

        logger.info(
            "Cleanup completed",
            cleaned_runs=cleaned_count,
            cutoff_date=cutoff_date.isoformat(),
            retention_days=retention_days,
        )

        return {
            "status": "completed",
            "cleaned_runs": cleaned_count,
            "cutoff_date": cutoff_date.isoformat(),
            "retention_days": retention_days,
        }

    except Exception as e:
        logger.error("Cleanup task failed", error=str(e), exc_info=True)
        db.rollback()
        raise

    finally:
        db.close()


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The exception that occurred

    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_errors = [
        "ConnectionError",
        "TimeoutError",
        "TemporaryFailure",
        "PlatformUnavailableError",
        "PlatformTimeoutError",
        "PlatformRateLimitError",
    ]

    non_retryable_errors = [
        "ValueError",  # Configuration errors
        "PlatformAuthenticationError",
        "AuditConfigurationError",
    ]

    error_name = type(error).__name__

    # Check for non-retryable errors first
    if error_name in non_retryable_errors:
        return False

    # Check for explicitly retryable errors
    if error_name in retryable_errors:
        return True

    # Check error message for retryable patterns
    error_message = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "service unavailable",
    ]

    return any(pattern in error_message for pattern in retryable_patterns)


# === Celery Signal Handlers ===


@task_prerun.connect
def task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds
):
    """Handle task pre-run signal for logging and metrics"""
    logger.info(
        "Task starting", task_name=task.name, task_id=task_id, args=args, kwargs=kwargs
    )


@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kwds,
):
    """Handle task post-run signal for logging and metrics"""
    logger.info(
        "Task completed",
        task_name=task.name,
        task_id=task_id,
        state=state,
        return_value_type=type(retval).__name__ if retval else None,
    )


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds
):
    """Handle task failure signal for logging and metrics"""
    logger.error(
        "Task failed",
        task_name=sender.name,
        task_id=task_id,
        exception=str(exception),
        exception_type=type(exception).__name__,
    )
