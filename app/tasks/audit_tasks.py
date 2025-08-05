import asyncio
import uuid
from typing import Any, Dict, cast

from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.audit import AuditRun
from app.models.question import Question
from app.models.response import Response
from app.services.ai_platforms.openai_client import OpenAIPlatform
from app.services.question_engine import QuestionEngine
from app.utils.logger import add_audit_context, add_platform_context, get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True)
def run_audit_task(self, audit_run_id: str) -> None:
    """Execute an audit run task with structured logging and error handling"""
    asyncio.run(run_audit_async(self, audit_run_id))


async def run_audit_async(self, audit_run_id: str) -> None:
    # Set up logging context
    context_logger = logger.bind(**add_audit_context(audit_run_id))
    context_logger.info("Starting audit task", task_id=self.request.id)

    db = SessionLocal()

    try:
        audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
        if not audit_run:
            context_logger.error("Audit run not found", audit_run_id=audit_run_id)
            return

        context_logger.info("Audit run found, starting execution")
        audit_run.status = "running"
        db.commit()

        # Initialize Question Engine
        question_engine = QuestionEngine()

        # Generate questions
        questions = await question_engine.generate_questions(
            client_brand=audit_run.client.name,
            competitors=audit_run.client.competitors,
            industry=audit_run.client.industry,
            product_type=audit_run.client.product_type,
            audit_run_id=uuid.UUID(audit_run.id),
        )

        # Store questions in the database
        for q in questions:
            db_question = Question(**q.model_dump())
            db.add(db_question)
        db.commit()

        # Initialize platform with logging context
        platform_logger = context_logger.bind(**add_platform_context("openai"))
        openai_platform = OpenAIPlatform()

        for question in questions:
            platform_logger.info(
                "Querying AI platform", question=question.question_text
            )

            # For testing, use a mock response when API key is dummy
            from app.core.config import settings

            if settings.OPENAI_API_KEY == "dummy_key":
                platform_logger.info("Using mock response (dummy API key)")
                raw_response = {
                    "choices": [
                        {
                            "message": {
                                "content": "Mock response: Salesforce is a popular CRM."
                            }
                        }
                    ]
                }
            else:
                # Production: use async call
                platform_logger.info("Making real API call")
                raw_response = await openai_platform.query(question.question_text)

            text_response = openai_platform.extract_text_response(
                cast(Dict[str, Any], raw_response)
            )

            platform_logger.info(
                "AI platform response received", response_length=len(text_response)
            )

            response = Response(
                id=uuid.uuid4().hex,
                audit_run_id=audit_run.id,
                question_id=question.id,
                response=text_response,
                raw_response=raw_response,
                platform="openai",
            )
            db.add(response)

        audit_run.status = "completed"
        db.commit()

        context_logger.info("Audit task completed successfully")

    except Exception as e:
        context_logger.error("Audit task failed", error=str(e), exc_info=True)
        try:
            audit_run.status = "failed"
            audit_run.error_log = str(e)
            db.commit()
        except Exception as commit_error:
            context_logger.error(
                "Failed to update audit run status", error=str(commit_error)
            )

        # Re-raise to trigger Celery retry mechanism if configured
        raise
    finally:
        db.close()
        context_logger.info("Database connection closed")
