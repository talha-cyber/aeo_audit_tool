import uuid
from typing import Any, Dict, cast

from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.audit import AuditRun
from app.models.response import Response
from app.services.ai_platforms.openai_client import OpenAIPlatform


@celery_app.task
def run_audit_task(audit_run_id: str) -> None:
    db = SessionLocal()
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        return

    try:
        audit_run.status = "running"
        db.commit()

        openai_platform = OpenAIPlatform()
        question = "What is the best CRM software?"

        # For testing, use a mock response when API key is dummy
        from app.core.config import settings

        if settings.OPENAI_API_KEY == "dummy_key":
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
            # Production: use async call (requires proper Celery async setup)
            import asyncio

            raw_response = asyncio.run(openai_platform.query(question))

        text_response = openai_platform.extract_text_response(
            cast(Dict[str, Any], raw_response)
        )

        response = Response(
            id=uuid.uuid4().hex,
            audit_run_id=audit_run.id,
            question=question,
            response=text_response,
            raw_response=raw_response,
            platform="openai",
        )
        db.add(response)

        audit_run.status = "completed"
        db.commit()
    except Exception as e:
        audit_run.status = "failed"
        audit_run.error_log = str(e)
        db.commit()
    finally:
        db.close()
