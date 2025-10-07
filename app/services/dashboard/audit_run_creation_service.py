"""Service for creating real audit runs with persona-driven question generation."""

import uuid
from datetime import datetime
from typing import List
from uuid import UUID

from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import (
    AuditRunProgressView,
    AuditRunView,
    CreateAuditRunRequest,
    CreateAuditRunResponse,
)
from app.models.audit import AuditRun, Client
from app.models.persona import Persona
from app.tasks.audit_tasks import run_audit_task
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_audit_run(
    db: Session,
    payload: CreateAuditRunRequest,
    owner_id: str,
    *,
    act_as_client_id: str | None = None,
) -> CreateAuditRunResponse:
    """
    Create a new audit run with persona selection and trigger question generation.

    Args:
        db: Database session
        payload: Audit run creation request
        owner_id: User ID creating the run

    Returns:
        CreateAuditRunResponse with the created run

    Raises:
        ValueError: If personas not found or don't belong to user
    """
    # Validate personas exist and belong to user
    persona_ids: List[UUID] = []
    for raw_id in payload.persona_ids:
        if isinstance(raw_id, uuid.UUID):
            persona_ids.append(raw_id)
        else:
            try:
                persona_ids.append(UUID(str(raw_id)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid persona identifier: {raw_id}") from error

    persona_query = db.query(Persona).filter(
        Persona.id.in_(persona_ids),
        Persona.owner_id == owner_id,
    )
    resolved_client_id = payload.client_id or act_as_client_id
    if resolved_client_id:
        persona_query = persona_query.filter(Persona.client_id == resolved_client_id)

    personas = persona_query.all()

    if len(personas) != len(payload.persona_ids):
        found_ids = {p.id for p in personas}
        missing_ids = set(payload.persona_ids) - found_ids
        raise ValueError(
            f"Personas not found or not accessible: {', '.join(missing_ids)}"
        )

    # Get or create client if client_id provided
    client = None
    resolved_client_id = payload.client_id or act_as_client_id
    if resolved_client_id:
        client = db.query(Client).filter(Client.id == resolved_client_id).first()
        if not client:
            raise ValueError(f"Client not found: {resolved_client_id}")

    persona_mode = personas[0].mode if personas else None
    persona_payloads = [
        {
            "id": str(p.id),
            "name": p.name,
            "role": p.role,
            "driver": p.driver,
            "voice": p.voice,
            "mode": p.mode,
            "priority": p.priority,
            "segment": p.segment,
            "key_need": p.key_need,
            "context_keys": list(p.contexts or []),
        }
        for p in personas
    ]

    # Create audit run
    audit_run = AuditRun(
        id=uuid.uuid4().hex,
        client_id=client.id if client else None,
        config={
            "name": payload.name,
            "platforms": payload.platforms,
            "question_count": payload.question_count,
            "persona_ids": payload.persona_ids,
            "personas": persona_payloads,
            "persona_mode": persona_mode,
            "persona_owner": owner_id,
            "owner_id": owner_id,
            "client": {
                "id": client.id,
                "name": client.name,
                "industry": getattr(client, "industry", None),
                "product_type": getattr(client, "product_type", None),
                "competitors": getattr(client, "competitors", None),
            }
            if client
            else None,
        },
        status="pending",
        started_at=datetime.utcnow(),
    )

    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)

    # Trigger background task to run the audit
    try:
        run_audit_task.delay(audit_run.id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "audit_run.enqueue_failed",
            run_id=audit_run.id,
            error=str(exc),
        )

    # Build response
    run_view = AuditRunView(
        id=audit_run.id,
        name=payload.name,
        status=audit_run.status,
        startedAt=audit_run.started_at,
        completedAt=audit_run.completed_at,
        progress=AuditRunProgressView(
            done=0, total=payload.question_count, updatedAt=datetime.utcnow()
        ),
        issues=[],
    )

    return CreateAuditRunResponse(run=run_view)
