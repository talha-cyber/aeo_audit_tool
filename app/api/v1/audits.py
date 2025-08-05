import uuid
from typing import Generator, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun, Client
from app.tasks.audit_tasks import run_audit_task

router = APIRouter(prefix="/audits", tags=["audits"])


class AuditRunResponse(BaseModel):
    id: str
    status: str
    error_log: Optional[str] = None


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/configs/{client_id}/run", response_model=AuditRunResponse)
async def trigger_audit_run(
    client_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> AuditRun:
    """Trigger immediate audit run"""

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    audit_run = AuditRun(
        id=uuid.uuid4().hex,
        client_id=client.id,
        config={"platforms": ["openai"]},
        status="pending",
    )
    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)

    background_tasks.add_task(run_audit_task, audit_run.id)

    return audit_run


@router.get("/runs/{run_id}/status", response_model=AuditRunResponse)
async def get_audit_status(run_id: str, db: Session = Depends(get_db)) -> AuditRun:
    """Get audit run status"""

    audit_run = db.query(AuditRun).filter(AuditRun.id == run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    return cast(AuditRun, audit_run)
