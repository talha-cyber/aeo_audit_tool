import uuid
from typing import Generator, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun
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


@router.post("/configs/{config_id}/run", response_model=AuditRunResponse)
async def trigger_audit_run(
    config_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> AuditRun:
    """Trigger immediate audit run"""

    # For now, we'll create a dummy AuditRun, since we don't have AuditConfig yet
    audit_run = AuditRun(
        id=uuid.uuid4().hex,
        client_id=1,  # Dummy client_id
        config={},
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
