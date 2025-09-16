import uuid
from typing import Generator, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun, Client
from app.models.report import Report
from app.services.report_generator import ReportGenerator
from app.tasks.audit_tasks import run_audit_task

router = APIRouter(prefix="/audits", tags=["audits"])


class AuditRunResponse(BaseModel):
    id: str
    status: str
    error_log: Optional[str] = None


class ReportGenerationResponse(BaseModel):
    message: str
    audit_run_id: str
    report_type: str


class ReportStatusResponse(BaseModel):
    id: str
    audit_run_id: str
    report_type: str
    file_path: Optional[str]
    generated_at: Optional[str]


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
        config={
            "platforms": ["openai"],
            "client": {
                "id": client.id,
                "name": client.name,
                "industry": getattr(client, "industry", None),
                "product_type": getattr(client, "product_type", None),
                "competitors": getattr(client, "competitors", None),
            },
        },
        status="pending",
    )
    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)

    # Enqueue Celery task (do not run in FastAPI background thread)
    run_audit_task.delay(audit_run.id)

    return audit_run


@router.get("/runs/{run_id}/status", response_model=AuditRunResponse)
async def get_audit_status(run_id: str, db: Session = Depends(get_db)) -> AuditRun:
    """Get audit run status"""

    audit_run = db.query(AuditRun).filter(AuditRun.id == run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    return cast(AuditRun, audit_run)


@router.post("/runs/{run_id}/generate-report", response_model=ReportGenerationResponse)
async def trigger_generate_report(
    run_id: str,
    report_type: str = "v2_comprehensive",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    Trigger the generation of a PDF report for a completed audit run.
    For reliability in this environment, generate synchronously.
    """
    audit_run = db.query(AuditRun).filter(AuditRun.id == run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    if audit_run.status != "completed":
        detail_message = (
            f"Audit run status is '{audit_run.status}'. "
            "Must be 'completed' to generate a report."
        )
        raise HTTPException(status_code=400, detail=detail_message)

    # Synchronous generation to ensure immediate availability
    generator = ReportGenerator(db_session=db)
    try:
        _ = generator.generate_audit_report(
            audit_run_id=run_id, report_type=report_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return {
        "message": "Report generated successfully.",
        "audit_run_id": run_id,
        "report_type": report_type,
    }


@router.get("/runs/{run_id}/report", response_model=ReportStatusResponse)
async def get_report_status(run_id: str, db: Session = Depends(get_db)):
    """
    Get the status and file path of a generated report.
    """
    report_query = db.query(Report).filter(Report.audit_run_id == run_id)
    ordering = desc(Report.generated_at)
    report = report_query.order_by(ordering).first()
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Report not found. It may not have been generated yet.",
        )

    return {
        "id": report.id,
        "audit_run_id": report.audit_run_id,
        "report_type": report.report_type,
        "file_path": report.file_path,
        "generated_at": report.generated_at.isoformat()
        if report.generated_at
        else None,
    }
