import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.v1.security import _authorize
from app.db.session import SessionLocal
from app.models.audit import AuditRun, Client
from app.models.persona import Persona
from app.security.auth.decorators import require_capability
from app.tasks.audit_tasks import run_audit_task
from app.utils.logger import get_logger

router = APIRouter(prefix="/admin", tags=["admin"])
logger = get_logger(__name__)

_ADMIN_RATE_LIMIT = 20
_ADMIN_RATE_WINDOW_SECONDS = 60
_rate_lock = asyncio.Lock()
_rate_state: Dict[str, tuple[float, int]] = {}


class AdminAuditActionRequest(BaseModel):
    action: str = Field(description="The administrative action to perform")


class AdminTenantActionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    action: str = Field(description="Tenant-scoped administrative action")
    value: Optional[str] = Field(default=None, description="Generic value payload")
    platform: Optional[str] = Field(default=None, description="Platform identifier")
    enabled: Optional[bool] = Field(default=None, description="Boolean toggle value")
    flag: Optional[str] = Field(default=None, description="Feature flag identifier")


class AdminActionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_client_scope(user: Dict[str, Any], client_id: Optional[str]) -> None:
    act_as = user.get("act_as_client_id")
    if act_as and client_id and act_as != client_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Act-as context cannot operate on this tenant",
        )


def _load_client(db: Session, client_id: str) -> Client:
    client = db.query(Client).filter(Client.id == client_id).one_or_none()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Client not found"
        )
    return client


@router.post("/audits/{run_id}", response_model=AdminActionResponse)
@require_capability("admin")
async def admin_audit_action(
    run_id: str,
    action: Optional[str] = Query(default=None),
    payload: Optional[AdminAuditActionRequest] = Body(default=None),
    user: Dict[str, Any] = Depends(_authorize),
    db: Session = Depends(get_db),
) -> AdminActionResponse:
    actor_id = str(user.get("sub"))
    await _enforce_admin_rate(actor_id)

    audit_run = db.query(AuditRun).filter(AuditRun.id == run_id).one_or_none()
    if audit_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Audit run not found"
        )

    _ensure_client_scope(user, audit_run.client_id)

    requested_action = (payload.action if payload else action) or action
    if not requested_action:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing action"
        )

    action_normalized = requested_action.lower()
    if action_normalized not in {"force_retry", "force_rerun"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported audit action"
        )

    run_audit_task.delay(run_id)

    logger.info(
        "admin.audit_action",
        actor=actor_id,
        action=action_normalized,
        run_id=run_id,
        client_id=audit_run.client_id,
    )
    return AdminActionResponse(
        status="queued", action=action_normalized, details={"runId": run_id}
    )


@router.post("/tenants/{client_id}", response_model=AdminActionResponse)
@require_capability("admin")
async def admin_tenant_action(
    client_id: str,
    action: Optional[str] = Query(default=None),
    payload: Optional[AdminTenantActionRequest] = Body(default=None),
    user: Dict[str, Any] = Depends(_authorize),
    db: Session = Depends(get_db),
) -> AdminActionResponse:
    actor_id = str(user.get("sub"))
    await _enforce_admin_rate(actor_id)

    client = _load_client(db, client_id)
    _ensure_client_scope(user, client.id)

    admin_settings = dict(client.admin_settings or {})
    effective_action = (payload.action if payload else action) or action
    if not effective_action:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing action"
        )

    action = effective_action.lower()

    if action == "set_qe_version":
        if not payload or not payload.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing value for qe_version",
            )
        admin_settings.setdefault("qe_version", payload.value)
        admin_settings["qe_version"] = payload.value
        response_details = {"qeVersion": payload.value}
    elif action == "toggle_platform":
        if not payload or not payload.platform or payload.enabled is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Platform and enabled flag required",
            )
        overrides = admin_settings.setdefault("platform_overrides", {})
        overrides[payload.platform] = bool(payload.enabled)
        response_details = {"platform": payload.platform, "enabled": payload.enabled}
    elif action == "set_feature_flag":
        if not payload or not payload.flag or payload.enabled is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Flag and enabled required",
            )
        flags = admin_settings.setdefault("feature_flags", {})
        flags[payload.flag] = bool(payload.enabled)
        response_details = {"flag": payload.flag, "enabled": payload.enabled}
    elif action == "seed_demo_data":
        response_details = _seed_demo_data(db, client)
    elif action == "wipe_demo_data":
        response_details = _wipe_demo_data(db, client)
    elif action == "export_debug_bundle":
        bundle = {
            "client": client.id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        response_details = {"bundle": bundle}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported tenant action"
        )

    client.admin_settings = admin_settings
    db.add(client)
    db.commit()

    logger.info(
        "admin.tenant_action",
        actor=actor_id,
        client_id=client.id,
        action=action,
        details=response_details,
    )

    return AdminActionResponse(status="ok", action=action, details=response_details)


async def _enforce_admin_rate(actor_id: str) -> None:
    now = time.time()
    async with _rate_lock:
        reset_at, count = _rate_state.get(actor_id, (0.0, 0))
        if now >= reset_at:
            reset_at = now + _ADMIN_RATE_WINDOW_SECONDS
            count = 0

        if count >= _ADMIN_RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many admin actions. Please wait",
            )

        _rate_state[actor_id] = (reset_at, count + 1)


def _seed_demo_data(db: Session, client: Client) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    created = 0
    templates = [
        ("completed", "completed", timedelta(hours=-6)),
        ("running", "running", timedelta(hours=-1)),
        ("pending", "pending", timedelta()),
    ]
    for suffix, status, offset in templates:
        run_id = f"seed-{client.id}-{suffix}"
        exists = db.query(AuditRun.id).filter(AuditRun.id == run_id).first()
        if exists:
            continue
        run = AuditRun(
            id=run_id,
            client_id=client.id,
            status=status,
            started_at=now + offset if status != "pending" else None,
            completed_at=now if status == "completed" else None,
            config={
                "client": {"id": client.id, "name": client.name},
                "platforms": ["openai", "claude"],
            },
            processed_questions=25 if status != "pending" else 0,
            total_questions=40,
            progress_data={
                "processed": 25 if status != "pending" else 0,
                "total": 40,
                "updated_at": now.isoformat(),
            },
        )
        db.add(run)
        created += 1

    logger.info(
        "admin.seed_demo_data",
        client_id=client.id,
        runs_created=created,
    )
    return {"seeded": True, "runsCreated": created}


def _wipe_demo_data(db: Session, client: Client) -> Dict[str, Any]:
    runs_deleted = db.query(AuditRun).filter(AuditRun.client_id == client.id).delete()
    personas_deleted = (
        db.query(Persona).filter(Persona.client_id == client.id).delete()
        if hasattr(Persona, "client_id")
        else 0
    )
    logger.info(
        "admin.wipe_demo_data",
        client_id=client.id,
        runs_deleted=runs_deleted,
        personas_deleted=personas_deleted,
    )
    return {
        "wiped": True,
        "runsDeleted": runs_deleted,
        "personasDeleted": personas_deleted,
    }
