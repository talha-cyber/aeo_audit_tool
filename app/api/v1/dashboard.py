"""Dashboard-facing API endpoints bridging backend data to the frontend."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import (
    AuditRunDetailView,
    AuditRunView,
    AuditSummaryView,
    ComparisonMatrixView,
    CreateAuditRunRequest,
    CreateAuditRunResponse,
    InsightView,
    LaunchTestRunRequest,
    LaunchTestRunResponse,
    PersonaCatalogView,
    PersonaCloneRequest,
    PersonaComposeRequest,
    PersonaLibraryEntryView,
    PersonaLibraryResponse,
    PersonaUpdateRequest,
    PersonaView,
    ReportSummaryView,
    SettingsView,
    WidgetView,
)
from app.api.v1.security import _authorize
from app.db.session import SessionLocal
from app.models.audit import Client
from app.services.dashboard.audit_run_creation_service import create_audit_run
from app.services.dashboard.audit_run_service import list_runs as list_run_views
from app.services.dashboard.audit_summary_service import list_audit_programs
from app.services.dashboard.comparison_service import (
    get_comparison_matrix as get_comparison_view,
)
from app.services.dashboard.insight_service import list_insights as list_insight_views
from app.services.dashboard.persona_service import (
    clone_persona,
    create_persona,
    delete_persona,
    export_persona_catalog,
    list_persona_library,
    update_persona,
)
from app.services.dashboard.persona_service import (
    list_personas as list_persona_views,
)
from app.services.dashboard.report_service import list_reports as list_report_views
from app.services.dashboard.run_detail_service import get_run_detail
from app.services.dashboard.settings_service import get_settings as get_settings_view
from app.services.dashboard.test_run_service import launch_simulated_run
from app.services.dashboard.widget_service import list_widgets as list_widget_views
from app.services.question_engine_v2.persona_extractor import PersonaCatalogError
from app.services.question_engine_v2.schemas import PersonaMode

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _resolve_client_context(
    user: dict,
    client_id: str | None = None,
    payload_client_id: str | None = None,
    db: Session | None = None,
) -> str:
    """Determine the active client context for dashboard operations."""
    act_as_client = user.get("act_as_client_id")

    requested = payload_client_id or client_id
    if act_as_client and requested and requested != act_as_client:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Act-as session cannot access a different client",
        )

    for candidate in (
        payload_client_id,
        client_id,
        act_as_client,
        user.get("client_id"),
        user.get("tenant_id"),
    ):
        if candidate:
            return str(candidate)

    if db is not None:
        internal = (
            db.query(Client)
            .filter(Client.is_internal.is_(True))
            .order_by(Client.name)
            .first()
        )
        if internal is None:
            internal = db.query(Client).order_by(Client.name).first()
        if internal is not None:
            return str(internal.id)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Client context required"
    )


@router.get("/audits", response_model=List[AuditSummaryView])
async def get_audit_programs(
    db: Session = Depends(get_db),
    limit: int | None = None,
    exclude_internal: bool = Query(default=False, alias="excludeInternal"),
) -> List[AuditSummaryView]:
    return list_audit_programs(db, limit=limit, exclude_internal=exclude_internal)


@router.get("/audits/runs", response_model=List[AuditRunView])
async def list_audit_runs(
    db: Session = Depends(get_db),
    limit: int | None = None,
    exclude_internal: bool = Query(default=False, alias="excludeInternal"),
) -> List[AuditRunView]:
    return list_run_views(db, limit=limit, exclude_internal=exclude_internal)


@router.get("/audits/run/{run_id}", response_model=AuditRunDetailView)
async def get_audit_run(
    run_id: str, db: Session = Depends(get_db)
) -> AuditRunDetailView:
    try:
        return get_run_detail(db, run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Audit run not found") from None


@router.post("/audits/test-run", response_model=LaunchTestRunResponse)
async def launch_test_run(payload: LaunchTestRunRequest) -> LaunchTestRunResponse:
    return launch_simulated_run(payload)


@router.post("/audits/run", response_model=CreateAuditRunResponse)
async def create_new_audit_run(
    payload: CreateAuditRunRequest,
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> CreateAuditRunResponse:
    """Create a new audit run with persona selection and platform configuration."""
    owner_id = user.get("sub")
    requested_client = payload.client_id
    payload.client_id = _resolve_client_context(
        user,
        client_id=requested_client,
        payload_client_id=requested_client,
        db=db,
    )
    try:
        return create_audit_run(
            db,
            payload,
            owner_id,
            act_as_client_id=user.get("act_as_client_id"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports", response_model=List[ReportSummaryView])
async def list_reports(
    db: Session = Depends(get_db),
    limit: int | None = None,
    exclude_internal: bool = Query(default=False, alias="excludeInternal"),
) -> List[ReportSummaryView]:
    return list_report_views(db, limit=limit, exclude_internal=exclude_internal)


@router.get("/insights", response_model=List[InsightView])
async def list_insights(limit: int | None = None) -> List[InsightView]:
    return list_insight_views(limit=limit)


@router.get("/personas", response_model=List[PersonaView])
async def list_personas(
    mode: PersonaMode = PersonaMode.B2C,
    client_id: str | None = Query(default=None, alias="clientId"),
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> List[PersonaView]:
    owner_id = user.get("sub")
    active_client_id = _resolve_client_context(user, client_id, db=db)
    return list_persona_views(
        db,
        mode,
        owner_id=owner_id,
        client_id=active_client_id,
    )


@router.get("/personas/catalog", response_model=PersonaCatalogView)
async def get_persona_catalog(
    mode: PersonaMode = PersonaMode.B2C,
) -> PersonaCatalogView:
    return export_persona_catalog(mode)


@router.get("/personas/library", response_model=PersonaLibraryResponse)
async def get_persona_library(
    mode: PersonaMode = PersonaMode.B2C,
    client_id: str | None = Query(default=None, alias="clientId"),
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> PersonaLibraryResponse:
    owner_id = user.get("sub")
    active_client_id = _resolve_client_context(user, client_id, db=db)
    return list_persona_library(db, owner_id, active_client_id, mode)


@router.post("/personas/custom", response_model=PersonaLibraryEntryView)
async def compose_persona(
    payload: PersonaComposeRequest,
    client_id: str | None = None,
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> PersonaLibraryEntryView:
    try:
        payload.owner_id = user.get("sub")
        payload.client_id = _resolve_client_context(
            user, client_id, payload.client_id, db=db
        )
        return create_persona(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PersonaCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/personas/{persona_id}", response_model=PersonaLibraryEntryView)
async def edit_persona(
    persona_id: str,
    payload: PersonaUpdateRequest,
    client_id: str | None = None,
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> PersonaLibraryEntryView:
    try:
        payload.owner_id = user.get("sub")
        payload.client_id = _resolve_client_context(
            user, client_id, payload.client_id, db=db
        )
        return update_persona(db, persona_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PersonaCatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/personas/{persona_id}/clone", response_model=PersonaLibraryEntryView)
async def clone_persona_endpoint(
    persona_id: str,
    payload: PersonaCloneRequest,
    client_id: str | None = Query(default=None, alias="clientId"),
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> PersonaLibraryEntryView:
    try:
        owner_id = user.get("sub")
        active_client_id = _resolve_client_context(
            user, client_id, payload.client_id, db=db
        )
        return clone_persona(
            db,
            persona_id,
            owner_id,
            active_client_id,
            name=payload.name,
        )
    except PersonaCatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/personas/{persona_id}", status_code=204)
async def delete_persona_endpoint(
    persona_id: str,
    client_id: str | None = Query(default=None, alias="clientId"),
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> Response:
    owner_id = user.get("sub")
    active_client_id = _resolve_client_context(user, client_id, db=db)
    removed = delete_persona(db, persona_id, owner_id, active_client_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Persona not found")
    return Response(status_code=204)


@router.get("/embeds/widgets", response_model=List[WidgetView])
async def list_widgets() -> List[WidgetView]:
    return list_widget_views()


@router.get("/comparisons/matrix", response_model=ComparisonMatrixView)
async def get_comparison_matrix(
    db: Session = Depends(get_db),
    exclude_internal: bool = Query(default=False, alias="excludeInternal"),
) -> ComparisonMatrixView:
    return get_comparison_view(db, exclude_internal=exclude_internal)


@router.get("/settings", response_model=SettingsView)
async def get_settings() -> SettingsView:
    return get_settings_view()
