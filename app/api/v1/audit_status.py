"""
Enhanced API endpoints for comprehensive audit status monitoring.

This module provides detailed audit run status tracking, real-time progress updates,
platform monitoring, metrics, and debugging information.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun
from app.models.response import Response as ResponseModel
from app.services.platform_manager import PlatformManager
from app.services.progress_tracker import create_progress_tracker
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/audit-status", tags=["audit-status"])


# === Response Models ===


class PlatformStatusModel(BaseModel):
    """Model for platform status information"""

    name: str
    status: str  # "available", "unavailable", "rate_limited", "error"
    response_time_ms: Optional[int] = None
    rate_limit_remaining: Optional[int] = None
    last_error: Optional[str] = None
    last_check: Optional[datetime] = None


class ProgressStageModel(BaseModel):
    """Model for progress stage information"""

    stage: str
    status: str  # "pending", "in_progress", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class ProgressSnapshotModel(BaseModel):
    """Model for comprehensive progress snapshot"""

    current_stage: str
    overall_progress_percent: float
    questions_generated: int
    questions_processed: int
    total_questions: int
    platforms_active: int
    brands_detected: int
    estimated_completion: Optional[datetime] = None
    stages: List[ProgressStageModel]
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class AuditRunDetailModel(BaseModel):
    """Detailed audit run information"""

    id: str
    client_id: str
    client_name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_questions: Optional[int] = None
    processed_questions: Optional[int] = None
    progress_data: Optional[Dict[str, Any]] = None
    platform_stats: Optional[Dict[str, Any]] = None
    error_log: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Computed fields
    duration_minutes: Optional[float] = None
    average_response_time_ms: Optional[float] = None
    success_rate: Optional[float] = None


class SystemStatusModel(BaseModel):
    """Overall system status"""

    status: str  # "healthy", "degraded", "unavailable"
    timestamp: datetime
    platforms: List[PlatformStatusModel]
    active_audits: int
    pending_audits: int
    total_audits_today: int
    average_processing_time_minutes: Optional[float] = None
    system_load: Optional[float] = None


class MetricsModel(BaseModel):
    """Audit metrics and analytics"""

    total_audits_completed: int
    total_audits_failed: int
    average_audit_duration_minutes: float
    average_questions_per_audit: float
    platform_success_rates: Dict[str, float]
    brand_detection_accuracy: Optional[float] = None
    cost_analytics: Optional[Dict[str, Any]] = None


# === Dependencies ===


def get_db() -> Generator[Session, None, None]:
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_platform_manager() -> PlatformManager:
    """Platform manager dependency"""
    return PlatformManager()


# === API Endpoints ===


@router.get("/audit-runs/{audit_run_id}", response_model=AuditRunDetailModel)
async def get_audit_run_status(
    audit_run_id: str, db: Session = Depends(get_db)
) -> AuditRunDetailModel:
    """
    Get detailed status information for a specific audit run.

    Provides comprehensive information including progress, timing,
    platform statistics, and error details.
    """
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    # Get client information
    client_name = audit_run.client.name if audit_run.client else None

    # Calculate computed fields
    duration_minutes = None
    if audit_run.started_at and audit_run.completed_at:
        duration = audit_run.completed_at - audit_run.started_at
        duration_minutes = duration.total_seconds() / 60

    # Calculate average response time from responses
    responses = (
        db.query(ResponseModel).filter(ResponseModel.audit_run_id == audit_run_id).all()
    )

    average_response_time_ms = None
    success_rate = None

    if responses:
        # Calculate average response time
        valid_times = [r.processing_time_ms for r in responses if r.processing_time_ms]
        if valid_times:
            average_response_time_ms = sum(valid_times) / len(valid_times)

        # Calculate success rate (responses with valid content vs total questions)
        successful_responses = sum(
            1 for r in responses if r.response_text and len(r.response_text.strip()) > 0
        )
        success_rate = (successful_responses / len(responses)) * 100 if responses else 0

    return AuditRunDetailModel(
        id=audit_run.id,
        client_id=audit_run.client_id,
        client_name=client_name,
        status=audit_run.status,
        started_at=audit_run.started_at,
        completed_at=audit_run.completed_at,
        total_questions=audit_run.total_questions,
        processed_questions=audit_run.processed_questions,
        progress_data=audit_run.progress_data,
        platform_stats=audit_run.platform_stats,
        error_log=audit_run.error_log,
        created_at=audit_run.created_at,
        updated_at=audit_run.updated_at,
        duration_minutes=duration_minutes,
        average_response_time_ms=average_response_time_ms,
        success_rate=success_rate,
    )


@router.get("/audit-runs/{audit_run_id}/progress", response_model=ProgressSnapshotModel)
async def get_audit_progress(
    audit_run_id: str, db: Session = Depends(get_db)
) -> ProgressSnapshotModel:
    """
    Get real-time progress information for an active audit run.

    Provides detailed progress tracking including stage information,
    performance metrics, and estimated completion time.
    """
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    # Create progress tracker to get current status
    progress_tracker = create_progress_tracker(db, audit_run_id)

    try:
        # Get current progress snapshot
        snapshot = await progress_tracker.get_progress_snapshot()

        # Convert to API model
        stages = []
        for stage_name, stage_data in snapshot.stages.items():
            stage_model = ProgressStageModel(
                stage=stage_name,
                status=stage_data.get("status", "pending"),
                started_at=stage_data.get("started_at"),
                completed_at=stage_data.get("completed_at"),
                duration_seconds=stage_data.get("duration_seconds"),
                error_message=stage_data.get("error_message"),
            )
            stages.append(stage_model)

        return ProgressSnapshotModel(
            current_stage=snapshot.current_stage.value,
            overall_progress_percent=snapshot.overall_progress_percent,
            questions_generated=snapshot.questions_generated,
            questions_processed=snapshot.questions_processed,
            total_questions=snapshot.total_questions,
            platforms_active=snapshot.platforms_active,
            brands_detected=snapshot.brands_detected,
            estimated_completion=snapshot.estimated_completion,
            stages=stages,
            performance_metrics=snapshot.performance_metrics,
        )

    except Exception as e:
        logger.error(f"Failed to get progress for audit {audit_run_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve progress information"
        )


@router.get("/audit-runs", response_model=List[AuditRunDetailModel])
async def list_audit_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
) -> List[AuditRunDetailModel]:
    """
    List audit runs with optional filtering and pagination.

    Supports filtering by status and client, with pagination controls.
    """
    query = db.query(AuditRun)

    # Apply filters
    if status:
        query = query.filter(AuditRun.status == status)
    if client_id:
        query = query.filter(AuditRun.client_id == client_id)

    # Apply pagination and ordering
    audit_runs = (
        query.order_by(desc(AuditRun.created_at)).offset(offset).limit(limit).all()
    )

    # Convert to detailed models
    results = []
    for audit_run in audit_runs:
        # Basic conversion - could be optimized with bulk queries
        client_name = audit_run.client.name if audit_run.client else None

        duration_minutes = None
        if audit_run.started_at and audit_run.completed_at:
            duration = audit_run.completed_at - audit_run.started_at
            duration_minutes = duration.total_seconds() / 60

        results.append(
            AuditRunDetailModel(
                id=audit_run.id,
                client_id=audit_run.client_id,
                client_name=client_name,
                status=audit_run.status,
                started_at=audit_run.started_at,
                completed_at=audit_run.completed_at,
                total_questions=audit_run.total_questions,
                processed_questions=audit_run.processed_questions,
                progress_data=audit_run.progress_data,
                platform_stats=audit_run.platform_stats,
                error_log=audit_run.error_log,
                created_at=audit_run.created_at,
                updated_at=audit_run.updated_at,
                duration_minutes=duration_minutes,
            )
        )

    return results


@router.get("/system", response_model=SystemStatusModel)
async def get_system_status(
    platform_manager: PlatformManager = Depends(get_platform_manager),
    db: Session = Depends(get_db),
) -> SystemStatusModel:
    """
    Get overall system status including platform health and audit statistics.

    Provides a comprehensive view of system health, active operations,
    and performance metrics.
    """
    # Check platform status
    platform_statuses = []
    overall_system_healthy = True

    for platform_name in platform_manager.get_available_platforms():
        try:
            platform = platform_manager.get_platform(platform_name)

            # Perform health check
            health_result = await platform.health_check()

            status = "available" if health_result else "unavailable"
            if not health_result:
                overall_system_healthy = False

            platform_status = PlatformStatusModel(
                name=platform_name, status=status, last_check=datetime.now(timezone.utc)
            )

        except Exception as e:
            overall_system_healthy = False
            platform_status = PlatformStatusModel(
                name=platform_name,
                status="error",
                last_error=str(e),
                last_check=datetime.now(timezone.utc),
            )

        platform_statuses.append(platform_status)

    # Get audit statistics
    today = datetime.now(timezone.utc).date()

    active_audits = (
        db.query(AuditRun).filter(AuditRun.status.in_(["pending", "running"])).count()
    )

    pending_audits = db.query(AuditRun).filter(AuditRun.status == "pending").count()

    total_audits_today = db.query(AuditRun).filter(AuditRun.created_at >= today).count()

    # Calculate average processing time
    completed_audits = (
        db.query(AuditRun)
        .filter(
            AuditRun.status == "completed",
            AuditRun.started_at.isnot(None),
            AuditRun.completed_at.isnot(None),
        )
        .limit(100)
        .all()
    )

    average_processing_time_minutes = None
    if completed_audits:
        total_time = sum(
            (audit.completed_at - audit.started_at).total_seconds()
            for audit in completed_audits
        )
        average_processing_time_minutes = (total_time / len(completed_audits)) / 60

    # Determine overall system status
    if not overall_system_healthy:
        system_status = "degraded"
    elif active_audits > 10:  # Configurable threshold
        system_status = "degraded"
    else:
        system_status = "healthy"

    return SystemStatusModel(
        status=system_status,
        timestamp=datetime.now(timezone.utc),
        platforms=platform_statuses,
        active_audits=active_audits,
        pending_audits=pending_audits,
        total_audits_today=total_audits_today,
        average_processing_time_minutes=average_processing_time_minutes,
    )


def _calculate_average_duration(audits: List[AuditRun]) -> float:
    """Calculate average duration from completed audits."""
    if not audits:
        return 0.0

    valid_durations = [
        (audit.completed_at - audit.started_at).total_seconds() / 60
        for audit in audits
        if audit.started_at and audit.completed_at
    ]
    return sum(valid_durations) / len(valid_durations) if valid_durations else 0.0


def _calculate_average_questions(audits: List[AuditRun]) -> float:
    """Calculate average questions per audit."""
    if not audits:
        return 0.0

    valid_counts = [
        audit.total_questions for audit in audits if audit.total_questions
    ]
    return sum(valid_counts) / len(valid_counts) if valid_counts else 0.0


def _calculate_platform_success_rates(audits: List[AuditRun]) -> Dict[str, float]:
    """Calculate platform success rates from audit statistics."""
    platform_stats = {}

    for audit in audits:
        if audit.platform_stats:
            for platform, stats in audit.platform_stats.items():
                if platform not in platform_stats:
                    platform_stats[platform] = {"total": 0, "successful": 0}

                platform_stats[platform]["total"] += stats.get("total_requests", 0)
                platform_stats[platform]["successful"] += stats.get(
                    "successful_requests", 0
                )

    return {
        platform: (stats["successful"] / stats["total"]) * 100
        for platform, stats in platform_stats.items()
        if stats["total"] > 0
    }


@router.get("/metrics", response_model=MetricsModel)
async def get_audit_metrics(
    days: int = Query(
        7, ge=1, le=365, description="Number of days to include in metrics"
    ),
    db: Session = Depends(get_db),
) -> MetricsModel:
    """
    Get comprehensive audit metrics and analytics.

    Provides performance metrics, success rates, and cost analytics
    for the specified time period.
    """
    from datetime import timedelta

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Get audit statistics
    completed_audits = (
        db.query(AuditRun)
        .filter(AuditRun.status == "completed", AuditRun.created_at >= cutoff_date)
        .all()
    )

    failed_audits = (
        db.query(AuditRun)
        .filter(AuditRun.status == "failed", AuditRun.created_at >= cutoff_date)
        .count()
    )

    return MetricsModel(
        total_audits_completed=len(completed_audits),
        total_audits_failed=failed_audits,
        average_audit_duration_minutes=_calculate_average_duration(completed_audits),
        average_questions_per_audit=_calculate_average_questions(completed_audits),
        platform_success_rates=_calculate_platform_success_rates(completed_audits),
    )


@router.get("/health")
async def health_check(
    response: Response,
    platform_manager: PlatformManager = Depends(get_platform_manager),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Comprehensive health check for the audit system.

    Checks database connectivity, platform availability, and system resources.
    Sets appropriate HTTP status codes based on health.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
    }

    overall_healthy = True

    # Check database connectivity
    try:
        db.execute("SELECT 1")
        health_status["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        overall_healthy = False
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Check platform availability
    platform_health = {}
    for platform_name in platform_manager.get_available_platforms():
        try:
            platform = platform_manager.get_platform(platform_name)
            is_healthy = await platform.health_check()

            platform_health[platform_name] = {
                "status": "healthy" if is_healthy else "unhealthy"
            }

            if not is_healthy:
                overall_healthy = False

        except Exception as e:
            overall_healthy = False
            platform_health[platform_name] = {"status": "unhealthy", "error": str(e)}

    health_status["components"]["platforms"] = platform_health

    # Set overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        response.status_code = 503

    return health_status
