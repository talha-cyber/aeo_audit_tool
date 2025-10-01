"""Audit run projections for dashboard endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session, selectinload

from app.api.v1.dashboard_schemas import AuditIssueView, AuditRunProgressView, AuditRunView
from app.models.audit import AuditRun

__all__ = ["list_runs", "get_run", "map_audit_run_to_view"]


def list_runs(db: Session, *, limit: Optional[int] = None) -> List[AuditRunView]:
    """Return recent audit runs with progress data."""

    query = (
        db.query(AuditRun)
        .options(selectinload(AuditRun.client))
        .order_by(desc(AuditRun.started_at), desc(AuditRun.completed_at))
    )
    if limit:
        query = query.limit(limit)

    runs = query.all()
    return [map_audit_run_to_view(run) for run in runs]


def get_run(db: Session, run_id: str) -> AuditRunView:
    """Fetch a single audit run."""

    run = (
        db.query(AuditRun)
        .options(selectinload(AuditRun.client))
        .filter(AuditRun.id == run_id)
        .first()
    )
    if not run:
        raise ValueError(f"Audit run {run_id} not found")
    return map_audit_run_to_view(run)


def map_audit_run_to_view(run: AuditRun) -> AuditRunView:
    """Convert an AuditRun ORM object into the dashboard view model."""

    return AuditRunView(
        id=run.id,
        name=_resolve_run_name(run),
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        progress=_extract_progress(run),
        issues=_extract_issues(run),
    )


def _resolve_run_name(run: AuditRun) -> str:
    config = run.config or {}
    if isinstance(config, dict):
        if config.get("name"):
            return str(config["name"])
        client = config.get("client")
        if isinstance(client, dict) and client.get("name"):
            return f"{client['name']} audit"

    if run.client and run.client.name:
        return f"{run.client.name} audit"

    return f"Audit {run.id[:8]}"


def _extract_progress(run: AuditRun) -> AuditRunProgressView:
    done = run.processed_questions or 0
    total = run.total_questions or done
    updated_at = None

    progress_data = run.progress_data
    if isinstance(progress_data, dict):
        done = progress_data.get("questions_processed", done)
        total = progress_data.get("total_questions", total)
        updated_at = _parse_datetime(progress_data.get("updated_at") or progress_data.get("last_updated"))

    return AuditRunProgressView(done=done, total=total or 0, updated_at=updated_at)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _extract_issues(run: AuditRun) -> List[AuditIssueView]:
    issues: List[AuditIssueView] = []

    platform_stats = run.platform_stats
    if isinstance(platform_stats, dict):
        for platform, stats in platform_stats.items():
            if not isinstance(stats, dict):
                continue

            error_count = stats.get("error_count") or stats.get("errors") or 0
            if error_count:
                severity = "high" if error_count > 3 else "medium"
                issues.append(
                    AuditIssueView(
                        id=f"{platform}:errors",
                        label=f"{platform} reported {error_count} errors",
                        severity=severity,
                    )
                )

            status = stats.get("status")
            if status and status not in {"ok", "healthy", "available"}:
                issues.append(
                    AuditIssueView(
                        id=f"{platform}:status",
                        label=f"{platform} status {status}",
                        severity="medium",
                    )
                )

            last_error = stats.get("last_error")
            if last_error:
                issues.append(
                    AuditIssueView(
                        id=f"{platform}:last_error",
                        label=str(last_error),
                        severity="medium",
                    )
                )

    progress_data = run.progress_data
    if isinstance(progress_data, dict):
        for item in progress_data.get("issues", []) or []:
            if isinstance(item, dict) and item.get("label"):
                issues.append(
                    AuditIssueView(
                        id=item.get("id") or f"{run.id}:issue:{len(issues)}",
                        label=item.get("label"),
                        severity=item.get("severity") or "medium",
                    )
                )

    if run.error_log:
        issues.append(
            AuditIssueView(
                id=f"{run.id}:error_log",
                label=run.error_log[:180],
                severity="high",
            )
        )

    return issues
