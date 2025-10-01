"""Audit program projections for dashboard consumption."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session, selectinload

from app.api.v1.dashboard_schemas import AuditOwnerView, AuditSummaryView
from app.models.scheduling import JobExecution, JobType, ScheduledJob
from app.services.dashboard.utils import cron_to_label, safe_ratio
from app.services.dashboard.static_data import default_audit_programs


def list_audit_programs(db: Session, *, limit: Optional[int] = None) -> List[AuditSummaryView]:
    """Return configured audit programs with computed health metadata."""

    query = (
        db.query(ScheduledJob)
        .options(
            selectinload(ScheduledJob.executions).selectinload(JobExecution.audit_run)
        )
        .filter(ScheduledJob.job_type.in_({JobType.AUDIT}))
        .order_by(desc(ScheduledJob.created_at))
    )

    if limit:
        query = query.limit(limit)

    jobs: Iterable[ScheduledJob] = query.all()

    summaries: List[AuditSummaryView] = []
    for job in jobs:
        if job.job_type != JobType.AUDIT:
            continue

        payload = job.job_data or {}

        owner_info = payload.get("owner") if isinstance(payload, dict) else None
        owner_view = AuditOwnerView(
            name=_coalesce_owner_name(owner_info, job.created_by),
            email=_coalesce_owner_email(owner_info),
        )

        cadence_label = _resolve_cadence(job, payload)
        platforms = _resolve_platforms(payload)

        last_run_at = job.last_run_at or _infer_last_run(job.executions)

        health_score = _calculate_health_score(job)

        summaries.append(
            AuditSummaryView(
                id=job.job_id,
                name=job.name,
                cadence=cadence_label,
                owner=owner_view,
                platforms=platforms,
                last_run=last_run_at,
                health_score=health_score,
            )
        )

    if summaries:
        return summaries

    return hydrate_from_static()


def _coalesce_owner_name(owner_payload: Optional[dict], fallback: Optional[str]) -> Optional[str]:
    if isinstance(owner_payload, dict):
        name = owner_payload.get("name") or owner_payload.get("displayName")
        if name:
            return name
    return fallback


def _coalesce_owner_email(owner_payload: Optional[dict]) -> Optional[str]:
    if isinstance(owner_payload, dict):
        email = owner_payload.get("email")
        if isinstance(email, str) and email:
            return email
    return None


def _resolve_cadence(job: ScheduledJob, payload: dict) -> str:
    cadence_info = payload.get("cadence") if isinstance(payload, dict) else None

    if isinstance(cadence_info, dict):
        label = cadence_info.get("label") or cadence_info.get("cron")
        if isinstance(label, str) and label.strip():
            return label
    elif isinstance(cadence_info, str) and cadence_info.strip():
        return cadence_info

    trigger_config = job.trigger_config or {}
    trigger_type = trigger_config.get("trigger_type")

    if trigger_type == "cron":
        return cron_to_label(trigger_config.get("expression", ""))
    if trigger_type == "interval":
        minutes = trigger_config.get("minutes")
        seconds = trigger_config.get("seconds")
        if isinstance(minutes, (int, float)):
            return f"Every {int(minutes)} minutes"
        if isinstance(seconds, (int, float)):
            return f"Every {int(seconds)} seconds"
        return "Interval schedule"
    if trigger_type == "date":
        when = trigger_config.get("run_at")
        if isinstance(when, str):
            return f"One-time on {when}"
        return "One-time"

    return "On demand"


def _resolve_platforms(payload: dict) -> List[str]:
    platforms = payload.get("platforms") if isinstance(payload, dict) else None
    if isinstance(platforms, list):
        return [str(p) for p in platforms if isinstance(p, str)]

    config_platforms = payload.get("config", {}).get("platforms") if isinstance(payload, dict) else None
    if isinstance(config_platforms, list):
        return [str(p) for p in config_platforms if isinstance(p, str)]

    return []


def _infer_last_run(executions: Iterable[JobExecution]) -> Optional[datetime]:
    latest: Optional[JobExecution] = None
    for execution in executions:
        candidate = execution.completed_at or execution.started_at or execution.scheduled_time
        if not candidate:
            continue
        if not latest:
            latest = execution
            continue
        latest_candidate = latest.completed_at or latest.started_at or latest.scheduled_time
        if latest_candidate is None or candidate > latest_candidate:
            latest = execution

    if latest:
        return latest.completed_at or latest.started_at or latest.scheduled_time
    return None


def _calculate_health_score(job: ScheduledJob) -> Optional[float]:
    if not job.run_count:
        return None

    success_rate = job.success_rate
    failure_ratio = safe_ratio(job.failure_count, job.run_count) or 0.0

    raw_score = success_rate - (failure_ratio * 0.5)
    bounded = max(0.0, min(1.0, raw_score))
    return round(bounded * 100, 2)


def hydrate_from_static() -> List[AuditSummaryView]:
    """Fallback when DB lacks configured audit programs."""

    summaries: List[AuditSummaryView] = []
    for config in default_audit_programs():
        payload = config.get("payload", {})
        owner = payload.get("owner", {})
        summaries.append(
            AuditSummaryView(
                id=config.get("name", "audit-program"),
                name=config.get("name", "Audit Program"),
                cadence=_resolve_cadence_stub(config, payload),
                owner=AuditOwnerView(
                    name=owner.get("name"),
                    email=owner.get("email"),
                ),
                platforms=_resolve_platforms(payload),
                last_run=None,
                health_score=None,
            )
        )
    return summaries


def _resolve_cadence_stub(config: dict, payload: dict) -> str:
    cadence = payload.get("cadence", {})
    if isinstance(cadence, dict) and cadence.get("label"):
        return cadence["label"]
    trigger = config.get("trigger_config", {})
    return cron_to_label(trigger.get("expression", ""))
