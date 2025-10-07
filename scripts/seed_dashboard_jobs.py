#!/usr/bin/env python3
"""Seed representative dashboard audit programs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun, Client
from app.models.scheduling import (
    JobType,
    ScheduledJob,
    ScheduledJobStatus,
    TriggerType,
)
from app.services.dashboard.static_data import default_audit_programs
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _create_job_from_config(db: Session, config: dict) -> ScheduledJob:
    name = config.get("name", "Audit Program")
    owner = (config.get("payload") or {}).get("owner", {})
    trigger_config = config.get("trigger_config") or {}

    job = ScheduledJob(
        name=name,
        description=config.get("description"),
        job_type=JobType.AUDIT,
        trigger_type=TriggerType(trigger_config.get("trigger_type", "cron")),
        trigger_config=trigger_config,
        job_config={"payload": config.get("payload", {})},
        status=ScheduledJobStatus.ACTIVE,
        priority=config.get("priority", 5),
        created_by=owner.get("name"),
        client_id=(config.get("payload") or {}).get("client_id"),
    )

    now = datetime.now(timezone.utc)
    job.created_at = now
    job.updated_at = now
    return job


def _seed_internal_preview_runs(db: Session) -> None:
    client = db.query(Client).filter(Client.is_internal.is_(True)).one_or_none()
    if client is None:
        logger.info("Internal preview client not found; skipping run seeding")
        return

    existing_ids = {
        row.id
        for row in db.query(AuditRun.id).filter(AuditRun.client_id == client.id).all()
    }

    now = datetime.now(timezone.utc)
    run_payloads = [
        {
            "id": "internal-preview-completed",
            "status": "completed",
            "started_at": now - timedelta(days=3),
            "completed_at": now - timedelta(days=3, hours=-2),
            "config": {
                "client": {"id": client.id, "name": client.name},
                "platforms": ["openai", "claude"],
                "persona_ids": [],
            },
            "processed_questions": 42,
            "total_questions": 42,
        },
        {
            "id": "internal-preview-running",
            "status": "running",
            "started_at": now - timedelta(hours=1),
            "completed_at": None,
            "config": {
                "client": {"id": client.id, "name": client.name},
                "platforms": ["openai"],
                "persona_ids": [],
            },
            "processed_questions": 18,
            "total_questions": 40,
        },
        {
            "id": "internal-preview-pending",
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "config": {
                "client": {"id": client.id, "name": client.name},
                "platforms": ["claude"],
                "persona_ids": [],
            },
            "processed_questions": 0,
            "total_questions": 30,
        },
    ]

    created = 0
    for payload in run_payloads:
        if payload["id"] in existing_ids:
            continue
        run = AuditRun(
            id=payload["id"],
            client_id=client.id,
            status=payload["status"],
            started_at=payload["started_at"],
            completed_at=payload["completed_at"],
            config=payload["config"],
            processed_questions=payload["processed_questions"],
            total_questions=payload["total_questions"],
        )
        run.platform_stats = {"openai": {"completed": payload["processed_questions"]}}
        run.progress_data = {
            "updated_at": now.isoformat(),
            "processed": payload["processed_questions"],
            "total": payload["total_questions"],
        }
        db.add(run)
        created += 1

    if created:
        logger.info("Seeded %s internal preview audit runs", created)


def seed_dashboard_jobs() -> None:
    logger.info("Seeding dashboard audit programs")
    db = SessionLocal()
    try:
        existing_names = {
            row.name
            for row in db.query(ScheduledJob.name)
            .filter(ScheduledJob.job_type == JobType.AUDIT)
            .all()
        }

        created = 0
        for config in default_audit_programs():
            name = config.get("name")
            if name in existing_names:
                logger.info("Skipping existing audit program", name=name)
                continue

            job = _create_job_from_config(db, config)
            db.add(job)
            created += 1
            logger.info("Queued dashboard audit program", name=name)

        if created:
            db.commit()
            logger.info("Seeded %s dashboard audit programs", created)
        else:
            logger.info("No new dashboard audit programs were seeded")

        _seed_internal_preview_runs(db)
        db.commit()
    except Exception:
        logger.exception("Failed to seed dashboard audit programs")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_dashboard_jobs()
