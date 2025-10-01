#!/usr/bin/env python3
"""Seed representative dashboard audit programs."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
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
    except Exception:
        logger.exception("Failed to seed dashboard audit programs")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_dashboard_jobs()
