from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.models.audit import Client
from app.models.scheduling import JobType, ScheduledJob
from app.services.dashboard.audit_summary_service import list_audit_programs


def _setup_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal(), engine


def test_list_audit_programs_can_exclude_internal_clients() -> None:
    session, engine = _setup_db()
    try:
        external_client = Client(id="client-public", name="Public", is_internal=False)
        internal_client = Client(
            id="client-internal", name="Internal", is_internal=True
        )
        session.add_all([external_client, internal_client])
        session.commit()

        job_external = ScheduledJob(
            id="job-public",
            name="Public Audit",
            job_type=JobType.AUDIT,
            job_config={"client_id": external_client.id},
            trigger_type="cron",
            trigger_config={"expression": "0 0 * * *"},
            client_id=external_client.id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        job_internal = ScheduledJob(
            id="job-internal",
            name="Internal Audit",
            job_type=JobType.AUDIT,
            job_config={"client_id": internal_client.id},
            trigger_type="cron",
            trigger_config={"expression": "0 0 * * *"},
            client_id=internal_client.id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add_all([job_external, job_internal])
        session.commit()

        all_programs = list_audit_programs(session)
        assert {program.id for program in all_programs} == {
            "job-public",
            "job-internal",
        }

        filtered_programs = list_audit_programs(session, exclude_internal=True)
        ids = {program.id for program in filtered_programs}
        assert ids == {"job-public"}
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
