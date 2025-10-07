from __future__ import annotations

from datetime import datetime

from sqlalchemy.engine import Engine

from app.db.base_class import Base
from app.models.audit import AuditRun, Client
from app.models.report import Report
from app.services.dashboard.report_service import list_reports


def _ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def _drop_schema(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)


def test_list_reports_excludes_internal(db_engine: Engine, db_session) -> None:
    _ensure_schema(db_engine)
    try:
        internal = Client(id="client-internal", name="Internal", is_internal=True)
        external = Client(id="client-external", name="External", is_internal=False)
        run_internal = AuditRun(
            id="run-internal", client=internal, status="completed", config={}
        )
        run_external = AuditRun(
            id="run-external", client=external, status="completed", config={}
        )
        report_internal = Report(
            id="report-internal",
            audit_run=run_internal,
            generated_at=datetime.utcnow(),
            file_path="internal.pdf",
        )
        report_external = Report(
            id="report-external",
            audit_run=run_external,
            generated_at=datetime.utcnow(),
            file_path="external.pdf",
        )
        db_session.add_all(
            [
                internal,
                external,
                run_internal,
                run_external,
                report_internal,
                report_external,
            ]
        )
        db_session.commit()

        all_reports = list_reports(db_session)
        assert {report.id for report in all_reports} == {
            "report-internal",
            "report-external",
        }

        filtered = list_reports(db_session, exclude_internal=True)
        assert {report.id for report in filtered} == {"report-external"}
    finally:
        _drop_schema(db_engine)
