from __future__ import annotations

from sqlalchemy.engine import Engine

from app.db.base_class import Base
from app.models.audit import AuditRun, Client
from app.services.dashboard.audit_run_service import list_runs


def _ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def _drop_schema(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)


def test_list_runs_excludes_internal(db_engine: Engine, db_session) -> None:
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
        db_session.add_all([internal, external, run_internal, run_external])
        db_session.commit()

        all_runs = list_runs(db_session)
        all_ids = {run.id for run in all_runs}
        assert {"run-internal", "run-external"} == all_ids

        filtered = list_runs(db_session, exclude_internal=True)
        filtered_ids = {run.id for run in filtered}
        assert filtered_ids == {"run-external"}
    finally:
        _drop_schema(db_engine)
