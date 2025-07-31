import time
import uuid
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.audits import get_db
from app.db.base import Base
from app.main import app
from app.tasks import audit_tasks

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db() -> Generator:
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Also override the SessionLocal used by the Celery task

audit_tasks.SessionLocal = TestingSessionLocal

client = TestClient(app)


@pytest.mark.integration
def test_audit_run_integration() -> None:
    # Step 1: Trigger an audit run
    config_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/audits/configs/{config_id}/run")
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Step 2: Poll for status until completed or failed
    timeout = 60  # seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_response = client.get(f"/api/v1/audits/runs/{run_id}/status")
        assert status_response.status_code == 200
        status = status_response.json()["status"]
        if status in ["completed", "failed"]:
            break
        time.sleep(1)

    # Step 3: Verify the response in the database
    db = TestingSessionLocal()
    from app.models.audit import AuditRun
    from app.models.response import Response

    audit_run = db.query(AuditRun).filter_by(id=run_id).first()
    if audit_run.status == "failed":
        pytest.fail(f"Audit failed with error: {audit_run.error_log}")

    db_response = db.query(Response).filter_by(audit_run_id=run_id).first()
    assert db_response is not None
    assert db_response.platform == "openai"
    db.close()
