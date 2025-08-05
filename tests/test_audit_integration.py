import asyncio
import time
import uuid
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.audits import get_db
from app.db.base import Base
from app.main import app
from app.models.audit import AuditRun, Client
from app.tasks import audit_tasks

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.drop_all(bind=engine)
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


@pytest.fixture
def mock_question_engine():
    with patch("app.tasks.audit_tasks.QuestionEngine") as mock:
        mock_instance = mock.return_value
        mock_instance.generate_questions = MagicMock(return_value=asyncio.Future())
        mock_instance.generate_questions.return_value.set_result(
            [
                MagicMock(
                    id=uuid.uuid4().hex,
                    question_text="What is the best CRM?",
                    model_dump=lambda: {
                        "id": uuid.uuid4().hex,
                        "question_text": "What is the best CRM?",
                    },
                ),
            ]
        )
        yield mock_instance


@pytest.mark.integration
def test_audit_run_integration(mock_question_engine) -> None:
    # Step 1: Create a client
    db = TestingSessionLocal()
    test_client = Client(
        id=uuid.uuid4().hex,
        name="Test Client",
        industry="SaaS",
        product_type="CRM",
        competitors=["CompA", "CompB"],
    )
    db.add(test_client)
    db.commit()

    # Step 2: Trigger an audit run
    response = client.post(f"/api/v1/audits/configs/{test_client.id}/run")
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Step 3: Poll for status until completed or failed
    timeout = 60  # seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_response = client.get(f"/api/v1/audits/runs/{run_id}/status")
        assert status_response.status_code == 200
        status = status_response.json()["status"]
        if status in ["completed", "failed"]:
            break
        time.sleep(1)

    # Step 4: Verify the response in the database
    from app.models.question import Question
    from app.models.response import Response

    audit_run = db.query(AuditRun).filter_by(id=run_id).first()
    if audit_run.status == "failed":
        pytest.fail(f"Audit failed with error: {audit_run.error_log}")

    # Verify that the QuestionEngine was called with the correct arguments
    mock_question_engine.generate_questions.assert_called_once()

    db_question = db.query(Question).first()
    assert db_question is not None

    db_response = db.query(Response).filter_by(audit_run_id=run_id).first()
    assert db_response is not None
    assert db_response.platform == "openai"
    db.close()
