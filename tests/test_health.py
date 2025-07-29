# tests/test_health.py
from typing import Generator

import pytest
from fastapi.testclient import TestClient  # Use FastAPI's TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Reusable TestClient for the module."""
    with TestClient(app) as c:
        yield c


def test_health_status(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_health_content_type(client: TestClient) -> None:
    res = client.get("/health")
    assert res.headers["content-type"].startswith("application/json")
