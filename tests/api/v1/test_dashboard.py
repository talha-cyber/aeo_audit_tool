"""API surface tests for the dashboard routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dashboard_test_run_endpoint_returns_mock_run() -> None:
    payload = {
        "scenarioId": "saas_weekly",
        "questionCount": 12,
        "platforms": ["chatgpt", "claude"],
    }

    response = client.post("/api/v1/dashboard/audits/test-run", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "run" in data
    run = data["run"]

    assert run["status"] == "running"
    assert run["progress"]["done"] == 0
    assert run["progress"]["total"] == payload["questionCount"]
    assert isinstance(run.get("startedAt"), str)
    assert run["issues"], "expected issue explaining mock providers"
