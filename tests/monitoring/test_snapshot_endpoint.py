from fastapi.testclient import TestClient

from app.main import app


def test_monitoring_snapshot_endpoint():
    client = TestClient(app)
    r = client.get("/api/v1/monitoring/snapshot")
    assert r.status_code == 200
    data = r.json()
    assert "dlq" in data
    assert "process" in data
    assert "cpu_ratio" in data["process"]
