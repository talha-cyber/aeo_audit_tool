from fastapi.testclient import TestClient

from app.main import app


def test_correlation_id_header_present():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") is not None
