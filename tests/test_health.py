from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test that the health endpoint returns 200 OK with correct JSON response"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_content_type() -> None:
    """Test that the health endpoint returns JSON content type"""
    response = client.get("/health")

    assert response.headers["content-type"] == "application/json"
