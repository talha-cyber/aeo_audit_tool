from fastapi.testclient import TestClient

from app.main import app
from app.security.auth.jwt_handler import get_jwt_handler


def test_protected_route_requires_jwt():
    client = TestClient(app)
    # Missing token -> 401
    r = client.get("/api/v1/secure/ping")
    assert r.status_code == 401

    # Valid token -> 200
    token = get_jwt_handler().create_token("user123")
    r = client.get("/api/v1/secure/ping", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json().get("user") == "user123"


def test_security_headers_present():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    # Basic headers
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "no-referrer"
