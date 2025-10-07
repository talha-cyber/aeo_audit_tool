from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

for key in {
    "DATABASE_URL",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "GOOGLE_AI_API_KEY",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_SERVER",
    "POSTGRES_PORT",
    "REDIS_HOST",
    "REDIS_PORT",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "APP_NAME",
    "APP_ENV",
    "CORS_ALLOW_ORIGINS",
    "SECRET_KEY",
}:
    os.environ.pop(key, None)

from app.db.base import Base
from app.main import app
from app.models.audit import AuditRun, Client
from app.security.auth.jwt_handler import get_jwt_handler
from app.tasks.audit_tasks import run_audit_task

try:
    from app.api.v1 import admin as admin_routes
except ImportError:  # pragma: no cover
    admin_routes = None


def _make_token(*, subject: str, capabilities: list[str] | None = None) -> str:
    handler = get_jwt_handler()
    return handler.create_token(subject, claims={"capabilities": capabilities or []})


@pytest.fixture()
def in_memory_db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@contextmanager
def override_admin_db(session: Session):
    if admin_routes is None:
        raise RuntimeError("admin routes not available")

    def _get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[admin_routes.get_db] = _get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(admin_routes.get_db, None)


@pytest.fixture(autouse=True)
def reset_rate_state():
    if admin_routes is not None and hasattr(admin_routes, "_rate_state"):
        admin_routes._rate_state.clear()  # type: ignore[attr-defined]
    yield
    if admin_routes is not None and hasattr(admin_routes, "_rate_state"):
        admin_routes._rate_state.clear()  # type: ignore[attr-defined]


@pytest.fixture()
def client(in_memory_db: Session):
    with override_admin_db(in_memory_db):
        yield TestClient(app)


def test_admin_requires_capability(client: TestClient, in_memory_db: Session) -> None:
    token = _make_token(subject="user@example.com", capabilities=["support"])
    response = client.post(
        "/api/v1/admin/audits/run-1",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": "force_retry"},
    )
    assert response.status_code == 403


def test_force_retry_invokes_task(
    client: TestClient, in_memory_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    client_record = Client(id="client-admin", name="Admin Client", is_internal=False)
    audit_run = AuditRun(
        id="run-admin", client=client_record, status="failed", config={}
    )
    in_memory_db.add_all([client_record, audit_run])
    in_memory_db.commit()

    called = {}

    def fake_delay(run_id: str) -> None:
        called["run_id"] = run_id

    monkeypatch.setattr(run_audit_task, "delay", fake_delay)

    response = client.post(
        "/api/v1/admin/audits/run-admin",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "force_retry"},
    )
    assert response.status_code == 200
    assert called["run_id"] == "run-admin"


def test_set_qe_version_updates_admin_settings(
    client: TestClient, in_memory_db: Session
) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-qe", name="Tenant QE", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "set_qe_version", "value": "v2"},
    )
    assert response.status_code == 200
    in_memory_db.refresh(tenant)
    assert tenant.admin_settings.get("qe_version") == "v2"


def test_toggle_platform_updates_settings(
    client: TestClient, in_memory_db: Session
) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-platform", name="Tenant Platform", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "toggle_platform", "platform": "claude", "enabled": False},
    )
    assert response.status_code == 200
    in_memory_db.refresh(tenant)
    overrides = tenant.admin_settings.get("platform_overrides", {})
    assert overrides.get("claude") is False


def test_set_feature_flag_updates_settings(
    client: TestClient, in_memory_db: Session
) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-flag", name="Tenant Flag", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "set_feature_flag", "flag": "beta_qe", "enabled": True},
    )
    assert response.status_code == 200
    in_memory_db.refresh(tenant)
    flags = tenant.admin_settings.get("feature_flags", {})
    assert flags.get("beta_qe") is True


def test_seed_demo_data_creates_runs(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-seed", name="Tenant Seed", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "seed_demo_data"},
    )
    assert response.status_code == 200
    runs = in_memory_db.query(AuditRun).filter(AuditRun.client_id == tenant.id).all()
    assert len(runs) >= 3


def test_wipe_demo_data_removes_runs(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-wipe", name="Tenant Wipe", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "seed_demo_data"},
    )
    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "wipe_demo_data"},
    )
    assert response.status_code == 200
    remaining = (
        in_memory_db.query(AuditRun).filter(AuditRun.client_id == tenant.id).count()
    )
    assert remaining == 0


def test_export_debug_bundle(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-export", name="Tenant Export", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    response = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "export_debug_bundle"},
    )
    assert response.status_code == 200
    bundle = response.json()["details"].get("bundle")
    assert bundle["client"] == tenant.id
    assert "exported_at" in bundle


def test_admin_rate_limit(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-rate", name="Tenant Rate", is_internal=False)
    in_memory_db.add(tenant)
    in_memory_db.commit()

    for _ in range(20):
        resp = client.post(
            f"/api/v1/admin/tenants/{tenant.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"action": "set_qe_version", "value": "v2"},
        )
        assert resp.status_code == 200

    limited = client.post(
        f"/api/v1/admin/tenants/{tenant.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "set_qe_version", "value": "v2"},
    )
    assert limited.status_code == 429
