from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

# Ensure test runs with minimal environment to keep settings lightweight.
_ENV_KEYS_TO_CLEAR = {
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
}
for _key in _ENV_KEYS_TO_CLEAR:
    os.environ.pop(_key, None)

from app.db.base import Base
from app.main import app
from app.models.audit import AuditRun, Client
from app.security.auth.jwt_handler import get_jwt_handler
from app.tasks.audit_tasks import run_audit_task

try:
    from app.api.v1 import admin as admin_routes
    from app.api.v1 import security as security_routes
except ImportError:  # pragma: no cover
    security_routes = None
    admin_routes = None


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    info = warning = error = critical = debug


def _noop_processor(*args, **kwargs):
    return lambda *a, **k: None


if "structlog" not in sys.modules:
    structlog_module = types.ModuleType("structlog")
    structlog_module.get_logger = lambda *args, **kwargs: _DummyLogger()
    structlog_module.processors = types.SimpleNamespace(
        TimeStamper=lambda fmt=None: _noop_processor(),
        CallsiteParameterAdder=lambda *a, **k: _noop_processor(),
        CallsiteParameter=types.SimpleNamespace(FILENAME="filename", LINENO="lineno"),
    )
    structlog_module.stdlib = types.SimpleNamespace(
        add_logger_name=_noop_processor,
        add_log_level=_noop_processor,
        BoundLogger=object,
        wrap_logger=_noop_processor,
        LoggerFactory=lambda: object,
    )
    structlog_module.dev = types.SimpleNamespace(
        set_exc_info=_noop_processor,
        ConsoleRenderer=lambda **kwargs: _noop_processor(),
    )
    structlog_module.configure = lambda **kwargs: None
    structlog_module.configure_once = lambda **kwargs: None
    sys.modules["structlog"] = structlog_module

if "structlog.types" not in sys.modules:
    types_module = types.ModuleType("structlog.types")
    types_module.Processor = object
    sys.modules["structlog.types"] = types_module


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
def override_security_db(session: Session):
    if security_routes is None:
        raise RuntimeError("security routes module not available")

    def _get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[security_routes.get_db] = _get_db
    if admin_routes is not None:
        app.dependency_overrides[admin_routes.get_db] = _get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(security_routes.get_db, None)
        if admin_routes is not None:
            app.dependency_overrides.pop(admin_routes.get_db, None)


@pytest.fixture(autouse=True)
def _reset_rate_state():
    from app.api.v1 import security as security_routes

    if hasattr(security_routes, "_rate_state"):
        security_routes._rate_state.clear()  # type: ignore[attr-defined]
    yield
    if hasattr(security_routes, "_rate_state"):
        security_routes._rate_state.clear()  # type: ignore[attr-defined]


@pytest.fixture()
def client(in_memory_db: Session):
    with override_security_db(in_memory_db):
        yield TestClient(app)


def _make_token(*, subject: str, capabilities: list[str] | None = None) -> str:
    handler = get_jwt_handler()
    return handler.create_token(
        subject=subject,
        claims={"capabilities": capabilities or []},
    )


def test_admin_can_impersonate(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    internal_client = Client(
        id="internal-client",
        name="Preview Tenant",
        is_internal=True,
    )
    in_memory_db.add(internal_client)
    in_memory_db.commit()

    response = client.post(
        "/api/v1/security/impersonate",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"target_client_id": internal_client.id},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "token" in payload
    assert "expiresAt" in payload

    handler = get_jwt_handler()
    claims = handler.verify_token(payload["token"])
    assert claims.get("act_as", {}).get("client_id") == internal_client.id


def test_non_admin_cannot_impersonate(
    client: TestClient, in_memory_db: Session
) -> None:
    user_token = _make_token(subject="user@example.com", capabilities=["support"])
    preview_client = Client(id="client-x", name="Client X", is_internal=False)
    in_memory_db.add(preview_client)
    in_memory_db.commit()

    response = client.post(
        "/api/v1/security/impersonate",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"target_client_id": preview_client.id},
    )
    assert response.status_code == 403


def test_impersonation_rate_limit(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    preview_client = Client(id="client-rate", name="Client Rate", is_internal=True)
    in_memory_db.add(preview_client)
    in_memory_db.commit()

    for _ in range(5):
        resp = client.post(
            "/api/v1/security/impersonate",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"target_client_id": preview_client.id},
        )
        assert resp.status_code == 200

    sixth = client.post(
        "/api/v1/security/impersonate",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"target_client_id": preview_client.id},
    )
    assert sixth.status_code == 429


def test_admin_can_force_retry_audit(
    client: TestClient, in_memory_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    client_record = Client(id="client-retry", name="Retry Client", is_internal=False)
    audit_run = AuditRun(
        id="run-retry",
        client_id=client_record.id,
        status="failed",
        config={"client": {"id": client_record.id, "name": client_record.name}},
    )
    in_memory_db.add_all([client_record, audit_run])
    in_memory_db.commit()

    called = {}

    def _fake_delay(run_id: str) -> None:
        called["run_id"] = run_id

    monkeypatch.setattr(run_audit_task, "delay", _fake_delay)

    response = client.post(
        f"/api/v1/admin/audits/{audit_run.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "force_retry"},
    )

    assert response.status_code == 200
    assert called["run_id"] == audit_run.id


def test_admin_can_set_qe_version(client: TestClient, in_memory_db: Session) -> None:
    admin_token = _make_token(subject="admin@example.com", capabilities=["admin"])
    tenant = Client(id="tenant-1", name="Tenant", is_internal=False)
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
