from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

# keep environment minimal
for key in (
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
):
    os.environ.pop(key, None)

from app.api.v1 import dashboard as dashboard_routes
from app.db.base import Base
from app.main import app
from app.models.audit import Client
from app.security.auth.jwt_handler import get_jwt_handler
from app.services.dashboard.persona_store import PersonaLibraryStore


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
def override_dashboard_db(session: Session):
    def _get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[dashboard_routes.get_db] = _get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(dashboard_routes.get_db, None)


@pytest.fixture()
def client(in_memory_db: Session):
    with override_dashboard_db(in_memory_db):
        yield TestClient(app)


def _seed_persona(
    store: PersonaLibraryStore, *, owner_id: str, client_id: str, name: str
) -> None:
    record = store.new_record(
        owner_id=owner_id,
        client_id=client_id,
        mode="b2c",
        name=name,
        segment="Enterprise",
        priority="primary",
        key_need="Need",
        journey_stage=[{"stage": "Awareness", "question": "?", "coverage": 1.0}],
        role="operations_lead",
        driver="efficiency",
        voice=None,
        contexts=["awareness"],
        meta={"source": "test", "clientId": client_id},
    )
    store.save(record)


def test_persona_library_scoped_by_act_as(
    client: TestClient, in_memory_db: Session
) -> None:
    admin_id = "admin@example.com"
    primary_client = Client(
        id="client-primary", name="Primary Client", is_internal=False
    )
    preview_client = Client(
        id="client-preview", name="Preview Client", is_internal=True
    )
    in_memory_db.add_all([primary_client, preview_client])
    in_memory_db.commit()

    store = PersonaLibraryStore(in_memory_db)
    _seed_persona(
        store, owner_id=admin_id, client_id=primary_client.id, name="Primary Persona"
    )
    _seed_persona(
        store, owner_id=admin_id, client_id=preview_client.id, name="Preview Persona"
    )

    handler = get_jwt_handler()
    impersonation_token = handler.mint_act_as_token(
        subject=admin_id,
        actor_id=admin_id,
        target_client_id=preview_client.id,
        ttl_seconds=900,
    )

    response = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c"},
        headers={"Authorization": f"Bearer {impersonation_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    names = {persona["name"] for persona in data["personas"]}
    assert names == {"Preview Persona"}

    other_token = handler.mint_act_as_token(
        subject=admin_id,
        actor_id=admin_id,
        target_client_id=primary_client.id,
        ttl_seconds=900,
    )
    other_response = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c"},
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert other_response.status_code == 200
    other_names = {persona["name"] for persona in other_response.json()["personas"]}
    assert other_names == {"Primary Persona"}


def test_persona_library_rejects_conflicting_client_context(
    client: TestClient, in_memory_db: Session
) -> None:
    admin_id = "admin@example.com"
    client_record = Client(id="client-a", name="Client A", is_internal=False)
    in_memory_db.add(client_record)
    in_memory_db.commit()

    handler = get_jwt_handler()
    act_as_token = handler.mint_act_as_token(
        subject=admin_id,
        actor_id=admin_id,
        target_client_id=client_record.id,
        ttl_seconds=900,
    )

    response = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c", "clientId": "other-client"},
        headers={"Authorization": f"Bearer {act_as_token}"},
    )
    assert response.status_code == 403
