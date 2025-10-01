import sys
import types
import uuid

from fastapi.testclient import TestClient

from app.main import app


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


client = TestClient(app)


def test_dashboard_persona_catalog_default_mode() -> None:
    response = client.get("/api/v1/dashboard/personas/catalog")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "b2c"
    assert isinstance(data["voices"], list)
    assert any(voice["key"] == "value_shopper" for voice in data["voices"])  # ensures preset surface


def test_dashboard_personas_catalog_mode_switch() -> None:
    response = client.get("/api/v1/dashboard/personas/catalog", params={"mode": "b2b"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "b2b"
    assert data["voices"]


def test_dashboard_personas_list_returns_resolved_personas() -> None:
    owner_id = f"test-owner-{uuid.uuid4().hex}"
    response = client.get(
        "/api/v1/dashboard/personas",
        params={"mode": "b2c", "ownerId": owner_id},
    )
    assert response.status_code == 200
    personas = response.json()
    assert isinstance(personas, list)
    assert personas
    assert all("journeyStage" in persona for persona in personas)


def test_dashboard_persona_custom_from_voice() -> None:
    owner_id = f"test-owner-{uuid.uuid4().hex}"
    payload = {"mode": "b2c", "ownerId": owner_id, "voice": "value_shopper"}
    response = client.post("/api/v1/dashboard/personas/custom", json=payload)
    assert response.status_code == 200
    persona = response.json()
    assert persona["priority"] == "secondary"
    assert persona["journeyStage"]
    stages = {stage["stage"] for stage in persona["journeyStage"]}
    assert "Evaluation" in stages or "Purchase" in stages
    assert persona["ownerId"] == owner_id

    library_response = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c", "ownerId": owner_id},
    )
    assert library_response.status_code == 200
    data = library_response.json()
    assert len(data["personas"]) == 1


def test_dashboard_persona_custom_rejects_incomplete_payload() -> None:
    payload = {"mode": "b2c", "contexts": []}
    response = client.post("/api/v1/dashboard/personas/custom", json=payload)
    assert response.status_code == 400


def test_dashboard_persona_update_clone_and_delete_cycle() -> None:
    owner_id = f"test-owner-{uuid.uuid4().hex}"

    create_payload = {
        "mode": "b2c",
        "ownerId": owner_id,
        "voice": "value_shopper",
        "name": "Value Shopper",
        "priority": "primary",
    }
    create_response = client.post("/api/v1/dashboard/personas/custom", json=create_payload)
    assert create_response.status_code == 200
    created = create_response.json()
    persona_id = created["id"]

    update_payload = {
        "mode": "b2c",
        "ownerId": owner_id,
        "name": "Updated Shopper",
        "contexts": ["purchase"],
    }
    update_response = client.patch(
        f"/api/v1/dashboard/personas/{persona_id}", json=update_payload
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["name"] == "Updated Shopper"
    assert len(updated["contextKeys"]) == 1

    clone_response = client.post(
        f"/api/v1/dashboard/personas/{persona_id}/clone",
        json={"ownerId": owner_id, "mode": "b2c", "name": "Clone Persona"},
    )
    assert clone_response.status_code == 200
    clone = clone_response.json()
    assert clone["name"] == "Clone Persona"
    assert clone["id"] != persona_id

    library_response = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c", "ownerId": owner_id},
    )
    assert library_response.status_code == 200
    assert len(library_response.json()["personas"]) == 2

    delete_response = client.delete(
        f"/api/v1/dashboard/personas/{persona_id}", params={"ownerId": owner_id}
    )
    assert delete_response.status_code == 204

    final_library = client.get(
        "/api/v1/dashboard/personas/library",
        params={"mode": "b2c", "ownerId": owner_id},
    )
    assert final_library.status_code == 200
    assert len(final_library.json()["personas"]) == 1
