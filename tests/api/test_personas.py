import sys
import types

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


def test_persona_modes_endpoint():
    response = client.get("/api/v1/personas/modes")
    assert response.status_code == 200
    data = response.json()
    assert "modes" in data
    assert "b2c" in data["modes"]


def test_persona_catalog_endpoint_default_mode():
    response = client.get("/api/v1/personas/catalog")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "b2c"
    assert "roles" in data


def test_persona_catalog_invalid_mode():
    response = client.get("/api/v1/personas/catalog", params={"mode": "invalid"})
    assert response.status_code == 400
