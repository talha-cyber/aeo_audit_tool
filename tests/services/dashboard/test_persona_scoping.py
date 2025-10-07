from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import Mock

try:
    import pydantic  # type: ignore  # noqa: F401
except ImportError:
    pydantic_module = types.ModuleType("pydantic")

    def _field(default=None, **_: object) -> object:
        return default

    def _identity_decorator(*_args: object, **_kwargs: object):
        def _decorator(func):
            return func

        return _decorator

    class _BaseModel:  # minimal stub
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self, *args, **kwargs):  # noqa: D401 - stub dump
            return self.__dict__.copy()

    pydantic_module.BaseModel = _BaseModel
    pydantic_module.Field = _field
    pydantic_module.field_validator = _identity_decorator
    pydantic_module.model_validator = _identity_decorator
    pydantic_module.BeforeValidator = lambda fn: fn
    pydantic_module.PostgresDsn = str
    pydantic_module.ValidationError = Exception
    pydantic_module.ConfigDict = dict
    sys.modules["pydantic"] = pydantic_module

try:
    import pydantic_settings  # type: ignore  # noqa: F401
except ImportError:
    settings_module = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    settings_module.BaseSettings = _BaseSettings
    settings_module.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = settings_module

try:
    import pydantic_core  # type: ignore  # noqa: F401
except ImportError:
    core_module = types.ModuleType("pydantic_core")
    sys.modules["pydantic_core"] = core_module

if "pydantic_core.core_schema" not in sys.modules:
    core_schema_module = types.ModuleType("pydantic_core.core_schema")

    class _ValidationInfo:  # noqa: D401 - minimal stub for ValidationInfo
        """Placeholder ValidationInfo used in config settings."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    core_schema_module.ValidationInfo = _ValidationInfo
    sys.modules["pydantic_core.core_schema"] = core_schema_module


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    info = warning = error = critical = debug


def _noop_processor(*args, **kwargs):
    return lambda *_a, **_k: None


try:
    import structlog  # type: ignore  # noqa: F401
except ImportError:
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

try:
    import prometheus_client  # type: ignore  # noqa: F401
except ImportError:
    prometheus_module = types.ModuleType("prometheus_client")

    class _Metric:
        def __init__(self, *args, **kwargs):  # noqa: D401
            self._args = args
            self._kwargs = kwargs

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

    prometheus_module.Counter = _Metric
    prometheus_module.Gauge = _Metric
    prometheus_module.Histogram = _Metric
    sys.modules["prometheus_client"] = prometheus_module

try:
    import vaderSentiment  # type: ignore  # noqa: F401
except ImportError:
    vader_module = types.ModuleType("vaderSentiment")
    sys.modules["vaderSentiment"] = vader_module

if "vaderSentiment.vaderSentiment" not in sys.modules:
    vader_submodule = types.ModuleType("vaderSentiment.vaderSentiment")

    class _SentimentIntensityAnalyzer:
        def polarity_scores(self, *_args, **_kwargs):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    vader_submodule.SentimentIntensityAnalyzer = _SentimentIntensityAnalyzer
    sys.modules["vaderSentiment.vaderSentiment"] = vader_submodule
try:
    import numpy  # type: ignore  # noqa: F401
except ImportError:
    numpy_module = types.ModuleType("numpy")

    def _as_array(data, dtype=None):  # noqa: D401 - simple stub
        return data

    def _dot(a, b):  # noqa: D401 - simple stub
        return 0.0

    numpy_module.array = _as_array
    numpy_module.dot = _dot
    numpy_module.var = lambda data: 0.0
    numpy_module.mean = lambda data, axis=None: 0.0
    numpy_module.std = lambda data, axis=None: 0.0
    numpy_module.zeros = lambda shape, dtype=None: [0.0] * (
        shape[0] if isinstance(shape, tuple) else shape
    )
    numpy_module.float32 = float
    numpy_module.float64 = float
    numpy_module.linalg = types.SimpleNamespace(norm=lambda x: 1.0)
    sys.modules["numpy"] = numpy_module


try:
    import openai  # type: ignore  # noqa: F401
except ImportError:
    openai_module = types.ModuleType("openai")

    class _EmbeddingsClient:
        async def create(self, input, model=None):  # noqa: D401 - stub async client
            class _Response:
                data = [
                    types.SimpleNamespace(embedding=[0.0] * len(str(item)))
                    for item in input
                ]

            return _Response()

    class _AsyncOpenAI:
        def __init__(self, api_key=None):  # noqa: D401 - stub initializer
            self.embeddings = _EmbeddingsClient()

    openai_module.AsyncOpenAI = _AsyncOpenAI
    openai_module.APIError = Exception
    openai_module.APITimeoutError = Exception
    openai_module.RateLimitError = Exception

    openai_types_module = types.ModuleType("openai.types")
    openai_types_chat_module = types.ModuleType("openai.types.chat")

    class _ChatCompletion:  # noqa: D401 - stub chat completion container
        def __init__(self, choices=None):
            self.choices = choices or []

    openai_types_chat_module.ChatCompletion = _ChatCompletion
    openai_types_module.chat = openai_types_chat_module

    openai_module.types = openai_types_module
    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = openai_types_module
    sys.modules["openai.types.chat"] = openai_types_chat_module

try:
    import cachetools  # type: ignore  # noqa: F401
except ImportError:
    cachetools_module = types.ModuleType("cachetools")

    class _TTLCache(dict):
        def __init__(self, maxsize=0, ttl=0):  # noqa: D401 - stub cache
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl

    cachetools_module.TTLCache = _TTLCache
    sys.modules["cachetools"] = cachetools_module


try:
    import rapidfuzz  # type: ignore  # noqa: F401
except ImportError:
    rapidfuzz_module = types.ModuleType("rapidfuzz")
    fuzz_module = types.ModuleType("rapidfuzz.fuzz")

    def _full_ratio(*args, **kwargs):  # noqa: D401 - stub ratio
        return 100

    fuzz_module.ratio = _full_ratio
    fuzz_module.partial_ratio = _full_ratio
    fuzz_module.token_sort_ratio = _full_ratio
    fuzz_module.token_set_ratio = _full_ratio

    rapidfuzz_module.fuzz = fuzz_module
    sys.modules["rapidfuzz"] = rapidfuzz_module
    sys.modules["rapidfuzz.fuzz"] = fuzz_module


try:
    import httpx  # type: ignore  # noqa: F401
except ImportError:
    httpx_module = types.ModuleType("httpx")

    class _AsyncClient:
        async def __aenter__(self):  # noqa: D401 - stub context enter
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: D401 - stub exit
            return False

        async def post(self, *args, **kwargs):  # noqa: D401 - stub post
            return types.SimpleNamespace(status_code=200, json=lambda: {})

        async def get(self, *args, **kwargs):  # noqa: D401 - stub get
            return types.SimpleNamespace(status_code=200, json=lambda: {})

    httpx_module.AsyncClient = _AsyncClient
    httpx_module.TimeoutException = Exception
    httpx_module.HTTPError = Exception
    sys.modules["httpx"] = httpx_module


try:
    import redis  # type: ignore  # noqa: F401
except ImportError:
    redis_module = types.ModuleType("redis")
    sys.modules["redis"] = redis_module

if "redis.asyncio" not in sys.modules:
    redis_async_module = types.ModuleType("redis.asyncio")

    class _RedisClient:
        async def script_load(self, *_args, **_kwargs):  # noqa: D401 - stub loader
            return "sha"

        async def evalsha(self, *_args, **_kwargs):  # noqa: D401 - stub eval
            return (1, 0)

    def _from_url(*_args, **_kwargs):  # noqa: D401 - stub factory
        return _RedisClient()

    redis_async_module.Redis = _RedisClient
    redis_async_module.RedisError = Exception
    redis_async_module.from_url = _from_url
    sys.modules["redis.asyncio"] = redis_async_module
    redis_module.asyncio = redis_async_module


try:
    import tenacity  # type: ignore  # noqa: F401
except ImportError:
    tenacity_module = types.ModuleType("tenacity")

    def _identity_decorator_factory(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    tenacity_module.retry = _identity_decorator_factory
    tenacity_module.wait_exponential = lambda *args, **kwargs: None
    tenacity_module.stop_after_attempt = lambda *args, **kwargs: None
    tenacity_module.retry_if_exception_type = lambda *args, **kwargs: None
    tenacity_module.before_sleep_log = lambda *args, **kwargs: None
    tenacity_module.RetryError = Exception
    sys.modules["tenacity"] = tenacity_module


try:
    import jinja2  # type: ignore  # noqa: F401
except ImportError:
    jinja2_module = types.ModuleType("jinja2")

    class _Environment:  # noqa: D401 - stub template env
        def __init__(self, *args, **kwargs):
            pass

        def get_template(self, *_args, **_kwargs):
            class _Template:
                def render(self, **_render_kwargs):
                    return ""

            return _Template()

    class _FileSystemLoader:  # noqa: D401 - stub loader
        def __init__(self, *args, **kwargs):
            pass

    def _select_autoescape(*_args, **_kwargs):
        return lambda text: text

    jinja2_module.Environment = _Environment
    jinja2_module.FileSystemLoader = _FileSystemLoader
    jinja2_module.select_autoescape = _select_autoescape
    sys.modules["jinja2"] = jinja2_module


try:
    import jsonschema  # type: ignore  # noqa: F401
except ImportError:
    jsonschema_module = types.ModuleType("jsonschema")

    class _Draft7Validator:  # noqa: D401 - stub validator
        def __init__(self, schema=None):
            self.schema = schema or {}

        def validate(self, _instance):
            return True

    jsonschema_module.Draft7Validator = _Draft7Validator
    sys.modules["jsonschema"] = jsonschema_module


try:
    import fastapi  # type: ignore  # noqa: F401
except ImportError:
    fastapi_module = types.ModuleType("fastapi")

    class _HTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Request:  # noqa: D401 - stub request
        def __init__(self, headers=None):
            self.headers = headers or {}

    fastapi_module.HTTPException = _HTTPException
    fastapi_module.Request = _Request
    sys.modules["fastapi"] = fastapi_module

if "fastapi.responses" not in sys.modules:
    fastapi_responses_module = types.ModuleType("fastapi.responses")

    class _JSONResponse:  # noqa: D401 - stub JSON response
        def __init__(self, content=None, status_code: int = 200):
            self.content = content
            self.status_code = status_code

    fastapi_responses_module.JSONResponse = _JSONResponse
    sys.modules["fastapi.responses"] = fastapi_responses_module

try:
    import starlette  # type: ignore  # noqa: F401
except ImportError:
    starlette_module = types.ModuleType("starlette")
    sys.modules["starlette"] = starlette_module
if "starlette.middleware" not in sys.modules:
    starlette_middleware_module = types.ModuleType("starlette.middleware")
    sys.modules["starlette.middleware"] = starlette_middleware_module
if "starlette.middleware.base" not in sys.modules:
    starlette_middleware_base_module = types.ModuleType("starlette.middleware.base")

    class _BaseHTTPMiddleware:  # noqa: D401 - stub middleware
        def __init__(self, app, dispatch=None):
            self.app = app
            self.dispatch = dispatch

    starlette_middleware_base_module.BaseHTTPMiddleware = _BaseHTTPMiddleware
    sys.modules["starlette.middleware.base"] = starlette_middleware_base_module

from sqlalchemy.engine import Engine

from app.api.v1.dashboard_schemas import CreateAuditRunRequest
from app.db.base_class import Base
from app.models.audit import AuditRun, Client
from app.models.question import Question
from app.services.audit_processor import AuditProcessor
from app.services.dashboard.audit_run_creation_service import create_audit_run
from app.services.dashboard.persona_store import PersonaLibraryStore


def _ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def _drop_schema(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)


def test_persona_library_scoped_by_client(db_engine: Engine, db_session) -> None:
    _ensure_schema(db_engine)
    try:
        internal_client = Client(id="client-internal", name="Internal Preview")
        external_client = Client(id="client-external", name="External Tenant")
        db_session.add_all([internal_client, external_client])
        db_session.commit()

        store = PersonaLibraryStore(db_session)

        internal_record = store.new_record(
            owner_id="admin",
            client_id=internal_client.id,
            mode="b2c",
            name="Ops Admin",
            segment="Enterprise",
            priority="primary",
            key_need="Need clarity",
            journey_stage=[],
            role="operations_lead",
            driver="efficiency",
            voice=None,
            contexts=["awareness"],
            meta={"source": "test"},
        )
        store.save(internal_record)

        external_record = store.new_record(
            owner_id="admin",
            client_id=external_client.id,
            mode="b2c",
            name="Marketing Lead",
            segment="SMB",
            priority="secondary",
            key_need="Need guidance",
            journey_stage=[],
            role="marketing_lead",
            driver="growth",
            voice=None,
            contexts=["consideration"],
            meta={"source": "test"},
        )
        store.save(external_record)

        scoped_personas = store.list(
            owner_id="admin",
            client_id=internal_client.id,
            mode="b2c",
        )

        assert len(scoped_personas) == 1
        persona = scoped_personas[0]
        assert persona.client_id == internal_client.id
        assert persona.name == "Ops Admin"

        # Ensure other owners do not see personas from a different client
        other_owner_personas = store.list(
            owner_id="other",
            client_id=internal_client.id,
            mode="b2c",
        )
        assert other_owner_personas == []
    finally:
        _drop_schema(db_engine)


def test_persist_questions_captures_persona_metadata(
    db_engine: Engine, db_session
) -> None:
    _ensure_schema(db_engine)
    try:
        client = Client(id="client-meta", name="Metadata Client")
        run = AuditRun(
            id="run-meta",
            client=client,
            config={"client": {"id": client.id, "name": client.name}},
            status="pending",
        )
        db_session.add_all([client, run])
        db_session.commit()

        processor = AuditProcessor(db_session, platform_manager=Mock())

        persona_payload = {
            "id": "persona-123",
            "name": "Ops Admin",
            "role": "operations_lead",
            "driver": "efficiency",
            "contexts": ["awareness", "consideration"],
            "mode": "b2c",
        }

        asyncio.run(
            processor._persist_questions(  # type: ignore[attr-defined]
                run.id,
                [
                    {
                        "question": "How do I roll out AI assistants?",
                        "category": "persona",
                        "question_type": "persona_specific",
                        "priority_score": 0.8,
                        "target_brand": client.name,
                        "provider": "qe_v2",
                        "metadata": {"persona": persona_payload},
                    }
                ],
            )
        )

        stored_questions = db_session.query(Question).all()
        assert stored_questions
        question = stored_questions[0]
        assert question.persona == persona_payload["id"]
        assert question.role == persona_payload["role"]
        assert question.driver == persona_payload["driver"]
        assert question.context_stage == persona_payload["contexts"][0]
        assert question.question_metadata["persona"]["name"] == persona_payload["name"]
    finally:
        _drop_schema(db_engine)


def test_create_audit_run_injects_persona_context(
    db_engine: Engine, db_session
) -> None:
    _ensure_schema(db_engine)
    try:
        client = Client(id="client-run", name="Persona Client", is_internal=False)
        db_session.add(client)
        db_session.commit()

        store = PersonaLibraryStore(db_session)
        persona = store.new_record(
            owner_id="admin@example.com",
            client_id=client.id,
            mode="b2c",
            name="Discovery Analyst",
            segment="Enterprise",
            priority="primary",
            key_need="Spot opportunities",
            journey_stage=[{"stage": "Awareness", "question": "?", "coverage": 1.0}],
            role="analyst",
            driver="growth",
            voice="insight_voice",
            contexts=["awareness"],
            meta={"source": "seed", "clientId": client.id},
        )
        store.save(persona)

        payload = CreateAuditRunRequest(
            name="Discovery Program",
            personaIds=[str(persona.id)],
            platforms=["openai"],
            questionCount=12,
            clientId=client.id,
        )

        response = create_audit_run(
            db_session,
            payload,
            owner_id="admin@example.com",
            act_as_client_id=client.id,
        )

        created = db_session.get(AuditRun, response.run.id)
        assert created is not None
        config = created.config or {}
        assert config.get("persona_mode") == "b2c"
        personas = config.get("personas") or []
        assert personas, "Personas missing from config"
        entry = personas[0]
        assert entry["id"] == str(persona.id)
        assert entry["context_keys"] == ["awareness"]
        assert entry["mode"] == "b2c"
    finally:
        _drop_schema(db_engine)
