import sys
import types
import uuid

import pytest


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

from app.services.question_engine_v2.providers.base import ProviderExecutionContext
from app.services.question_engine_v2.providers.dynamic_provider import DynamicProviderV2
from app.services.question_engine_v2.providers.template_provider import (
    TemplateProviderV2,
)
from app.services.question_engine_v2.schemas import (
    DynamicProviderOptions,
    PersonaMode,
    PersonaResolution,
    QuestionEngineRequest,
    QuotaConfig,
    SeedMixConfig,
    TemplateProviderOptions,
)


@pytest.fixture
def sample_request() -> QuestionEngineRequest:
    return QuestionEngineRequest(
        client_brand="ClientX",
        competitors=["CompetitorY"],
        industry="SaaS",
        product_type="CRM",
        audit_run_id=uuid.uuid4(),
        language="en",
    )


@pytest.fixture
def sample_persona() -> PersonaResolution:
    return PersonaResolution(
        mode=PersonaMode.B2C,
        role="skeptic",
        driver="deal_hunter",
        contexts=["evaluation"],
        emotional_anchor="fear_of_overpaying",
        voice="value_shopper",
    )


@pytest.mark.asyncio
async def test_template_provider_renders_catalog_templates(
    sample_request, sample_persona
):
    provider = TemplateProviderV2(options=TemplateProviderOptions(max_per_persona=5))
    context = ProviderExecutionContext(
        request=sample_request,
        personas=[sample_persona],
        persona_request=None,
        quotas=QuotaConfig(total=4, per_persona_min=1),
        seed_mix=None,
        provider_config=None,
        router=None,
        cache=None,
        extras={},
    )

    questions, metadata = await provider._generate(context)

    # Catalog currently yields one deterministic template per persona/competitor combo
    assert len(questions) == 1
    assert all(q.provider == provider.name for q in questions)
    assert questions[0].persona == sample_persona.voice
    assert metadata["persona_count"] == 1


@pytest.mark.asyncio
async def test_template_provider_respects_total_quota(sample_request, sample_persona):
    provider = TemplateProviderV2(options=TemplateProviderOptions(max_per_persona=1))
    context = ProviderExecutionContext(
        request=sample_request,
        personas=[sample_persona],
        persona_request=None,
        quotas=QuotaConfig(total=1, per_persona_min=1),
        seed_mix=None,
        provider_config=None,
        router=None,
        cache=None,
        extras={},
    )

    questions, _ = await provider._generate(context)
    assert len(questions) == 1


class _StubRouter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def complete_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        return self.payload


@pytest.mark.asyncio
async def test_dynamic_provider_invokes_router_with_schema(
    sample_request, sample_persona
):
    payload = {
        "questions": [
            {
                "text": "How does ClientX compare?",
                "category": "comparison",
                "seed_type": "unseeded",
                "persona": sample_persona.voice,
                "context_stage": "evaluation",
            }
        ]
    }
    router = _StubRouter(payload)
    provider = DynamicProviderV2(
        options=DynamicProviderOptions(model="openai:gpt-test"),
        router=router,
    )
    context = ProviderExecutionContext(
        request=sample_request,
        personas=[sample_persona],
        persona_request=None,
        quotas=QuotaConfig(total=3, per_persona_min=1),
        seed_mix=SeedMixConfig(unseeded=0.5, competitor=0.3, brand=0.2),
        provider_config=None,
        router=router,
        cache=None,
        extras={},
    )

    questions, metadata = await provider._generate(context)

    assert len(router.calls) == 1
    prompt, schema, call_kwargs = router.calls[0]
    assert "Seed mix" in prompt
    assert (
        schema["properties"]["questions"]["minItems"] == context.quotas.per_persona_min
    )
    assert len(questions) == 1
    assert questions[0].provider == provider.name
    assert metadata["count"] == 1


@pytest.mark.asyncio
async def test_dynamic_provider_requires_router(sample_request, sample_persona):
    provider = DynamicProviderV2(options=DynamicProviderOptions())
    context = ProviderExecutionContext(
        request=sample_request,
        personas=[sample_persona],
        persona_request=None,
        quotas=QuotaConfig(total=2, per_persona_min=1),
        seed_mix=None,
        provider_config=None,
        router=None,
        cache=None,
        extras={},
    )

    with pytest.raises(RuntimeError):
        await provider._generate(context)
