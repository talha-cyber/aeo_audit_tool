import asyncio
import sys
import types
import uuid

import pytest

from app.services.providers import Question
from app.services.question_engine_v2.engine import QuestionEngineV2
from app.services.question_engine_v2.evaluator.answer_eval import AnswerSatisfactionEvaluator
from app.services.question_engine_v2.providers.base import BaseProviderV2, ProviderExecutionContext
from app.services.question_engine_v2.schemas import (
    PersonaMode,
    PersonaRequest,
    PersonaResolution,
    PersonaSelection,
    ProviderConfig,
    QuestionEngineRequest,
    QuotaConfig,
    SeedMixConfig,
)


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


class _StubPersonaExtractor:
    def resolve_personas(self, mode, voices=None, selections=None):
        return [
            PersonaResolution(
                mode=PersonaMode.B2C,
                role="skeptic",
                driver="deal_hunter",
                contexts=["evaluation"],
                emotional_anchor="fear_of_overpaying",
                voice="value_shopper",
            )
        ]


class _StubProvider(BaseProviderV2):
    name = "stub_provider"
    provider_key = "dynamic"

    async def _generate(self, context: ProviderExecutionContext):
        question = Question(
            question_text="How does ClientX compare?",
            category="comparison",
            provider=self.name,
            seed_type="unseeded",
            role="skeptic",
            driver="deal_hunter",
            context_stage="evaluation",
        )
        question.persona = "value_shopper"
        dup_question = Question(
            question_text="How does ClientX compare?",
            category="comparison",
            provider=self.name,
            seed_type="competitor",
            role="skeptic",
            driver="deal_hunter",
            context_stage="evaluation",
        )
        dup_question.persona = "value_shopper"
        return [question, dup_question], {"stub": True}


class _StubRouter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def complete_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        return self.payload


@pytest.fixture()
def engine():
    provider = _StubProvider()
    persona_extractor = _StubPersonaExtractor()
    return QuestionEngineV2(providers=[provider], persona_extractor=persona_extractor)


@pytest.fixture()
def request_payload():
    return QuestionEngineRequest(
        client_brand="ClientX",
        competitors=["CompetitorY"],
        industry="SaaS",
        product_type="CRM",
        audit_run_id=uuid.uuid4(),
        personas=PersonaRequest(
            mode=PersonaMode.B2C,
            voices=["value_shopper"],
            overrides=[
                PersonaSelection(
                    role="skeptic",
                    driver="deal_hunter",
                    contexts=["evaluation"],
                )
            ],
        ),
        seed_mix=SeedMixConfig(unseeded=0.5, competitor=0.3, brand=0.2),
        quotas=QuotaConfig(total=1, per_persona_min=1),
        providers=ProviderConfig(),
    )


@pytest.mark.asyncio
async def test_engine_generate_applies_constraints_and_scoring(engine, request_payload):
    questions = await engine.generate(request_payload)
    assert len(questions) == 1
    question = questions[0]
    assert question.priority_score > 0
    assert question.persona == "value_shopper"


@pytest.fixture()
def persona():
    return PersonaResolution(
        mode=PersonaMode.B2C,
        role="skeptic",
        driver="deal_hunter",
        contexts=["evaluation"],
        emotional_anchor="fear_of_overpaying",
        voice="value_shopper",
    )


@pytest.mark.asyncio
async def test_engine_evaluate_response(persona):
    router = _StubRouter({"status": "satisfied", "score": 0.9})
    evaluator = AnswerSatisfactionEvaluator(router=router, model="openai:gpt-score")
    engine = QuestionEngineV2(providers=[], evaluator=evaluator)

    question = Question(
        question_text="Why choose ClientX?",
        category="comparison",
        provider="template",
        persona=persona.voice,
        role=persona.role,
        driver=persona.driver,
        context_stage="evaluation",
    )

    result = await engine.evaluate_response(
        client_brand="ClientX",
        question=question,
        response_text="Because it saves money",
        persona=persona,
    )

    assert result["status"] == "satisfied"
    assert result["score"] == 0.9
    assert router.calls
