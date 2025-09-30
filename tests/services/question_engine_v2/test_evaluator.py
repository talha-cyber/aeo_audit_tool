import sys
import types

import pytest

from app.services.question_engine_v2.evaluator.answer_eval import AnswerSatisfactionEvaluator
from app.services.question_engine_v2.schemas import PersonaMode, PersonaResolution


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


class _StubRouter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        return self.response


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
async def test_evaluator_calls_router_with_schema(persona):
    router = _StubRouter({"status": "satisfied", "score": 0.9, "rationale": "Addresses ROI"})
    evaluator = AnswerSatisfactionEvaluator(router=router, model="openai:gpt-score")

    result = await evaluator.evaluate(
        client_brand="ClientX",
        question_text="Why choose ClientX?",
        response_text="Because it offers the best ROI",
        persona=persona,
    )

    assert result["status"] == "satisfied"
    assert result["score"] == 0.9
    assert router.calls
    prompt, schema, kwargs = router.calls[0]
    assert "fear_of_overpaying" in prompt
    assert schema["properties"]["status"]["enum"] == ["satisfied", "partial", "unsatisfied"]
    assert kwargs["model"] == "openai:gpt-score"


@pytest.mark.asyncio
async def test_evaluator_requires_router(persona):
    evaluator = AnswerSatisfactionEvaluator(router=None)
    with pytest.raises(RuntimeError):
        await evaluator.evaluate(
            client_brand="ClientX",
            question_text="Why choose ClientX?",
            response_text="Because it offers the best ROI",
            persona=persona,
        )
