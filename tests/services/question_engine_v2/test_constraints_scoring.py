import sys
import types

import pytest

from app.services.providers import Question
from app.services.question_engine_v2.constraints import enforce_constraints
from app.services.question_engine_v2.schemas import QuotaConfig, SeedMixConfig
from app.services.question_engine_v2.scoring import score_questions


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


@pytest.fixture
def sample_questions():
    return [
        Question(
            question_text="How does brand compare?",
            category="comparison",
            provider="dynamic",
            seed_type="unseeded",
            role="skeptic",
            driver="deal_hunter",
            context_stage="evaluation",
        ),
        Question(
            question_text="How does brand compare?",
            category="comparison",
            provider="dynamic",
            seed_type="unseeded",
            role="skeptic",
            driver="deal_hunter",
            context_stage="evaluation",
        ),
        Question(
            question_text="What is competitor pricing for ClientX?",
            category="pricing",
            provider="template",
            seed_type="competitor",
            role="cfo",
            driver="roi_tco",
            context_stage="validation",
        ),
        Question(
            question_text="Why choose ClientX for loyalty programs?",
            category="features",
            provider="template",
            seed_type="brand",
            role="advocate",
            driver="values_aligned",
            context_stage="loyalty",
        ),
    ]


def test_enforce_constraints_applies_dedupe_and_seed_mix(sample_questions):
    quotas = QuotaConfig(total=3, per_persona_min=1)
    seed_mix = SeedMixConfig(unseeded=0.4, competitor=0.3, brand=0.3)

    constrained = enforce_constraints(
        sample_questions, quotas=quotas, seed_mix=seed_mix
    )

    assert len(constrained) == 3
    texts = {q.question_text for q in constrained}
    assert len(texts) == 3  # dedupe applied
    assert any(q.seed_type == "competitor" for q in constrained)
    assert any(q.seed_type == "brand" for q in constrained)


def test_score_questions_applies_weights(sample_questions):
    scored = score_questions(sample_questions)

    first = scored[0]
    assert first.priority_score > 0
    brand_question = next(q for q in scored if q.seed_type == "brand")
    assert brand_question.priority_score < first.priority_score

    dynamic_scores = [q.priority_score for q in scored if q.provider == "dynamic"]
    template_scores = [q.priority_score for q in scored if q.provider == "template"]
    assert dynamic_scores and template_scores
    assert max(dynamic_scores) >= max(template_scores)
