"""Persona-aware question engine v2 package."""

from app.services.question_engine_v2.engine import QuestionEngineV2
from app.services.question_engine_v2.evaluator.answer_eval import AnswerSatisfactionEvaluator
from app.services.question_engine_v2.providers.dynamic_provider import DynamicProviderV2
from app.services.question_engine_v2.providers.template_provider import TemplateProviderV2
from app.services.question_engine_v2.router import LLMRouter


def build_default_engine(
    *, enable_dynamic: bool = False, include_evaluator: bool = True
) -> QuestionEngineV2:
    """Construct a default v2 question engine instance."""

    providers = [TemplateProviderV2()]
    router: LLMRouter | None = None
    if enable_dynamic or include_evaluator:
        router = LLMRouter()
    if enable_dynamic:
        providers.append(DynamicProviderV2(router=router))

    evaluator = AnswerSatisfactionEvaluator(router=router) if include_evaluator else None

    engine = QuestionEngineV2(providers=providers, evaluator=evaluator)
    return engine


__all__ = ["QuestionEngineV2", "build_default_engine"]
