"""Orchestrator for the persona-aware question engine v2."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from app.services import metrics
from app.services.providers import Question
from app.services.question_engine_v2.constraints import enforce_constraints
from app.services.question_engine_v2.evaluator.answer_eval import (
    AnswerSatisfactionEvaluator,
)
from app.services.question_engine_v2.persona_extractor import PersonaExtractor
from app.services.question_engine_v2.providers.base import (
    BaseProviderV2,
    ProviderExecutionContext,
)
from app.services.question_engine_v2.schemas import (
    PersonaRequest,
    PersonaResolution,
    ProviderConfig,
    QuestionEngineRequest,
)
from app.services.question_engine_v2.scoring import score_questions
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionEngineV2:
    """Entry point for the new persona- and intent-aware question engine."""

    def __init__(
        self,
        providers: Iterable[BaseProviderV2] | None = None,
        persona_extractor: PersonaExtractor | None = None,
        evaluator: AnswerSatisfactionEvaluator | None = None,
    ) -> None:
        self._providers: List[BaseProviderV2] = list(providers or [])
        self._persona_extractor = persona_extractor or PersonaExtractor()
        self._evaluator = evaluator
        if not self._providers:
            logger.debug(
                "QuestionEngineV2 initialized without providers; awaiting wiring"
            )
        else:
            logger.info(
                "QuestionEngineV2 initialized",
                provider_count=len(self._providers),
                providers=[
                    getattr(provider, "name", provider.__class__.__name__)
                    for provider in self._providers
                ],
            )

    async def generate(self, request: QuestionEngineRequest) -> List[Question]:
        """Generate questions for the provided request."""
        personas = self._resolve_personas(request.personas)
        context = ProviderExecutionContext(
            request=request,
            personas=personas,
            persona_request=request.personas,
            quotas=request.quotas,
            seed_mix=request.seed_mix,
            provider_config=request.providers,
            router=None,
            cache=None,
            extras={},
        )

        enabled_providers = self._select_providers(context, request.providers)
        logger.info(
            "QE v2 providers selected",
            providers=[p.name for p in enabled_providers],
            persona_count=len(personas),
        )

        results = await asyncio.gather(
            *[provider._generate(context) for provider in enabled_providers]
        )

        questions = [q for q_list, _ in results for q in q_list]
        questions = enforce_constraints(
            questions,
            quotas=request.quotas,
            seed_mix=request.seed_mix,
        )
        questions = score_questions(questions)
        self._record_metrics(request, questions)
        return questions

    async def evaluate_response(
        self,
        *,
        client_brand: str,
        question: Question,
        response_text: str,
        persona: PersonaResolution | Dict[str, Any],
        provider: str | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        """Evaluate satisfaction for a single response."""
        if not self._evaluator:
            logger.info("Satisfaction evaluator disabled; skipping evaluation")
            return {}

        result = await self._evaluator.evaluate(
            client_brand=client_brand,
            question_text=question.question_text,
            response_text=response_text,
            persona=persona,
            provider=provider,
            model_override=model_override,
        )
        if isinstance(persona, PersonaResolution):
            driver = persona.driver
        else:
            driver = persona.get("driver")
        if driver:
            metrics.QE_V2_SATISFACTION_SCORE.labels(driver=driver).set(
                result.get("score", 0.0)
            )
            metrics.QE_V2_SATISFACTION_STATUS_TOTAL.labels(
                driver=driver,
                status=result.get("status", "unknown"),
            ).inc()
        return result

    def _resolve_personas(
        self, persona_request: PersonaRequest | None
    ) -> List[PersonaExtractor.PERSONA_CLASS]:
        if persona_request is None:
            logger.warning(
                "No persona request provided; falling back to empty personas"
            )
            return []
        voices: Sequence[str] | None = persona_request.voices
        overrides = [selection.model_dump() for selection in persona_request.overrides]
        personas = self._persona_extractor.resolve_personas(
            mode=persona_request.mode,
            voices=voices,
            selections=overrides,
        )
        logger.debug("Resolved personas", count=len(personas))
        return personas

    def _select_providers(
        self,
        context: ProviderExecutionContext,
        provider_config: ProviderConfig | None,
    ) -> List[BaseProviderV2]:
        providers = []
        for provider in self._providers:
            if provider_config:
                if (
                    provider.provider_key == "template"
                    and not provider_config.template.enabled
                ):
                    continue
                if (
                    provider.provider_key == "dynamic"
                    and not provider_config.dynamic.enabled
                ):
                    continue
            if provider.can_handle(context):
                providers.append(provider)
        return providers

    def _record_metrics(
        self, request: QuestionEngineRequest, questions: List[Question]
    ) -> None:
        run_id = str(request.audit_run_id)
        seed_counts: Counter[str] = Counter()
        for question in questions:
            provider = question.provider or "unknown"
            role = question.role or "unknown"
            driver = question.driver or "unknown"
            context_stage = question.context_stage or "unknown"
            seed_type = (question.seed_type or "unseeded").lower()

            metrics.QE_V2_QUESTIONS_GENERATED_TOTAL.labels(
                provider=provider,
                role=role,
                driver=driver,
                context_stage=context_stage,
                seed_type=seed_type,
            ).inc()
            seed_counts[seed_type] += 1

        total = len(questions) or 1
        for seed, count in seed_counts.items():
            metrics.QE_V2_SEED_MIX_RATIO.labels(
                run_id=run_id, seed_type=seed
            ).set(count / total)
