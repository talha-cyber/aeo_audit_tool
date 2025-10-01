"""Dynamic LLM-backed question provider for engine v2."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.providers import Question
from app.services.question_engine_v2.providers.base import (
    BaseProviderV2,
    ProviderExecutionContext,
)
from app.services.question_engine_v2.router import LLMRouter
from app.services.question_engine_v2.schemas import DynamicProviderOptions
from app.utils.logger import get_logger


class DynamicProviderV2(BaseProviderV2):
    """Routes LLM requests via provider-agnostic router."""

    name = "dynamic_provider_v2"
    provider_key = "dynamic"

    def __init__(
        self, options: DynamicProviderOptions | None = None, router: Any | None = None
    ) -> None:
        super().__init__()
        self.options = options or DynamicProviderOptions()
        self.logger = get_logger(__name__)
        self.router: Optional[LLMRouter] = None
        if router is not None:
            self.with_router(router)

    def update_options(self, options: DynamicProviderOptions) -> None:
        """Apply runtime configuration supplied by the engine."""
        self.options = options

    async def _generate(
        self, context: ProviderExecutionContext
    ) -> tuple[List[Question], Dict[str, Any]]:
        if not self.router and context.router:
            self.router = context.router
        if not self.router:
            raise RuntimeError("DynamicProviderV2 requires an LLMRouter instance")

        prompt = self._build_prompt(context)
        schema = self._build_schema(context)

        try:
            payload = await self.router.complete_json(
                prompt,
                schema,
                provider=self.options.provider,
                model=self.options.model,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
        except Exception as exc:
            self.logger.error(
                "Dynamic provider failed",
                audit_run=str(context.request.audit_run_id),
                error=str(exc),
            )
            return [], {"error": str(exc)}

        questions = self._parse_questions(payload, context)
        metadata = {
            "provider": self.provider_key,
            "model": self.options.model,
            "provider_key": self.options.provider,
            "count": len(questions),
        }
        return questions, metadata

    def _build_prompt(self, context: ProviderExecutionContext) -> str:
        personas = ", ".join(
            f"{persona.voice or persona.role}:{'/'.join(persona.contexts)}"
            for persona in context.personas
        )
        seed_mix = context.seed_mix.model_dump() if context.seed_mix else {}
        quota_total = context.quotas.total if context.quotas else None
        competitors = ", ".join(context.request.competitors)
        return (
            "You are generating audit questions for a competitive intelligence audit.\n"
            f"Client brand: {context.request.client_brand}.\n"
            f"Industry: {context.request.industry}. Product: {context.request.product_type}.\n"
            f"Personas (role:contexts): {personas}.\n"
            f"Competitors: {competitors}.\n"
            f"Seed mix targets: {seed_mix}. Total quota: {quota_total}.\n"
            "Questions must be diverse across personas, contexts, and seed types."
            " Return JSON matching the schema exactly."
        )

    def _build_schema(self, context: ProviderExecutionContext) -> Dict[str, Any]:
        quota_total = context.quotas.total if context.quotas else None
        min_items = context.quotas.per_persona_min if context.quotas else None
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": min_items or 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "category": {"type": "string"},
                            "seed_type": {"type": "string"},
                            "persona": {"type": "string"},
                            "driver": {"type": "string"},
                            "context_stage": {"type": "string"},
                        },
                        "required": ["text", "category", "persona"],
                    },
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        }

    def _parse_questions(
        self, payload: Dict[str, Any], context: ProviderExecutionContext
    ) -> List[Question]:
        questions: List[Question] = []
        for item in payload.get("questions", []):
            text = item.get("text")
            if not text:
                continue
            question = Question(
                question_text=text,
                category=item.get("category", "dynamic"),
                provider=self.name,
                metadata=self._build_question_metadata(item.get("metadata", {})),
            )
            question.seed_type = item.get("seed_type")
            question.persona = item.get("persona")
            questions.append(question)
        return questions
