"""Answer satisfaction evaluation for question engine v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.services.question_engine_v2.router import LLMRouter
from app.services.question_engine_v2.schemas import PersonaResolution
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_NAME = "satisfaction_prompt.j2"


class AnswerSatisfactionEvaluator:
    """Scores responses against persona emotional anchors."""

    def __init__(
        self,
        router: LLMRouter | None = None,
        *,
        model: Optional[str] = None,
        template_dir: Optional[Path] = None,
    ) -> None:
        self._router = router
        self._model = model
        prompts_dir = template_dir or Path(__file__).resolve().parent.parent / "prompts"
        self._env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape(enabled_extensions=(".j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template = self._env.get_template(PROMPT_NAME)
        logger.debug("AnswerSatisfactionEvaluator initialized", router=router)

    def set_router(self, router: LLMRouter) -> None:
        self._router = router

    async def evaluate(
        self,
        *,
        client_brand: str,
        question_text: str,
        response_text: str,
        persona: PersonaResolution | Dict[str, Any],
        provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate LLM response satisfaction for a persona."""
        if not self._router:
            raise RuntimeError(
                "AnswerSatisfactionEvaluator requires an LLMRouter instance"
            )

        persona_data = self._persona_fields(persona)
        prompt = self._template.render(
            client_brand=client_brand,
            question_text=question_text,
            response_text=response_text,
            **persona_data,
        )

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["satisfied", "partial", "unsatisfied"],
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "rationale": {"type": "string"},
            },
            "required": ["status", "score"],
            "additionalProperties": False,
        }

        result = await self._router.complete_json(
            prompt,
            schema,
            provider=provider,
            model=model_override or self._model,
            max_attempts=2,
        )
        logger.debug(
            "Satisfaction evaluation completed",
            status=result.get("status"),
            score=result.get("score"),
        )
        return result

    def _persona_fields(
        self, persona: PersonaResolution | Dict[str, Any]
    ) -> Dict[str, Any]:
        if isinstance(persona, PersonaResolution):
            data = persona.model_dump()
        else:
            data = dict(persona)
        context_stage = data.get("context_stage")
        contexts = data.get("contexts") or []
        if not context_stage and contexts:
            context_stage = contexts[0]
        return {
            "role": data.get("role", "persona"),
            "driver": data.get("driver", ""),
            "emotional_anchor": data.get("emotional_anchor", ""),
            "context_stage": context_stage or "evaluation",
        }


__all__ = ["AnswerSatisfactionEvaluator"]
