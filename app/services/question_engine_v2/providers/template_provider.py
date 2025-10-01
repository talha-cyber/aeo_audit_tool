"""Template-based question provider for engine v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.services.providers import Question
from app.services.question_engine_v2.providers.base import (
    BaseProviderV2,
    ProviderExecutionContext,
)
from app.services.question_engine_v2.schemas import TemplateProviderOptions
from app.utils.logger import get_logger


class TemplateProviderV2(BaseProviderV2):
    """Generates deterministic questions from catalogs and templates."""

    name = "template_provider_v2"
    provider_key = "template"

    def __init__(self, options: TemplateProviderOptions | None = None) -> None:
        super().__init__()
        self.options = options or TemplateProviderOptions()
        self.logger = get_logger(__name__)
        self._template_cache: Dict[str, List[Tuple[str, str, str, str]]] = {}
        self._load_catalogs()

    def update_options(self, options: TemplateProviderOptions) -> None:
        """Apply runtime configuration supplied by the engine."""
        self.options = options

    async def _generate(
        self, context: ProviderExecutionContext
    ) -> tuple[List[Question], Dict[str, Any]]:
        """Render persona-aware templates deterministically."""
        if not context.personas:
            self.logger.warning(
                "Template provider invoked without personas",
                audit_run=str(context.request.audit_run_id),
            )
            return [], {"warning": "no_personas"}

        total_quota = (
            context.quotas.total if context.quotas and context.quotas.total else None
        )
        per_persona_cap = self.options.max_per_persona

        questions: List[Question] = []
        for persona in context.personas:
            persona_key = persona.mode.value
            templates = self._templates_for(persona_key, persona)
            persona_questions = self._render_templates(
                templates,
                context=context,
                persona=persona,
            )

            if per_persona_cap:
                persona_questions = persona_questions[:per_persona_cap]

            questions.extend(persona_questions)

        if total_quota is not None:
            questions = questions[:total_quota]

        metadata = {
            "provider": self.provider_key,
            "persona_count": len(context.personas),
            "total_quota": total_quota,
        }
        return questions, metadata

    def _load_catalogs(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent / "catalogs" / "templates"
        if not base_dir.exists():
            self.logger.warning(
                "Template catalog directory missing", path=str(base_dir)
            )
            return
        for path in base_dir.glob("*.txt"):
            mode = path.stem
            entries: List[Tuple[str, str, str, str]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    key, template = line.split("=", 1)
                    role, driver, context_stage = key.split(":", 3)
                    entries.append((role, driver, context_stage, template))
                except ValueError:
                    self.logger.warning(
                        "Malformed template row skipped", mode=mode, row=line
                    )
            self._template_cache[mode] = entries

    def _templates_for(self, mode: str, persona) -> Iterable[Tuple[str, str, str, str]]:
        entries = self._template_cache.get(mode, [])
        for role, driver, context_stage, template in entries:
            if role == persona.role and driver == persona.driver:
                yield role, driver, context_stage, template

    def _render_templates(
        self,
        templates: Iterable[Tuple[str, str, str, str]],
        *,
        context: ProviderExecutionContext,
        persona,
    ) -> List[Question]:
        rendered: List[Question] = []
        base_context = {
            "client_brand": context.request.client_brand,
            "industry": context.request.industry,
            "product_type": context.request.product_type,
        }
        competitors = context.request.competitors or []

        for role, driver, template_context, template_str in templates:
            persona_label = persona.voice or persona.role
            for competitor in competitors or [None]:
                values = base_context | {
                    "role": role,
                    "driver": driver,
                    "context_stage": template_context,
                    "persona": persona_label,
                }
                if competitor:
                    values["competitor"] = competitor
                try:
                    text = template_str.format(**values)
                except KeyError as exc:
                    self.logger.warning(
                        "Template missing key", template=template_str, missing=str(exc)
                    )
                    continue
                question = Question(
                    question_text=text,
                    category="template",
                    provider=self.name,
                    metadata=self._build_question_metadata(
                        {
                            "role": role,
                            "driver": driver,
                            "context_stage": template_context,
                            "persona": persona_label,
                            "competitor": competitor,
                        }
                    ),
                )
                question.role = role
                question.driver = driver
                question.context_stage = template_context
                question.persona = persona_label
                rendered.append(question)
        return rendered


__all__ = ["TemplateProviderV2"]
