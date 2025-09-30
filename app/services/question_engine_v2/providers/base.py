"""Common provider interfaces for question engine v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from app.services.providers import ProviderResult, Question
from app.services.question_engine_v2.schemas import (
    PersonaMode,
    PersonaRequest,
    PersonaResolution,
    ProviderConfig,
    QuotaConfig,
    QuestionEngineRequest,
    SeedMixConfig,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProviderExecutionContext(BaseModel):
    """Runtime payload supplied to providers during generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: QuestionEngineRequest
    personas: List[PersonaResolution]
    persona_request: Optional[PersonaRequest] = None
    quotas: Optional[QuotaConfig] = None
    seed_mix: Optional[SeedMixConfig] = None
    provider_config: Optional[ProviderConfig] = None
    router: Any = None
    cache: Any = None
    extras: Dict[str, Any] = {}


class BaseProviderV2(ABC):
    """Abstract base class for persona-aware question providers."""

    name: str = "provider_v2"
    provider_key: str = "base"
    supported_modes: Optional[List[PersonaMode]] = None

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def with_router(self, router: Any) -> None:
        """Inject an LLM router instance."""
        self.config["router"] = router

    def with_cache(self, cache: Any) -> None:
        """Inject cache dependency."""
        self.config["cache"] = cache

    def can_handle(self, context: ProviderExecutionContext) -> bool:
        """Return True if provider should execute for the supplied context."""
        if self.supported_modes and context.personas:
            persona_modes = {persona.mode for persona in context.personas}
            if not persona_modes.intersection(self.supported_modes):
                self.logger.debug(
                    "Provider skipped due to unsupported persona mode",
                    provider=self.name,
                    modes=[mode.value for mode in persona_modes],
                )
                return False
        return True

    async def generate(self, context: ProviderExecutionContext) -> ProviderResult:
        """Execute provider and wrap results in ProviderResult."""
        self.logger.debug(
            "Provider generate invoked",
            provider=self.name,
            personas=len(context.personas),
            audit_run=str(context.request.audit_run_id),
        )
        questions, metadata = await self._generate(context)
        return ProviderResult(questions=questions, metadata=metadata or {})

    @abstractmethod
    async def _generate(
        self, context: ProviderExecutionContext
    ) -> tuple[List[Question], Dict[str, Any]]:
        """Core implementation to be provided by subclasses."""

    def _build_question_metadata(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = base.copy() if base else {}
        metadata.setdefault("provider", self.provider_key)
        return metadata


__all__ = [
    "BaseProviderV2",
    "ProviderExecutionContext",
]
