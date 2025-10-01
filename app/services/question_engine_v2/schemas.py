"""Schemas and enums used by question engine v2."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.providers import Question


class PersonaMode(str, Enum):
    """Supported persona catalog modes."""

    B2C = "b2c"
    B2B = "b2b"


class RoleCatalogItem(BaseModel):
    """Role definition captured in persona catalogs."""

    label: str
    description: Optional[str] = None
    default_context_stage: Optional[str] = Field(
        default=None, description="Fallback context stage when none supplied."
    )


class DriverCatalogItem(BaseModel):
    """Driver definition including emotional anchor and optional weight."""

    label: str
    emotional_anchor: Optional[str] = None
    weight: Optional[float] = None


class ContextCatalogItem(BaseModel):
    """Context stage metadata used for quota weighting."""

    label: str
    priority: Optional[float] = None


class VoiceCatalogItem(BaseModel):
    """Preset persona combining role, driver, and context stages."""

    role: str
    driver: str
    contexts: List[str]

    @field_validator("contexts")
    def _ensure_contexts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("contexts must contain at least one stage")
        return value


class PersonaCatalog(BaseModel):
    """Complete catalog bundle for a persona mode."""

    mode: PersonaMode
    roles: Dict[str, RoleCatalogItem]
    drivers: Dict[str, DriverCatalogItem]
    contexts: Dict[str, ContextCatalogItem]
    voices: Dict[str, VoiceCatalogItem]


class PersonaResolution(BaseModel):
    """Resolved persona ready for quota planning."""

    mode: PersonaMode
    role: str
    driver: str
    contexts: List[str]
    emotional_anchor: Optional[str] = None
    voice: Optional[str] = None


class PersonaSelection(BaseModel):
    """Explicit persona specification from request payloads."""

    role: str
    driver: str
    contexts: List[str]
    voice: Optional[str] = None

    @field_validator("contexts")
    def _non_empty_contexts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("contexts must contain at least one entry")
        return value


class PersonaRequest(BaseModel):
    """Persona configuration passed into the engine."""

    mode: PersonaMode
    voices: List[str] = Field(default_factory=list)
    overrides: List[PersonaSelection] = Field(default_factory=list)


class SeedMixConfig(BaseModel):
    """Desired distribution across seed types."""

    unseeded: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    competitor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    brand: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class QuotaConfig(BaseModel):
    """Quota targets for question generation."""

    total: Optional[int] = Field(default=None, ge=1)
    per_persona_min: Optional[int] = Field(default=None, ge=0)
    per_context_min: Optional[int] = Field(default=None, ge=0)


class TemplateProviderOptions(BaseModel):
    """Configuration flags for template provider."""

    enabled: bool = True
    max_per_persona: Optional[int] = Field(default=None, ge=1)


class DynamicProviderOptions(BaseModel):
    """Configuration flags for dynamic provider."""

    enabled: bool = True
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=32)


class ProviderConfig(BaseModel):
    """Wrapper for provider-specific configuration."""

    template: TemplateProviderOptions = Field(default_factory=TemplateProviderOptions)
    dynamic: DynamicProviderOptions = Field(default_factory=DynamicProviderOptions)


class QuestionEngineRequest(BaseModel):
    """Incoming request payload for question generation."""

    client_brand: str
    competitors: List[str]
    industry: str
    product_type: str
    audit_run_id: uuid.UUID
    language: str = "en"
    market: Optional[str] = None
    personas: Optional[PersonaRequest] = None
    seed_mix: Optional[SeedMixConfig] = None
    quotas: Optional[QuotaConfig] = None
    providers: Optional[ProviderConfig] = None
    engine_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionEngineResponse(BaseModel):
    """Standardized response envelope for generated questions."""

    questions: List[Question]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionEngineConfig(BaseModel):
    """Placeholder config schema for engine initialization."""

    settings: Optional[dict] = None


__all__ = [
    "ContextCatalogItem",
    "DriverCatalogItem",
    "DynamicProviderOptions",
    "PersonaCatalog",
    "PersonaMode",
    "PersonaRequest",
    "PersonaResolution",
    "PersonaSelection",
    "ProviderConfig",
    "QuotaConfig",
    "QuestionEngineConfig",
    "QuestionEngineRequest",
    "QuestionEngineResponse",
    "RoleCatalogItem",
    "SeedMixConfig",
    "TemplateProviderOptions",
    "VoiceCatalogItem",
]
