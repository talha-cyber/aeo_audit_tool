"""Pydantic response schemas for the dashboard-facing API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.services.question_engine_v2.schemas import PersonaMode

AuditRunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
IssueSeverity = Literal["low", "medium", "high"]
SentimentLabel = Literal["positive", "neutral", "negative"]
InsightKind = Literal["opportunity", "risk", "signal"]
InsightImpact = Literal["low", "medium", "high"]
PersonaPriority = Literal["primary", "secondary"]
WidgetStatus = Literal["draft", "published"]


class BaseApiModel(BaseModel):
    """Base class enabling alias-friendly export."""

    model_config = ConfigDict(populate_by_name=True)


class AuditIssueView(BaseApiModel):
    id: str
    label: str
    severity: IssueSeverity


class AuditRunProgressView(BaseApiModel):
    done: int
    total: int = Field(ge=0)
    updated_at: Optional[datetime] = Field(default=None, alias="updatedAt")


class AuditRunView(BaseApiModel):
    id: str
    name: str
    status: AuditRunStatus
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    completed_at: Optional[datetime] = Field(default=None, alias="completedAt")
    progress: AuditRunProgressView
    issues: List[AuditIssueView] = []


class BrandMentionView(BaseApiModel):
    brand: str
    frequency: int = Field(ge=0)
    sentiment: SentimentLabel


class AuditQuestionView(BaseApiModel):
    id: str
    prompt: str
    platform: str
    sentiment: SentimentLabel
    mentions: List[BrandMentionView] = []


class AuditRunDetailView(BaseApiModel):
    run: AuditRunView
    questions: List[AuditQuestionView]


class CoverageView(BaseApiModel):
    completed: int = Field(ge=0)
    total: int = Field(ge=0)


class ReportSummaryView(BaseApiModel):
    id: str
    title: str
    generated_at: datetime = Field(alias="generatedAt")
    audit_id: str = Field(alias="auditId")
    coverage: CoverageView


class InsightView(BaseApiModel):
    id: str
    title: str
    kind: InsightKind
    summary: str
    detected_at: datetime = Field(alias="detectedAt")
    impact: InsightImpact


class PersonaStageView(BaseApiModel):
    stage: str
    question: str
    coverage: float = Field(ge=0.0, le=1.0)


class PersonaView(BaseApiModel):
    id: str
    name: str
    segment: str
    priority: PersonaPriority
    key_need: str = Field(alias="keyNeed")
    journey_stage: List[PersonaStageView] = Field(alias="journeyStage")
    meta: Optional[Dict[str, Any]] = None


class PersonaLibraryEntryView(PersonaView):
    owner_id: str = Field(alias="ownerId")
    mode: PersonaMode
    role: str
    driver: str
    voice: Optional[str] = None
    context_keys: List[str] = Field(alias="contextKeys")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class PersonaCatalogRoleView(BaseApiModel):
    key: str
    label: str
    description: Optional[str] = None
    default_context_stage: Optional[str] = Field(
        default=None, alias="defaultContextStage"
    )


class PersonaCatalogDriverView(BaseApiModel):
    key: str
    label: str
    emotional_anchor: Optional[str] = Field(
        default=None, alias="emotionalAnchor"
    )
    weight: Optional[float] = None


class PersonaCatalogContextView(BaseApiModel):
    key: str
    label: str
    priority: Optional[float] = None


class PersonaCatalogVoiceView(BaseApiModel):
    key: str
    role: str
    driver: str
    contexts: List[str]


class PersonaCatalogView(BaseApiModel):
    mode: PersonaMode
    roles: List[PersonaCatalogRoleView]
    drivers: List[PersonaCatalogDriverView]
    contexts: List[PersonaCatalogContextView]
    voices: List[PersonaCatalogVoiceView]


class PersonaComposeRequest(BaseApiModel):
    mode: PersonaMode = PersonaMode.B2C
    owner_id: Optional[str] = Field(default=None, alias="ownerId")
    voice: Optional[str] = None
    role: Optional[str] = None
    driver: Optional[str] = None
    contexts: Optional[List[str]] = None
    name: Optional[str] = None
    segment: Optional[str] = None
    priority: Optional[PersonaPriority] = None
    key_need: Optional[str] = Field(default=None, alias="keyNeed")
    journey_stage: Optional[List[PersonaStageView]] = Field(
        default=None, alias="journeyStage"
    )


class PersonaUpdateRequest(BaseApiModel):
    owner_id: Optional[str] = Field(default=None, alias="ownerId")
    mode: PersonaMode = PersonaMode.B2C
    voice: Optional[str] = None
    role: Optional[str] = None
    driver: Optional[str] = None
    contexts: Optional[List[str]] = None
    name: Optional[str] = None
    segment: Optional[str] = None
    priority: Optional[PersonaPriority] = None
    key_need: Optional[str] = Field(default=None, alias="keyNeed")
    journey_stage: Optional[List[PersonaStageView]] = Field(
        default=None, alias="journeyStage"
    )


class PersonaCloneRequest(BaseApiModel):
    owner_id: Optional[str] = Field(default=None, alias="ownerId")
    mode: PersonaMode = PersonaMode.B2C
    name: Optional[str] = None


class PersonaLibraryResponse(BaseApiModel):
    personas: List[PersonaLibraryEntryView]


class WidgetView(BaseApiModel):
    id: str
    name: str
    preview: str
    status: WidgetStatus


class ComparisonSignalView(BaseApiModel):
    label: str
    weights: List[float]


class ComparisonMatrixView(BaseApiModel):
    competitors: List[str]
    signals: List[ComparisonSignalView]


class BrandingSettingsView(BaseApiModel):
    primary_color: str = Field(alias="primaryColor")
    logo_url: Optional[HttpUrl] = Field(default=None, alias="logoUrl")
    tone: str


class MemberView(BaseApiModel):
    id: str
    name: str
    role: str
    email: str


class BillingSettingsView(BaseApiModel):
    plan: str
    renews_on: datetime = Field(alias="renewsOn")


class IntegrationView(BaseApiModel):
    id: str
    name: str
    connected: bool



class AuditOwnerView(BaseApiModel):
    name: Optional[str] = None
    email: Optional[str] = None


class SettingsView(BaseApiModel):
    branding: BrandingSettingsView
    members: List[MemberView]
    billing: BillingSettingsView
    integrations: List[IntegrationView]


class AuditSummaryView(BaseApiModel):
    id: str
    name: str
    cadence: str
    owner: AuditOwnerView
    platforms: List[str]
    last_run: Optional[datetime] = Field(default=None, alias="lastRun")
    health_score: Optional[float] = Field(default=None, alias="healthScore")


class LaunchTestRunRequest(BaseApiModel):
    scenario_id: str = Field(alias="scenarioId")
    question_count: int = Field(alias="questionCount", ge=1)
    platforms: List[str]


class LaunchTestRunResponse(BaseApiModel):
    run: AuditRunView


__all__ = [
    "AuditIssueView",
    "AuditRunProgressView",
    "AuditRunView",
    "AuditRunDetailView",
    "BrandMentionView",
    "AuditQuestionView",
    "CoverageView",
    "ReportSummaryView",
    "InsightView",
    "PersonaStageView",
    "PersonaView",
    "WidgetView",
    "ComparisonSignalView",
    "ComparisonMatrixView",
    "BrandingSettingsView",
    "MemberView",
    "BillingSettingsView",
    "IntegrationView",
    "AuditOwnerView",
    "SettingsView",
    "AuditSummaryView",
    "LaunchTestRunRequest",
    "LaunchTestRunResponse",
    "AuditRunStatus",
    "IssueSeverity",
    "SentimentLabel",
    "InsightKind",
    "InsightImpact",
]
