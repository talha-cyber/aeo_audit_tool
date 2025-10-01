"""Audit run detail assembly."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, selectinload

from app.api.v1.dashboard_schemas import (
    AuditQuestionView,
    AuditRunDetailView,
    BrandMentionView,
    SentimentLabel,
)
from app.models.audit import AuditRun
from app.models.question import Question
from app.models.response import Response
from app.services.dashboard.audit_run_service import map_audit_run_to_view

__all__ = ["get_run_detail"]


def get_run_detail(db: Session, run_id: str) -> AuditRunDetailView:
    """Return detailed question/response breakdown for an audit run."""

    run = (
        db.query(AuditRun)
        .options(
            selectinload(AuditRun.questions).selectinload(Question.responses),
            selectinload(AuditRun.responses),
            selectinload(AuditRun.client),
        )
        .filter(AuditRun.id == run_id)
        .first()
    )
    if not run:
        raise ValueError(f"Audit run {run_id} not found")

    question_views = _build_question_views(run)

    return AuditRunDetailView(run=map_audit_run_to_view(run), questions=question_views)


def _build_question_views(run: AuditRun) -> List[AuditQuestionView]:
    views: List[AuditQuestionView] = []

    for question in run.questions:
        response = _select_primary_response(question, run.responses)
        sentiment = _extract_sentiment(response)
        mentions = _extract_mentions(response)

        views.append(
            AuditQuestionView(
                id=question.id,
                prompt=question.question_text,
                platform=_resolve_platform(question, response),
                sentiment=sentiment,
                mentions=mentions,
            )
        )

    return views


def _select_primary_response(question: Question, fallback_responses: List[Response]) -> Optional[Response]:
    if question.responses:
        return question.responses[0]
    for response in fallback_responses:
        if response.question_id == question.id:
            return response
    return None


def _resolve_platform(question: Question, response: Optional[Response]) -> str:
    if question.provider:
        return question.provider
    if response and response.platform:
        return response.platform
    return "unknown"


def _extract_sentiment(response: Optional[Response]) -> SentimentLabel:
    if not response:
        return "neutral"

    metadata = response.response_metadata if isinstance(response.response_metadata, dict) else {}
    sentinel = metadata.get("sentiment") or response.emotional_satisfaction

    if isinstance(sentinel, str):
        lower = sentinel.lower()
        if lower in {"positive", "neutral", "negative"}:
            return lower  # type: ignore[return-value]
    return "neutral"


def _extract_mentions(response: Optional[Response]) -> List[BrandMentionView]:
    mentions_payload = response.brand_mentions if response else None
    if not mentions_payload:
        return []

    normalized: List[BrandMentionView] = []

    if isinstance(mentions_payload, list):
        for item in mentions_payload:
            if not isinstance(item, dict):
                continue
            brand = item.get("brand") or item.get("name")
            if not brand:
                continue
            normalized.append(
                BrandMentionView(
                    brand=str(brand),
                    frequency=int(item.get("frequency") or item.get("count") or 0),
                    sentiment=_sanitize_sentiment(item.get("sentiment")),
                )
            )
        return normalized

    if isinstance(mentions_payload, dict):
        for brand, details in mentions_payload.items():
            if not isinstance(details, dict):
                continue
            normalized.append(
                BrandMentionView(
                    brand=str(brand),
                    frequency=int(details.get("count") or details.get("frequency") or 0),
                    sentiment=_sanitize_sentiment(details.get("sentiment")),
                )
            )
    return normalized


def _sanitize_sentiment(value: Any) -> SentimentLabel:
    if isinstance(value, str) and value.lower() in {"positive", "neutral", "negative"}:
        return value.lower()  # type: ignore[return-value]
    return "neutral"
