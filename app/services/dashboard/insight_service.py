"""Insight feeds exposed to the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import InsightImpact, InsightKind, InsightView
from app.organism.brain.central_intelligence import CentralIntelligence, SystemInsight
from app.services.dashboard.static_data import default_insights
from app.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["list_insights"]


def list_insights(db: Session | None = None, *, limit: Optional[int] = None) -> List[InsightView]:
    """Return recent competitive insights for the dashboard feed."""

    views = _collect_central_intelligence_insights(limit=limit)
    if views:
        return views

    logger.debug("Falling back to static insight fixtures")
    data = default_insights()
    if limit is not None:
        data = data[:limit]

    return [
        InsightView(
            id=payload["id"],
            title=payload["title"],
            kind=_coerce_kind(payload.get("kind")),
            summary=payload["summary"],
            detected_at=_parse_datetime(payload.get("detected_at")),
            impact=_coerce_impact(payload.get("impact")),
        )
        for payload in data
    ]


def _collect_central_intelligence_insights(limit: Optional[int]) -> List[InsightView]:
    try:
        brain = CentralIntelligence()
        insights = brain.get_active_insights()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Central intelligence unavailable", error=str(exc))
        return []

    if not insights:
        return []

    if limit is not None:
        insights = insights[:limit]

    views: List[InsightView] = []
    for insight in insights:
        views.append(
            InsightView(
                id=insight.id,
                title=insight.description,
                kind=_coerce_kind(insight.category),
                summary=insight.description,
                detected_at=insight.timestamp,
                impact=_coerce_impact(_select_highest_impact(insight)),
            )
        )
    return views


def _parse_datetime(value: Optional[str]) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


def _coerce_kind(value: Optional[str]) -> InsightKind:
    mapping = {"risk": "risk", "opportunity": "opportunity", "signal": "signal"}
    return mapping.get((value or "").lower(), "signal")  # type: ignore[return-value]


def _coerce_impact(value: Optional[str]) -> InsightImpact:
    mapping = {"low": "low", "medium": "medium", "high": "high"}
    return mapping.get((value or "").lower(), "medium")  # type: ignore[return-value]


def _select_highest_impact(insight: SystemInsight) -> str:
    impact = insight.impact_assessment or {}
    if not impact:
        return "medium"
    highest = max(impact.values())
    if highest >= 0.75:
        return "high"
    if highest >= 0.4:
        return "medium"
    return "low"
