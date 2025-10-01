"""Competitive comparison matrix calculations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import ComparisonMatrixView, ComparisonSignalView
from app.models.response import Response

__all__ = ["get_comparison_matrix"]


def get_comparison_matrix(db: Session | None = None) -> ComparisonMatrixView:
    """Aggregate share-of-voice metrics into a comparison matrix."""

    if db is None:
        return _fallback_matrix()

    brand_totals: Dict[str, float] = defaultdict(float)
    brand_positive: Dict[str, float] = defaultdict(float)
    brand_platforms: Dict[str, set[str]] = defaultdict(set)
    all_platforms: set[str] = set()

    for mentions, platform in _iter_mentions(db):
        all_platforms.add(platform)
        for brand, frequency, sentiment in mentions:
            if frequency <= 0:
                continue
            brand_totals[brand] += frequency
            if sentiment == "positive":
                brand_positive[brand] += frequency
            brand_platforms[brand].add(platform)

    if not brand_totals:
        return _fallback_matrix()

    competitors = sorted(brand_totals, key=brand_totals.get, reverse=True)
    total_volume = sum(brand_totals.values()) or 1.0
    total_platforms = max(len(all_platforms), 1)

    share_of_voice = [round(brand_totals[name] / total_volume, 4) for name in competitors]
    positive_ratio = [
        round(brand_positive.get(name, 0.0) / brand_totals[name], 4) if brand_totals[name] else 0.0
        for name in competitors
    ]
    platform_coverage = [
        round(len(brand_platforms.get(name, set())) / total_platforms, 4)
        for name in competitors
    ]

    signals = [
        ComparisonSignalView(label="Share of Voice", weights=share_of_voice),
        ComparisonSignalView(label="Positive Sentiment", weights=positive_ratio),
        ComparisonSignalView(label="Platform Coverage", weights=platform_coverage),
    ]
    return ComparisonMatrixView(competitors=competitors, signals=signals)


def _iter_mentions(db: Session) -> Iterable[Tuple[List[Tuple[str, float, str]], str]]:
    statement = select(Response.brand_mentions, Response.platform).where(
        Response.brand_mentions.isnot(None)
    )
    for payload, platform in db.execute(statement):
        normalized = _normalize_mentions(payload)
        if normalized:
            yield normalized, platform or "unknown"


def _normalize_mentions(payload: object) -> List[Tuple[str, float, str]]:
    mentions: List[Tuple[str, float, str]] = []

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            brand = item.get("brand") or item.get("name")
            if not brand:
                continue
            frequency = float(item.get("frequency") or item.get("count") or 0)
            sentiment = str(item.get("sentiment") or "neutral").lower()
            mentions.append((str(brand), frequency, _sanitize_sentiment(sentiment)))
    elif isinstance(payload, dict):
        for brand, details in payload.items():
            if not isinstance(details, dict):
                continue
            frequency = float(details.get("frequency") or details.get("count") or 0)
            sentiment = str(details.get("sentiment") or "neutral").lower()
            mentions.append((str(brand), frequency, _sanitize_sentiment(sentiment)))

    return [entry for entry in mentions if entry[1] > 0]


def _sanitize_sentiment(label: str) -> str:
    if label in {"positive", "neutral", "negative"}:
        return label
    return "neutral"


def _fallback_matrix() -> ComparisonMatrixView:
    competitors = ["Primary Brand", "Competitor X", "Competitor Y"]
    signals = [
        ComparisonSignalView(label="Share of Voice", weights=[0.62, 0.21, 0.17]),
        ComparisonSignalView(label="Positive Sentiment", weights=[0.48, 0.32, 0.20]),
        ComparisonSignalView(label="Platform Coverage", weights=[0.71, 0.18, 0.11]),
    ]
    return ComparisonMatrixView(competitors=competitors, signals=signals)
