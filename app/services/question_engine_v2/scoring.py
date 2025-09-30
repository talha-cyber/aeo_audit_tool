"""Scoring and prioritization for question engine v2."""

from __future__ import annotations

from typing import Dict, Iterable, List

from app.services.providers import Question
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DRIVER_WEIGHTS: Dict[str, float] = {
    "deal_hunter": 3.0,
    "values_aligned": 2.5,
    "convenience": 2.0,
    "roi_tco": 4.0,
    "trust_compliance": 3.5,
    "scalability": 3.0,
}

DEFAULT_CONTEXT_WEIGHTS: Dict[str, float] = {
    "awareness": 1.5,
    "evaluation": 2.5,
    "purchase": 3.0,
    "loyalty": 2.0,
    "validation": 2.5,
    "negotiation": 3.0,
    "rollout": 2.0,
    "expansion": 2.5,
}

DEFAULT_SEED_WEIGHTS: Dict[str, float] = {
    "unseeded": 2.5,
    "competitor": 2.0,
    "brand": 1.5,
}

DEFAULT_PROVIDER_BONUS: Dict[str, float] = {
    "template": 0.5,
    "dynamic": 1.0,
}


def score_questions(
    questions: Iterable[Question],
    *,
    driver_weights: Dict[str, float] | None = None,
    context_weights: Dict[str, float] | None = None,
    seed_weights: Dict[str, float] | None = None,
    provider_bonus: Dict[str, float] | None = None,
) -> List[Question]:
    """Assign priority scores based on driver, context, seed, and provider."""

    driver_weights = driver_weights or DEFAULT_DRIVER_WEIGHTS
    context_weights = context_weights or DEFAULT_CONTEXT_WEIGHTS
    seed_weights = seed_weights or DEFAULT_SEED_WEIGHTS
    provider_bonus = provider_bonus or DEFAULT_PROVIDER_BONUS

    scored: List[Question] = []
    for question in questions:
        driver = (question.driver or "").lower()
        context = (question.context_stage or "").lower()
        seed = (question.seed_type or "unseeded").lower()
        provider = question.provider

        score = 0.0
        score += driver_weights.get(driver, 1.0)
        score += context_weights.get(context, 1.0)
        score += seed_weights.get(seed, 1.0)
        score += provider_bonus.get(provider, 0.0)

        question.priority_score = score
        scored.append(question)

    return scored


__all__ = ["score_questions"]
