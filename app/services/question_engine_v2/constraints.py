"""Constraint enforcement helpers for question engine v2."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List

from app.services.providers import Question
from app.services.question_engine_v2.schemas import QuotaConfig, SeedMixConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEED_TARGETS: Dict[str, float] = {
    "unseeded": 0.4,
    "competitor": 0.3,
    "brand": 0.3,
}


def enforce_constraints(
    questions: Iterable[Question],
    quotas: QuotaConfig | None = None,
    seed_mix: SeedMixConfig | None = None,
    max_length: int = 180,
) -> List[Question]:
    """Apply seed-mix, length, and dedupe constraints to generated questions."""

    question_list = list(questions)
    logger.debug(
        "Applying constraints",
        question_count=len(question_list),
        quotas=quotas.model_dump() if quotas else None,
        seed_mix=seed_mix.model_dump() if seed_mix else None,
    )

    constrained = _dedupe_questions(question_list)
    constrained = _enforce_length(constrained, max_length=max_length)
    constrained = _enforce_seed_mix(constrained, quotas, seed_mix)

    total_quota = quotas.total if quotas and quotas.total else None
    if total_quota is not None:
        constrained = constrained[:total_quota]

    return constrained


def _dedupe_questions(questions: Iterable[Question]) -> List[Question]:
    seen = OrderedDict()
    for question in questions:
        key = question.question_text.strip().lower()
        if key not in seen:
            seen[key] = question
    return list(seen.values())


def _enforce_length(
    questions: Iterable[Question], *, max_length: int
) -> List[Question]:
    if max_length <= 0:
        return list(questions)
    filtered: List[Question] = []
    for question in questions:
        if len(question.question_text.split()) <= max_length:
            filtered.append(question)
    return filtered


def _enforce_seed_mix(
    questions: List[Question],
    quotas: QuotaConfig | None,
    seed_mix: SeedMixConfig | None,
) -> List[Question]:
    if not questions:
        return []

    targets = DEFAULT_SEED_TARGETS.copy()
    if seed_mix:
        data = seed_mix.model_dump(exclude_none=True)
        if data:
            total = sum(data.values()) or 1.0
            targets = {k: v / total for k, v in data.items()}

    bucketed: Dict[str, List[Question]] = {}
    for question in questions:
        key = (question.seed_type or "unseeded").lower()
        bucketed.setdefault(key, []).append(question)

    total_quota = quotas.total if quotas and quotas.total else len(questions)
    allocation: Dict[str, int] = {}
    remaining = total_quota
    for seed, ratio in targets.items():
        count = min(int(round(total_quota * ratio)), len(bucketed.get(seed, [])))
        allocation[seed] = count
        remaining -= count

    # Redistribute remaining slots to buckets with headroom
    if remaining > 0:
        for seed, items in bucketed.items():
            available = len(items) - allocation.get(seed, 0)
            if available <= 0:
                continue
            add = min(available, remaining)
            allocation[seed] = allocation.get(seed, 0) + add
            remaining -= add
            if remaining <= 0:
                break

    selected: List[Question] = []
    for seed, items in bucketed.items():
        take = allocation.get(seed, 0)
        if take:
            selected.extend(items[:take])

    # Fill any remaining slots with leftover questions preserving original order
    if len(selected) < len(questions):
        used_ids = {id(q) for q in selected}
        for question in questions:
            if len(selected) >= total_quota:
                break
            if id(question) not in used_ids:
                selected.append(question)
                used_ids.add(id(question))

    return selected


__all__ = ["enforce_constraints"]
