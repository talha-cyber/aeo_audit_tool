"""Typed structures shared across dashboard services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HealthScoreComponents:
    """Breakdown of metrics contributing to an audit program health score."""

    success_rate: float
    timeliness: float
    error_penalty: float
