"""Utility helpers for dashboard service projections."""

from __future__ import annotations

from datetime import time
from typing import Optional


_CRON_PRESETS = {
    "0 0 * * *": "Daily at 00:00",
    "0 9 * * *": "Daily at 09:00",
    "0 12 * * *": "Daily at 12:00",
    "0 9 1 * *": "Monthly on day 1 at 09:00",
    "0 6 * * 1": "Weekly on Monday at 06:00",
}


def cron_to_label(cron_expression: str) -> str:
    """Return a human-friendly label for a cron expression."""
    expr = (cron_expression or "").strip()
    if not expr:
        return "Custom schedule"

    preset = _CRON_PRESETS.get(expr)
    if preset:
        return preset

    parts = expr.split()
    if len(parts) < 5:
        return expr

    minute, hour, day_of_month, month, day_of_week = parts[:5]

    def _fmt_time(m: str, h: str) -> Optional[str]:
        if m.isdigit() and h.isdigit():
            try:
                return time(int(h) % 24, int(m) % 60).strftime("%H:%M")
            except ValueError:
                return None
        return None

    label = "Custom schedule"
    time_str = _fmt_time(minute, hour)

    if day_of_month == "*" and day_of_week == "*":
        label = "Daily"
    elif day_of_month != "*" and day_of_week == "*":
        label = f"Monthly on day {day_of_month}"
    elif day_of_week != "*" and day_of_month == "*":
        label = f"Weekly on {day_of_week}"

    if time_str:
        label = f"{label} at {time_str}"

    return label


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator / denominator guarding against division by zero."""
    if denominator == 0:
        return None
    return numerator / denominator
