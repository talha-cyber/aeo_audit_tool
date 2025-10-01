"""Portal widget catalogue service."""

from __future__ import annotations

from typing import List

from app.api.v1.dashboard_schemas import WidgetView
from app.services.dashboard.static_data import default_widgets

__all__ = ["list_widgets"]


def list_widgets() -> List[WidgetView]:
    """Return available portal widgets for embedding."""

    return [WidgetView(**item) for item in default_widgets()]
