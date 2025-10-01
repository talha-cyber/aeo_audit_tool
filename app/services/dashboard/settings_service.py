"""Dashboard settings aggregation."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import SettingsView
from app.services.dashboard.static_data import default_settings

__all__ = ["get_settings"]


def get_settings(db: Session | None = None) -> SettingsView:
    """Return branding, membership, and integration settings."""

    payload = default_settings()
    return SettingsView.model_validate(payload)
