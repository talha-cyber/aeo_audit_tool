from __future__ import annotations

from typing import Any, Dict

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Lightweight alert manager stub.

    Integrate with your existing alerting stack (e.g., Sentry, PagerDuty) here.
    """

    def fire(
        self,
        title: str,
        *,
        severity: str = "warning",
        context: Dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        logger.warning("resilience_alert", title=title, severity=severity, **ctx)
