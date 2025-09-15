from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import sentry_sdk

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """Minimal notification sender: logs, optional webhook."""

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        self.webhook_url = webhook_url or getattr(settings, "ALERT_WEBHOOK_URL", None)

    def notify(
        self,
        title: str,
        *,
        severity: str = "warning",
        context: Dict[str, Any] | None = None,
    ) -> None:
        payload = {"title": title, "severity": severity, "context": context or {}}
        logger.warning("alert_notification", **payload)
        # Also capture in Sentry (if configured)
        try:
            if settings.SENTRY_DSN:
                sentry_sdk.capture_message(f"{title} ({severity})")
        except Exception as e:  # noqa: BLE001
            logger.error("alert_sentry_error", error=str(e))
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=payload, timeout=5)
            except Exception as e:  # noqa: BLE001
                logger.error("alert_webhook_error", error=str(e))
