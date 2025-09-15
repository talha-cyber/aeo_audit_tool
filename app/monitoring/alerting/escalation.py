from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EscalationPolicy:
    cooldown_seconds: int = 300
    last_fired: Dict[str, datetime] = None

    def __post_init__(self) -> None:
        if self.last_fired is None:
            self.last_fired = {}

    def on_alert(self, rule) -> None:
        now = datetime.utcnow()
        last = self.last_fired.get(rule.name)
        if not last or (now - last) > timedelta(seconds=self.cooldown_seconds):
            logger.warning("alert_escalation", rule=rule.name, severity=rule.severity)
            self.last_fired[rule.name] = now
