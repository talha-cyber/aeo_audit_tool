from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from app.utils.logger import get_logger

from .escalation import EscalationPolicy
from .notification import NotificationManager

logger = get_logger(__name__)


@dataclass
class ThresholdRule:
    name: str
    metric_getter: Callable[[], float]
    threshold: float
    comparison: str  # ">" or "<"
    severity: str = "warning"
    description: str = ""

    def triggered(self) -> bool:
        value = float(self.metric_getter())
        if self.comparison == ">":
            return value > self.threshold
        return value < self.threshold


class AlertManager:
    def __init__(
        self,
        notifier: Optional[NotificationManager] = None,
        escalation: Optional[EscalationPolicy] = None,
    ):
        self.notifier = notifier or NotificationManager()
        self.escalation = escalation or EscalationPolicy()
        self.rules: list[ThresholdRule] = []

    def add_rule(self, rule: ThresholdRule) -> None:
        self.rules.append(rule)

    def evaluate(self) -> int:
        fired = 0
        for rule in self.rules:
            try:
                if rule.triggered():
                    fired += 1
                    self.notifier.notify(
                        title=f"Alert: {rule.name}",
                        severity=rule.severity,
                        context={"description": rule.description},
                    )
                    self.escalation.on_alert(rule)
            except Exception as e:  # noqa: BLE001
                logger.error("alert_rule_error", rule=rule.name, error=str(e))
        return fired
