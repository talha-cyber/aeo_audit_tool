from app.monitoring.alerting.alert_manager import AlertManager, ThresholdRule


def test_alert_manager_triggers_rule(monkeypatch):
    fired = {"n": 0}

    class DummyNotifier:
        def notify(self, title: str, *, severity: str = "warning", context=None):
            fired["n"] += 1

    mgr = AlertManager(notifier=DummyNotifier())
    # metric getter returns 0.5; threshold 0.8 with comparison '<' should trigger
    mgr.add_rule(
        ThresholdRule(
            name="low_rate", metric_getter=lambda: 0.5, threshold=0.8, comparison="<"
        )
    )
    count = mgr.evaluate()
    assert count == 1
    assert fired["n"] == 1
