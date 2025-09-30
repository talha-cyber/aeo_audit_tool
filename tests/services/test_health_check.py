from datetime import datetime, timedelta, timezone

import pytest

from app.services.health_check import (
    ComponentHealth,
    ComponentType,
    HealthChecker,
    HealthMetric,
    HealthStatus,
    SystemHealth,
)


@pytest.mark.asyncio
async def test_get_health_history_filters_window():
    checker = HealthChecker()
    checker._history.clear()

    base_metric = HealthMetric(name="ping", value=1, unit="ms")

    old_component = ComponentHealth(
        name="database",
        component_type=ComponentType.DATABASE,
        status=HealthStatus.HEALTHY,
        metrics=[base_metric],
    )
    recent_component = ComponentHealth(
        name="queue",
        component_type=ComponentType.QUEUE,
        status=HealthStatus.DEGRADED,
        metrics=[base_metric],
    )

    now = datetime.now(timezone.utc)
    oldest = SystemHealth(
        status=HealthStatus.HEALTHY,
        timestamp=now - timedelta(hours=4),
        components=[old_component],
        summary={},
    )
    recent = SystemHealth(
        status=HealthStatus.DEGRADED,
        timestamp=now - timedelta(minutes=30),
        components=[recent_component],
        summary={"note": "queue backlog"},
        alert_level="medium",
    )

    checker._history.extend([oldest, recent])

    history = await checker.get_health_history(hours=2)

    assert len(history) == 1
    assert history[0]["status"] == HealthStatus.DEGRADED.value
    assert history[0]["alert_level"] == "medium"
    assert history[0]["timestamp"] == recent.timestamp.isoformat()
