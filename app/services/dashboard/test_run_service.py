"""Simulated audit test-run launcher for dashboard interactions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.api.v1.dashboard_schemas import (
    AuditIssueView,
    AuditRunProgressView,
    AuditRunView,
    LaunchTestRunRequest,
    LaunchTestRunResponse,
)

__all__ = ["launch_simulated_run"]


def launch_simulated_run(payload: LaunchTestRunRequest) -> LaunchTestRunResponse:
    """Create a deterministic mocked audit run for dashboard testing.

    The frontend relies on this endpoint while the full execution pipeline is
    still wired to mock providers. We synthesise an `AuditRunView` so the UI can
    immediately display progress without having to wait on Celery workers.
    """
    now = datetime.now(timezone.utc)

    progress = AuditRunProgressView(
        done=0,
        total=payload.question_count,
        updated_at=now,
    )

    run = AuditRunView(
        id=f"testrun-{uuid4().hex[:12]}",
        name=_derive_run_name(payload),
        status="running",
        started_at=now,
        completed_at=None,
        progress=progress,
        issues=[
            AuditIssueView(
                id="mock-provider",
                label="Test run uses mock providers; real execution not yet wired",
                severity="low",
            )
        ],
    )

    return LaunchTestRunResponse(run=run)


def _derive_run_name(payload: LaunchTestRunRequest) -> str:
    platforms = ", ".join(payload.platforms[:3])
    if len(payload.platforms) > 3:
        platforms += ", ..."
    return f"Simulation - {payload.scenario_id} ({platforms})"
