"""Report summary projections for dashboard endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session, selectinload

from app.api.v1.dashboard_schemas import CoverageView, ReportSummaryView
from app.models.audit import AuditRun
from app.models.report import Report
from app.services.dashboard.static_data import default_report_summaries

__all__ = ["list_reports"]


def list_reports(db: Session, *, limit: Optional[int] = None) -> List[ReportSummaryView]:
    """Return generated reports in reverse chronological order."""

    query = db.query(Report).options(selectinload(Report.audit_run)).order_by(desc(Report.generated_at))
    if limit:
        query = query.limit(limit)

    reports: Iterable[Report] = query.all()
    summaries = [
        ReportSummaryView(
            id=report.id,
            title=report_title(report),
            generated_at=report.generated_at or datetime.now(),
            audit_id=report.audit_run_id,
            coverage=_report_coverage(report),
        )
        for report in reports
    ]

    if summaries:
        return summaries

    return _fallback_reports()


def _report_coverage(report: Report) -> CoverageView:
    run: Optional[AuditRun] = report.audit_run
    completed = getattr(run, "processed_questions", 0) or 0
    total = getattr(run, "total_questions", completed) or 0

    if report.file_path and "coverage" in (report.template_version or ""):
        pass  # placeholder for future template hints

    if total == 0:
        total = completed

    return CoverageView(completed=completed, total=total or completed)


def report_title(report: Report) -> str:
    if report.report_type:
        return f"{report.report_type.replace('_', ' ').title()}"
    return report.file_path.split("/")[-1] if report.file_path else f"Report {report.id[:8]}"


def _fallback_reports() -> List[ReportSummaryView]:
    summaries: List[ReportSummaryView] = []
    for payload in default_report_summaries():
        generated_at = payload.get("generated_at")
        timestamp = datetime.fromisoformat(generated_at) if isinstance(generated_at, str) else datetime.now()
        coverage = payload.get("coverage", {})
        summaries.append(
            ReportSummaryView(
                id=str(payload.get("id")),
                title=payload.get("title", "Report"),
                generated_at=timestamp,
                audit_id=str(payload.get("audit_id", "")),
                coverage=CoverageView(
                    completed=int(coverage.get("completed", 0)),
                    total=int(coverage.get("total", coverage.get("completed", 0))),
                ),
            )
        )
    return summaries
