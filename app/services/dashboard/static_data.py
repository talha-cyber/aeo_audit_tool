"""Static fixtures used to bootstrap dashboard responses."""

from __future__ import annotations

from typing import Any, Dict, List


def default_audit_programs() -> List[Dict[str, Any]]:
    """Return representative audit program configs for initial seeding."""

    return [
        {
            "name": "US Agency Benchmark",
            "job_type": "audit",
            "trigger_config": {"trigger_type": "cron", "expression": "0 9 * * 1"},
            "payload": {
                "client_id": "client-us-agency",
                "owner": {"name": "Strategy Ops", "email": "ops@agency.com"},
                "cadence": {"label": "Weekly", "cron": "0 9 * * 1"},
                "platforms": ["openai", "anthropic", "perplexity"],
                "question_bank_id": "bank-weekly-visibility",
                "execution": {"sla_minutes": 90, "batch_size": 24},
            },
        },
        {
            "name": "Feature Comparison North America",
            "job_type": "audit",
            "trigger_config": {"trigger_type": "cron", "expression": "0 12 1 * *"},
            "payload": {
                "client_id": "client-feature-na",
                "owner": {"name": "Product Marketing", "email": "pm@agency.com"},
                "cadence": {"label": "Monthly", "cron": "0 12 1 * *"},
                "platforms": ["openai", "vertex"],
                "question_bank_id": "bank-monthly-feature",
                "execution": {"sla_minutes": 240, "batch_size": 48},
            },
        },
    ]


def default_report_summaries() -> List[Dict[str, Any]]:
    """Return representative report metadata for initial responses."""

    return [
        {
            "id": "report-001",
            "title": "Executive Summary — July",
            "generated_at": "2025-07-31T09:00:00+00:00",
            "audit_id": "run-101",
            "coverage": {"completed": 112, "total": 120},
        },
        {
            "id": "report-002",
            "title": "Competitive Deep Dive — Pricing",
            "generated_at": "2025-07-01T12:00:00+00:00",
            "audit_id": "run-100",
            "coverage": {"completed": 96, "total": 98},
        },
    ]


def default_insights() -> List[Dict[str, Any]]:
    """Return representative insight payloads."""

    return [
        {
            "id": "insight-001",
            "title": "Claude ranks Competitor Z first for onboarding",
            "kind": "risk",
            "summary": "Competitor Z dominates onboarding guidance across LLMs.",
            "detected_at": "2025-07-30T16:30:00+00:00",
            "impact": "high",
        },
        {
            "id": "insight-002",
            "title": "ChatGPT promotes our case study in comparison queries",
            "kind": "opportunity",
            "summary": "Leverage this asset in client portal and QBR decks.",
            "detected_at": "2025-07-29T10:00:00+00:00",
            "impact": "medium",
        },
    ]


def default_personas() -> List[Dict[str, Any]]:
    return [
        {
            "id": "persona-1",
            "name": "Growth Marketer",
            "segment": "Performance Agencies",
            "priority": "primary",
            "key_need": "Proves ROI of AI presence to clients.",
            "journey_stage": [
                {"stage": "Discover", "question": "Who is leading AI visibility?", "coverage": 0.82},
                {"stage": "Evaluate", "question": "What keywords drive conversions?", "coverage": 0.67},
                {"stage": "Decide", "question": "Which assets to amplify?", "coverage": 0.54},
            ],
        },
        {
            "id": "persona-2",
            "name": "VP of Client Services",
            "segment": "Agency Leadership",
            "priority": "secondary",
            "key_need": "Keeps clients confident during QBRs.",
            "journey_stage": [
                {"stage": "Discover", "question": "Where are we falling behind?", "coverage": 0.48},
                {"stage": "Evaluate", "question": "Which competitors trend up?", "coverage": 0.41},
            ],
        },
    ]


def default_widgets() -> List[Dict[str, Any]]:
    return [
        {"id": "widget-1", "name": "AEO Heatmap", "preview": "9×9 matrix", "status": "draft"},
        {"id": "widget-2", "name": "Weekly Signal Digest", "preview": "Email embed", "status": "published"},
    ]


def default_settings() -> Dict[str, Any]:
    return {
        "branding": {
            "primaryColor": "#111214",
            "logoUrl": None,
            "tone": "Measured and confident",
        },
        "members": [
            {"id": "member-1", "name": "Jamie", "role": "Admin", "email": "jamie@agency.com"},
            {"id": "member-2", "name": "Morgan", "role": "Editor", "email": "morgan@agency.com"},
        ],
        "billing": {"plan": "Agency Pro", "renewsOn": "2025-09-15T00:00:00+00:00"},
        "integrations": [
            {"id": "int-1", "name": "Slack", "connected": True},
            {"id": "int-2", "name": "HubSpot", "connected": False},
            {"id": "int-3", "name": "Google Drive", "connected": True},
        ],
    }


__all__ = [
    "default_audit_programs",
    "default_report_summaries",
    "default_insights",
    "default_personas",
    "default_widgets",
    "default_settings",
]


__all__ = [
    "default_audit_programs",
    "default_report_summaries",
    "default_insights",
]
