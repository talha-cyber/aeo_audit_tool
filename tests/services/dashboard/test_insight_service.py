import pytest

pytest.importorskip("pydantic", reason="pydantic wheels unavailable on this interpreter")

from app.services.dashboard import insight_service


def test_insight_fallback(monkeypatch):
    class DummyBrain:
        def get_active_insights(self):
            return []

    monkeypatch.setattr(insight_service, "CentralIntelligence", lambda: DummyBrain())

    insights = insight_service.list_insights(limit=1)
    assert len(insights) == 1
    first = insights[0]
    assert first.kind in {"risk", "opportunity", "signal"}
    assert first.impact in {"low", "medium", "high"}


def test_insight_kind_coercion():
    assert insight_service._coerce_kind("risk") == "risk"  # type: ignore[attr-defined]
    assert insight_service._coerce_kind("unknown") == "signal"  # type: ignore[attr-defined]


def test_insight_impact_coercion():
    assert insight_service._coerce_impact("high") == "high"  # type: ignore[attr-defined]
    assert insight_service._coerce_impact("mystery") == "medium"  # type: ignore[attr-defined]
