from __future__ import annotations

from sqlalchemy.engine import Engine

from app.db.base_class import Base
from app.models.audit import AuditRun, Client
from app.models.response import Response
from app.services.dashboard.comparison_service import get_comparison_matrix


def _ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def _drop_schema(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)


def test_comparison_matrix_from_responses(db_engine: Engine, db_session) -> None:
    _ensure_schema(db_engine)
    try:
        client = Client(id="client-1", name="Primary Brand", industry="AI")
        run = AuditRun(
            id="run-1",
            client=client,
            config={"name": "Test run"},
            status="completed",
        )

        response_a = Response(
            id="resp-1",
            audit_run=run,
            question_id="q1",
            platform="chatgpt",
            response_text="",
            raw_response={},
            brand_mentions=[
                {"brand": "Primary Brand", "frequency": 3, "sentiment": "positive"},
                {"brand": "Competitor X", "frequency": 1, "sentiment": "negative"},
            ],
        )

        response_b = Response(
            id="resp-2",
            audit_run=run,
            question_id="q2",
            platform="claude",
            response_text="",
            raw_response={},
            brand_mentions=[
                {"brand": "Primary Brand", "frequency": 1, "sentiment": "neutral"},
                {"brand": "Competitor Y", "frequency": 2, "sentiment": "positive"},
            ],
        )

        db_session.add_all([client, run, response_a, response_b])
        db_session.commit()

        matrix = get_comparison_matrix(db_session)

        assert matrix.competitors[0] == "Primary Brand"
        assert len(matrix.signals) == 3
        sov_signal = next(signal for signal in matrix.signals if signal.label == "Share of Voice")
        assert round(sum(sov_signal.weights), 4) == 1.0

        positive_signal = next(signal for signal in matrix.signals if signal.label == "Positive Sentiment")
        primary_positive = positive_signal.weights[matrix.competitors.index("Primary Brand")]
        assert primary_positive > 0.0

        coverage_signal = next(signal for signal in matrix.signals if signal.label == "Platform Coverage")
        assert coverage_signal.weights[matrix.competitors.index("Primary Brand")] == 1.0
    finally:
        _drop_schema(db_engine)


def test_comparison_matrix_fallback_without_session() -> None:
    matrix = get_comparison_matrix(None)
    assert matrix.competitors
    assert matrix.signals
