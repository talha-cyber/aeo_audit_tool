import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.providers import (
    ProviderResult,
    Question,
    QuestionContext,
    QuestionProvider,
)
from app.services.providers.template_provider import TemplateProvider
from app.services.question_engine import QuestionEngine


@pytest.fixture
def question_context():
    """Fixture to provide a sample QuestionContext."""
    return QuestionContext(
        client_brand="TestClient",
        competitors=["CompA", "CompB"],
        industry="SaaS",
        product_type="CRM",
        audit_run_id=uuid.uuid4(),
    )


@pytest.fixture
def mock_dynamic_provider():
    """Fixture to create a mock DynamicProvider."""
    mock_provider = AsyncMock(spec=QuestionProvider)
    mock_provider.name = "mock_dynamic_provider"
    mock_provider.can_handle.return_value = True
    mock_provider.generate.return_value = ProviderResult(
        questions=[
            Question(
                question_text="Dynamic question 1?",
                category="dynamic",
                priority_score=10,
                provider="mock_dynamic_provider",
            )
        ],
        metadata={"source": "dynamic"},
    )
    return mock_provider


@pytest.mark.asyncio
async def test_question_engine_orchestration(
    question_context, mock_dynamic_provider, monkeypatch
):
    """
    Test that the QuestionEngine correctly orchestrates multiple providers.
    """
    # Arrange
    # Using a real TemplateProvider and a mocked DynamicProvider
    template_provider = TemplateProvider()
    template_provider._generate_questions_sync = MagicMock(
        return_value=[
            Question(
                question_text="Template question 1?",
                category="comparison",
                provider="template_provider",
                priority_score=5,
            ),
            Question(
                question_text="Template question 2?",
                category="alternatives",
                provider="template_provider",
                priority_score=8,
            ),
        ]
    )
    providers = [template_provider, mock_dynamic_provider]
    engine = QuestionEngine(providers=providers)

    # Act
    results = await engine.generate_questions(
        client_brand=question_context.client_brand,
        competitors=question_context.competitors,
        industry=question_context.industry,
        product_type=question_context.product_type,
        audit_run_id=question_context.audit_run_id,
        max_questions=5,
    )

    # Assert
    # 1. Both providers should have been called
    mock_dynamic_provider.can_handle.assert_called_with(question_context)
    mock_dynamic_provider.generate.assert_awaited_with(question_context)

    # 2. Results should be merged and prioritized
    assert len(results) > 0

    # 3. Check for questions from both sources
    assert any(q.provider == "template_provider" for q in results)
    assert any(q.category == "dynamic" for q in results)

    # 4. Check that dynamic question is first due to high priority
    assert results[0].question_text == "Dynamic question 1?"

    # 5. Check the total number of questions
    assert len(results) == 3  # 2 from template, 1 from dynamic
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_question_engine_handles_provider_failure(
    question_context, mock_dynamic_provider, monkeypatch
):
    """
    Test that the QuestionEngine gracefully handles a failure in one provider.
    """
    # Arrange
    # Sabotage one of the providers
    mock_dynamic_provider.generate.side_effect = Exception("API is down")

    providers = [TemplateProvider(), mock_dynamic_provider]
    engine = QuestionEngine(providers=providers)

    # Act
    results = await engine.generate_questions(
        client_brand=question_context.client_brand,
        competitors=question_context.competitors,
        industry=question_context.industry,
        product_type=question_context.product_type,
        audit_run_id=question_context.audit_run_id,
        max_questions=5,
    )

    # Assert
    # 1. The failing provider was called
    mock_dynamic_provider.generate.assert_awaited_with(question_context)

    # 2. The engine should still return results from the healthy provider
    assert len(results) > 0

    # 3. All returned questions should have the correct provider
    assert all(q.provider == "template_provider" for q in results)
    assert len(results) > 0
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_question_engine_deduplication(question_context, monkeypatch):
    """
    Test that the QuestionEngine deduplicates questions from different providers.
    """
    # Arrange
    # Create two mock providers that return an overlapping question
    provider1 = AsyncMock(spec=QuestionProvider)
    provider1.name = "provider1"
    provider1.can_handle.return_value = True
    provider1.generate.return_value = ProviderResult(
        questions=[
            Question(
                question_text="What is the best CRM?",
                category="template",
                priority_score=5,
                provider="provider1",
            )
        ],
        metadata={},
    )

    provider2 = AsyncMock(spec=QuestionProvider)
    provider2.name = "provider2"
    provider2.can_handle.return_value = True
    provider2.generate.return_value = ProviderResult(
        questions=[
            Question(
                question_text="what is the best crm?",
                category="dynamic",
                priority_score=10,
                provider="provider2",
            )
        ],
        metadata={},
    )

    engine = QuestionEngine(providers=[provider1, provider2])

    # Act
    results = await engine.generate_questions(
        client_brand=question_context.client_brand,
        competitors=question_context.competitors,
        industry=question_context.industry,
        product_type=question_context.product_type,
        audit_run_id=question_context.audit_run_id,
    )

    # Assert
    # 1. Should only have one instance of the question
    assert len(results) == 1

    # 2. It should keep the one with the higher priority score
    assert results[0].category == "dynamic"
    assert results[0].priority_score >= 8
    monkeypatch.undo()
