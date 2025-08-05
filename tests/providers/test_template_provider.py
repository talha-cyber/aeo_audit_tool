import uuid
from unittest.mock import MagicMock

import pytest

from app.services.providers import Question, QuestionContext
from app.services.providers.template_provider import TemplateProvider


@pytest.fixture
def mock_template_provider():
    """Fixture to create a TemplateProvider with a mocked generation method."""
    provider = TemplateProvider()
    provider._generate_questions_sync = MagicMock(
        return_value=[
            Question(
                question_text="What is the best CRM?",
                category="comparison",
                provider="template_provider",
            ),
            Question(
                question_text="What are alternatives to Salesforce?",
                category="alternatives",
                provider="template_provider",
            ),
        ]
    )
    return provider


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


def test_template_provider_name(mock_template_provider):
    """Test that the provider has the correct name."""
    assert mock_template_provider.name == "template_provider"


def test_template_provider_can_handle(mock_template_provider, question_context):
    """Test that the provider can always handle a context."""
    assert mock_template_provider.can_handle(question_context) is True


@pytest.mark.asyncio
async def test_template_provider_health_check(mock_template_provider):
    """Test that the provider's health check always returns True."""
    assert await mock_template_provider.health_check() is True


@pytest.mark.asyncio
async def test_template_provider_generate(mock_template_provider, question_context):
    """Test the generate method of the TemplateProvider."""
    result = await mock_template_provider.generate(question_context)

    # Verify that the sync generation method was called
    mock_template_provider._generate_questions_sync.assert_called_once()

    # Verify the structure of the result
    assert result.metadata["source"] == "template_provider"
    assert len(result.questions) == 2
    assert result.questions[0].question_text == "What is the best CRM?"

    # Check that the 'category' is NOT updated to 'template'
    # The provider should preserve the original category
    assert result.questions[0].category == "comparison"
    assert result.questions[1].category == "alternatives"
