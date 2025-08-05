import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from app.services.providers import Question, QuestionContext
from app.services.providers.dynamic_provider import DynamicProvider


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
    """Fixture to create a DynamicProvider with mocked internal components."""
    with (
        patch(
            "app.services.providers.dynamic_provider.TrendsAdapter",
            new_callable=MagicMock,
        ) as mock_trends,
        patch(
            "app.services.providers.dynamic_provider.PromptBuilder",
            new_callable=MagicMock,
        ) as mock_prompt,
        patch(
            "app.services.providers.dynamic_provider.LLMClient", new_callable=MagicMock
        ) as mock_llm,
        patch(
            "app.services.providers.dynamic_provider.PostProcessor",
            new_callable=MagicMock,
        ) as mock_post,
    ):
        # Configure mocks
        mock_trends.return_value.fetch_seeds = AsyncMock(
            return_value=["seed1", "seed2"]
        )
        mock_prompt.return_value.build.return_value = "This is a test prompt."
        mock_llm.return_value.generate_questions = AsyncMock(
            return_value=ChatCompletion(
                id="test_completion",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content="Question 1\nQuestion 2",
                            role="assistant",
                        ),
                    )
                ],
                created=12345,
                model="test_model",
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=50,
                    prompt_tokens=100,
                    total_tokens=150,
                ),
            )
        )
        mock_llm.return_value._calculate_cost.return_value = 0.0001
        mock_post.return_value.process.return_value = [
            Question(
                question_text="Question 1",
                category="dynamic",
                provider="dynamic_provider",
            ),
            Question(
                question_text="Question 2",
                category="dynamic",
                provider="dynamic_provider",
            ),
        ]

        provider = DynamicProvider()
        # Attach mocks to the provider instance for easy access in tests
        provider._trends_adapter = mock_trends.return_value
        provider._prompt_builder = mock_prompt.return_value
        provider._llm_client = mock_llm.return_value
        provider._post_processor = mock_post.return_value

        yield provider


@pytest.mark.asyncio
@patch("app.utils.cache.CacheManager.get", new_callable=AsyncMock)
@patch("app.utils.cache.CacheManager.set", new_callable=AsyncMock)
async def test_generate_cache_miss(
    mock_cache_set, mock_cache_get, mock_dynamic_provider, question_context
):
    """Test the generate method on a cache miss."""
    mock_cache_get.return_value = None  # Ensure cache miss

    result = await mock_dynamic_provider.generate(question_context)

    # Assert that the full pipeline was executed
    mock_dynamic_provider._trends_adapter.fetch_seeds.assert_awaited_once()
    mock_dynamic_provider._prompt_builder.build.assert_called_once()
    mock_dynamic_provider._llm_client.generate_questions.assert_awaited_once()
    mock_dynamic_provider._post_processor.process.assert_called_once()

    # Assert that the result was cached
    mock_cache_set.assert_awaited_once()

    # Assert the result is correct
    assert result.metadata["cache_hit"] is False
    assert len(result.questions) == 2
    assert result.questions[0].category == "dynamic"
    assert result.questions[0].cost is not None
    assert result.questions[0].tokens is not None
    assert result.metadata["cost"] > 0


@pytest.mark.asyncio
@patch("app.utils.cache.CacheManager.get", new_callable=AsyncMock)
@patch("app.utils.cache.CacheManager.set", new_callable=AsyncMock)
async def test_generate_cache_hit(
    mock_cache_set, mock_cache_get, mock_dynamic_provider, question_context
):
    """Test the generate method on a cache hit."""
    cached_questions = [
        Question(
            question_text="Cached Question",
            category="dynamic",
            provider="dynamic_provider",
        )
    ]
    mock_cache_get.return_value = [q.model_dump() for q in cached_questions]

    result = await mock_dynamic_provider.generate(question_context)

    # Assert that the pipeline was NOT executed
    mock_dynamic_provider._trends_adapter.fetch_seeds.assert_not_awaited()
    mock_dynamic_provider._llm_client.generate_questions.assert_not_awaited()

    # Assert that the cache was not set again
    mock_cache_set.assert_not_awaited()

    # Assert the result is from the cache
    assert result.metadata["cache_hit"] is True
    assert result.questions == cached_questions


def test_dynamic_provider_name(mock_dynamic_provider):
    """Test that the provider has the correct name."""
    assert mock_dynamic_provider.name == "dynamic_provider"


def test_can_handle_when_enabled(question_context):
    """Test can_handle when the provider is enabled in config."""
    with patch(
        "app.services.providers.dynamic_provider.settings.DYNAMIC_Q_ENABLED", True
    ):
        provider = DynamicProvider()
        assert provider.can_handle(question_context) is True


def test_can_handle_when_disabled(question_context):
    """Test can_handle when the provider is disabled in config."""
    with patch(
        "app.services.providers.dynamic_provider.settings.DYNAMIC_Q_ENABLED", False
    ):
        provider = DynamicProvider()
        assert provider.can_handle(question_context) is False
