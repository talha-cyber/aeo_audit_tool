from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.v1.providers.health import get_question_engine
from app.main import app
from app.services.providers import QuestionProvider
from app.services.question_engine import QuestionEngine

# Create a TestClient instance
client = TestClient(app)


@pytest.fixture
def mock_question_engine():
    """Fixture to mock the QuestionEngine and its providers."""
    # Create mock providers
    healthy_provider = AsyncMock(spec=QuestionProvider)
    healthy_provider.name = "healthy_provider"
    healthy_provider.health_check.return_value = True

    unhealthy_provider = AsyncMock(spec=QuestionProvider)
    unhealthy_provider.name = "unhealthy_provider"
    unhealthy_provider.health_check.return_value = False

    # Create a mock QuestionEngine that uses these providers
    mock_engine = MagicMock(spec=QuestionEngine)
    mock_engine.providers = [healthy_provider, unhealthy_provider]

    return mock_engine


def test_health_check_all_healthy():
    """Test the health endpoint when all providers are healthy."""
    # Arrange
    healthy_provider = AsyncMock(spec=QuestionProvider)
    healthy_provider.name = "healthy_provider"
    healthy_provider.health_check.return_value = True

    mock_engine = MagicMock(spec=QuestionEngine)
    mock_engine.providers = [healthy_provider]

    app.dependency_overrides[get_question_engine] = lambda: mock_engine

    # Act
    response = client.get("/api/v1/providers/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data == {"healthy_provider": {"status": "ok"}}
    # Clean up the override
    app.dependency_overrides = {}


def test_health_check_one_unhealthy():
    """Test the health endpoint when one provider is unhealthy."""
    # Arrange
    healthy_provider = AsyncMock(spec=QuestionProvider)
    healthy_provider.name = "healthy_provider"
    healthy_provider.health_check.return_value = True

    unhealthy_provider = AsyncMock(spec=QuestionProvider)
    unhealthy_provider.name = "unhealthy_provider"
    unhealthy_provider.health_check.return_value = False

    mock_engine = MagicMock(spec=QuestionEngine)
    mock_engine.providers = [healthy_provider, unhealthy_provider]
    app.dependency_overrides[get_question_engine] = lambda: mock_engine

    # Act
    response = client.get("/api/v1/providers/health")

    # Assert
    assert response.status_code == 503
    data = response.json()
    assert data == {
        "healthy_provider": {"status": "ok"},
        "unhealthy_provider": {"status": "error"},
    }
    app.dependency_overrides = {}


def test_health_check_with_exception():
    """Test the health endpoint when a provider's health_check raises an exception."""
    # Arrange
    exception_provider = AsyncMock(spec=QuestionProvider)
    exception_provider.name = "exception_provider"
    exception_provider.health_check.side_effect = Exception("Connection failed")

    mock_engine = MagicMock(spec=QuestionEngine)
    mock_engine.providers = [exception_provider]

    app.dependency_overrides[get_question_engine] = lambda: mock_engine

    # Act
    response = client.get("/api/v1/providers/health")

    # Assert
    assert response.status_code == 503
    data = response.json()
    assert data == {"exception_provider": {"status": "error"}}
    app.dependency_overrides = {}
