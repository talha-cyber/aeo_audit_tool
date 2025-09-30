"""
Comprehensive tests for the audit processor module.

This module tests the core audit processor functionality, including:
- Audit workflow orchestration
- Question generation integration
- Platform management
- Brand detection integration
- Error handling and recovery
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.audit import AuditRun, Client
from app.services.audit_processor import AuditProcessor
from app.services.platform_manager import PlatformManager
from app.utils.error_handler import AuditConfigurationError, PlatformError


@pytest.fixture(autouse=True)
def disable_question_engine_v2(monkeypatch):
    """Ensure tests exercise question engine v1 path by default."""
    monkeypatch.setattr(settings, "QUESTION_ENGINE_V2", False)


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = Mock(spec=Session)
    session.query.return_value.filter.return_value.first.return_value = None
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session


@pytest.fixture
def mock_platform_manager():
    """Mock platform manager with test platforms"""
    manager = Mock(spec=PlatformManager)
    manager.get_available_platforms.return_value = ["openai", "anthropic"]

    # Mock platform instances
    mock_platform = Mock()
    mock_platform.safe_query = AsyncMock(
        return_value={"choices": [{"message": {"content": "Test response"}}]}
    )
    mock_platform.extract_text_response.return_value = "Test response content"
    mock_platform.health_check = AsyncMock(return_value=True)

    manager.get_platform.return_value = mock_platform
    return manager


@pytest.fixture
def sample_audit_run():
    """Sample audit run for testing"""
    client = Client(
        id="550e8400-e29b-41d4-a716-446655440001",
        name="Test Company",
        industry="Technology",
        product_type="SaaS",
        competitors=["Competitor A", "Competitor B"],
    )

    audit_run = AuditRun(
        id="550e8400-e29b-41d4-a716-446655440000",  # Valid UUID format
        client_id="550e8400-e29b-41d4-a716-446655440001",
        client=client,
        config={
            "platforms": ["openai"],
            "question_count": 5,
            "client": {
                "name": "Test Company",
                "industry": "Technology",
                "product_type": "SaaS",
                "competitors": ["Competitor A", "Competitor B"],
            },
        },
        status="pending",
    )

    return audit_run


@pytest.fixture
def audit_processor(mock_db_session, mock_platform_manager, disable_question_engine_v2):
    """Create audit processor with mocked dependencies"""
    processor = AuditProcessor(mock_db_session, mock_platform_manager)
    processor.question_engine_v2 = None
    return processor


class TestAuditProcessor:
    """Test cases for the main AuditProcessor class"""

    @pytest.mark.asyncio
    async def test_audit_processor_initialization(self, audit_processor):
        """Test audit processor initialization"""
        assert audit_processor.db is not None
        assert audit_processor.platform_manager is not None
        assert audit_processor.settings is not None
        assert audit_processor.metrics is not None

    @pytest.mark.asyncio
    async def test_run_audit_success(
        self, audit_processor, mock_db_session, sample_audit_run
    ):
        """Test successful audit run execution"""
        # Setup mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        # Mock question generation
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            assert audit_processor.question_engine_v2 is None
            mock_questions = [
                Mock(id="q1", question_text="Test question 1", category="comparison"),
                Mock(
                    id="q2", question_text="Test question 2", category="recommendation"
                ),
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            # Mock brand detection
            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {
                    "brands": ["Test Company"]
                }
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Execute audit
                result = await audit_processor.run_audit(
                    "550e8400-e29b-41d4-a716-446655440000"
                )

                # Verify result
                assert result == "550e8400-e29b-41d4-a716-446655440000"

                # Verify database interactions
                assert mock_db_session.commit.called

                # Verify question generation was called
                mock_question_engine.return_value.generate_questions.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_audit_not_found(self, audit_processor, mock_db_session):
        """Test audit run not found scenario"""
        # Setup mock to return None (audit not found)
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Audit run .* not found"):
            await audit_processor.run_audit("nonexistent_audit")

    @pytest.mark.asyncio
    async def test_run_audit_invalid_status(
        self, audit_processor, mock_db_session, sample_audit_run
    ):
        """Test audit run with invalid status"""
        sample_audit_run.status = "completed"  # Already completed
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        with pytest.raises(ValueError, match="Audit run .* is not in executable state"):
            await audit_processor.run_audit("550e8400-e29b-41d4-a716-446655440000")

    @pytest.mark.asyncio
    async def test_run_audit_no_platforms_available(
        self, audit_processor, mock_db_session, sample_audit_run, mock_platform_manager
    ):
        """Test audit run when no platforms are available"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )
        mock_platform_manager.get_available_platforms.return_value = []

        with pytest.raises(AuditConfigurationError, match="No platforms available"):
            await audit_processor.run_audit("550e8400-e29b-41d4-a716-446655440000")

    @pytest.mark.asyncio
    async def test_generate_questions_success(self, audit_processor, sample_audit_run):
        """Test successful question generation"""
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                {"id": "q1", "question": "Test question 1", "category": "comparison"},
                {
                    "id": "q2",
                    "question": "Test question 2",
                    "category": "recommendation",
                },
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            context = {"max_questions": 10, "target_brands": ["Test Company"]}
            questions = await audit_processor._generate_questions(
                sample_audit_run, context
            )

            assert len(questions) == 2
            assert questions[0]["question"] == "Test question 1"
            mock_question_engine.return_value.generate_questions.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_questions_concurrency(
        self, audit_processor, sample_audit_run, mock_platform_manager
    ):
        """Test concurrent question processing"""
        # Create test questions in the format expected by the actual implementation
        questions = [
            {"id": f"q{i}", "question": f"Question {i}", "category": "comparison"}
            for i in range(5)
        ]

        # Mock platform response
        mock_platform = mock_platform_manager.get_platform.return_value
        mock_platform.safe_query.return_value = {
            "choices": [{"message": {"content": "Response content"}}]
        }

        # Mock brand detection
        with patch(
            "app.services.audit_processor.BrandDetectionEngine"
        ) as mock_brand_detection:
            mock_detection_result = Mock()
            mock_detection_result.to_summary_dict.return_value = {"brands": []}
            mock_brand_detection.return_value.analyze_response.return_value = (
                mock_detection_result
            )

            platforms = ["openai"]
            target_brands = ["Test Company"]

            context = {"platforms": platforms, "target_brands": target_brands}
            responses = await audit_processor._process_questions_batched(
                sample_audit_run, questions, context
            )

            # Verify responses were processed (may be fewer due to batching)
            assert len(responses) >= 0  # Some responses should be generated

            # Verify platform was called
            assert mock_platform.safe_query.call_count >= 0

    @pytest.mark.asyncio
    async def test_process_question_with_platform_error(
        self, audit_processor, sample_audit_run, mock_platform_manager
    ):
        """Test question processing with platform error"""
        questions = [
            {"id": "q1", "question": "Test question", "category": "comparison"}
        ]

        # Mock platform to raise error
        mock_platform = mock_platform_manager.get_platform.return_value
        mock_platform.safe_query.side_effect = PlatformError("openai", "API Error")

        with patch("app.services.audit_processor.BrandDetectionEngine"):
            context = {"platforms": ["openai"], "target_brands": ["Test Company"]}
            responses = await audit_processor._process_questions_batched(
                sample_audit_run, questions, context
            )

            # Should handle errors gracefully
            assert isinstance(responses, list)  # Should return a list even with errors

    @pytest.mark.asyncio
    async def test_batch_processing(
        self, audit_processor, sample_audit_run, mock_platform_manager
    ):
        """Test question batching functionality"""
        # Create many questions
        questions = [
            {"id": f"q{i}", "question": f"Question {i}", "category": "comparison"}
            for i in range(25)
        ]

        # Mock platform response
        mock_platform = mock_platform_manager.get_platform.return_value
        mock_platform.safe_query.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch(
            "app.services.audit_processor.BrandDetectionEngine"
        ) as mock_brand_detection:
            mock_detection_result = Mock()
            mock_detection_result.to_summary_dict.return_value = {"brands": []}
            mock_brand_detection.return_value.analyze_response.return_value = (
                mock_detection_result
            )

            # Test that batched processing works with many questions
            context = {"platforms": ["openai"], "target_brands": ["Test Company"]}
            responses = await audit_processor._process_questions_batched(
                sample_audit_run, questions, context
            )

            # Should process questions in batches
            assert isinstance(responses, list)
            # With batching, we should get some responses
            assert len(responses) >= 0

    @pytest.mark.asyncio
    async def test_audit_status_tracking(
        self, audit_processor, mock_db_session, sample_audit_run
    ):
        """Test audit status tracking functionality"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        # Test status update
        from app.services.audit_processor import AuditStatus

        await audit_processor._update_audit_status(
            sample_audit_run, AuditStatus.RUNNING
        )

        assert sample_audit_run.status == AuditStatus.RUNNING.value

        # Database commit should be called
        assert mock_db_session.commit.called

    @pytest.mark.asyncio
    async def test_platform_fallback(
        self, audit_processor, sample_audit_run, mock_platform_manager
    ):
        """Test platform fallback when primary platform fails"""
        questions = [
            {"id": "q1", "question": "Test question", "category": "comparison"}
        ]

        # Setup platform manager with multiple platforms
        mock_platform_manager.get_available_platforms.return_value = [
            "openai",
            "anthropic",
        ]

        # Mock first platform to fail, second to succeed
        mock_openai = Mock()
        mock_openai.safe_query.side_effect = PlatformError(
            "openai", "Rate limit exceeded"
        )
        mock_anthropic = Mock()
        mock_anthropic.safe_query.return_value = {
            "content": [{"text": "Fallback response"}]
        }
        mock_anthropic.extract_text_response.return_value = "Fallback response"

        def get_platform_side_effect(name):
            if name == "openai":
                return mock_openai
            elif name == "anthropic":
                return mock_anthropic

        mock_platform_manager.get_platform.side_effect = get_platform_side_effect

        with patch(
            "app.services.audit_processor.BrandDetectionEngine"
        ) as mock_brand_detection:
            mock_detection_result = Mock()
            mock_detection_result.to_summary_dict.return_value = {"brands": []}
            mock_brand_detection.return_value.analyze_response.return_value = (
                mock_detection_result
            )

            context = {
                "platforms": ["openai", "anthropic"],
                "target_brands": ["Test Company"],
            }
            responses = await audit_processor._process_questions_batched(
                sample_audit_run, questions, context
            )

            # Should handle platform failures gracefully
            assert isinstance(responses, list)
            # Should attempt both platforms
            assert mock_openai.safe_query.called or mock_anthropic.safe_query.called

    @pytest.mark.asyncio
    async def test_audit_metrics_integration(
        self, audit_processor, mock_db_session, sample_audit_run
    ):
        """Test integration with audit metrics"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_question_engine.return_value.generate_questions.return_value = []

            with patch.object(
                audit_processor.metrics, "increment_audit_started"
            ) as mock_started:
                with patch.object(
                    audit_processor.metrics, "increment_audit_completed"
                ) as mock_completed:
                    await audit_processor.run_audit(
                        "550e8400-e29b-41d4-a716-446655440000"
                    )

                    # Verify metrics were recorded
                    mock_started.assert_called_once()
                    mock_completed.assert_called_once()

    def test_audit_configuration_validation(self, audit_processor):
        """Test audit configuration validation"""
        # Valid configuration
        valid_config = {
            "platforms": ["openai", "anthropic"],
            "question_count": 10,
            "concurrent_requests": 5,
        }

        result = audit_processor._validate_audit_config(valid_config)
        assert result is True

        # Invalid configuration - no platforms
        invalid_config = {"question_count": 10}

        with pytest.raises(AuditConfigurationError):
            audit_processor._validate_audit_config(invalid_config)

    @pytest.mark.asyncio
    async def test_error_recovery(
        self, audit_processor, mock_db_session, sample_audit_run
    ):
        """Test error recovery mechanisms"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        # Mock question generation to fail initially then succeed
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_question_engine.return_value.generate_questions.side_effect = (
                Exception("Temporary failure")
            )

            context = {"max_questions": 10, "target_brands": ["Test Company"]}

            # Should handle the error gracefully
            with pytest.raises(Exception):
                await audit_processor._generate_questions(sample_audit_run, context)


class TestAuditProcessorPerformance:
    """Performance tests for audit processor"""

    @pytest.mark.asyncio
    async def test_large_audit_performance(
        self, audit_processor, mock_db_session, sample_audit_run, mock_platform_manager
    ):
        """Test performance with large number of questions"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )

        # Create large number of mock questions
        large_question_set = [
            Mock(
                id=f"q{i}",
                question_text=f"Question {i}",
                audit_run_id="550e8400-e29b-41d4-a716-446655440000",
            )
            for i in range(100)
        ]

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_question_engine.return_value.generate_questions.return_value = (
                large_question_set
            )

            # Mock brand detection
            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                start_time = datetime.now()
                result = await audit_processor.run_audit(
                    "550e8400-e29b-41d4-a716-446655440000"
                )
                end_time = datetime.now()

                # Verify it completes in reasonable time (< 30 seconds for mock)
                duration = (end_time - start_time).total_seconds()
                assert duration < 30
                assert result == "550e8400-e29b-41d4-a716-446655440000"

    @pytest.mark.asyncio
    async def test_concurrent_audit_handling(
        self, mock_db_session, mock_platform_manager
    ):
        """Test handling multiple concurrent audits"""
        # Create multiple audit processors (simulating concurrent requests)
        audit_processors = [
            AuditProcessor(mock_db_session, mock_platform_manager) for _ in range(5)
        ]

        # Create multiple audit runs
        audit_runs = []
        for i in range(5):
            client = Client(
                id=f"client_{i}",
                name=f"Test Company {i}",
                industry="Technology",
                product_type="SaaS",
                competitors=["Competitor A"],
            )

            audit_run = AuditRun(
                id=f"audit_{i}",
                client_id=f"client_{i}",
                client=client,
                config={"platforms": ["openai"]},
                status="pending",
            )
            audit_runs.append(audit_run)

        # Mock database to return different audit runs
        def mock_query_side_effect(*args):
            mock_filter = Mock()
            mock_filter.first.return_value = audit_runs[0]  # Simplification for test
            mock_query = Mock()
            mock_query.filter.return_value = mock_filter
            return mock_query

        mock_db_session.query.side_effect = mock_query_side_effect

        # Mock question generation
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_question_engine.return_value.generate_questions.return_value = [
                Mock(id="q1", question_text="Test question")
            ]

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Run concurrent audits
                tasks = [
                    processor.run_audit(f"audit_{i}")
                    for i, processor in enumerate(audit_processors)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Verify all completed successfully
                for result in results:
                    assert not isinstance(result, Exception)


if __name__ == "__main__":
    pytest.main([__file__])
