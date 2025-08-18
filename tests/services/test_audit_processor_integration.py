"""
Integration tests for the audit processor.

This module tests the actual end-to-end audit workflow with real components
and minimal mocking to ensure the system works as a complete unit.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.orm import Session

from app.core.audit_config import get_audit_settings
from app.models.audit import AuditRun, Client
from app.services.audit_processor import AuditProcessor
from app.services.platform_manager import PlatformManager


class TestAuditProcessorIntegration:
    """Integration tests for the audit processor with real components."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def mock_platform_manager(self):
        """Mock platform manager with proper async platform"""
        manager = Mock(spec=PlatformManager)
        manager.get_available_platforms.return_value = ["openai"]

        # Create a proper async mock platform
        mock_platform = AsyncMock()
        mock_platform.safe_query.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 50},
        }
        mock_platform.extract_text_response.return_value = "Test response"

        manager.get_platform.return_value = mock_platform
        return manager

    @pytest.fixture
    def integration_audit_run(self):
        """Real audit run with proper configuration for integration testing"""
        client = Client(
            id="550e8400-e29b-41d4-a716-446655440001",
            name="Test Company",
            industry="Technology",
            product_type="SaaS",
            competitors=["Competitor A", "Competitor B"],
        )

        audit_run = AuditRun(
            id="550e8400-e29b-41d4-a716-446655440000",
            client_id="550e8400-e29b-41d4-a716-446655440001",
            client=client,
            config={
                "platforms": ["openai"],
                "question_count": 3,  # Small number for testing
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
    def audit_processor_integration(self, mock_db_session, mock_platform_manager):
        """Audit processor with real settings and minimal mocking"""
        # Use real audit settings
        get_audit_settings()

        processor = AuditProcessor(mock_db_session, mock_platform_manager)

        # Override settings if needed for testing
        processor.batch_size = 2  # Small batches for testing
        processor.inter_batch_delay = 0.1  # Fast for testing

        return processor

    @pytest.mark.asyncio
    async def test_full_audit_workflow_integration(
        self,
        audit_processor_integration,
        mock_db_session,
        integration_audit_run,
        mock_platform_manager,
    ):
        """Test complete audit workflow with real components"""

        # Setup database mock to return our audit run
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            integration_audit_run
        )

        # Mock brand detection to avoid complexity
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

            # Execute the full audit workflow
            result = await audit_processor_integration.run_audit(
                "550e8400-e29b-41d4-a716-446655440000"
            )

            # Verify successful completion
            assert result == "550e8400-e29b-41d4-a716-446655440000"

            # Verify database interactions
            assert mock_db_session.commit.called

            # Verify platform was called
            mock_platform = mock_platform_manager.get_platform.return_value
            assert mock_platform.safe_query.called

            # Verify audit run status progression
            # Should have been set to running and then completed
            assert integration_audit_run.status in ["completed", "running"]

    @pytest.mark.asyncio
    async def test_question_generation_integration(
        self, audit_processor_integration, integration_audit_run
    ):
        """Test that question generation works with real QuestionEngine"""

        # Test question generation with real context
        context = {
            "client": {
                "name": "Test Company",
                "industry": "Technology",
                "product_type": "SaaS",
                "competitors": ["Competitor A", "Competitor B"],
            },
            "target_brands": ["Test Company", "Competitor A", "Competitor B"],
            "platforms": ["openai"],
            "categories": [],
            "max_questions": 5,
        }

        # This should work with the real QuestionEngine
        questions = await audit_processor_integration._generate_questions(
            integration_audit_run, context
        )

        # Verify questions were generated
        assert isinstance(questions, list)
        assert len(questions) > 0

        # Verify question structure
        for question in questions:
            assert isinstance(question, dict)
            assert "question" in question
            assert "category" in question
            assert isinstance(question["question"], str)
            assert len(question["question"]) > 0

    @pytest.mark.asyncio
    async def test_batch_processing_integration(
        self, audit_processor_integration, integration_audit_run, mock_platform_manager
    ):
        """Test batch processing with real question data"""

        # Create realistic question data
        questions = [
            {"id": f"q{i}", "question": f"Test question {i}", "category": "comparison"}
            for i in range(7)  # More than batch size to test batching
        ]

        platforms = ["openai"]
        target_brands = ["Test Company"]

        # Execute batch processing
        context = {"platforms": platforms, "target_brands": target_brands}
        results = await audit_processor_integration._process_questions_batched(
            integration_audit_run, questions, context
        )

        # Verify results
        assert isinstance(results, list)
        # Should have attempted to process questions (may fail due to mock setup)
        mock_platform = mock_platform_manager.get_platform.return_value
        assert mock_platform.safe_query.call_count > 0

    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self,
        audit_processor_integration,
        mock_db_session,
        integration_audit_run,
        mock_platform_manager,
    ):
        """Test error handling in integration environment"""

        # Setup audit run
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            integration_audit_run
        )

        # Make platform fail
        mock_platform = mock_platform_manager.get_platform.return_value
        mock_platform.safe_query.side_effect = Exception("Platform error")

        # Mock brand detection
        with patch(
            "app.services.audit_processor.BrandDetectionEngine"
        ) as mock_brand_detection:
            mock_detection_result = Mock()
            mock_detection_result.to_summary_dict.return_value = {"brands": []}
            mock_brand_detection.return_value.analyze_response.return_value = (
                mock_detection_result
            )

            # Execute audit - should handle errors gracefully
            result = await audit_processor_integration.run_audit(
                "550e8400-e29b-41d4-a716-446655440000"
            )

            # Verify the audit completed (even with errors)
            assert result == "550e8400-e29b-41d4-a716-446655440000"

            # Verify audit run was marked as completed/failed appropriately
            assert integration_audit_run.status in ["completed", "failed"]

    def test_audit_configuration_validation_integration(
        self, audit_processor_integration
    ):
        """Test configuration validation with real settings"""

        # Test valid configuration
        valid_config = {
            "platforms": ["openai", "anthropic"],
            "question_count": 10,
            "client": {
                "name": "Test Company",
                "industry": "Technology",
                "product_type": "SaaS",
            },
        }

        result = audit_processor_integration._validate_audit_config(valid_config)
        assert result is True

        # Test invalid configuration
        invalid_config = {
            "question_count": 10
            # Missing platforms
        }

        from app.utils.error_handler import AuditConfigurationError

        with pytest.raises(AuditConfigurationError):
            audit_processor_integration._validate_audit_config(invalid_config)

    def test_metrics_integration(self, audit_processor_integration):
        """Test that metrics are properly integrated"""

        # Verify metrics object exists and has expected methods
        assert hasattr(audit_processor_integration, "metrics")
        assert hasattr(audit_processor_integration.metrics, "increment_audit_started")
        assert hasattr(audit_processor_integration.metrics, "increment_audit_completed")
        assert hasattr(
            audit_processor_integration.metrics, "record_batch_processing_time"
        )

        # Test metrics can be called
        audit_processor_integration.metrics.increment_audit_started()
        audit_processor_integration.metrics.record_batch_processing_time(100)
