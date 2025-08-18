"""
Integration tests for the complete audit processor system.

This module tests the full audit processor workflow including:
- End-to-end audit execution
- Component integration
- Error handling and recovery
- Performance under load
- Real-world scenarios
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.orm import Session

from app.models.audit import AuditRun, Client
from app.services.audit_processor import AuditProcessor
from app.services.platform_manager import PlatformManager
from app.utils.error_handler import AuditConfigurationError, PlatformError


class TestAuditProcessorIntegration:
    """Integration tests for the complete audit processor system"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session with transaction support"""
        session = Mock(spec=Session)
        session.query.return_value.filter.return_value.first.return_value = None
        session.commit = Mock()
        session.rollback = Mock()
        session.add = Mock()
        session.close = Mock()
        return session

    @pytest.fixture
    def mock_platform_manager(self):
        """Mock platform manager with realistic behavior"""
        manager = Mock(spec=PlatformManager)
        manager.get_available_platforms.return_value = ["openai", "anthropic", "google"]

        # Create different mock platforms with different behaviors
        platforms = {}

        # OpenAI platform mock
        openai_platform = Mock()
        openai_platform.query = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "OpenAI response about competitive landscape"
                        }
                    }
                ]
            }
        )
        openai_platform.extract_text_response.return_value = (
            "OpenAI response about competitive landscape"
        )
        openai_platform.health_check = AsyncMock(return_value=True)
        platforms["openai"] = openai_platform

        # Anthropic platform mock
        anthropic_platform = Mock()
        anthropic_platform.query = AsyncMock(
            return_value={
                "content": [{"text": "Anthropic analysis of brand positioning"}]
            }
        )
        anthropic_platform.extract_text_response.return_value = (
            "Anthropic analysis of brand positioning"
        )
        anthropic_platform.health_check = AsyncMock(return_value=True)
        platforms["anthropic"] = anthropic_platform

        # Google platform mock (sometimes fails)
        google_platform = Mock()
        google_platform.query = AsyncMock(
            side_effect=[
                PlatformError("google", "Rate limit exceeded"),
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Google AI insights"}]}}
                    ]
                },
            ]
        )
        google_platform.extract_text_response.return_value = "Google AI insights"
        google_platform.health_check = AsyncMock(return_value=True)
        platforms["google"] = google_platform

        def get_platform_side_effect(name):
            return platforms.get(name)

        manager.get_platform.side_effect = get_platform_side_effect
        return manager

    @pytest.fixture
    def complete_audit_run(self):
        """Complete audit run with client and configuration"""
        client = Client(
            id="client_456",
            name="TechCorp Inc",
            industry="Technology",
            product_type="Enterprise Software",
            competitors=["Salesforce", "Microsoft", "Oracle", "SAP"],
        )

        audit_run = AuditRun(
            id="audit_integration_test",
            client_id="client_456",
            client=client,
            config={
                "platforms": ["openai", "anthropic", "google"],
                "question_count": 15,
                "concurrent_requests": 3,
                "include_competitive_analysis": True,
                "brand_detection_enabled": True,
            },
            status="pending",
            total_questions=0,
            processed_questions=0,
        )

        return audit_run

    @pytest.mark.asyncio
    async def test_complete_audit_workflow(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test complete audit workflow from start to finish"""
        # Setup database mock
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        # Mock question generation
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id=f"q_{i}",
                    question_text=f"Question {i}: How does TechCorp compare to {competitor}?",
                    category="comparison",
                    priority_score=0.8,
                    target_brand="TechCorp Inc",
                    audit_run_id="audit_integration_test",
                )
                for i, competitor in enumerate(complete_audit_run.client.competitors, 1)
            ]

            # Add some recommendation questions
            mock_questions.extend(
                [
                    Mock(
                        id=f"q_{i+5}",
                        question_text=f"Recommendation question {i}",
                        category="recommendation",
                        priority_score=0.7,
                        target_brand="TechCorp Inc",
                        audit_run_id="audit_integration_test",
                    )
                    for i in range(3)
                ]
            )

            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            # Mock brand detection
            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_results = [
                    Mock(
                        to_summary_dict=Mock(
                            return_value={
                                "brands": [
                                    {
                                        "name": "TechCorp Inc",
                                        "mentions": 2,
                                        "sentiment": "positive",
                                    },
                                    {
                                        "name": "Salesforce",
                                        "mentions": 1,
                                        "sentiment": "neutral",
                                    },
                                ],
                                "total_mentions": 3,
                                "sentiment_score": 0.6,
                            }
                        )
                    )
                    for _ in mock_questions
                ]

                mock_brand_detection.return_value.analyze_response.side_effect = (
                    mock_detection_results
                )

                # Create audit processor and run the audit
                audit_processor = AuditProcessor(mock_db_session, mock_platform_manager)

                # Execute the audit
                result = await audit_processor.run_audit("audit_integration_test")

                # Verify successful completion
                assert result == "audit_integration_test"

                # Verify audit run status was updated
                assert complete_audit_run.status == "completed"
                assert complete_audit_run.started_at is not None
                assert complete_audit_run.completed_at is not None

                # Verify questions were generated
                mock_question_engine.return_value.generate_questions.assert_called_once()

                # Verify database interactions
                assert mock_db_session.add.call_count >= len(
                    mock_questions
                )  # Questions + responses
                assert mock_db_session.commit.call_count >= 1

    @pytest.mark.asyncio
    async def test_audit_with_platform_failures_and_recovery(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test audit handling platform failures and recovery"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        # Configure platform manager with some failing platforms
        mock_platform_manager.get_available_platforms.return_value = [
            "openai",
            "anthropic",
            "google",
        ]

        # Make Google platform always fail
        google_platform = mock_platform_manager.get_platform("google")
        google_platform.query = AsyncMock(
            side_effect=PlatformError("google", "Service unavailable")
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id=f"q_{i}",
                    question_text=f"Test question {i}",
                    category="comparison",
                    audit_run_id="audit_integration_test",
                )
                for i in range(5)
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                audit_processor = AuditProcessor(mock_db_session, mock_platform_manager)

                # Execute audit - should handle platform failures gracefully
                result = await audit_processor.run_audit("audit_integration_test")

                # Should still complete successfully using working platforms
                assert result == "audit_integration_test"
                assert complete_audit_run.status == "completed"

                # Verify platform errors were handled (responses from working platforms only)
                openai_platform = mock_platform_manager.get_platform("openai")
                anthropic_platform = mock_platform_manager.get_platform("anthropic")

                # Both working platforms should have been called
                assert openai_platform.query.call_count > 0
                assert anthropic_platform.query.call_count > 0

    @pytest.mark.asyncio
    async def test_audit_with_progress_tracking(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test audit with comprehensive progress tracking"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id=f"q_{i}",
                    question_text=f"Question {i}",
                    audit_run_id="audit_integration_test",
                )
                for i in range(3)
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Mock progress tracker
                with patch(
                    "app.services.audit_processor.create_progress_tracker"
                ) as mock_create_tracker:
                    mock_tracker = Mock()
                    mock_tracker.initialize = AsyncMock()
                    mock_tracker.start_stage = AsyncMock()
                    mock_tracker.complete_stage = AsyncMock()
                    mock_tracker.update_question_progress = AsyncMock()
                    mock_tracker.finalize_tracking = AsyncMock()
                    mock_create_tracker.return_value = mock_tracker

                    audit_processor = AuditProcessor(
                        mock_db_session, mock_platform_manager
                    )
                    await audit_processor.run_audit("audit_integration_test")

                    # Verify progress tracking was used
                    mock_tracker.initialize.assert_called_once()
                    mock_tracker.finalize_tracking.assert_called_once()
                    assert mock_tracker.start_stage.call_count >= 1
                    assert mock_tracker.complete_stage.call_count >= 1

    @pytest.mark.asyncio
    async def test_audit_with_context_logging(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test audit with proper context logging"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id="q_1",
                    question_text="Test question",
                    audit_run_id="audit_integration_test",
                )
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Mock context logging
                with patch(
                    "app.services.audit_processor.add_audit_context"
                ) as mock_add_context:
                    mock_context_manager = Mock()
                    mock_context_manager.__enter__ = Mock(
                        return_value=mock_context_manager
                    )
                    mock_context_manager.__exit__ = Mock(return_value=None)
                    mock_add_context.return_value = mock_context_manager

                    audit_processor = AuditProcessor(
                        mock_db_session, mock_platform_manager
                    )
                    await audit_processor.run_audit("audit_integration_test")

                    # Verify context logging was used
                    mock_add_context.assert_called_with(
                        audit_run_id="audit_integration_test"
                    )

    @pytest.mark.asyncio
    async def test_audit_configuration_validation(
        self, mock_db_session, mock_platform_manager
    ):
        """Test audit configuration validation"""
        # Create audit run with invalid configuration
        invalid_client = Client(
            id="client_invalid",
            name="Invalid Corp",
            industry="Technology",
            product_type="Software",
            competitors=[],  # No competitors
        )

        invalid_audit_run = AuditRun(
            id="audit_invalid",
            client_id="client_invalid",
            client=invalid_client,
            config={},  # Empty configuration
            status="pending",
        )

        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            invalid_audit_run
        )

        audit_processor = AuditProcessor(mock_db_session, mock_platform_manager)

        # Should raise configuration error
        with pytest.raises(AuditConfigurationError):
            await audit_processor.run_audit("audit_invalid")

    @pytest.mark.asyncio
    async def test_concurrent_audit_processing(
        self, mock_db_session, mock_platform_manager
    ):
        """Test concurrent processing of multiple audits"""
        # Create multiple audit runs
        audit_runs = []
        for i in range(3):
            client = Client(
                id=f"client_{i}",
                name=f"Company {i}",
                industry="Technology",
                product_type="Software",
                competitors=["Competitor A", "Competitor B"],
            )

            audit_run = AuditRun(
                id=f"audit_concurrent_{i}",
                client_id=f"client_{i}",
                client=client,
                config={"platforms": ["openai"], "question_count": 2},
                status="pending",
            )
            audit_runs.append(audit_run)

        # Mock database to return different audit runs
        def mock_query_side_effect(*args):
            # This is a simplified mock - in reality each query would return different audit runs
            return Mock(
                filter=Mock(return_value=Mock(first=Mock(return_value=audit_runs[0])))
            )

        mock_db_session.query.side_effect = mock_query_side_effect

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(id="q_1", question_text="Test question", audit_run_id="test")
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Create multiple audit processors
                audit_processors = [
                    AuditProcessor(mock_db_session, mock_platform_manager)
                    for _ in range(3)
                ]

                # Run concurrent audits
                tasks = [
                    processor.run_audit(f"audit_concurrent_{i}")
                    for i, processor in enumerate(audit_processors)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # All should complete successfully
                for result in results:
                    assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_large_scale_audit_performance(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test performance with large-scale audit"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        # Create large number of questions
        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            large_question_set = [
                Mock(
                    id=f"q_{i}",
                    question_text=f"Performance test question {i}",
                    category="comparison" if i % 2 == 0 else "recommendation",
                    audit_run_id="audit_integration_test",
                )
                for i in range(50)  # Large question set
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                large_question_set
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                audit_processor = AuditProcessor(mock_db_session, mock_platform_manager)

                start_time = datetime.now()
                result = await audit_processor.run_audit("audit_integration_test")
                end_time = datetime.now()

                # Verify completion
                assert result == "audit_integration_test"

                # Performance should be reasonable (under 60 seconds for mock execution)
                duration = (end_time - start_time).total_seconds()
                assert duration < 60, f"Audit took too long: {duration} seconds"

                # Verify all questions were processed
                assert complete_audit_run.total_questions == len(large_question_set)

    @pytest.mark.asyncio
    async def test_audit_metrics_integration(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test integration with audit metrics system"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id="q_1",
                    question_text="Test question",
                    audit_run_id="audit_integration_test",
                )
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                # Mock metrics
                with patch(
                    "app.services.audit_processor.get_audit_metrics"
                ) as mock_get_metrics:
                    mock_metrics = Mock()
                    mock_get_metrics.return_value = mock_metrics

                    audit_processor = AuditProcessor(
                        mock_db_session, mock_platform_manager
                    )
                    await audit_processor.run_audit("audit_integration_test")

                    # Verify metrics were recorded
                    assert mock_metrics.record_audit_started.called
                    assert mock_metrics.record_audit_completed.called

    @pytest.mark.asyncio
    async def test_error_recovery_and_partial_completion(
        self, mock_db_session, mock_platform_manager, complete_audit_run
    ):
        """Test error recovery and partial audit completion"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            complete_audit_run
        )

        with patch(
            "app.services.audit_processor.QuestionEngine"
        ) as mock_question_engine:
            mock_questions = [
                Mock(
                    id=f"q_{i}",
                    question_text=f"Question {i}",
                    audit_run_id="audit_integration_test",
                )
                for i in range(5)
            ]
            mock_question_engine.return_value.generate_questions.return_value = (
                mock_questions
            )

            # Configure platforms to fail intermittently
            openai_platform = mock_platform_manager.get_platform("openai")
            openai_platform.query = AsyncMock(
                side_effect=[
                    {"choices": [{"message": {"content": "Success 1"}}]},
                    PlatformError("openai", "Temporary failure"),
                    {"choices": [{"message": {"content": "Success 2"}}]},
                    PlatformError("openai", "Another failure"),
                    {"choices": [{"message": {"content": "Success 3"}}]},
                ]
            )

            with patch(
                "app.services.audit_processor.BrandDetectionEngine"
            ) as mock_brand_detection:
                mock_detection_result = Mock()
                mock_detection_result.to_summary_dict.return_value = {"brands": []}
                mock_brand_detection.return_value.analyze_response.return_value = (
                    mock_detection_result
                )

                audit_processor = AuditProcessor(mock_db_session, mock_platform_manager)

                # Should complete despite some failures
                result = await audit_processor.run_audit("audit_integration_test")

                assert result == "audit_integration_test"
                # Some responses should have been generated despite failures
                assert mock_db_session.add.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
