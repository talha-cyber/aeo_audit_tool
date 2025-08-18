"""
Tests for the progress tracking system.

This module tests the comprehensive progress tracking functionality including:
- Progress initialization and tracking
- Stage management
- Metrics calculation
- Time estimation
- Error handling
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from app.models.audit import AuditRun
from app.services.progress_tracker import (
    ProgressSnapshot,
    ProgressStage,
    ProgressTracker,
    create_progress_tracker,
)


class TestProgressTracker:
    """Test cases for the ProgressTracker class"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock(spec=Session)
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def sample_audit_run(self):
        """Sample audit run for testing"""
        return AuditRun(
            id="audit_123",
            client_id="client_123",
            status="pending",
            total_questions=10,
            processed_questions=0,
        )

    @pytest.fixture
    def progress_tracker(self, mock_db_session, sample_audit_run):
        """Create progress tracker instance"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            sample_audit_run
        )
        return ProgressTracker(mock_db_session, "audit_123")

    @pytest.mark.asyncio
    async def test_progress_tracker_initialization(self, progress_tracker):
        """Test progress tracker initialization"""
        await progress_tracker.initialize()

        assert progress_tracker.audit_run_id == "audit_123"
        assert progress_tracker.current_stage == ProgressStage.INITIALIZING
        assert progress_tracker.start_time is not None

    @pytest.mark.asyncio
    async def test_stage_progression(self, progress_tracker):
        """Test stage progression and timing"""
        await progress_tracker.initialize()

        # Start question generation
        await progress_tracker.start_stage(ProgressStage.GENERATING_QUESTIONS)
        assert progress_tracker.current_stage == ProgressStage.GENERATING_QUESTIONS

        # Complete question generation
        await progress_tracker.complete_stage(ProgressStage.GENERATING_QUESTIONS)

        # Start question processing
        await progress_tracker.start_stage(ProgressStage.PROCESSING_QUESTIONS)
        assert progress_tracker.current_stage == ProgressStage.PROCESSING_QUESTIONS

        # Verify stage timing is recorded
        snapshot = await progress_tracker.get_progress_snapshot()
        assert ProgressStage.GENERATING_QUESTIONS.value in snapshot.stages
        assert (
            snapshot.stages[ProgressStage.GENERATING_QUESTIONS.value]["status"]
            == "completed"
        )

    @pytest.mark.asyncio
    async def test_progress_calculation(self, progress_tracker, sample_audit_run):
        """Test progress percentage calculation"""
        sample_audit_run.total_questions = 20
        sample_audit_run.processed_questions = 5

        await progress_tracker.initialize()
        await progress_tracker.update_question_progress(5, 20)

        snapshot = await progress_tracker.get_progress_snapshot()
        assert snapshot.overall_progress_percent == 25.0  # 5/20 = 25%
        assert snapshot.questions_processed == 5
        assert snapshot.total_questions == 20

    @pytest.mark.asyncio
    async def test_time_estimation(self, progress_tracker, sample_audit_run):
        """Test estimated completion time calculation"""
        sample_audit_run.total_questions = 10
        sample_audit_run.processed_questions = 5

        await progress_tracker.initialize()

        # Simulate some processing time
        await asyncio.sleep(0.1)
        await progress_tracker.update_question_progress(5, 10)

        snapshot = await progress_tracker.get_progress_snapshot()

        # Should have an estimated completion time
        assert snapshot.estimated_completion is not None
        assert snapshot.estimated_completion > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_platform_tracking(self, progress_tracker):
        """Test platform activity tracking"""
        await progress_tracker.initialize()

        # Record platform activity
        await progress_tracker.record_platform_activity("openai", "active")
        await progress_tracker.record_platform_activity("anthropic", "active")

        snapshot = await progress_tracker.get_progress_snapshot()
        assert snapshot.platforms_active == 2

    @pytest.mark.asyncio
    async def test_brand_detection_tracking(self, progress_tracker):
        """Test brand detection tracking"""
        await progress_tracker.initialize()

        # Record brand detections
        await progress_tracker.record_brand_detection(
            {"brand": "Company A", "count": 3}
        )
        await progress_tracker.record_brand_detection(
            {"brand": "Company B", "count": 2}
        )

        snapshot = await progress_tracker.get_progress_snapshot()
        assert snapshot.brands_detected == 2

    @pytest.mark.asyncio
    async def test_performance_metrics(self, progress_tracker):
        """Test performance metrics tracking"""
        await progress_tracker.initialize()

        # Record performance metrics
        await progress_tracker.record_performance_metric("avg_response_time", 1500.0)
        await progress_tracker.record_performance_metric("total_tokens", 50000)

        snapshot = await progress_tracker.get_progress_snapshot()
        assert "avg_response_time" in snapshot.performance_metrics
        assert "total_tokens" in snapshot.performance_metrics
        assert snapshot.performance_metrics["avg_response_time"] == 1500.0

    @pytest.mark.asyncio
    async def test_error_recording(self, progress_tracker):
        """Test error recording and handling"""
        await progress_tracker.initialize()

        # Record an error
        await progress_tracker.record_error(
            stage=ProgressStage.PROCESSING_QUESTIONS,
            error_message="Platform timeout",
            error_type="PlatformTimeoutError",
        )

        snapshot = await progress_tracker.get_progress_snapshot()

        # Check that error is recorded in stage data
        stage_data = snapshot.stages.get(ProgressStage.PROCESSING_QUESTIONS.value, {})
        assert "error_message" in stage_data
        assert stage_data["error_message"] == "Platform timeout"

    @pytest.mark.asyncio
    async def test_stage_failure_handling(self, progress_tracker):
        """Test handling of stage failures"""
        await progress_tracker.initialize()

        # Start and fail a stage
        await progress_tracker.start_stage(ProgressStage.GENERATING_QUESTIONS)
        await progress_tracker.fail_stage(
            ProgressStage.GENERATING_QUESTIONS, "Question generation failed"
        )

        snapshot = await progress_tracker.get_progress_snapshot()
        stage_data = snapshot.stages[ProgressStage.GENERATING_QUESTIONS.value]

        assert stage_data["status"] == "failed"
        assert "error_message" in stage_data

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, progress_tracker):
        """Test concurrent progress updates"""
        await progress_tracker.initialize()

        # Simulate concurrent updates
        async def update_progress(value):
            await progress_tracker.update_question_progress(value, 10)
            await progress_tracker.record_performance_metric(
                f"metric_{value}", value * 100
            )

        # Run concurrent updates
        await asyncio.gather(*[update_progress(i) for i in range(1, 6)])

        snapshot = await progress_tracker.get_progress_snapshot()

        # Verify final state is consistent
        assert snapshot.questions_processed == 5  # Last update
        assert len(snapshot.performance_metrics) == 5

    @pytest.mark.asyncio
    async def test_database_persistence(
        self, progress_tracker, mock_db_session, sample_audit_run
    ):
        """Test that progress is persisted to database"""
        await progress_tracker.initialize()
        await progress_tracker.update_question_progress(5, 10)

        # Verify database was updated
        assert sample_audit_run.processed_questions == 5
        assert sample_audit_run.total_questions == 10
        assert sample_audit_run.progress_data is not None
        assert mock_db_session.commit.called

    @pytest.mark.asyncio
    async def test_finalize_tracking(self, progress_tracker, sample_audit_run):
        """Test finalizing progress tracking"""
        await progress_tracker.initialize()
        await progress_tracker.start_stage(ProgressStage.PROCESSING_QUESTIONS)

        # Finalize with success
        await progress_tracker.finalize_tracking(
            final_stage=ProgressStage.COMPLETED, success=True
        )

        assert sample_audit_run.status == "completed"
        assert sample_audit_run.completed_at is not None

        snapshot = await progress_tracker.get_progress_snapshot()
        assert snapshot.current_stage == ProgressStage.COMPLETED
        assert snapshot.overall_progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_finalize_tracking_with_failure(
        self, progress_tracker, sample_audit_run
    ):
        """Test finalizing progress tracking with failure"""
        await progress_tracker.initialize()

        # Finalize with failure
        await progress_tracker.finalize_tracking(
            final_stage=ProgressStage.FAILED, success=False
        )

        assert sample_audit_run.status == "failed"
        assert sample_audit_run.completed_at is not None

        snapshot = await progress_tracker.get_progress_snapshot()
        assert snapshot.current_stage == ProgressStage.FAILED


class TestProgressSnapshot:
    """Test cases for the ProgressSnapshot class"""

    def test_progress_snapshot_creation(self):
        """Test progress snapshot creation and serialization"""
        snapshot = ProgressSnapshot(
            current_stage=ProgressStage.PROCESSING_QUESTIONS,
            overall_progress_percent=50.0,
            questions_generated=10,
            questions_processed=5,
            total_questions=10,
            platforms_active=2,
            brands_detected=3,
            estimated_completion=datetime.now(timezone.utc) + timedelta(minutes=30),
            stages={
                "generating_questions": {
                    "status": "completed",
                    "started_at": datetime.now(timezone.utc),
                    "completed_at": datetime.now(timezone.utc),
                    "duration_seconds": 45.5,
                }
            },
            performance_metrics={"avg_response_time": 1200.0, "total_tokens": 25000},
        )

        # Test serialization
        snapshot_dict = snapshot.to_dict()

        assert snapshot_dict["current_stage"] == "processing_questions"
        assert snapshot_dict["overall_progress_percent"] == 50.0
        assert "stages" in snapshot_dict
        assert "performance_metrics" in snapshot_dict

    def test_progress_snapshot_validation(self):
        """Test progress snapshot validation"""
        # Valid snapshot
        snapshot = ProgressSnapshot(
            current_stage=ProgressStage.INITIALIZING,
            overall_progress_percent=0.0,
            questions_generated=0,
            questions_processed=0,
            total_questions=0,
            platforms_active=0,
            brands_detected=0,
        )

        assert snapshot.current_stage == ProgressStage.INITIALIZING
        assert snapshot.overall_progress_percent == 0.0


class TestProgressTrackerFactory:
    """Test cases for progress tracker factory functions"""

    @pytest.mark.asyncio
    async def test_create_progress_tracker(self):
        """Test progress tracker factory function"""
        mock_db = Mock(spec=Session)
        mock_audit_run = AuditRun(id="audit_123", status="pending")
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_audit_run
        )

        tracker = create_progress_tracker(mock_db, "audit_123")

        assert isinstance(tracker, ProgressTracker)
        assert tracker.audit_run_id == "audit_123"

    @pytest.mark.asyncio
    async def test_create_progress_tracker_not_found(self):
        """Test progress tracker factory with non-existent audit"""
        mock_db = Mock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Audit run not found"):
            create_progress_tracker(mock_db, "nonexistent_audit")


class TestProgressTrackerIntegration:
    """Integration tests for progress tracker"""

    @pytest.mark.asyncio
    async def test_full_audit_progress_flow(self):
        """Test complete audit progress tracking flow"""
        mock_db = Mock(spec=Session)
        audit_run = AuditRun(
            id="audit_123", status="pending", total_questions=20, processed_questions=0
        )
        mock_db.query.return_value.filter.return_value.first.return_value = audit_run

        tracker = ProgressTracker(mock_db, "audit_123")

        # Initialize
        await tracker.initialize()

        # Generate questions
        await tracker.start_stage(ProgressStage.GENERATING_QUESTIONS)
        await tracker.update_question_progress(0, 20)
        await tracker.complete_stage(ProgressStage.GENERATING_QUESTIONS)

        # Process questions
        await tracker.start_stage(ProgressStage.PROCESSING_QUESTIONS)

        # Simulate processing questions in batches
        for processed in [5, 10, 15, 20]:
            await tracker.update_question_progress(processed, 20)
            await tracker.record_performance_metric(
                "avg_response_time", 1000 + processed * 50
            )

            if processed == 10:
                await tracker.record_platform_activity("openai", "active")
                await tracker.record_brand_detection({"brand": "Company A", "count": 2})

        await tracker.complete_stage(ProgressStage.PROCESSING_QUESTIONS)

        # Brand detection
        await tracker.start_stage(ProgressStage.DETECTING_BRANDS)
        await tracker.record_brand_detection({"brand": "Company B", "count": 1})
        await tracker.complete_stage(ProgressStage.DETECTING_BRANDS)

        # Finalize
        await tracker.finalize_tracking(ProgressStage.COMPLETED, success=True)

        # Verify final state
        final_snapshot = await tracker.get_progress_snapshot()

        assert final_snapshot.current_stage == ProgressStage.COMPLETED
        assert final_snapshot.overall_progress_percent == 100.0
        assert final_snapshot.questions_processed == 20
        assert final_snapshot.brands_detected == 2
        assert audit_run.status == "completed"

    @pytest.mark.asyncio
    async def test_progress_tracker_with_errors(self):
        """Test progress tracker handling errors gracefully"""
        mock_db = Mock(spec=Session)
        audit_run = AuditRun(id="audit_123", status="pending")
        mock_db.query.return_value.filter.return_value.first.return_value = audit_run

        # Mock database commit to fail
        mock_db.commit.side_effect = Exception("Database error")

        tracker = ProgressTracker(mock_db, "audit_123")

        # Should handle database errors gracefully
        await tracker.initialize()

        # Updates should not raise exceptions even if database fails
        await tracker.update_question_progress(5, 10)
        await tracker.record_performance_metric("test_metric", 100)

        # Verify rollback is called on error
        assert mock_db.rollback.called


if __name__ == "__main__":
    pytest.main([__file__])
