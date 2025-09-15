"""
Tests for the main scheduler engine.

Tests scheduler lifecycle, job management, execution coordination,
and integration between all scheduling system components.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from app.models.scheduling import (
    JobExecutionStatus,
    ScheduledJob,
    ScheduledJobStatus,
    TriggerType,
)
from app.services.scheduling.engine import (
    JobDefinition,
    SchedulerEngine,
    SchedulerStatus,
)
from app.services.scheduling.execution_manager import ExecutionManager
from app.services.scheduling.repository import SchedulingRepository
from app.services.scheduling.triggers.factory import TriggerFactory


@pytest.fixture
def mock_repository():
    """Create mock scheduling repository"""
    repo = Mock(spec=SchedulingRepository)
    return repo


@pytest.fixture
def mock_execution_manager():
    """Create mock execution manager"""
    manager = Mock(spec=ExecutionManager)
    manager.get_active_executions = Mock(return_value=[])
    manager.get_health_status = Mock(
        return_value={"is_healthy": True, "active_executions": 0, "stuck_executions": 0}
    )
    return manager


@pytest.fixture
def mock_trigger_factory():
    """Create mock trigger factory"""
    factory = Mock(spec=TriggerFactory)
    return factory


@pytest.fixture
def scheduler_engine(mock_repository, mock_execution_manager, mock_trigger_factory):
    """Create scheduler engine with mocked dependencies"""
    engine = SchedulerEngine(
        repository=mock_repository,
        execution_manager=mock_execution_manager,
        trigger_factory=mock_trigger_factory,
        poll_interval=1,  # Short interval for testing
    )
    return engine


@pytest.fixture
def sample_job_definition():
    """Create sample job definition for testing"""
    return JobDefinition(
        name="test_job",
        job_type="audit_execution",
        trigger_config={"trigger_type": "cron", "expression": "0 9 * * *"},
        job_data={"client_id": "123"},
        description="Test job for unit testing",
        priority=5,
        timeout_seconds=3600,
        max_retries=3,
    )


class TestSchedulerEngineInitialization:
    """Test suite for SchedulerEngine initialization"""

    def test_scheduler_engine_initialization(self):
        """Test scheduler engine initialization with defaults"""
        engine = SchedulerEngine()

        assert engine.status == SchedulerStatus.STOPPED
        assert engine.scheduler_id is not None
        assert engine.started_at is None
        assert engine.poll_interval == 30
        assert engine._job_handlers == {}

    def test_scheduler_engine_initialization_with_params(
        self, mock_repository, mock_execution_manager
    ):
        """Test scheduler engine initialization with custom parameters"""
        engine = SchedulerEngine(
            repository=mock_repository,
            execution_manager=mock_execution_manager,
            poll_interval=60,
        )

        assert engine.repository == mock_repository
        assert engine.execution_manager == mock_execution_manager
        assert engine.poll_interval == 60

    def test_scheduler_engine_execution_callbacks_setup(
        self, scheduler_engine, mock_execution_manager
    ):
        """Test that execution manager callbacks are properly set up"""
        # Verify callbacks were added to execution manager
        mock_execution_manager.add_execution_callback.assert_has_calls(
            [
                call("on_success", scheduler_engine._on_execution_success),
                call("on_failure", scheduler_engine._on_execution_failure),
                call("on_complete", scheduler_engine._on_execution_complete),
            ]
        )


class TestSchedulerEngineLifecycle:
    """Test suite for scheduler engine lifecycle management"""

    @pytest.mark.asyncio
    async def test_start_scheduler_success(self, scheduler_engine, mock_repository):
        """Test successful scheduler start"""
        # Mock lock acquisition
        mock_repository.acquire_scheduler_lock.return_value = "lock-123"

        await scheduler_engine.start()

        assert scheduler_engine.status == SchedulerStatus.RUNNING
        assert scheduler_engine.started_at is not None
        assert scheduler_engine._scheduler_task is not None

        # Clean up
        await scheduler_engine.stop()

    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(self, scheduler_engine):
        """Test starting scheduler when already running"""
        scheduler_engine.status = SchedulerStatus.RUNNING

        with pytest.raises(RuntimeError, match="already running"):
            await scheduler_engine.start()

    @pytest.mark.asyncio
    async def test_start_scheduler_lock_acquisition_failed(
        self, scheduler_engine, mock_repository
    ):
        """Test scheduler start with lock acquisition failure"""
        # Mock lock acquisition failure
        mock_repository.acquire_scheduler_lock.return_value = None

        with pytest.raises(RuntimeError, match="Could not acquire scheduler lock"):
            await scheduler_engine.start()

    @pytest.mark.asyncio
    async def test_stop_scheduler_success(self, scheduler_engine, mock_repository):
        """Test successful scheduler stop"""
        # Start scheduler first
        mock_repository.acquire_scheduler_lock.return_value = "lock-123"
        await scheduler_engine.start()

        # Mock the task to complete quickly
        scheduler_engine._scheduler_task = AsyncMock()

        await scheduler_engine.stop()

        assert scheduler_engine.status == SchedulerStatus.STOPPED
        mock_repository.release_scheduler_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_scheduler_already_stopped(self, scheduler_engine):
        """Test stopping scheduler when already stopped"""
        assert scheduler_engine.status == SchedulerStatus.STOPPED

        # Should not raise error
        await scheduler_engine.stop()

        assert scheduler_engine.status == SchedulerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pause_scheduler(self, scheduler_engine, mock_repository):
        """Test pausing scheduler"""
        # Start scheduler first
        mock_repository.acquire_scheduler_lock.return_value = "lock-123"
        await scheduler_engine.start()

        await scheduler_engine.pause()

        assert scheduler_engine.status == SchedulerStatus.PAUSED

        await scheduler_engine.stop()

    @pytest.mark.asyncio
    async def test_resume_scheduler(self, scheduler_engine, mock_repository):
        """Test resuming paused scheduler"""
        # Start and pause scheduler
        mock_repository.acquire_scheduler_lock.return_value = "lock-123"
        await scheduler_engine.start()
        await scheduler_engine.pause()

        await scheduler_engine.resume()

        assert scheduler_engine.status == SchedulerStatus.RUNNING

        await scheduler_engine.stop()


class TestJobScheduling:
    """Test suite for job scheduling operations"""

    @pytest.mark.asyncio
    async def test_schedule_job_success(
        self,
        scheduler_engine,
        mock_repository,
        mock_trigger_factory,
        sample_job_definition,
    ):
        """Test successful job scheduling"""
        # Mock trigger creation
        mock_trigger = AsyncMock()
        next_run_time = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_trigger.get_next_run_time.return_value = next_run_time
        mock_trigger_factory.create_trigger.return_value = mock_trigger

        # Mock job creation
        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = "job-123"
        mock_job.name = sample_job_definition.name
        mock_repository.create_job.return_value = mock_job

        result = await scheduler_engine.schedule_job(sample_job_definition)

        assert result == "job-123"
        mock_trigger_factory.create_trigger.assert_called_once()
        mock_repository.create_job.assert_called_once()

        # Verify job data structure
        create_call_args = mock_repository.create_job.call_args[0][0]
        assert create_call_args["name"] == sample_job_definition.name
        assert create_call_args["job_type"] == sample_job_definition.job_type
        assert create_call_args["trigger_type"] == TriggerType.CRON
        assert create_call_args["next_run_time"] == next_run_time

    @pytest.mark.asyncio
    async def test_schedule_job_invalid_trigger(
        self, scheduler_engine, mock_trigger_factory, sample_job_definition
    ):
        """Test job scheduling with invalid trigger configuration"""
        # Mock trigger creation failure
        mock_trigger_factory.create_trigger.side_effect = Exception(
            "Invalid trigger config"
        )

        with pytest.raises(Exception, match="Invalid trigger config"):
            await scheduler_engine.schedule_job(sample_job_definition)

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, scheduler_engine, mock_repository):
        """Test successful job cancellation"""
        job_id = "job-123"

        # Mock job retrieval
        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = job_id
        mock_job.name = "test_job"
        mock_repository.get_job.return_value = mock_job

        # Mock job update
        mock_repository.update_job.return_value = mock_job

        result = await scheduler_engine.cancel_job(job_id)

        assert result is True
        mock_repository.get_job.assert_called_once_with(job_id)
        mock_repository.update_job.assert_called_once()

        # Verify update call
        update_call = mock_repository.update_job.call_args[0]
        assert update_call[0] == job_id  # job_id
        assert update_call[1]["status"] == ScheduledJobStatus.CANCELLED
        assert update_call[1]["next_run_time"] is None

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, scheduler_engine, mock_repository):
        """Test cancelling non-existent job"""
        job_id = "nonexistent-job"

        # Mock job not found
        mock_repository.get_job.return_value = None

        result = await scheduler_engine.cancel_job(job_id)

        assert result is False
        mock_repository.update_job.assert_not_called()


class TestScheduledJobStatusAndListing:
    """Test suite for job status and listing operations"""

    @pytest.mark.asyncio
    async def test_get_job_status_success(
        self,
        scheduler_engine,
        mock_repository,
        mock_execution_manager,
        mock_trigger_factory,
    ):
        """Test successful job status retrieval"""
        job_id = "job-123"

        # Mock job with executions
        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = job_id
        mock_job.name = "test_job"
        mock_job.description = "Test job"
        mock_job.status = ScheduledJobStatus.ACTIVE
        mock_job.trigger_type = TriggerType.CRON
        mock_job.trigger_config = {"trigger_type": "cron", "expression": "0 9 * * *"}
        mock_job.next_run_time = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_job.created_at = datetime.now(timezone.utc) - timedelta(days=1)
        mock_job.updated_at = datetime.now(timezone.utc)
        mock_job.priority = 5
        mock_job.timeout_seconds = 3600
        mock_job.max_retries = 3
        mock_job.tags = ["test"]

        mock_repository.get_job_with_executions.return_value = mock_job

        # Mock recent executions
        mock_executions = [Mock() for _ in range(2)]
        for i, exec in enumerate(mock_executions):
            exec.execution_id = f"exec-{i}"
            exec.status = JobExecutionStatus.SUCCESS
            exec.started_at = datetime.now(timezone.utc) - timedelta(hours=i + 1)
            exec.finished_at = (
                datetime.now(timezone.utc)
                - timedelta(hours=i + 1)
                + timedelta(minutes=30)
            )
            exec.runtime_seconds = 1800

        mock_repository.get_job_executions.return_value = mock_executions

        # Mock active execution
        mock_context = Mock()
        mock_context.job_id = job_id
        mock_context.execution_id = "exec-active"
        mock_context.started_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        mock_execution_manager.get_active_executions.return_value = [mock_context]

        # Mock trigger info
        mock_trigger = Mock()
        mock_trigger.get_trigger_info.return_value = {
            "type": "cron",
            "description": "Daily at 9 AM",
        }
        mock_trigger_factory.create_trigger.return_value = mock_trigger

        result = await scheduler_engine.get_job_status(job_id)

        assert result is not None
        assert result["job_id"] == job_id
        assert result["name"] == "test_job"
        assert result["status"] == ScheduledJobStatus.ACTIVE.value
        assert result["trigger_type"] == TriggerType.CRON.value
        assert "active_execution" in result
        assert result["active_execution"]["execution_id"] == "exec-active"
        assert len(result["recent_executions"]) == 2

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, scheduler_engine, mock_repository):
        """Test job status retrieval for non-existent job"""
        job_id = "nonexistent-job"

        mock_repository.get_job_with_executions.return_value = None

        result = await scheduler_engine.get_job_status(job_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_jobs_by_status(self, scheduler_engine, mock_repository):
        """Test listing jobs by status"""
        # Mock jobs
        mock_jobs = []
        for i in range(3):
            job = Mock(spec=ScheduledJob)
            job.job_id = f"job-{i}"
            job.name = f"test_job_{i}"
            job.status = ScheduledJobStatus.ACTIVE
            job.trigger_type = TriggerType.CRON
            job.next_run_time = datetime.now(timezone.utc) + timedelta(hours=i + 1)
            job.priority = 5
            job.tags = ["test"]
            mock_jobs.append(job)

        mock_repository.get_jobs_by_status.return_value = mock_jobs

        # Mock latest executions for each job
        for job in mock_jobs:
            mock_execution = Mock()
            mock_execution.status = JobExecutionStatus.SUCCESS
            mock_execution.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
            mock_execution.runtime_seconds = 300
            mock_repository.get_job_executions.return_value = [mock_execution]

        result = await scheduler_engine.list_jobs(status=ScheduledJobStatus.ACTIVE)

        assert len(result) == 3
        for i, job_info in enumerate(result):
            assert job_info["job_id"] == f"job-{i}"
            assert job_info["name"] == f"test_job_{i}"
            assert job_info["status"] == ScheduledJobStatus.ACTIVE.value
            assert "latest_execution" in job_info


class TestJobExecution:
    """Test suite for job execution functionality"""

    def test_register_job_handler(self, scheduler_engine):
        """Test registering job type handlers"""

        def test_handler(job_data, context):
            return {"result": "success"}

        scheduler_engine.register_job_handler("test_type", test_handler)

        assert "test_type" in scheduler_engine._job_handlers
        assert scheduler_engine._job_handlers["test_type"] == test_handler

    @pytest.mark.asyncio
    async def test_execute_job_success(
        self, scheduler_engine, mock_execution_manager, mock_trigger_factory
    ):
        """Test successful job execution"""
        # Create mock job
        job = Mock(spec=ScheduledJob)
        job.job_id = "job-123"
        job.name = "test_job"
        job.job_type = "test_type"
        job.job_data = {"key": "value"}
        job.trigger_config = {"trigger_type": "manual"}
        job.last_run_time = None

        # Mock trigger for next run time calculation
        mock_trigger = AsyncMock()
        mock_trigger.get_next_run_time.return_value = None  # No next run
        mock_trigger_factory.create_trigger.return_value = mock_trigger

        # Register handler
        async def test_handler(job_data, context):
            return {"result": "success"}

        scheduler_engine.register_job_handler("test_type", test_handler)

        # Mock execution manager context
        mock_context = Mock()
        mock_execution_manager.track_execution.return_value.__aenter__ = AsyncMock(
            return_value=mock_context
        )
        mock_execution_manager.track_execution.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        await scheduler_engine._execute_job(job)

        # Verify execution manager was used
        mock_execution_manager.track_execution.assert_called_once_with(job)

    @pytest.mark.asyncio
    async def test_execute_job_no_handler(self, scheduler_engine, mock_repository):
        """Test job execution with no registered handler"""
        # Create mock job
        job = Mock(spec=ScheduledJob)
        job.job_id = "job-123"
        job.name = "test_job"
        job.job_type = "unknown_type"

        # Mock repository update (for next run time)
        mock_repository.update_job.return_value = job

        # Execute job - should log error and return without executing
        await scheduler_engine._execute_job(job)

        # Verify job was updated (next run time calculation still happens)
        mock_repository.update_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_next_run_time_success(
        self, scheduler_engine, mock_trigger_factory
    ):
        """Test successful next run time calculation"""
        job = Mock(spec=ScheduledJob)
        job.trigger_config = {"trigger_type": "cron", "expression": "0 9 * * *"}
        job.last_run_time = datetime.now(timezone.utc) - timedelta(days=1)

        # Mock trigger
        next_run_time = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_trigger = AsyncMock()
        mock_trigger.get_next_run_time.return_value = next_run_time
        mock_trigger_factory.create_trigger.return_value = mock_trigger

        result = await scheduler_engine._calculate_next_run_time(job)

        assert result == next_run_time
        mock_trigger_factory.create_trigger.assert_called_once_with(job.trigger_config)
        mock_trigger.get_next_run_time.assert_called_once_with(job.last_run_time)

    @pytest.mark.asyncio
    async def test_calculate_next_run_time_error(
        self, scheduler_engine, mock_trigger_factory
    ):
        """Test next run time calculation with error"""
        job = Mock(spec=ScheduledJob)
        job.job_id = "job-123"
        job.trigger_config = {"trigger_type": "cron", "expression": "invalid"}

        # Mock trigger creation failure
        mock_trigger_factory.create_trigger.side_effect = Exception("Invalid trigger")

        result = await scheduler_engine._calculate_next_run_time(job)

        assert result is None


class TestExecutionCallbacks:
    """Test suite for execution callback handling"""

    @pytest.mark.asyncio
    async def test_on_execution_success(self, scheduler_engine):
        """Test successful execution callback"""
        context = Mock()
        context.job_id = "job-123"
        context.execution_id = "exec-456"

        result = {"key": "value"}

        # Should not raise error
        await scheduler_engine._on_execution_success(context, result)

        # Check stats were updated
        assert scheduler_engine._stats["jobs_executed"] == 1

    @pytest.mark.asyncio
    async def test_on_execution_failure(self, scheduler_engine):
        """Test failed execution callback"""
        context = Mock()
        context.job_id = "job-123"
        context.execution_id = "exec-456"

        error_message = "Task failed"
        error_details = {"error_type": "RuntimeError"}

        # Should not raise error
        await scheduler_engine._on_execution_failure(
            context, error_message, error_details
        )

        # Check stats were updated
        assert scheduler_engine._stats["jobs_failed"] == 1

    @pytest.mark.asyncio
    async def test_on_execution_complete(self, scheduler_engine):
        """Test execution complete callback"""
        context = Mock()
        context.job_id = "job-123"
        context.execution_id = "exec-456"

        status = JobExecutionStatus.SUCCESS
        runtime_seconds = 125.5

        # Should not raise error
        await scheduler_engine._on_execution_complete(context, status, runtime_seconds)


class TestSchedulerStatus:
    """Test suite for scheduler status and health checks"""

    def test_get_scheduler_status(self, scheduler_engine, mock_execution_manager):
        """Test getting scheduler status"""
        scheduler_engine.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        scheduler_engine._stats["uptime_seconds"] = 7200
        scheduler_engine._stats["last_poll"] = datetime.now(timezone.utc) - timedelta(
            minutes=1
        )
        scheduler_engine._job_handlers["test_type"] = Mock()

        mock_execution_manager.get_active_executions.return_value = [Mock(), Mock()]

        result = scheduler_engine.get_scheduler_status()

        assert result["scheduler_id"] == scheduler_engine.scheduler_id
        assert result["status"] == SchedulerStatus.STOPPED.value
        assert result["uptime_seconds"] == 7200
        assert result["active_executions"] == 2
        assert result["registered_job_types"] == ["test_type"]
        assert "last_poll" in result

    @pytest.mark.asyncio
    async def test_get_health_status(
        self, scheduler_engine, mock_execution_manager, mock_repository
    ):
        """Test getting comprehensive health status"""
        # Mock execution manager health
        mock_execution_manager.get_health_status.return_value = {
            "is_healthy": True,
            "active_executions": 1,
            "stuck_executions": 0,
        }

        # Mock repository health check
        mock_repository.health_check.return_value = {
            "database_connection": True,
            "scheduler_healthy": True,
            "active_jobs": 5,
            "issues": [],
        }

        result = await scheduler_engine.get_health_status()

        assert result["overall_healthy"] is False  # Scheduler is stopped
        assert "scheduler" in result
        assert "executions" in result
        assert "repository" in result
        assert "checked_at" in result

    @pytest.mark.asyncio
    async def test_get_health_status_running(
        self, scheduler_engine, mock_execution_manager, mock_repository
    ):
        """Test health status when scheduler is running"""
        # Set scheduler to running status
        scheduler_engine.status = SchedulerStatus.RUNNING
        scheduler_engine._stats["uptime_seconds"] = 3600

        # Mock healthy components
        mock_execution_manager.get_health_status.return_value = {
            "is_healthy": True,
            "active_executions": 1,
            "stuck_executions": 0,
        }

        mock_repository.health_check.return_value = {
            "database_connection": True,
            "scheduler_healthy": True,
            "active_jobs": 5,
            "issues": [],
        }

        result = await scheduler_engine.get_health_status()

        assert result["overall_healthy"] is True
        assert result["scheduler"]["is_healthy"] is True


class TestSchedulerPolling:
    """Test suite for scheduler polling functionality"""

    @pytest.mark.asyncio
    async def test_poll_and_execute_jobs_no_jobs(
        self, scheduler_engine, mock_repository
    ):
        """Test polling when no jobs are due"""
        mock_repository.get_jobs_due_for_execution.return_value = []

        # Should not raise error
        await scheduler_engine._poll_and_execute_jobs()

        mock_repository.get_jobs_due_for_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_and_execute_jobs_with_jobs(
        self, scheduler_engine, mock_repository
    ):
        """Test polling and executing due jobs"""
        # Create mock due jobs
        job1 = Mock(spec=ScheduledJob)
        job1.job_id = "job-1"
        job1.name = "test_job_1"
        job1.job_type = "test_type"

        job2 = Mock(spec=ScheduledJob)
        job2.job_id = "job-2"
        job2.name = "test_job_2"
        job2.job_type = "test_type"

        mock_repository.get_jobs_due_for_execution.return_value = [job1, job2]

        # Mock _execute_job to avoid actual execution
        with patch.object(scheduler_engine, "_execute_job") as mock_execute:
            await scheduler_engine._poll_and_execute_jobs()

            # Verify both jobs were executed
            assert mock_execute.call_count == 2
            mock_execute.assert_has_calls([call(job1), call(job2)])

    @pytest.mark.asyncio
    async def test_poll_and_execute_jobs_execution_error(
        self, scheduler_engine, mock_repository
    ):
        """Test polling with job execution error"""
        # Create mock job that will fail execution
        job = Mock(spec=ScheduledJob)
        job.job_id = "job-1"
        job.name = "failing_job"
        job.job_type = "test_type"

        mock_repository.get_jobs_due_for_execution.return_value = [job]

        # Mock _execute_job to raise error
        with patch.object(scheduler_engine, "_execute_job") as mock_execute:
            mock_execute.side_effect = Exception("Execution failed")

            # Should not raise error (should be caught and logged)
            await scheduler_engine._poll_and_execute_jobs()

            mock_execute.assert_called_once_with(job)
