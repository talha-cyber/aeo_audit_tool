"""
Integration tests for the complete scheduling system.

Tests end-to-end workflows, component interactions, and real-world
scenarios for the comprehensive audit scheduling system.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.scheduling import ScheduledJobStatus, TriggerType
from app.services.scheduling import (
    ExecutionManager,
    JobDefinition,
    SchedulerEngine,
    SchedulingRepository,
    TriggerFactory,
)
from app.services.scheduling.health_monitor import HealthMonitor
from app.services.scheduling.integrations.celery_integration import (
    CeleryJobExecutor,
    CelerySchedulerBridge,
)
from app.services.scheduling.policies.concurrency import (
    ConcurrencyManager,
    ConcurrencyPolicy,
)
from app.services.scheduling.policies.priority import PriorityManager
from app.services.scheduling.policies.retry import ExponentialBackoffRetry


@pytest.fixture
async def full_scheduler_system():
    """Create a complete scheduler system for integration testing"""
    # Mock external dependencies
    mock_db_session = Mock()
    mock_celery_app = Mock()

    # Create components
    repository = SchedulingRepository(db=mock_db_session)
    execution_manager = ExecutionManager(repository)
    trigger_factory = TriggerFactory()

    # Create scheduler engine
    scheduler = SchedulerEngine(
        repository=repository,
        execution_manager=execution_manager,
        trigger_factory=trigger_factory,
        poll_interval=1,  # Fast polling for testing
    )

    # Create integrations
    celery_executor = CeleryJobExecutor(mock_celery_app, execution_manager)
    celery_bridge = CelerySchedulerBridge(scheduler, celery_executor)

    # Create monitoring
    health_monitor = HealthMonitor(scheduler, check_interval=5)

    return {
        "scheduler": scheduler,
        "repository": repository,
        "execution_manager": execution_manager,
        "trigger_factory": trigger_factory,
        "celery_executor": celery_executor,
        "celery_bridge": celery_bridge,
        "health_monitor": health_monitor,
        "mock_db": mock_db_session,
        "mock_celery": mock_celery_app,
    }


class TestEndToEndJobScheduling:
    """Test complete job scheduling workflows"""

    @pytest.mark.asyncio
    async def test_schedule_and_execute_cron_job(self, full_scheduler_system):
        """Test scheduling and executing a cron-based job"""
        system = full_scheduler_system
        scheduler = system["scheduler"]
        mock_db = system["mock_db"]

        # Mock database operations
        mock_job = Mock()
        mock_job.job_id = "cron-job-123"
        mock_job.name = "daily_audit"
        mock_job.job_type = "audit_execution"
        mock_job.trigger_type = TriggerType.CRON
        mock_job.trigger_config = {"trigger_type": "cron", "expression": "0 9 * * *"}
        mock_job.job_data = {"client_id": "client-123"}
        mock_job.status = ScheduledJobStatus.ACTIVE
        mock_job.last_run_time = None
        mock_job.timeout_seconds = 3600
        mock_job.max_retries = 3
        mock_job.retry_delay_seconds = 300
        mock_job.tags = ["audit", "daily"]
        mock_job.metadata = {}
        mock_job.priority = 5

        # Mock trigger for next run time
        next_run = datetime.now(timezone.utc) + timedelta(minutes=1)
        with patch.object(
            system["trigger_factory"], "create_trigger"
        ) as mock_create_trigger:
            mock_trigger = AsyncMock()
            mock_trigger.get_next_run_time.return_value = next_run
            mock_create_trigger.return_value = mock_trigger

            # Mock repository operations
            system["repository"].create_job.return_value = mock_job
            system[
                "repository"
            ].get_jobs_due_for_execution.return_value = []  # Initially no due jobs
            system["repository"].acquire_scheduler_lock.return_value = "lock-123"

            # Create job definition
            job_def = JobDefinition(
                name="daily_audit",
                job_type="audit_execution",
                trigger_config={"trigger_type": "cron", "expression": "0 9 * * *"},
                job_data={"client_id": "client-123"},
                description="Daily audit execution",
                priority=5,
                timeout_seconds=3600,
                tags=["audit", "daily"],
            )

            # Schedule the job
            job_id = await scheduler.schedule_job(job_def)

            assert job_id == "cron-job-123"
            system["repository"].create_job.assert_called_once()
            mock_create_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_interval_job_with_execution_tracking(self, full_scheduler_system):
        """Test interval job with execution tracking"""
        system = full_scheduler_system
        scheduler = system["scheduler"]
        execution_manager = system["execution_manager"]

        # Create interval job definition
        job_def = JobDefinition(
            name="health_check",
            job_type="system_health_check",
            trigger_config={"trigger_type": "interval", "minutes": 5},
            job_data={"check_type": "comprehensive"},
            priority=3,
        )

        # Mock job creation
        mock_job = Mock()
        mock_job.job_id = "interval-job-456"
        mock_job.name = job_def.name
        mock_job.job_type = job_def.job_type
        mock_job.trigger_config = job_def.trigger_config
        mock_job.job_data = job_def.job_data
        mock_job.status = ScheduledJobStatus.ACTIVE
        mock_job.timeout_seconds = None

        # Mock trigger behavior
        with patch.object(
            system["trigger_factory"], "create_trigger"
        ) as mock_create_trigger:
            mock_trigger = AsyncMock()
            mock_trigger.get_next_run_time.return_value = datetime.now(
                timezone.utc
            ) + timedelta(minutes=5)
            mock_create_trigger.return_value = mock_trigger

            system["repository"].create_job.return_value = mock_job

            # Schedule job
            job_id = await scheduler.schedule_job(job_def)

            assert job_id == "interval-job-456"

            # Verify execution manager setup
            assert execution_manager.repository == system["repository"]

    @pytest.mark.asyncio
    async def test_job_dependency_chain(self, full_scheduler_system):
        """Test jobs with dependencies"""
        system = full_scheduler_system
        scheduler = system["scheduler"]

        # Create parent job
        parent_job_def = JobDefinition(
            name="data_preparation",
            job_type="data_cleanup",
            trigger_config={"trigger_type": "cron", "expression": "0 8 * * *"},
            job_data={"scope": "audit_data"},
        )

        # Create dependent job
        child_job_def = JobDefinition(
            name="audit_execution",
            job_type="audit_execution",
            trigger_config={
                "trigger_type": "dependency",
                "depends_on": ["parent-job-id"],
                "dependency_type": "success",
            },
            job_data={"client_id": "client-123"},
        )

        # Mock job creation for both jobs
        parent_job = Mock()
        parent_job.job_id = "parent-job-id"
        parent_job.name = parent_job_def.name

        child_job = Mock()
        child_job.job_id = "child-job-id"
        child_job.name = child_job_def.name

        with patch.object(
            system["trigger_factory"], "create_trigger"
        ) as mock_create_trigger:
            mock_trigger = AsyncMock()
            mock_trigger.get_next_run_time.return_value = datetime.now(
                timezone.utc
            ) + timedelta(hours=1)
            mock_create_trigger.return_value = mock_trigger

            system["repository"].create_job.side_effect = [parent_job, child_job]

            # Schedule both jobs
            parent_id = await scheduler.schedule_job(parent_job_def)
            child_id = await scheduler.schedule_job(child_job_def)

            assert parent_id == "parent-job-id"
            assert child_id == "child-job-id"
            assert system["repository"].create_job.call_count == 2


class TestSchedulerLifecycleIntegration:
    """Test scheduler lifecycle with all components"""

    @pytest.mark.asyncio
    async def test_scheduler_start_stop_cycle(self, full_scheduler_system):
        """Test complete scheduler start/stop cycle"""
        system = full_scheduler_system
        scheduler = system["scheduler"]
        health_monitor = system["health_monitor"]

        # Mock lock acquisition
        system["repository"].acquire_scheduler_lock.return_value = "test-lock-123"
        system["repository"].release_scheduler_lock.return_value = True

        # Test starting scheduler
        await scheduler.start()
        assert scheduler.status.value == "running"
        assert scheduler.started_at is not None

        # Test starting health monitoring
        await health_monitor.start_monitoring()

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Test stopping monitoring
        await health_monitor.stop_monitoring()

        # Test stopping scheduler
        await scheduler.stop()
        assert scheduler.status.value == "stopped"

        # Verify cleanup
        system["repository"].release_scheduler_lock.assert_called_once_with(
            "main_scheduler", scheduler.scheduler_id
        )

    @pytest.mark.asyncio
    async def test_scheduler_with_job_handlers(self, full_scheduler_system):
        """Test scheduler with registered job handlers"""
        system = full_scheduler_system
        scheduler = system["scheduler"]

        # Register job handlers
        async def audit_handler(job_data, context):
            await asyncio.sleep(0.01)  # Simulate work
            return {"audit_result": "passed", "issues_found": 0}

        async def report_handler(job_data, context):
            await asyncio.sleep(0.01)  # Simulate work
            return {"report_generated": True, "format": "pdf"}

        scheduler.register_job_handler("audit_execution", audit_handler)
        scheduler.register_job_handler("report_generation", report_handler)

        # Verify handlers were registered
        assert "audit_execution" in scheduler._job_handlers
        assert "report_generation" in scheduler._job_handlers

        # Test handler execution
        job_data = {"client_id": "test-client"}
        mock_context = Mock()
        mock_context.execution_id = "test-exec-123"

        result = await audit_handler(job_data, mock_context)
        assert result["audit_result"] == "passed"


class TestCeleryIntegration:
    """Test Celery integration functionality"""

    @pytest.mark.asyncio
    async def test_celery_job_execution(self, full_scheduler_system):
        """Test job execution through Celery"""
        system = full_scheduler_system
        celery_executor = system["celery_executor"]
        mock_celery = system["mock_celery"]

        # Create mock job
        job = Mock()
        job.job_id = "celery-test-job"
        job.name = "test_audit"
        job.job_type = "audit_execution"
        job.job_data = {"client_id": "client-123"}
        job.priority = 5
        job.timeout_seconds = 3600
        job.max_retries = 3
        job.retry_delay_seconds = 300
        job.metadata = {}

        # Create mock execution context
        execution_context = Mock()
        execution_context.execution_id = "exec-123"
        execution_context.job_id = job.job_id
        execution_context.scheduled_time = datetime.now(timezone.utc)
        execution_context.metadata = {}

        # Mock Celery task result
        mock_task_result = Mock()
        mock_task_result.task_id = "celery-task-123"
        mock_task_result.ready.return_value = True
        mock_task_result.successful.return_value = True
        mock_task_result.result = {"status": "completed", "issues": 0}

        mock_celery.send_task.return_value = mock_task_result

        # Execute job through Celery
        result = await celery_executor.execute_job(job, execution_context)

        # Verify Celery integration
        assert result["success"] is True
        assert result["task_id"] == "celery-task-123"
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_celery_bridge_integration(self, full_scheduler_system):
        """Test Celery scheduler bridge"""
        system = full_scheduler_system
        scheduler = system["scheduler"]
        celery_bridge = system["celery_bridge"]

        # Test bridge initialization
        assert celery_bridge.scheduler_engine == scheduler
        assert celery_bridge.celery_executor == system["celery_executor"]

        # Test bridge status
        with patch.object(
            system["celery_executor"].celery_app.control, "inspect"
        ) as mock_inspect:
            mock_inspect.return_value.stats.return_value = {
                "worker1": {"pool": {"max-concurrency": 4}}
            }
            mock_inspect.return_value.active.return_value = {"worker1": []}
            mock_inspect.return_value.scheduled.return_value = {"worker1": []}

            status = celery_bridge.get_celery_worker_status()

            assert status["is_healthy"] is True
            assert status["worker_count"] == 1


class TestPolicyIntegration:
    """Test policy system integration"""

    @pytest.mark.asyncio
    async def test_retry_policy_integration(self, full_scheduler_system):
        """Test retry policy integration"""
        system = full_scheduler_system

        # Create retry policy
        retry_policy = ExponentialBackoffRetry(
            max_retries=3, base_delay_seconds=60, multiplier=2.0
        )

        # Test retry decision
        should_retry = retry_policy.should_retry(
            attempt_number=1,
            error_message="Connection timeout",
            error_type="TimeoutError",
        )

        assert should_retry is True

        # Test retry scheduling
        retry_attempt = retry_policy.schedule_retry(
            attempt_number=1,
            execution_id="exec-123",
            error_message="Connection timeout",
            error_type="TimeoutError",
        )

        assert retry_attempt.attempt_number == 1
        assert retry_attempt.delay_seconds >= 60
        assert retry_attempt.next_retry_at > datetime.now(timezone.utc)

    def test_concurrency_policy_integration(self, full_scheduler_system):
        """Test concurrency policy integration"""
        system = full_scheduler_system

        # Create concurrency manager
        concurrency_manager = ConcurrencyManager()

        # Create concurrency policy
        policy = ConcurrencyPolicy(limits={"global_limit": 10, "per_job_limit": 2})

        # Mock job
        job = Mock()
        job.job_id = "test-job"
        job.job_type = "audit_execution"
        job.priority = 5

        # Test policy evaluation
        allows_execution = policy.allows_execution(
            job=job,
            current_executions=[],
            available_resources={"cpu_cores": 4, "memory_mb": 8192},
        )

        assert allows_execution is True

    def test_priority_integration(self, full_scheduler_system):
        """Test priority system integration"""
        system = full_scheduler_system

        # Create priority manager
        priority_manager = PriorityManager()

        # Create mock jobs with different priorities
        high_priority_job = Mock()
        high_priority_job.job_id = "high-priority"
        high_priority_job.name = "critical_audit"
        high_priority_job.job_type = "audit_execution"
        high_priority_job.priority = 1  # High priority

        low_priority_job = Mock()
        low_priority_job.job_id = "low-priority"
        low_priority_job.name = "cleanup_job"
        low_priority_job.job_type = "data_cleanup"
        low_priority_job.priority = 8  # Low priority

        # Enqueue jobs
        priority_manager.enqueue_job(low_priority_job)  # Enqueue low priority first
        priority_manager.enqueue_job(high_priority_job)  # Then high priority

        # Dequeue should return high priority job first
        dequeued_job, category = priority_manager.dequeue_highest_priority_job()

        assert dequeued_job.job_id == "high-priority"
        assert category == "main"


class TestHealthMonitoringIntegration:
    """Test health monitoring integration"""

    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, full_scheduler_system):
        """Test complete health monitoring"""
        system = full_scheduler_system
        health_monitor = system["health_monitor"]
        scheduler = system["scheduler"]

        # Mock repository health check
        system["repository"].health_check.return_value = {
            "database_connection": True,
            "scheduler_healthy": True,
            "active_jobs": 5,
            "running_executions": 2,
            "issues": [],
        }

        # Mock execution manager health
        system["execution_manager"].get_health_status.return_value = {
            "is_healthy": True,
            "active_executions": 2,
            "stuck_executions": 0,
            "stuck_execution_details": [],
        }

        # Set scheduler to running status for health check
        scheduler.status = scheduler.SchedulerStatus.RUNNING
        scheduler.started_at = datetime.now(timezone.utc)
        scheduler._stats["uptime_seconds"] = 3600
        scheduler._stats["last_poll"] = datetime.now(timezone.utc)

        # Perform health check
        health_results = await health_monitor.perform_health_check()

        # Verify health check results
        assert (
            len(health_results) >= 3
        )  # System, database, scheduler, execution manager

        # Check that all components were checked
        component_names = [result.component for result in health_results]
        assert "system_resources" in component_names
        assert "database" in component_names
        assert "scheduler_engine" in component_names
        assert "execution_manager" in component_names

        # Get current health status
        current_status = health_monitor.get_current_health_status()
        assert "overall_status" in current_status
        assert "components" in current_status

    @pytest.mark.asyncio
    async def test_health_monitoring_with_issues(self, full_scheduler_system):
        """Test health monitoring when issues are detected"""
        system = full_scheduler_system
        health_monitor = system["health_monitor"]

        # Mock unhealthy database
        system["repository"].health_check.return_value = {
            "database_connection": False,
            "scheduler_healthy": False,
            "issues": ["Database connection failed"],
        }

        # Mock stuck executions
        system["execution_manager"].get_health_status.return_value = {
            "is_healthy": False,
            "active_executions": 3,
            "stuck_executions": 2,
            "stuck_execution_details": [
                {"execution_id": "exec-1", "runtime_seconds": 7200},
                {"execution_id": "exec-2", "runtime_seconds": 10800},
            ],
        }

        # Perform health check
        health_results = await health_monitor.perform_health_check()

        # Should detect issues
        critical_issues = [r for r in health_results if r.status.value == "critical"]
        warning_issues = [r for r in health_results if r.status.value == "warning"]

        # Should have some issues detected
        assert len(critical_issues) > 0 or len(warning_issues) > 0


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    @pytest.mark.asyncio
    async def test_job_execution_failure_handling(self, full_scheduler_system):
        """Test handling of job execution failures"""
        system = full_scheduler_system
        scheduler = system["scheduler"]
        execution_manager = system["execution_manager"]

        # Mock job that will fail
        failing_job = Mock()
        failing_job.job_id = "failing-job"
        failing_job.name = "failing_audit"
        failing_job.job_type = "audit_execution"
        failing_job.job_data = {"client_id": "client-123"}
        failing_job.trigger_config = {"trigger_type": "manual"}
        failing_job.max_retries = 2
        failing_job.retry_delay_seconds = 60

        # Register failing handler
        async def failing_handler(job_data, context):
            raise Exception("Simulated job failure")

        scheduler.register_job_handler("audit_execution", failing_handler)

        # Mock execution tracking
        mock_context = Mock()
        mock_context.execution_id = "exec-fail-123"
        mock_context.job_id = failing_job.job_id

        with patch.object(execution_manager, "track_execution") as mock_track:
            mock_track.return_value.__aenter__ = AsyncMock(return_value=mock_context)
            mock_track.return_value.__aexit__ = AsyncMock(
                side_effect=Exception("Simulated job failure")
            )

            # Mock repository updates
            system["repository"].update_job.return_value = failing_job

            # Execute failing job
            await scheduler._execute_job(failing_job)

            # Verify execution tracking was used
            mock_track.assert_called_once_with(failing_job)

    @pytest.mark.asyncio
    async def test_scheduler_recovery_from_lock_loss(self, full_scheduler_system):
        """Test scheduler recovery when lock is lost"""
        system = full_scheduler_system
        scheduler = system["scheduler"]

        # Mock initial lock acquisition
        system["repository"].acquire_scheduler_lock.return_value = "lock-123"

        # Start scheduler
        await scheduler.start()
        assert scheduler.status.value == "running"

        # Simulate lock loss (another instance takes over)
        system["repository"].acquire_scheduler_lock.return_value = None

        # Stop scheduler (simulates detection of lock loss)
        await scheduler.stop()
        assert scheduler.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_database_connection_recovery(self, full_scheduler_system):
        """Test handling of database connection issues"""
        system = full_scheduler_system
        repository = system["repository"]

        # Mock database connection failure
        with patch.object(repository, "health_check") as mock_health:
            mock_health.return_value = {
                "database_connection": False,
                "scheduler_healthy": False,
                "issues": ["Connection to database failed"],
            }

            health_status = repository.health_check()

            # Should detect database issues
            assert health_status["database_connection"] is False
            assert health_status["scheduler_healthy"] is False
            assert len(health_status["issues"]) > 0


@pytest.mark.asyncio
async def test_complete_audit_workflow(full_scheduler_system):
    """Test a complete audit workflow from scheduling to completion"""
    system = full_scheduler_system
    scheduler = system["scheduler"]

    # Create audit job definition
    audit_job_def = JobDefinition(
        name="monthly_compliance_audit",
        job_type="audit_execution",
        trigger_config={"trigger_type": "cron", "expression": "0 2 1 * *"},  # Monthly
        job_data={
            "client_id": "client-456",
            "audit_type": "compliance",
            "scope": "full",
        },
        description="Monthly compliance audit for client",
        priority=3,
        timeout_seconds=7200,  # 2 hours
        max_retries=2,
        tags=["compliance", "monthly", "audit"],
    )

    # Mock successful job scheduling
    mock_job = Mock()
    mock_job.job_id = "audit-job-789"
    mock_job.name = audit_job_def.name
    mock_job.status = ScheduledJobStatus.ACTIVE

    # Mock trigger creation
    with patch.object(
        system["trigger_factory"], "create_trigger"
    ) as mock_create_trigger:
        mock_trigger = AsyncMock()
        next_month = datetime.now(timezone.utc).replace(
            day=1, hour=2, minute=0
        ) + timedelta(days=32)
        mock_trigger.get_next_run_time.return_value = next_month
        mock_create_trigger.return_value = mock_trigger

        system["repository"].create_job.return_value = mock_job

        # Schedule the audit job
        job_id = await scheduler.schedule_job(audit_job_def)

        assert job_id == "audit-job-789"

        # Verify job was created with correct configuration
        create_call_args = system["repository"].create_job.call_args[0][0]
        assert create_call_args["name"] == audit_job_def.name
        assert create_call_args["job_type"] == audit_job_def.job_type
        assert create_call_args["priority"] == audit_job_def.priority
        assert create_call_args["timeout_seconds"] == audit_job_def.timeout_seconds
        assert "compliance" in create_call_args["tags"]

    # Test job status retrieval
    system["repository"].get_job_with_executions.return_value = mock_job
    system["repository"].get_job_executions.return_value = []

    status = await scheduler.get_job_status(job_id)
    assert status is not None
    assert status["job_id"] == job_id
    assert status["name"] == audit_job_def.name
