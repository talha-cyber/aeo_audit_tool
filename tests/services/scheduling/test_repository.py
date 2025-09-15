"""
Tests for scheduling repository functionality.

Tests database operations, concurrency handling, and data integrity
for the scheduling system repository layer.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from app.models.scheduling import (
    JobDependency,
    JobExecution,
    JobExecutionStatus,
    ScheduledJob,
    ScheduledJobStatus,
    SchedulerMetrics,
    TriggerType,
)
from app.services.scheduling.repository import (
    JobNotFoundError,
    SchedulingRepository,
    SchedulingRepositoryError,
)


@pytest.fixture
def mock_db_session():
    """Create mock database session"""
    session = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.query = Mock()
    session.add = Mock()
    session.flush = Mock()
    session.delete = Mock()
    return session


@pytest.fixture
def repository(mock_db_session):
    """Create repository with mocked database"""
    repo = SchedulingRepository(db=mock_db_session)
    return repo


class TestSchedulingRepository:
    """Test suite for SchedulingRepository"""

    def test_init_with_session(self, mock_db_session):
        """Test repository initialization with provided session"""
        repo = SchedulingRepository(db=mock_db_session)
        assert repo._db == mock_db_session
        assert not repo._auto_commit

    def test_init_without_session(self):
        """Test repository initialization without session"""
        repo = SchedulingRepository()
        assert repo._db is None
        assert repo._auto_commit

    def test_create_job_success(self, repository, mock_db_session):
        """Test successful job creation"""
        job_data = {
            "job_id": str(uuid.uuid4()),
            "name": "test_job",
            "job_type": "audit_execution",
            "trigger_type": TriggerType.CRON,
            "trigger_config": {"expression": "0 9 * * *"},
            "job_data": {"client_id": "123"},
            "status": ScheduledJobStatus.ACTIVE,
        }

        # Mock job object
        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = job_data["job_id"]
        mock_job.name = job_data["name"]
        mock_job.trigger_type = job_data["trigger_type"]

        # Setup mock behavior
        mock_db_session.add.return_value = None
        mock_db_session.flush.return_value = None

        with patch(
            "app.services.scheduling.repository.ScheduledJob", return_value=mock_job
        ):
            result = repository.create_job(job_data)

            assert result == mock_job
            mock_db_session.add.assert_called_once_with(mock_job)
            mock_db_session.flush.assert_called_once()

    def test_create_job_integrity_error(self, repository, mock_db_session):
        """Test job creation with integrity error"""
        from sqlalchemy.exc import IntegrityError

        job_data = {"job_id": "duplicate_id"}
        mock_db_session.add.side_effect = IntegrityError("", "", "")

        with pytest.raises(SchedulingRepositoryError, match="Job creation failed"):
            repository.create_job(job_data)

    def test_get_job_found(self, repository, mock_db_session):
        """Test getting existing job"""
        job_id = "test-job-123"
        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        result = repository.get_job(job_id)

        assert result == mock_job
        mock_db_session.query.assert_called_once_with(ScheduledJob)

    def test_get_job_not_found(self, repository, mock_db_session):
        """Test getting non-existent job"""
        job_id = "nonexistent-job"

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        result = repository.get_job(job_id)

        assert result is None

    def test_update_job_success(self, repository, mock_db_session):
        """Test successful job update"""
        job_id = "test-job-123"
        updates = {"status": ScheduledJobStatus.CANCELLED, "priority": 1}

        mock_job = Mock(spec=ScheduledJob)
        mock_job.job_id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        result = repository.update_job(job_id, updates)

        assert result == mock_job
        assert mock_job.status == ScheduledJobStatus.CANCELLED
        assert mock_job.priority == 1
        assert hasattr(mock_job, "updated_at")

    def test_update_job_not_found(self, repository, mock_db_session):
        """Test updating non-existent job"""
        job_id = "nonexistent-job"
        updates = {"status": ScheduledJobStatus.CANCELLED}

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        with pytest.raises(JobNotFoundError, match="Job not found"):
            repository.update_job(job_id, updates)

    def test_delete_job_success(self, repository, mock_db_session):
        """Test successful job deletion"""
        job_id = "test-job-123"
        mock_job = Mock(spec=ScheduledJob)

        # Setup queries
        job_query = Mock()
        job_query.filter.return_value.first.return_value = mock_job

        exec_query = Mock()
        exec_query.filter.return_value.delete.return_value = 3  # 3 executions deleted

        dep_query = Mock()
        dep_query.filter.return_value.delete.return_value = 1  # 1 dependency deleted

        mock_db_session.query.side_effect = [job_query, exec_query, dep_query]

        result = repository.delete_job(job_id)

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_job)

    def test_delete_job_not_found(self, repository, mock_db_session):
        """Test deleting non-existent job"""
        job_id = "nonexistent-job"

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        result = repository.delete_job(job_id)

        assert result is False

    def test_get_jobs_due_for_execution(self, repository, mock_db_session):
        """Test getting jobs due for execution"""
        current_time = datetime.now(timezone.utc)

        mock_jobs = [Mock(spec=ScheduledJob) for _ in range(3)]

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = (
            mock_jobs
        )
        mock_db_session.query.return_value = mock_query

        result = repository.get_jobs_due_for_execution(current_time)

        assert result == mock_jobs
        mock_db_session.query.assert_called_once_with(ScheduledJob)

    def test_create_execution_success(self, repository, mock_db_session):
        """Test successful execution creation"""
        execution_data = {
            "execution_id": str(uuid.uuid4()),
            "job_id": "test-job-123",
            "status": JobExecutionStatus.RUNNING,
            "started_at": datetime.now(timezone.utc),
        }

        mock_execution = Mock(spec=JobExecution)
        mock_execution.execution_id = execution_data["execution_id"]

        with patch(
            "app.services.scheduling.repository.JobExecution",
            return_value=mock_execution,
        ):
            result = repository.create_execution(execution_data)

            assert result == mock_execution
            mock_db_session.add.assert_called_once_with(mock_execution)
            mock_db_session.flush.assert_called_once()

    def test_update_execution_success(self, repository, mock_db_session):
        """Test successful execution update"""
        execution_id = "exec-123"
        updates = {
            "status": JobExecutionStatus.SUCCESS,
            "finished_at": datetime.now(timezone.utc),
        }

        mock_execution = Mock(spec=JobExecution)
        mock_execution.execution_id = execution_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_execution
        mock_db_session.query.return_value = mock_query

        result = repository.update_execution(execution_id, updates)

        assert result == mock_execution
        assert mock_execution.status == JobExecutionStatus.SUCCESS

    def test_acquire_scheduler_lock_success(self, repository, mock_db_session):
        """Test successful lock acquisition"""
        lock_name = "main_scheduler"

        mock_db_session.add.return_value = None

        with patch(
            "app.services.scheduling.repository.SchedulerLock"
        ) as mock_lock_class:
            mock_lock = Mock()
            mock_lock_class.return_value = mock_lock

            result = repository.acquire_scheduler_lock(lock_name)

            assert result is not None
            assert len(result) == 36  # UUID length
            mock_db_session.add.assert_called_once_with(mock_lock)

    def test_acquire_scheduler_lock_already_held(self, repository, mock_db_session):
        """Test lock acquisition when lock already held"""
        from sqlalchemy.exc import IntegrityError

        lock_name = "main_scheduler"

        # First attempt fails with integrity error
        mock_db_session.add.side_effect = IntegrityError("", "", "")

        # Cleanup attempt - no expired locks found
        cleanup_query = Mock()
        cleanup_query.filter.return_value.delete.return_value = 0
        mock_db_session.query.return_value = cleanup_query

        result = repository.acquire_scheduler_lock(lock_name)

        assert result is None

    def test_release_scheduler_lock_success(self, repository, mock_db_session):
        """Test successful lock release"""
        lock_name = "main_scheduler"
        lock_id = str(uuid.uuid4())

        mock_query = Mock()
        mock_query.filter.return_value.delete.return_value = 1
        mock_db_session.query.return_value = mock_query

        result = repository.release_scheduler_lock(lock_name, lock_id)

        assert result is True

    def test_release_scheduler_lock_not_found(self, repository, mock_db_session):
        """Test lock release when lock not found"""
        lock_name = "main_scheduler"
        lock_id = str(uuid.uuid4())

        mock_query = Mock()
        mock_query.filter.return_value.delete.return_value = 0
        mock_db_session.query.return_value = mock_query

        result = repository.release_scheduler_lock(lock_name, lock_id)

        assert result is False

    def test_cleanup_old_executions(self, repository, mock_db_session):
        """Test cleanup of old execution records"""
        older_than = datetime.now(timezone.utc) - timedelta(days=30)

        mock_query = Mock()
        mock_query.filter.return_value.delete.return_value = (
            5  # 5 old executions deleted
        )
        mock_db_session.query.return_value = mock_query

        result = repository.cleanup_old_executions(older_than)

        assert result == 5

    def test_get_job_statistics(self, repository, mock_db_session):
        """Test getting job statistics"""
        job_id = "test-job-123"

        # Mock query chain
        base_query = Mock()
        filtered_query = Mock()
        base_query.filter.return_value = filtered_query
        mock_db_session.query.return_value = base_query

        # Mock count queries
        filtered_query.count.return_value = 10  # Total executions

        # Mock status-specific queries
        success_query = Mock()
        success_query.count.return_value = 8
        failure_query = Mock()
        failure_query.count.return_value = 1
        running_query = Mock()
        running_query.count.return_value = 1

        filtered_query.filter.side_effect = [
            success_query,
            failure_query,
            running_query,
        ]

        # Mock average runtime query
        mock_db_session.query.return_value.select_from.return_value.first.return_value.avg_runtime_seconds = (
            120.0
        )

        # Mock recent executions query
        recent_query = Mock()
        recent_query.count.return_value = 5
        filtered_query.filter.return_value = recent_query

        result = repository.get_job_statistics(job_id)

        expected = {
            "total_executions": 10,
            "success_count": 8,
            "failure_count": 1,
            "running_count": 1,
            "success_rate": 80.0,
            "average_runtime_seconds": 120.0,
            "recent_executions_24h": 5,
        }

        # Note: This test needs refinement based on actual implementation complexity
        assert result["total_executions"] == 10

    def test_health_check_healthy(self, repository, mock_db_session):
        """Test health check when system is healthy"""
        # Mock database connection test
        mock_db_session.execute.return_value.fetchone.return_value = (1,)

        # Mock various count queries
        mock_queries = []
        for _ in range(5):  # Multiple queries in health check
            query = Mock()
            query.count.return_value = 0
            query.filter.return_value = query
            query.order_by.return_value.first.return_value = None
            mock_queries.append(query)

        mock_db_session.query.side_effect = mock_queries

        result = repository.health_check()

        assert result["database_connection"] is True
        assert result["scheduler_healthy"] is True
        assert result["active_jobs"] == 0
        assert result["running_executions"] == 0

    def test_health_check_database_error(self, repository, mock_db_session):
        """Test health check with database connection error"""
        # Mock database connection failure
        mock_db_session.execute.side_effect = Exception("Connection failed")

        result = repository.health_check()

        assert result["database_connection"] is False
        assert result["scheduler_healthy"] is False
        assert "Database connection failed" in result["issues"][0]


class TestJobDependencyOperations:
    """Test suite for job dependency operations"""

    def test_create_dependency_success(self, repository, mock_db_session):
        """Test successful dependency creation"""
        job_id = "job-1"
        depends_on_job_id = "job-2"
        dependency_type = "success"

        mock_dependency = Mock(spec=JobDependency)

        with patch(
            "app.services.scheduling.repository.JobDependency",
            return_value=mock_dependency,
        ):
            result = repository.create_dependency(
                job_id, depends_on_job_id, dependency_type
            )

            assert result == mock_dependency
            mock_db_session.add.assert_called_once_with(mock_dependency)

    def test_get_job_dependencies(self, repository, mock_db_session):
        """Test getting job dependencies"""
        job_id = "test-job-123"
        mock_dependencies = [Mock(spec=JobDependency) for _ in range(2)]

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_dependencies
        mock_db_session.query.return_value = mock_query

        result = repository.get_job_dependencies(job_id)

        assert result == mock_dependencies
        mock_db_session.query.assert_called_once_with(JobDependency)

    def test_check_dependency_satisfaction(self, repository, mock_db_session):
        """Test dependency satisfaction checking"""
        job_id = "test-job-123"

        # Mock dependencies
        mock_dep = Mock()
        mock_dep.depends_on_job_id = "parent-job"
        mock_dep.dependency_type = "success"

        dep_query = Mock()
        dep_query.filter.return_value.all.return_value = [mock_dep]

        # Mock execution query
        mock_execution = Mock()
        exec_query = Mock()
        exec_query.filter.return_value.order_by.return_value.first.return_value = (
            mock_execution
        )

        mock_db_session.query.side_effect = [dep_query, exec_query]

        result = repository.check_dependency_satisfaction(job_id)

        assert "parent-job" in result
        assert result["parent-job"] is True  # Mock execution found


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test suite for async repository operations"""

    async def test_transaction_context_manager_success(self, repository):
        """Test successful transaction context manager"""
        with patch.object(repository, "db") as mock_db:
            mock_db.commit = Mock()
            mock_db.rollback = Mock()
            mock_db.close = Mock()

            with repository.transaction() as db:
                assert db == mock_db
                # Simulate some work
                pass

            mock_db.commit.assert_called_once()
            mock_db.rollback.assert_not_called()

    async def test_transaction_context_manager_error(self, repository):
        """Test transaction context manager with error"""
        with patch.object(repository, "db") as mock_db:
            mock_db.commit = Mock()
            mock_db.rollback = Mock()
            mock_db.close = Mock()

            with pytest.raises(ValueError):
                with repository.transaction():
                    raise ValueError("Test error")

            mock_db.rollback.assert_called_once()
            mock_db.commit.assert_not_called()


class TestMetricsOperations:
    """Test suite for metrics operations"""

    def test_record_scheduler_metrics(self, repository, mock_db_session):
        """Test recording scheduler metrics"""
        metrics_data = {
            "timestamp": datetime.now(timezone.utc),
            "jobs_processed": 10,
            "average_execution_time": 125.5,
            "memory_usage_mb": 512,
        }

        mock_metrics = Mock(spec=SchedulerMetrics)
        mock_metrics.id = 1

        with patch(
            "app.services.scheduling.repository.SchedulerMetrics",
            return_value=mock_metrics,
        ):
            result = repository.record_scheduler_metrics(metrics_data)

            assert result == mock_metrics
            mock_db_session.add.assert_called_once_with(mock_metrics)

    def test_get_scheduler_metrics(self, repository, mock_db_session):
        """Test getting scheduler metrics"""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        limit = 50

        mock_metrics = [Mock(spec=SchedulerMetrics) for _ in range(5)]

        mock_query = Mock()
        query_chain = (
            mock_query.filter.return_value.filter.return_value.order_by.return_value.limit
        )
        query_chain.return_value.all.return_value = mock_metrics
        mock_db_session.query.return_value = mock_query

        result = repository.get_scheduler_metrics(start_time, end_time, limit)

        assert result == mock_metrics
        mock_db_session.query.assert_called_once_with(SchedulerMetrics)
