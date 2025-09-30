"""
Scheduling system database models.

Comprehensive models for persistent job scheduling, execution tracking,
and audit management. Designed for production use with proper relationships,
constraints, and recovery mechanisms.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates

from app.db.base_class import Base


class ScheduledJobStatus(str, Enum):
    """Status of a scheduled job"""

    ACTIVE = "active"  # Job is enabled and will be scheduled
    PAUSED = "paused"  # Job is temporarily disabled
    EXPIRED = "expired"  # Job has reached its end date
    DISABLED = "disabled"  # Job is permanently disabled
    CANCELLED = "cancelled"  # Job cancelled manually
    DELETED = "deleted"  # Job is marked for deletion


class JobExecutionStatus(str, Enum):
    """Status of a job execution"""

    PENDING = "pending"  # Execution is scheduled but not started
    RUNNING = "running"  # Execution is currently in progress
    SUCCESS = "completed"  # Execution completed successfully (alias)
    COMPLETED = SUCCESS
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Execution was cancelled
    TIMEOUT = "timeout"  # Execution timed out
    RETRYING = "retrying"  # Execution is being retried


class TriggerType(str, Enum):
    """Types of scheduling triggers"""

    CRON = "cron"  # Cron expression based
    INTERVAL = "interval"  # Fixed interval
    DATE = "date"  # One-time at specific date
    MANUAL = "manual"  # Manual trigger only
    DEPENDENCY = "dependency"  # Triggered by other job completion


class JobType(str, Enum):
    """Types of jobs that can be scheduled"""

    AUDIT = "audit"  # Full audit execution
    AUDIT_PARTIAL = "audit_partial"  # Partial audit (specific questions)
    REPORT_GENERATION = "report_generation"  # Report generation only
    DATA_CLEANUP = "data_cleanup"  # Data maintenance tasks
    SYSTEM_HEALTH = "system_health"  # System health checks
    CUSTOM = "custom"  # Custom job types


class JobPriority(int, Enum):
    """Job execution priority levels"""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class ScheduledJob(Base):
    """
    Main scheduled job configuration.

    Stores persistent job definitions including triggers, configurations,
    and execution policies. Designed for high reliability and auditability.
    """

    __tablename__ = "scheduled_jobs"

    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Job configuration
    job_type = Column(SQLEnum(JobType), nullable=False, index=True)
    job_config = Column(JSON, nullable=False)  # Job-specific configuration

    # Trigger configuration
    trigger_type = Column(SQLEnum(TriggerType), nullable=False)
    trigger_config = Column(JSON, nullable=False)  # Trigger-specific config

    # Status and lifecycle
    status = Column(
        SQLEnum(ScheduledJobStatus),
        nullable=False,
        default=ScheduledJobStatus.ACTIVE,
        index=True,
    )
    priority = Column(Integer, nullable=False, default=JobPriority.NORMAL.value)

    # Scheduling metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    created_by = Column(String(255), nullable=True)  # User/system that created the job

    # Execution tracking
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    run_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)

    # Execution policies
    max_retries = Column(Integer, nullable=False, default=3)
    retry_delay = Column(Integer, nullable=False, default=300)  # seconds
    execution_timeout = Column(Integer, nullable=False, default=3600)  # seconds
    max_concurrent = Column(
        Integer, nullable=False, default=1
    )  # max concurrent executions

    # Lifecycle constraints
    start_date = Column(
        DateTime(timezone=True), nullable=True
    )  # Job becomes active after this
    end_date = Column(DateTime(timezone=True), nullable=True)  # Job expires after this
    max_executions = Column(Integer, nullable=True)  # Max total executions

    # Client relationship (for audit jobs)
    client_id = Column(String, ForeignKey("client.id"), nullable=True, index=True)
    client = relationship("Client", backref="scheduled_jobs")

    # Execution history
    executions = relationship(
        "JobExecution", back_populates="job", cascade="all, delete-orphan"
    )
    dependencies = relationship(
        "JobDependency", foreign_keys="JobDependency.job_id", back_populates="job"
    )
    dependents = relationship(
        "JobDependency",
        foreign_keys="JobDependency.depends_on_id",
        back_populates="depends_on",
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_scheduled_jobs_status_next_run", "status", "next_run_at"),
        Index("idx_scheduled_jobs_type_status", "job_type", "status"),
        Index("idx_scheduled_jobs_client_type", "client_id", "job_type"),
        CheckConstraint("max_retries >= 0", name="ck_max_retries_positive"),
        CheckConstraint("retry_delay >= 0", name="ck_retry_delay_positive"),
        CheckConstraint("execution_timeout > 0", name="ck_execution_timeout_positive"),
        CheckConstraint("max_concurrent > 0", name="ck_max_concurrent_positive"),
        CheckConstraint(
            "end_date IS NULL OR start_date IS NULL OR end_date > start_date",
            name="ck_date_range_valid",
        ),
    )

    @validates("job_config", "trigger_config")
    def validate_json_config(self, key, value):
        """Validate JSON configuration fields"""
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a dictionary")
        return value

    @validates("priority")
    def validate_priority(self, key, priority):
        """Validate priority is within allowed range"""
        if not (1 <= priority <= 10):
            raise ValueError("Priority must be between 1 and 10")
        return priority

    # Compatibility alias for legacy code/tests that reference job_id
    @property
    def job_id(self) -> str:  # type: ignore[override]
        return self.id

    @job_id.setter
    def job_id(self, value: str) -> None:
        self.id = value

    @property
    def next_run_time(self) -> Optional[datetime]:
        return self.next_run_at

    @next_run_time.setter
    def next_run_time(self, value: Optional[datetime]) -> None:
        self.next_run_at = value

    @property
    def last_run_time(self) -> Optional[datetime]:
        return self.last_run_at

    @last_run_time.setter
    def last_run_time(self, value: Optional[datetime]) -> None:
        self.last_run_at = value

    @property
    def retry_delay_seconds(self) -> int:
        return self.retry_delay

    @retry_delay_seconds.setter
    def retry_delay_seconds(self, value: int) -> None:
        self.retry_delay = value

    @hybrid_property
    def is_active(self) -> bool:
        """Check if job is currently active"""
        now = datetime.now(timezone.utc)

        # Check status
        if self.status != ScheduledJobStatus.ACTIVE:
            return False

        # Check date constraints
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False

        # Check execution count limit
        if self.max_executions and self.run_count >= self.max_executions:
            return False

        return True

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate job success rate"""
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "trigger_type": self.trigger_type.value,
            "trigger_config": self.trigger_config,
            "job_config": self.job_config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "is_active": self.is_active,
            "client_id": self.client_id,
        }


class JobExecution(Base):
    """
    Individual job execution record.

    Tracks every execution attempt with detailed logging for debugging,
    performance analysis, and compliance requirements.
    """

    __tablename__ = "job_executions"

    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("scheduled_jobs.id"), nullable=False, index=True)
    execution_id = Column(
        String(255), nullable=False, unique=True, index=True
    )  # External execution ID (e.g., Celery task ID)

    # Execution metadata
    scheduled_time = Column(
        DateTime(timezone=True), nullable=False
    )  # When execution was supposed to start
    started_at = Column(
        DateTime(timezone=True), nullable=True
    )  # When execution actually started
    completed_at = Column(
        DateTime(timezone=True), nullable=True
    )  # When execution finished

    # Status and results
    status = Column(
        SQLEnum(JobExecutionStatus),
        nullable=False,
        default=JobExecutionStatus.PENDING,
        index=True,
    )
    exit_code = Column(Integer, nullable=True)  # Exit code from execution

    # Error handling
    error_message = Column(Text, nullable=True)  # Human-readable error message
    error_details = Column(JSON, nullable=True)  # Detailed error information
    retry_count = Column(
        Integer, nullable=False, default=0
    )  # Number of retries attempted

    # Performance tracking
    duration_seconds = Column(Integer, nullable=True)  # Execution duration
    resource_usage = Column(JSON, nullable=True)  # Resource usage statistics

    # Results and output
    result_data = Column(JSON, nullable=True)  # Execution results
    output_log = Column(Text, nullable=True)  # Execution output/logs

    # Context information
    triggered_by = Column(String(255), nullable=True)  # What triggered this execution
    execution_context = Column(JSON, nullable=True)  # Additional context data

    # Relationships
    job = relationship("ScheduledJob", back_populates="executions")
    audit_run_id = Column(
        String, ForeignKey("auditrun.id"), nullable=True
    )  # Link to created audit run
    audit_run = relationship("AuditRun", backref="job_executions")

    # Indexes for performance
    __table_args__ = (
        Index("idx_job_executions_job_status", "job_id", "status"),
        Index("idx_job_executions_scheduled_time", "scheduled_time"),
        Index("idx_job_executions_status_started", "status", "started_at"),
        Index("idx_job_executions_audit_run", "audit_run_id"),
    )

    @hybrid_property
    def is_running(self) -> bool:
        """Check if execution is currently running"""
        return self.status == JobExecutionStatus.RUNNING

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if execution has completed (successfully or not)"""
        return self.status in [
            JobExecutionStatus.COMPLETED,
            JobExecutionStatus.FAILED,
            JobExecutionStatus.CANCELLED,
            JobExecutionStatus.TIMEOUT,
        ]

    @hybrid_property
    def was_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == JobExecutionStatus.COMPLETED and (
            self.exit_code == 0 or self.exit_code is None
        )

    def calculate_duration(self) -> Optional[int]:
        """Calculate execution duration in seconds"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds())
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "duration_seconds": self.duration_seconds or self.calculate_duration(),
            "was_successful": self.was_successful,
            "triggered_by": self.triggered_by,
            "audit_run_id": self.audit_run_id,
        }


class JobDependency(Base):
    """
    Job dependency relationships.

    Enables complex workflows where jobs depend on successful completion
    of other jobs. Supports DAG validation and cycle detection.
    """

    __tablename__ = "job_dependencies"

    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Dependency relationship
    job_id = Column(String, ForeignKey("scheduled_jobs.id"), nullable=False, index=True)
    depends_on_id = Column(
        String, ForeignKey("scheduled_jobs.id"), nullable=False, index=True
    )

    # Dependency configuration
    dependency_type = Column(
        String(50), nullable=False, default="success"
    )  # success, completion, failure
    delay_seconds = Column(
        Integer, nullable=False, default=0
    )  # Delay after dependency completion

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    created_by = Column(String(255), nullable=True)

    # Relationships
    job = relationship(
        "ScheduledJob", foreign_keys=[job_id], back_populates="dependencies"
    )
    depends_on = relationship(
        "ScheduledJob", foreign_keys=[depends_on_id], back_populates="dependents"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("job_id", "depends_on_id", name="uq_job_dependency"),
        Index("idx_job_dependencies_depends_on", "depends_on_id"),
        CheckConstraint("job_id != depends_on_id", name="ck_no_self_dependency"),
        CheckConstraint("delay_seconds >= 0", name="ck_delay_seconds_positive"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "depends_on_id": self.depends_on_id,
            "dependency_type": self.dependency_type,
            "delay_seconds": self.delay_seconds,
            "created_at": self.created_at.isoformat(),
        }

    @property
    def depends_on_job_id(self) -> str:
        return self.depends_on_id

    @depends_on_job_id.setter
    def depends_on_job_id(self, value: str) -> None:
        self.depends_on_id = value


class SchedulerLock(Base):
    """
    Distributed scheduler locking mechanism.

    Prevents multiple scheduler instances from processing the same jobs
    in distributed environments. Implements leader election and failover.
    """

    __tablename__ = "scheduler_locks"

    # Lock identification
    lock_name = Column(String(255), primary_key=True)  # Name of the lock
    instance_id = Column(String(255), nullable=False)  # Instance holding the lock
    acquired_at = Column(
        DateTime(timezone=True), nullable=False
    )  # When lock was acquired
    expires_at = Column(DateTime(timezone=True), nullable=False)  # When lock expires
    heartbeat_at = Column(DateTime(timezone=True), nullable=False)  # Last heartbeat

    # Lock metadata
    lock_data = Column(JSON, nullable=True)  # Additional lock data

    # Indexes
    __table_args__ = (
        Index("idx_scheduler_locks_expires", "expires_at"),
        Index("idx_scheduler_locks_instance", "instance_id"),
    )

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if lock has expired"""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "lock_name": self.lock_name,
            "instance_id": self.instance_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "heartbeat_at": self.heartbeat_at.isoformat(),
            "is_expired": self.is_expired,
            "lock_data": self.lock_data,
        }


class SchedulerMetrics(Base):
    """
    Scheduler performance and health metrics.

    Tracks scheduler performance, job execution statistics, and system health
    for monitoring and alerting purposes.
    """

    __tablename__ = "scheduler_metrics"

    # Metric identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(255), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram

    # Metric data
    value = Column(JSON, nullable=False)  # Metric value (can be number, array, object)
    tags = Column(JSON, nullable=True)  # Metric tags/labels

    # Timing
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    instance_id = Column(String(255), nullable=False, index=True)

    # Indexes for time-series queries
    __table_args__ = (
        Index("idx_scheduler_metrics_name_time", "metric_name", "timestamp"),
        Index("idx_scheduler_metrics_instance_time", "instance_id", "timestamp"),
        Index("idx_scheduler_metrics_type_time", "metric_type", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "instance_id": self.instance_id,
        }
