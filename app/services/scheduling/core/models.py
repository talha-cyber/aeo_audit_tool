"""
Core scheduling data models and types.

Defines the main data structures and configuration objects used throughout
the scheduling system. Built for type safety and validation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from app.models.scheduling import (
    JobExecutionStatus,
    JobPriority,
    JobType,
    TriggerType,
)


class SchedulerMode(str, Enum):
    """Scheduler operation modes"""

    LEADER = "leader"  # Active scheduler processing jobs
    FOLLOWER = "follower"  # Standby scheduler monitoring
    STANDALONE = "standalone"  # Single instance mode
    DISABLED = "disabled"  # Scheduler disabled


class ExecutionStrategy(str, Enum):
    """Job execution strategies"""

    IMMEDIATE = "immediate"  # Execute immediately when triggered
    QUEUED = "queued"  # Queue for later execution
    BATCHED = "batched"  # Batch with other jobs
    PARALLEL = "parallel"  # Allow parallel execution


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler engine"""

    # Instance configuration
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instance_name: str = "scheduler"

    # Operational settings
    mode: SchedulerMode = SchedulerMode.STANDALONE
    poll_interval: int = 30  # seconds between job checks
    max_concurrent_jobs: int = 10
    execution_timeout: int = 3600  # default job timeout in seconds

    # Distributed settings
    enable_leader_election: bool = False
    leader_lease_duration: int = 60  # seconds
    heartbeat_interval: int = 15  # seconds

    # Database settings
    batch_size: int = 100  # jobs to process per batch
    cleanup_retention_days: int = 30  # days to keep execution history
    enable_metrics: bool = True

    # Error handling
    default_max_retries: int = 3
    default_retry_delay: int = 300  # seconds
    enable_dead_letter_queue: bool = True

    # Performance settings
    enable_job_caching: bool = True
    cache_refresh_interval: int = 300  # seconds
    enable_parallel_execution: bool = True

    def validate(self) -> None:
        """Validate configuration"""
        if self.poll_interval < 1:
            raise ValueError("poll_interval must be at least 1 second")
        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")
        if self.execution_timeout < 1:
            raise ValueError("execution_timeout must be at least 1 second")


class TriggerConfig(BaseModel):
    """Base configuration for job triggers"""

    trigger_type: TriggerType
    timezone: Optional[str] = None

    class Config:
        use_enum_values = True


class CronTriggerConfig(TriggerConfig):
    """Cron-based trigger configuration"""

    trigger_type: TriggerType = TriggerType.CRON
    expression: str = Field(..., description="Cron expression (5 or 6 fields)")

    @validator("expression")
    def validate_cron_expression(cls, v):
        """Basic cron expression validation"""
        fields = v.split()
        if len(fields) not in [5, 6]:
            raise ValueError("Cron expression must have 5 or 6 fields")
        return v


class IntervalTriggerConfig(TriggerConfig):
    """Interval-based trigger configuration"""

    trigger_type: TriggerType = TriggerType.INTERVAL
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None

    @validator("seconds", "minutes", "hours", "days")
    def validate_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Interval values must be positive")
        return v

    def total_seconds(self) -> int:
        """Calculate total interval in seconds"""
        total = 0
        if self.seconds:
            total += self.seconds
        if self.minutes:
            total += self.minutes * 60
        if self.hours:
            total += self.hours * 3600
        if self.days:
            total += self.days * 86400
        return total


class DateTriggerConfig(TriggerConfig):
    """One-time date-based trigger configuration"""

    trigger_type: TriggerType = TriggerType.DATE
    run_date: datetime = Field(..., description="When to run the job")

    @validator("run_date")
    def validate_future_date(cls, v):
        if v <= datetime.now(timezone.utc):
            raise ValueError("run_date must be in the future")
        return v


class DependencyTriggerConfig(TriggerConfig):
    """Dependency-based trigger configuration"""

    trigger_type: TriggerType = TriggerType.DEPENDENCY
    depends_on_jobs: List[str] = Field(
        ..., description="List of job IDs this job depends on"
    )
    dependency_type: str = "success"  # success, completion, failure
    delay_after_dependency: int = 0  # seconds to wait after dependency completion

    @validator("depends_on_jobs")
    def validate_dependencies(cls, v):
        if not v:
            raise ValueError("depends_on_jobs cannot be empty for dependency triggers")
        return v


class JobDefinition(BaseModel):
    """Complete job definition for scheduling"""

    # Basic job info
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    job_type: JobType

    # Job configuration
    job_config: Dict[str, Any] = Field(default_factory=dict)
    trigger_config: Union[
        CronTriggerConfig,
        IntervalTriggerConfig,
        DateTriggerConfig,
        DependencyTriggerConfig,
    ]

    # Execution settings
    priority: JobPriority = JobPriority.NORMAL
    max_retries: Optional[int] = None
    retry_delay: Optional[int] = None  # seconds
    execution_timeout: Optional[int] = None  # seconds
    max_concurrent: int = 1

    # Lifecycle settings
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_executions: Optional[int] = None

    # Context
    client_id: Optional[str] = None
    created_by: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    # Execution strategy
    execution_strategy: ExecutionStrategy = ExecutionStrategy.IMMEDIATE

    class Config:
        use_enum_values = True

    @validator("end_date")
    def validate_date_range(cls, v, values):
        if v and "start_date" in values and values["start_date"]:
            if v <= values["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v

    @validator("max_executions")
    def validate_max_executions(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_executions must be positive")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = self.dict()

        # Convert datetime objects to ISO strings
        if self.start_date:
            data["start_date"] = self.start_date.isoformat()
        if self.end_date:
            data["end_date"] = self.end_date.isoformat()

        return data


@dataclass
class ExecutionContext:
    """Context information for job execution"""

    job_id: str
    execution_id: str
    scheduled_time: datetime
    trigger_info: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    previous_failures: List[str] = field(default_factory=list)

    # Runtime context
    instance_id: Optional[str] = None
    execution_node: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "trigger_info": self.trigger_info,
            "retry_count": self.retry_count,
            "previous_failures": self.previous_failures,
            "instance_id": self.instance_id,
            "execution_node": self.execution_node,
            "resource_limits": self.resource_limits,
        }


@dataclass
class ExecutionResult:
    """Result of a job execution"""

    execution_id: str
    job_id: str
    status: JobExecutionStatus

    # Timing info
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Results
    exit_code: Optional[int] = None
    output: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Resource usage
    resource_usage: Optional[Dict[str, Any]] = None

    # Generated artifacts
    audit_run_id: Optional[str] = None  # For audit jobs
    report_id: Optional[str] = None  # For report jobs
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def was_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == JobExecutionStatus.COMPLETED and (
            self.exit_code == 0 or self.exit_code is None
        )

    @property
    def should_retry(self) -> bool:
        """Check if execution should be retried"""
        return self.status in [JobExecutionStatus.FAILED, JobExecutionStatus.TIMEOUT]

    def calculate_duration(self) -> Optional[int]:
        """Calculate execution duration in seconds"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds())
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "execution_id": self.execution_id,
            "job_id": self.job_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration_seconds or self.calculate_duration(),
            "exit_code": self.exit_code,
            "output": self.output,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "resource_usage": self.resource_usage,
            "audit_run_id": self.audit_run_id,
            "report_id": self.report_id,
            "artifacts": self.artifacts,
            "was_successful": self.was_successful,
        }


@dataclass
class SchedulerStatus:
    """Current status of the scheduler"""

    instance_id: str
    mode: SchedulerMode
    is_running: bool

    # Statistics
    total_jobs: int = 0
    active_jobs: int = 0
    running_executions: int = 0
    failed_jobs_last_hour: int = 0

    # Health info
    last_poll_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None
    uptime_seconds: Optional[int] = None

    # Performance metrics
    avg_execution_time: Optional[float] = None
    jobs_per_minute: Optional[float] = None
    success_rate: Optional[float] = None

    # Leader election (if enabled)
    is_leader: Optional[bool] = None
    leader_since: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "instance_id": self.instance_id,
            "mode": self.mode.value,
            "is_running": self.is_running,
            "total_jobs": self.total_jobs,
            "active_jobs": self.active_jobs,
            "running_executions": self.running_executions,
            "failed_jobs_last_hour": self.failed_jobs_last_hour,
            "last_poll_at": self.last_poll_at.isoformat()
            if self.last_poll_at
            else None,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat()
            if self.last_heartbeat_at
            else None,
            "uptime_seconds": self.uptime_seconds,
            "avg_execution_time": self.avg_execution_time,
            "jobs_per_minute": self.jobs_per_minute,
            "success_rate": self.success_rate,
            "is_leader": self.is_leader,
            "leader_since": self.leader_since.isoformat()
            if self.leader_since
            else None,
        }


@dataclass
class JobSchedule:
    """Represents when a job should next run"""

    job_id: str
    next_run_time: datetime
    priority: int
    execution_context: ExecutionContext

    def __lt__(self, other: "JobSchedule") -> bool:
        """Allow sorting by next run time and priority"""
        if self.next_run_time != other.next_run_time:
            return self.next_run_time < other.next_run_time
        return self.priority > other.priority  # Higher priority first


# Type definitions for job executors
JobExecutorFunction = Callable[[ExecutionContext], ExecutionResult]
AsyncJobExecutorFunction = Callable[[ExecutionContext], ExecutionResult]


# Events for scheduler lifecycle
class SchedulerEvent(str, Enum):
    """Events emitted by the scheduler"""

    STARTED = "scheduler_started"
    STOPPED = "scheduler_stopped"
    JOB_SCHEDULED = "job_scheduled"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    JOB_RETRYING = "job_retrying"
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"
    ERROR_OCCURRED = "error_occurred"
