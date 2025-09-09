"""
Create scheduling system tables

Revision ID: create_scheduling_tables
Revises: merge_heads
Create Date: 2025-09-09 16:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "create_scheduling_tables"
down_revision: Union[str, None] = None  # Will be set to latest revision
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create scheduling system tables"""

    # Create scheduled_jobs table
    op.create_table(
        "scheduled_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "job_type",
            sa.Enum(
                "AUDIT",
                "AUDIT_PARTIAL",
                "REPORT_GENERATION",
                "DATA_CLEANUP",
                "SYSTEM_HEALTH",
                "CUSTOM",
                name="jobtype",
            ),
            nullable=False,
        ),
        sa.Column("job_config", sa.JSON(), nullable=False),
        sa.Column(
            "trigger_type",
            sa.Enum(
                "CRON", "INTERVAL", "DATE", "MANUAL", "DEPENDENCY", name="triggertype"
            ),
            nullable=False,
        ),
        sa.Column("trigger_config", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "ACTIVE",
                "PAUSED",
                "EXPIRED",
                "DISABLED",
                "DELETED",
                name="scheduledjobstatus",
            ),
            nullable=False,
        ),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("run_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("retry_delay", sa.Integer(), nullable=False),
        sa.Column("execution_timeout", sa.Integer(), nullable=False),
        sa.Column("max_concurrent", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_executions", sa.Integer(), nullable=True),
        sa.Column("client_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["client.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("max_retries >= 0", name="ck_max_retries_positive"),
        sa.CheckConstraint("retry_delay >= 0", name="ck_retry_delay_positive"),
        sa.CheckConstraint(
            "execution_timeout > 0", name="ck_execution_timeout_positive"
        ),
        sa.CheckConstraint("max_concurrent > 0", name="ck_max_concurrent_positive"),
        sa.CheckConstraint(
            "end_date IS NULL OR start_date IS NULL OR end_date > start_date",
            name="ck_date_range_valid",
        ),
    )

    # Create indexes for scheduled_jobs
    op.create_index(
        "idx_scheduled_jobs_client_type", "scheduled_jobs", ["client_id", "job_type"]
    )
    op.create_index(
        "idx_scheduled_jobs_status_next_run",
        "scheduled_jobs",
        ["status", "next_run_at"],
    )
    op.create_index(
        "idx_scheduled_jobs_type_status", "scheduled_jobs", ["job_type", "status"]
    )
    op.create_index(op.f("ix_scheduled_jobs_job_type"), "scheduled_jobs", ["job_type"])
    op.create_index(op.f("ix_scheduled_jobs_name"), "scheduled_jobs", ["name"])
    op.create_index(
        op.f("ix_scheduled_jobs_next_run_at"), "scheduled_jobs", ["next_run_at"]
    )
    op.create_index(op.f("ix_scheduled_jobs_status"), "scheduled_jobs", ["status"])
    op.create_index(
        op.f("ix_scheduled_jobs_client_id"), "scheduled_jobs", ["client_id"]
    )

    # Create job_executions table
    op.create_table(
        "job_executions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("execution_id", sa.String(length=255), nullable=False),
        sa.Column("scheduled_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "RETRYING",
                name="jobexecutionstatus",
            ),
            nullable=False,
        ),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_details", sa.JSON(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("resource_usage", sa.JSON(), nullable=True),
        sa.Column("result_data", sa.JSON(), nullable=True),
        sa.Column("output_log", sa.Text(), nullable=True),
        sa.Column("triggered_by", sa.String(length=255), nullable=True),
        sa.Column("execution_context", sa.JSON(), nullable=True),
        sa.Column("audit_run_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["audit_run_id"],
            ["auditrun.id"],
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["scheduled_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("execution_id"),
    )

    # Create indexes for job_executions
    op.create_index("idx_job_executions_audit_run", "job_executions", ["audit_run_id"])
    op.create_index(
        "idx_job_executions_job_status", "job_executions", ["job_id", "status"]
    )
    op.create_index(
        "idx_job_executions_scheduled_time", "job_executions", ["scheduled_time"]
    )
    op.create_index(
        "idx_job_executions_status_started", "job_executions", ["status", "started_at"]
    )
    op.create_index(
        op.f("ix_job_executions_execution_id"), "job_executions", ["execution_id"]
    )
    op.create_index(op.f("ix_job_executions_job_id"), "job_executions", ["job_id"])
    op.create_index(op.f("ix_job_executions_status"), "job_executions", ["status"])

    # Create job_dependencies table
    op.create_table(
        "job_dependencies",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("depends_on_id", sa.String(), nullable=False),
        sa.Column("dependency_type", sa.String(length=50), nullable=False),
        sa.Column("delay_seconds", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["depends_on_id"],
            ["scheduled_jobs.id"],
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["scheduled_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "depends_on_id", name="uq_job_dependency"),
        sa.CheckConstraint("job_id != depends_on_id", name="ck_no_self_dependency"),
        sa.CheckConstraint("delay_seconds >= 0", name="ck_delay_seconds_positive"),
    )

    # Create indexes for job_dependencies
    op.create_index(
        "idx_job_dependencies_depends_on", "job_dependencies", ["depends_on_id"]
    )
    op.create_index(op.f("ix_job_dependencies_job_id"), "job_dependencies", ["job_id"])

    # Create scheduler_locks table
    op.create_table(
        "scheduler_locks",
        sa.Column("lock_name", sa.String(length=255), nullable=False),
        sa.Column("instance_id", sa.String(length=255), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lock_data", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("lock_name"),
    )

    # Create indexes for scheduler_locks
    op.create_index("idx_scheduler_locks_expires", "scheduler_locks", ["expires_at"])
    op.create_index("idx_scheduler_locks_instance", "scheduler_locks", ["instance_id"])

    # Create scheduler_metrics table
    op.create_table(
        "scheduler_metrics",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("metric_name", sa.String(length=255), nullable=False),
        sa.Column("metric_type", sa.String(length=50), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("instance_id", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for scheduler_metrics
    op.create_index(
        "idx_scheduler_metrics_instance_time",
        "scheduler_metrics",
        ["instance_id", "timestamp"],
    )
    op.create_index(
        "idx_scheduler_metrics_name_time",
        "scheduler_metrics",
        ["metric_name", "timestamp"],
    )
    op.create_index(
        "idx_scheduler_metrics_type_time",
        "scheduler_metrics",
        ["metric_type", "timestamp"],
    )
    op.create_index(
        op.f("ix_scheduler_metrics_instance_id"), "scheduler_metrics", ["instance_id"]
    )
    op.create_index(
        op.f("ix_scheduler_metrics_metric_name"), "scheduler_metrics", ["metric_name"]
    )
    op.create_index(
        op.f("ix_scheduler_metrics_timestamp"), "scheduler_metrics", ["timestamp"]
    )


def downgrade() -> None:
    """Drop scheduling system tables"""

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table("scheduler_metrics")
    op.drop_table("scheduler_locks")
    op.drop_table("job_dependencies")
    op.drop_table("job_executions")
    op.drop_table("scheduled_jobs")

    # Drop enums
    sa.Enum(name="scheduledjobstatus").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="jobexecutionstatus").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="triggertype").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="jobtype").drop(op.get_bind(), checkfirst=False)
