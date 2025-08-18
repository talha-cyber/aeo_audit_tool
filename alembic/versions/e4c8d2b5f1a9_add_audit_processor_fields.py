"""Add audit processor fields

Revision ID: e4c8d2b5f1a9
Revises: db76b1886c2a
Create Date: 2025-01-01 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "e4c8d2b5f1a9"
down_revision: Union[str, None] = "db76b1886c2a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### AuditRun model updates ###
    op.add_column(
        "auditrun", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "auditrun", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "auditrun", sa.Column("total_questions", sa.Integer(), nullable=True, default=0)
    )
    op.add_column(
        "auditrun",
        sa.Column("processed_questions", sa.Integer(), nullable=True, default=0),
    )
    op.add_column("auditrun", sa.Column("progress_data", sa.JSON(), nullable=True))
    op.add_column("auditrun", sa.Column("platform_stats", sa.JSON(), nullable=True))

    # ### Question model updates ###
    op.add_column("questions", sa.Column("audit_run_id", sa.String(), nullable=True))
    op.add_column("questions", sa.Column("question_type", sa.String(), nullable=True))
    op.add_column(
        "questions", sa.Column("priority_score", sa.Float(), nullable=True, default=0.0)
    )
    op.add_column("questions", sa.Column("target_brand", sa.String(), nullable=True))

    # ### Response model updates ###
    # Rename response column to response_text
    op.alter_column("responses", "response", new_column_name="response_text")
    op.add_column("responses", sa.Column("response_metadata", sa.JSON(), nullable=True))
    op.add_column(
        "responses", sa.Column("processing_time_ms", sa.Integer(), nullable=True)
    )

    # Create foreign key constraint for questions.audit_run_id
    op.create_foreign_key(
        "questions_audit_run_id_fkey", "questions", "auditrun", ["audit_run_id"], ["id"]
    )

    # Create indices for performance
    op.create_index("idx_audit_runs_status", "auditrun", ["status"])
    op.create_index(
        "idx_audit_runs_client_started", "auditrun", ["client_id", "started_at"]
    )
    op.create_index("idx_questions_audit_run", "questions", ["audit_run_id"])
    op.create_index("idx_responses_audit_run", "responses", ["audit_run_id"])
    op.create_index(
        "idx_responses_platform_time", "responses", ["platform", "created_at"]
    )


def downgrade() -> None:
    # Drop indices
    op.drop_index("idx_responses_platform_time", "responses")
    op.drop_index("idx_responses_audit_run", "responses")
    op.drop_index("idx_questions_audit_run", "questions")
    op.drop_index("idx_audit_runs_client_started", "auditrun")
    op.drop_index("idx_audit_runs_status", "auditrun")

    # Drop foreign key constraint
    op.drop_constraint("questions_audit_run_id_fkey", "questions", type_="foreignkey")

    # ### Response model rollback ###
    op.drop_column("responses", "processing_time_ms")
    op.drop_column("responses", "response_metadata")
    op.alter_column("responses", "response_text", new_column_name="response")

    # ### Question model rollback ###
    op.drop_column("questions", "target_brand")
    op.drop_column("questions", "priority_score")
    op.drop_column("questions", "question_type")
    op.drop_column("questions", "audit_run_id")

    # ### AuditRun model rollback ###
    op.drop_column("auditrun", "platform_stats")
    op.drop_column("auditrun", "progress_data")
    op.drop_column("auditrun", "processed_questions")
    op.drop_column("auditrun", "total_questions")
    op.drop_column("auditrun", "completed_at")
    op.drop_column("auditrun", "started_at")
