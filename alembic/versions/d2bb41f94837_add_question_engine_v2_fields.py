"""Add persona and satisfaction fields for question engine v2

Revision ID: d2bb41f94837
Revises: a047571e7507
Create Date: 2025-09-25 15:45:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2bb41f94837"
down_revision: Union[str, None] = "a047571e7507"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


QUESTION_TABLE = "questions"
RESPONSE_TABLE = "responses"


def upgrade() -> None:
    """Add persona metadata and satisfaction columns."""
    op.add_column(
        QUESTION_TABLE,
        sa.Column("provider_version", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("persona", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("role", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("driver", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("emotional_anchor", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("context_stage", sa.String(length=128), nullable=True),
    )
    op.add_column(
        QUESTION_TABLE,
        sa.Column("seed_type", sa.String(length=64), nullable=True),
    )

    op.create_index(
        "ix_questions_audit_run_persona",
        QUESTION_TABLE,
        ["audit_run_id", "persona"],
    )
    op.create_index(
        "ix_questions_driver_context_stage",
        QUESTION_TABLE,
        ["driver", "context_stage"],
    )

    op.add_column(
        RESPONSE_TABLE,
        sa.Column("emotional_satisfaction", sa.String(length=64), nullable=True),
    )
    op.add_column(
        RESPONSE_TABLE,
        sa.Column("satisfaction_score", sa.Float(), nullable=True),
    )
    op.add_column(
        RESPONSE_TABLE,
        sa.Column("satisfaction_model", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    """Revert persona metadata and satisfaction columns."""
    op.drop_column(RESPONSE_TABLE, "satisfaction_model")
    op.drop_column(RESPONSE_TABLE, "satisfaction_score")
    op.drop_column(RESPONSE_TABLE, "emotional_satisfaction")

    op.drop_index("ix_questions_driver_context_stage", table_name=QUESTION_TABLE)
    op.drop_index("ix_questions_audit_run_persona", table_name=QUESTION_TABLE)

    op.drop_column(QUESTION_TABLE, "seed_type")
    op.drop_column(QUESTION_TABLE, "context_stage")
    op.drop_column(QUESTION_TABLE, "emotional_anchor")
    op.drop_column(QUESTION_TABLE, "driver")
    op.drop_column(QUESTION_TABLE, "role")
    op.drop_column(QUESTION_TABLE, "persona")
    op.drop_column(QUESTION_TABLE, "provider_version")
