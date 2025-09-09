"""Add v2 report fields for sentiment and metadata

Revision ID: b8f5c2d1a9e3
Revises: ac553b04ff00
Create Date: 2025-08-28 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8f5c2d1a9e3"
down_revision: Union[str, None] = "ac553b04ff00"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add sentiment field to responses table for real sentiment analysis
    op.add_column(
        "responses",
        sa.Column("sentiment", sa.Float(), nullable=True),  # -1..+1 scale
    )

    # Add prompt basket version to audit runs for comparability
    op.add_column(
        "auditrun",
        sa.Column("prompt_basket_version", sa.String(length=64), nullable=True),
    )

    # Add report versioning fields to reports table
    op.add_column(
        "report",
        sa.Column("template_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "report",
        sa.Column("theme_key", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    # Remove added columns in reverse order
    op.drop_column("report", "theme_key")
    op.drop_column("report", "template_version")
    op.drop_column("auditrun", "prompt_basket_version")
    op.drop_column("responses", "sentiment")
