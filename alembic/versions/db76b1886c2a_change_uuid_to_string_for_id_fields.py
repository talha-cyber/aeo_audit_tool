"""Change UUID to String for ID fields

Revision ID: db76b1886c2a
Revises: cfbb21270e7d
Create Date: 2025-07-31 14:50:07.483583

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "db76b1886c2a"
down_revision: Union[str, None] = "cfbb21270e7d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("report_audit_run_id_fkey", "report", type_="foreignkey")
    op.drop_constraint("responses_audit_run_id_fkey", "responses", type_="foreignkey")
    op.alter_column(
        "auditrun",
        "id",
        existing_type=sa.UUID(),
        type_=sa.String(),
        existing_nullable=False,
    )
    op.alter_column(
        "responses",
        "id",
        existing_type=sa.UUID(),
        type_=sa.String(),
        existing_nullable=False,
    )
    op.alter_column(
        "responses",
        "audit_run_id",
        existing_type=sa.UUID(),
        type_=sa.String(),
        existing_nullable=True,
    )
    op.alter_column(
        "report",
        "audit_run_id",
        existing_type=sa.UUID(),
        type_=sa.String(),
        existing_nullable=True,
    )
    op.create_foreign_key(
        "report_audit_run_id_fkey", "report", "auditrun", ["audit_run_id"], ["id"]
    )
    op.create_foreign_key(
        "responses_audit_run_id_fkey", "responses", "auditrun", ["audit_run_id"], ["id"]
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "responses",
        "audit_run_id",
        existing_type=sa.String(),
        type_=sa.UUID(),
        existing_nullable=True,
    )
    op.alter_column(
        "responses",
        "id",
        existing_type=sa.String(),
        type_=sa.UUID(),
        existing_nullable=False,
    )
    op.alter_column(
        "auditrun",
        "id",
        existing_type=sa.String(),
        type_=sa.UUID(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###
