"""Add internal client flag and persona client scoping

Revision ID: f3d2c9c1d5ab
Revises: 6164b2d1831c
Create Date: 2025-10-04 12:15:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3d2c9c1d5ab"
down_revision: Union[str, None] = "6164b2d1831c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "client",
        sa.Column(
            "is_internal",
            sa.Boolean(),
            nullable=False,
            server_default=sa.sql.expression.false(),
        ),
    )
    op.add_column(
        "client",
        sa.Column(
            "admin_settings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )

    op.add_column(
        "personas",
        sa.Column("client_id", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_personas_client_id",
        "personas",
        ["client_id"],
        unique=False,
    )

    connection = op.get_bind()
    personas_table = sa.sql.table(
        "personas",
        sa.Column("client_id", sa.String(length=255)),
        sa.Column("owner_id", sa.String(length=255)),
    )
    connection.execute(
        personas_table.update()
        .where(personas_table.c.client_id.is_(None))
        .values(client_id=personas_table.c.owner_id)
    )

    op.alter_column(
        "personas",
        "client_id",
        existing_type=sa.String(length=255),
        nullable=False,
    )
    op.create_foreign_key(
        "fk_personas_client",
        "personas",
        "client",
        ["client_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_personas_client", "personas", type_="foreignkey")
    op.alter_column(
        "personas",
        "client_id",
        existing_type=sa.String(length=255),
        nullable=True,
    )
    op.drop_index("ix_personas_client_id", table_name="personas")
    op.drop_column("personas", "client_id")

    op.drop_column("client", "is_internal")
    op.drop_column("client", "admin_settings")
