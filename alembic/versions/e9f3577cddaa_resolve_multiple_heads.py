"""Resolve multiple heads

Revision ID: e9f3577cddaa
Revises: create_scheduling_tables, d2bb41f94837
Create Date: 2025-10-01 13:58:27.388883

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9f3577cddaa'
down_revision: Union[str, None] = ('create_scheduling_tables', 'd2bb41f94837')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
