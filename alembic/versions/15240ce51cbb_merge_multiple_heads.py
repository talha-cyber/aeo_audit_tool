"""merge multiple heads

Revision ID: 15240ce51cbb
Revises: b00e2b865774, e4c8d2b5f1a9
Create Date: 2025-08-26 11:17:01.495532

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "15240ce51cbb"
down_revision: Union[str, None] = ("b00e2b865774", "e4c8d2b5f1a9")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
