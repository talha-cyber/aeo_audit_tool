"""merge v2 report fields with existing changes

Revision ID: a047571e7507
Revises: 26909b33d4d0, b8f5c2d1a9e3
Create Date: 2025-08-28 17:03:20.176127

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "a047571e7507"
down_revision: Union[str, None] = ("26909b33d4d0", "b8f5c2d1a9e3")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
