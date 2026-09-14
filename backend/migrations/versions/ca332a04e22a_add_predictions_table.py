"""add predictions table

Revision ID: ca332a04e22a
Revises: 58046c4ffb32
Create Date: 2026-09-04 23:37:20.592479

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca332a04e22a'
down_revision: Union[str, Sequence[str], None] = '58046c4ffb32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("exam_score", sa.Integer(), nullable=False),
        sa.Column("years_exp", sa.Integer(), nullable=False),
        sa.Column("predicted_salary", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("predictions")
