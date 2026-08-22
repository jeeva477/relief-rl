"""create hazards table

Revision ID: 0001
Revises:
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hazards",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("radius_m", sa.Float(), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("hazard_type", sa.String(length=64), nullable=False),
        sa.Column("hard_constraint", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("ix_hazards_active", "hazards", ["active"])
    op.create_index("ix_hazards_lat_lng", "hazards", ["latitude", "longitude"])


def downgrade() -> None:
    op.drop_index("ix_hazards_lat_lng", table_name="hazards")
    op.drop_index("ix_hazards_active", table_name="hazards")
    op.drop_table("hazards")
