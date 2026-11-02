"""Add the IMDb title to the materialized movie ranking.

Revision ID: 4ad21a86c9b7
Revises: dd75da52fd4e
Create Date: 2026-10-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4ad21a86c9b7"
down_revision: Union[str, Sequence[str], None] = "dd75da52fd4e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add and backfill the required IMDb movie title."""
    op.add_column(
        "movie_ranking",
        sa.Column("movie_title", sa.String(length=512), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE movie_ranking AS ranking
            SET movie_title = imdb.primary_title
            FROM imdb_movies AS imdb
            WHERE imdb.tconst = ranking.imdb_id
            """
        )
    )
    op.alter_column(
        "movie_ranking",
        "movie_title",
        existing_type=sa.String(length=512),
        nullable=False,
    )


def downgrade() -> None:
    """Remove the duplicated display title from the ranking."""
    op.drop_column("movie_ranking", "movie_title")
