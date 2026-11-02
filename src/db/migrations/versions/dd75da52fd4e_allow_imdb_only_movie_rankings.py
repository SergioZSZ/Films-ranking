"""allow imdb only movie rankings

Revision ID: dd75da52fd4e
Revises: 26c3d9007a6e
Create Date: 2026-10-30 17:39:27.838995

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dd75da52fd4e'
down_revision: Union[str, Sequence[str], None] = '26c3d9007a6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(
        op.f('fk_movie_ranking_match'),
        'movie_ranking',
        type_='foreignkey',
    )
    op.drop_constraint(
        op.f('pk_movie_ranking'),
        'movie_ranking',
        type_='primary',
    )
    op.alter_column('movie_ranking', 'rt_id',
               existing_type=sa.VARCHAR(length=255),
               nullable=True)
    op.alter_column('movie_ranking', 'critic_score',
               existing_type=sa.DOUBLE_PRECISION(precision=53),
               nullable=True)
    op.create_primary_key(
        op.f('pk_movie_ranking'),
        'movie_ranking',
        ['imdb_id'],
    )
    op.create_foreign_key(op.f('fk_movie_ranking_imdb_id_imdb_movies'), 'movie_ranking', 'imdb_movies', ['imdb_id'], ['tconst'], ondelete='CASCADE')
    op.create_foreign_key(op.f('fk_movie_ranking_rt_id_rt_movies'), 'movie_ranking', 'rt_movies', ['rt_id'], ['rt_id'], ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('fk_movie_ranking_rt_id_rt_movies'), 'movie_ranking', type_='foreignkey')
    op.drop_constraint(op.f('fk_movie_ranking_imdb_id_imdb_movies'), 'movie_ranking', type_='foreignkey')
    op.drop_constraint(
        op.f('pk_movie_ranking'),
        'movie_ranking',
        type_='primary',
    )
    op.alter_column('movie_ranking', 'critic_score',
               existing_type=sa.DOUBLE_PRECISION(precision=53),
               nullable=False)
    op.alter_column('movie_ranking', 'rt_id',
               existing_type=sa.VARCHAR(length=255),
               nullable=False)
    op.create_primary_key(
        op.f('pk_movie_ranking'),
        'movie_ranking',
        ['imdb_id', 'rt_id'],
    )
    op.create_foreign_key(op.f('fk_movie_ranking_match'), 'movie_ranking', 'movie_matches', ['imdb_id', 'rt_id'], ['imdb_id', 'rt_id'], ondelete='CASCADE')
