"""IMDb to Rotten Tomatoes movie-match model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .imdb_movie import ImdbMovie
    from .rt_movie import RtMovie


class MovieMatch(Base):
    """An auditable candidate or verified match between both sources."""

    __tablename__ = "movie_matches"
    __table_args__ = (
        CheckConstraint(
            "match_confidence IS NULL OR match_confidence BETWEEN 0 AND 1",
            name="match_confidence_range",
        ),
        Index(
            "uq_movie_matches_verified_imdb_id",
            "imdb_id",
            unique=True,
            postgresql_where=text("verified IS TRUE"),
        ),
        Index(
            "uq_movie_matches_verified_rt_id",
            "rt_id",
            unique=True,
            postgresql_where=text("verified IS TRUE"),
        ),
    )

    imdb_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("imdb_movies.tconst", ondelete="CASCADE"),
        primary_key=True,
    )
    rt_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rt_movies.rt_id", ondelete="CASCADE"),
        primary_key=True,
    )
    match_method: Mapped[str] = mapped_column(String(64), nullable=False)
    match_confidence: Mapped[float | None] = mapped_column(Float)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    imdb_movie: Mapped[ImdbMovie] = relationship(back_populates="matches")
    rt_movie: Mapped[RtMovie] = relationship(back_populates="matches")

