"""IMDb movie and user-rating model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .movie_match import MovieMatch


class ImdbMovie(Base):
    """A movie from IMDb with its observed and Bayesian user ratings."""

    __tablename__ = "imdb_movies"
    __table_args__ = (
        CheckConstraint(
            "average_rating BETWEEN 0 AND 10",
            name="average_rating_range",
        ),
        CheckConstraint(
            "bayesian_rating BETWEEN 0 AND 10",
            name="bayesian_rating_range",
        ),
        CheckConstraint("num_votes >= 0", name="num_votes_non_negative"),
    )

    tconst: Mapped[str] = mapped_column(String(16), primary_key=True)
    primary_title: Mapped[str] = mapped_column(String(512), nullable=False)
    original_title: Mapped[str | None] = mapped_column(String(512))
    start_year: Mapped[int | None] = mapped_column(Integer)
    directors: Mapped[str | None] = mapped_column(Text)
    average_rating: Mapped[float] = mapped_column(Float, nullable=False)
    num_votes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bayesian_rating: Mapped[float] = mapped_column(Float, nullable=False)

    matches: Mapped[list[MovieMatch]] = relationship(
        back_populates="imdb_movie",
        cascade="all, delete-orphan",
    )

