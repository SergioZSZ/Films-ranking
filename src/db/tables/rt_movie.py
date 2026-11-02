"""Rotten Tomatoes movie model."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .movie_match import MovieMatch
    from .rt_review import RtReview


class RtMovie(Base):
    """A Rotten Tomatoes movie used for reviews and source matching."""

    __tablename__ = "rt_movies"
    __table_args__ = (
        CheckConstraint(
            "tomatometer_rating IS NULL OR "
            "tomatometer_rating BETWEEN 0 AND 100",
            name="tomatometer_rating_range",
        ),
        CheckConstraint(
            "tomatometer_count IS NULL OR tomatometer_count >= 0",
            name="tomatometer_count_non_negative",
        ),
    )

    rt_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    movie_title: Mapped[str] = mapped_column(String(512), nullable=False)
    original_release_date: Mapped[date | None] = mapped_column(Date)
    directors: Mapped[str | None] = mapped_column(Text)
    tomatometer_rating: Mapped[float | None] = mapped_column(Float)
    tomatometer_count: Mapped[int | None] = mapped_column(Integer)

    reviews: Mapped[list[RtReview]] = relationship(
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    matches: Mapped[list[MovieMatch]] = relationship(
        back_populates="rt_movie",
        cascade="all, delete-orphan",
    )

