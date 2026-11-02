"""Final auditable movie-ranking model."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MovieRanking(Base):
    """Final score plus every input needed to reproduce it."""

    __tablename__ = "movie_ranking"
    __table_args__ = (
        UniqueConstraint("rank_position", name="uq_movie_ranking_position"),
        CheckConstraint(
            "imdb_bayesian_rating BETWEEN 0 AND 10",
            name="imdb_bayesian_rating_range",
        ),
        CheckConstraint(
            "critic_score BETWEEN 0 AND 10",
            name="critic_score_range",
        ),
        CheckConstraint(
            "final_score BETWEEN 0 AND 10",
            name="final_score_range",
        ),
        CheckConstraint(
            "observed_review_count >= 0",
            name="observed_review_count_non_negative",
        ),
        CheckConstraint(
            "predicted_review_count >= 0",
            name="predicted_review_count_non_negative",
        ),
        CheckConstraint(
            "effective_critic_count >= 0",
            name="effective_critic_count_non_negative",
        ),
        CheckConstraint("alpha BETWEEN 0 AND 1", name="alpha_range"),
        CheckConstraint("k >= 0", name="k_non_negative"),
        CheckConstraint(
            "prediction_weight BETWEEN 0 AND 1",
            name="prediction_weight_range",
        ),
        CheckConstraint("rank_position > 0", name="rank_position_positive"),
    )

    imdb_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("imdb_movies.tconst", ondelete="CASCADE"),
        primary_key=True,
    )
    movie_title: Mapped[str] = mapped_column(String(512), nullable=False)
    rt_id: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("rt_movies.rt_id", ondelete="SET NULL"),
    )
    imdb_bayesian_rating: Mapped[float] = mapped_column(Float, nullable=False)
    critic_score: Mapped[float | None] = mapped_column(Float)
    observed_review_count: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_review_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_critic_count: Mapped[float] = mapped_column(Float, nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    k: Mapped[float] = mapped_column(Float, nullable=False, default=20.0)
    prediction_weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.80,
    )
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


