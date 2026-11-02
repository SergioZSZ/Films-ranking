"""Rotten Tomatoes critic-review model."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Identity,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .rt_movie import RtMovie


class RtReview(Base):
    """A professional review with observed or inferred score provenance."""

    __tablename__ = "rt_reviews"
    __table_args__ = (
        CheckConstraint(
            "review_type IN ('Fresh', 'Rotten')",
            name="review_type_values",
        ),
        CheckConstraint(
            "normalized_score IS NULL OR normalized_score BETWEEN 0 AND 10",
            name="normalized_score_range",
        ),
        CheckConstraint(
            "predicted_score IS NULL OR predicted_score BETWEEN 0 AND 10",
            name="predicted_score_range",
        ),
        CheckConstraint(
            "final_score IS NULL OR final_score BETWEEN 0 AND 10",
            name="final_score_range",
        ),
        CheckConstraint(
            "score_source IN ('observed', 'predicted', 'unavailable')",
            name="score_source_values",
        ),
        CheckConstraint(
            "score_source != 'observed' OR normalized_score IS NOT NULL",
            name="observed_score_present",
        ),
        CheckConstraint(
            "score_source != 'predicted' OR predicted_score IS NOT NULL",
            name="predicted_score_present",
        ),
        CheckConstraint(
            "score_source != 'unavailable' OR final_score IS NULL",
            name="unavailable_score_is_null",
        ),
        CheckConstraint(
            "char_length(source_hash) = 64",
            name="source_hash_length",
        ),
        CheckConstraint(
            "score_source != 'unavailable' OR unavailable_reason IS NOT NULL",
            name="unavailable_reason_present",
        ),
        CheckConstraint(
            "score_source = 'unavailable' OR unavailable_reason IS NULL",
            name="available_reason_is_null",
        ),
    )

    review_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    source_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    rotten_tomatoes_link: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rt_movies.rt_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    critic_name: Mapped[str | None] = mapped_column(String(255))
    publisher_name: Mapped[str | None] = mapped_column(String(255))
    review_date: Mapped[date | None] = mapped_column(Date)
    top_critic: Mapped[bool | None] = mapped_column(Boolean)
    review_type: Mapped[str] = mapped_column(String(6), nullable=False)
    review_score: Mapped[str | None] = mapped_column(String(64))
    normalized_score: Mapped[float | None] = mapped_column(Float)
    predicted_score: Mapped[float | None] = mapped_column(Float)
    final_score: Mapped[float | None] = mapped_column(Float)
    score_source: Mapped[str] = mapped_column(String(16), nullable=False)
    unavailable_reason: Mapped[str | None] = mapped_column(String(64))
    review_content: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(255))
    model_version: Mapped[str | None] = mapped_column(String(255))
    prediction_run_id: Mapped[str | None] = mapped_column(String(255))

    movie: Mapped[RtMovie] = relationship(back_populates="reviews")

