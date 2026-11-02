"""Pure calculation and deterministic ordering for the current ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class RankingParameters:
    alpha: float = 0.30
    k: float = 20.0
    prediction_weight: float = 0.80

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha debe estar entre 0 y 1")
        if self.k < 0:
            raise ValueError("k debe ser >= 0")
        if not 0 <= self.prediction_weight <= 1:
            raise ValueError("prediction_weight debe estar entre 0 y 1")


@dataclass(frozen=True)
class MovieScore:
    imdb_id: str
    movie_title: str
    rt_id: str | None
    imdb_bayesian_rating: float
    num_votes: int
    critic_score: float | None
    observed_review_count: int
    predicted_review_count: int
    effective_critic_count: float
    alpha: float
    k: float
    prediction_weight: float
    final_score: float
    rank_position: int | None = None


def _validate_score_sum(score_sum: float, count: int, field: str) -> None:
    if count < 0:
        raise ValueError(f"{field}_count debe ser >= 0")
    if score_sum < 0 or score_sum > count * 10:
        raise ValueError(f"{field}_sum no es compatible con notas en [0, 10]")


def calculate_movie_score(
    *,
    imdb_id: str,
    movie_title: str,
    imdb_bayesian_rating: float,
    num_votes: int,
    rt_id: str | None = None,
    observed_score_sum: float = 0.0,
    observed_review_count: int = 0,
    predicted_score_sum: float = 0.0,
    predicted_review_count: int = 0,
    parameters: RankingParameters | None = None,
) -> MovieScore:
    """Calculate one auditable score, falling back to IMDb without critics."""
    parameters = parameters or RankingParameters()
    if not movie_title.strip():
        raise ValueError("movie_title no puede estar vacío")
    if not 0 <= imdb_bayesian_rating <= 10:
        raise ValueError("imdb_bayesian_rating debe estar entre 0 y 10")
    if num_votes < 0:
        raise ValueError("num_votes debe ser >= 0")
    _validate_score_sum(
        observed_score_sum,
        observed_review_count,
        "observed_score",
    )
    _validate_score_sum(
        predicted_score_sum,
        predicted_review_count,
        "predicted_score",
    )
    if rt_id is None and (observed_review_count or predicted_review_count):
        raise ValueError("No puede haber críticas profesionales sin rt_id")

    effective_count = (
        observed_review_count
        + parameters.prediction_weight * predicted_review_count
    )
    critic_score: float | None = None
    final_score = imdb_bayesian_rating
    if effective_count > 0:
        critic_score = (
            observed_score_sum
            + parameters.prediction_weight * predicted_score_sum
        ) / effective_count
        confidence = effective_count / (effective_count + parameters.k)
        final_score = imdb_bayesian_rating + (
            parameters.alpha
            * confidence
            * (critic_score - imdb_bayesian_rating)
        )

    return MovieScore(
        imdb_id=imdb_id,
        movie_title=movie_title,
        rt_id=rt_id,
        imdb_bayesian_rating=imdb_bayesian_rating,
        num_votes=num_votes,
        critic_score=critic_score,
        observed_review_count=observed_review_count,
        predicted_review_count=predicted_review_count,
        effective_critic_count=effective_count,
        alpha=parameters.alpha,
        k=parameters.k,
        prediction_weight=parameters.prediction_weight,
        final_score=min(10.0, max(0.0, final_score)),
    )


def assign_rank_positions(scores: Iterable[MovieScore]) -> list[MovieScore]:
    """Sort scores deterministically and assign unique one-based positions."""
    ordered = sorted(
        scores,
        key=lambda score: (
            -score.final_score,
            -score.imdb_bayesian_rating,
            -score.num_votes,
            score.imdb_id,
        ),
    )
    return [
        replace(score, rank_position=position)
        for position, score in enumerate(ordered, start=1)
    ]
