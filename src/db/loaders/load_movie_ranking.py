"""Calculate and atomically replace the current all-IMDb movie ranking."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import Engine, and_, case, delete, func, insert, select
from sqlalchemy.sql import Select

from src.db.ranking.calculator import (
    MovieScore,
    RankingParameters,
    assign_rank_positions,
    calculate_movie_score,
)
from src.db.session import create_db_engine
from src.db.tables import ImdbMovie, MovieMatch, MovieRanking, RtReview

from .common import positive_int


@dataclass(frozen=True)
class RankingLoadStats:
    movies: int
    with_verified_rt: int
    with_critic_score: int
    imdb_only: int


def build_ranking_source_query() -> Select:
    """Build the all-IMDb query with optional verified RT review aggregates."""
    reviews = RtReview.__table__
    review_aggregates = (
        select(
            reviews.c.rotten_tomatoes_link.label("rt_id"),
            func.count()
            .filter(reviews.c.score_source == "observed")
            .label("observed_review_count"),
            func.coalesce(
                func.sum(
                    case(
                        (reviews.c.score_source == "observed", reviews.c.final_score),
                        else_=0.0,
                    )
                ),
                0.0,
            ).label("observed_score_sum"),
            func.count()
            .filter(reviews.c.score_source == "predicted")
            .label("predicted_review_count"),
            func.coalesce(
                func.sum(
                    case(
                        (reviews.c.score_source == "predicted", reviews.c.final_score),
                        else_=0.0,
                    )
                ),
                0.0,
            ).label("predicted_score_sum"),
        )
        .where(reviews.c.final_score.is_not(None))
        .group_by(reviews.c.rotten_tomatoes_link)
        .subquery("review_aggregates")
    )
    imdb = ImdbMovie.__table__
    matches = MovieMatch.__table__
    source = imdb.outerjoin(
        matches,
        and_(
            matches.c.imdb_id == imdb.c.tconst,
            matches.c.verified.is_(True),
        ),
    ).outerjoin(
        review_aggregates,
        review_aggregates.c.rt_id == matches.c.rt_id,
    )
    return select(
        imdb.c.tconst.label("imdb_id"),
        imdb.c.primary_title.label("movie_title"),
        imdb.c.bayesian_rating.label("imdb_bayesian_rating"),
        imdb.c.num_votes,
        matches.c.rt_id,
        func.coalesce(review_aggregates.c.observed_review_count, 0).label(
            "observed_review_count"
        ),
        func.coalesce(review_aggregates.c.observed_score_sum, 0.0).label(
            "observed_score_sum"
        ),
        func.coalesce(review_aggregates.c.predicted_review_count, 0).label(
            "predicted_review_count"
        ),
        func.coalesce(review_aggregates.c.predicted_score_sum, 0.0).label(
            "predicted_score_sum"
        ),
    ).select_from(source)


def ranking_to_record(score: MovieScore) -> dict[str, object]:
    """Map a ranked movie to the current ranking table."""
    if score.rank_position is None:
        raise ValueError("La película debe tener rank_position antes de guardarse")
    return {
        "imdb_id": score.imdb_id,
        "movie_title": score.movie_title,
        "rt_id": score.rt_id,
        "imdb_bayesian_rating": score.imdb_bayesian_rating,
        "critic_score": score.critic_score,
        "observed_review_count": score.observed_review_count,
        "predicted_review_count": score.predicted_review_count,
        "effective_critic_count": score.effective_critic_count,
        "alpha": score.alpha,
        "k": score.k,
        "prediction_weight": score.prediction_weight,
        "final_score": score.final_score,
        "rank_position": score.rank_position,
    }


def _calculate_scores(
    engine: Engine,
    parameters: RankingParameters,
) -> list[MovieScore]:
    with engine.connect() as connection:
        rows = connection.execute(build_ranking_source_query()).mappings()
        scores = [
            calculate_movie_score(
                imdb_id=row["imdb_id"],
                movie_title=row["movie_title"],
                imdb_bayesian_rating=float(row["imdb_bayesian_rating"]),
                num_votes=int(row["num_votes"]),
                rt_id=row["rt_id"],
                observed_score_sum=float(row["observed_score_sum"]),
                observed_review_count=int(row["observed_review_count"]),
                predicted_score_sum=float(row["predicted_score_sum"]),
                predicted_review_count=int(row["predicted_review_count"]),
                parameters=parameters,
            )
            for row in rows
        ]
    if not scores:
        raise RuntimeError("No hay películas IMDb; se cancela el reemplazo del ranking")
    return assign_rank_positions(scores)


def load_movie_ranking(
    engine: Engine,
    *,
    parameters: RankingParameters | None = None,
    batch_size: int = 2_500,
) -> RankingLoadStats:
    """Calculate all scores, then replace the current ranking atomically."""
    parameters = parameters or RankingParameters()
    scores = _calculate_scores(engine, parameters)
    records = [ranking_to_record(score) for score in scores]

    with engine.begin() as connection:
        connection.execute(delete(MovieRanking.__table__))
        for offset in range(0, len(records), batch_size):
            connection.execute(
                insert(MovieRanking.__table__),
                records[offset : offset + batch_size],
            )

    with_rt = sum(score.rt_id is not None for score in scores)
    with_critics = sum(score.critic_score is not None for score in scores)
    return RankingLoadStats(
        movies=len(scores),
        with_verified_rt=with_rt,
        with_critic_score=with_critics,
        imdb_only=len(scores) - with_rt,
    )


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("debe ser >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=_unit_float, default=0.30)
    parser.add_argument("--k", type=_non_negative_float, default=20.0)
    parser.add_argument("--prediction-weight", type=_unit_float, default=0.80)
    parser.add_argument("--batch-size", type=positive_int, default=2_500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parameters = RankingParameters(
        alpha=args.alpha,
        k=args.k,
        prediction_weight=args.prediction_weight,
    )
    engine = create_db_engine()
    try:
        stats = load_movie_ranking(
            engine,
            parameters=parameters,
            batch_size=args.batch_size,
        )
    finally:
        engine.dispose()
    print(
        f"movie_ranking: películas={stats.movies:,}, "
        f"con RT verificado={stats.with_verified_rt:,}, "
        f"con señal profesional={stats.with_critic_score:,}, "
        f"solo IMDb={stats.imdb_only:,}"
    )


if __name__ == "__main__":
    main()
