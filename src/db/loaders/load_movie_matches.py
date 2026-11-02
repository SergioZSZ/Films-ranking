"""Match loaded IMDb and Rotten Tomatoes movies and persist candidates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sqlalchemy import Engine, select

from src.db.matching.matcher import (
    ImdbFilm,
    MatchCandidate,
    RtFilm,
    UnmatchedFilm,
    match_movies,
)
from src.db.session import create_db_engine
from src.db.tables import ImdbMovie, MovieMatch, RtMovie

from .common import PROJECT_ROOT, positive_int, write_batch


DEFAULT_UNMATCHED_REPORT = (
    PROJECT_ROOT
    / "processed_datasets"
    / "matches"
    / "unmatched_rt_movies.csv"
)


@dataclass(frozen=True)
class MatchLoadStats:
    imdb_movies: int
    rt_movies: int
    candidates: int
    verified: int
    unmatched: int
    methods: dict[str, int]


def candidate_to_record(candidate: MatchCandidate) -> dict[str, object]:
    """Map a pure match candidate to `movie_matches` columns."""
    return {
        "imdb_id": candidate.imdb_id,
        "rt_id": candidate.rt_id,
        "match_method": candidate.match_method,
        "match_confidence": candidate.match_confidence,
        "verified": candidate.verified,
    }


def write_unmatched_report(
    output_path: Path,
    unmatched: Sequence[UnmatchedFilm],
) -> None:
    """Write RT movies without deterministic candidates to a UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("rt_id", "movie_title", "release_year", "reason"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "rt_id": movie.rt_id,
                "movie_title": movie.title,
                "release_year": movie.year,
                "reason": movie.reason,
            }
            for movie in unmatched
        )


def _read_imdb_movies(engine: Engine) -> list[ImdbFilm]:
    statement = select(
        ImdbMovie.tconst,
        ImdbMovie.primary_title,
        ImdbMovie.original_title,
        ImdbMovie.start_year,
        ImdbMovie.directors,
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings()
        return [
            ImdbFilm(
                row["tconst"],
                row["primary_title"],
                row["original_title"],
                row["start_year"],
                row["directors"],
            )
            for row in rows
        ]


def _read_rt_movies(engine: Engine) -> list[RtFilm]:
    statement = select(
        RtMovie.rt_id,
        RtMovie.movie_title,
        RtMovie.original_release_date,
        RtMovie.directors,
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings()
        return [
            RtFilm(
                row["rt_id"],
                row["movie_title"],
                (
                    row["original_release_date"].year
                    if row["original_release_date"] is not None
                    else None
                ),
                row["directors"],
            )
            for row in rows
        ]


def load_movie_matches(
    engine: Engine,
    *,
    unmatched_report: Path = DEFAULT_UNMATCHED_REPORT,
    batch_size: int = 5_000,
) -> MatchLoadStats:
    """Build deterministic candidates, upsert them and report unmatched RT rows."""
    imdb_movies = _read_imdb_movies(engine)
    rt_movies = _read_rt_movies(engine)
    result = match_movies(imdb_movies, rt_movies)
    records = [candidate_to_record(candidate) for candidate in result.candidates]

    for offset in range(0, len(records), batch_size):
        write_batch(
            engine,
            MovieMatch.__table__,
            records[offset : offset + batch_size],
            conflict_columns=("imdb_id", "rt_id"),
        )
    write_unmatched_report(unmatched_report, result.unmatched)

    methods = Counter(candidate.match_method for candidate in result.candidates)
    return MatchLoadStats(
        imdb_movies=len(imdb_movies),
        rt_movies=len(rt_movies),
        candidates=len(result.candidates),
        verified=sum(candidate.verified for candidate in result.candidates),
        unmatched=len(result.unmatched),
        methods=dict(sorted(methods.items())),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unmatched-report",
        type=Path,
        default=DEFAULT_UNMATCHED_REPORT,
    )
    parser.add_argument("--batch-size", type=positive_int, default=5_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    engine = create_db_engine()
    try:
        stats = load_movie_matches(
            engine,
            unmatched_report=args.unmatched_report,
            batch_size=args.batch_size,
        )
    finally:
        engine.dispose()

    print(
        f"movie_matches: IMDb={stats.imdb_movies:,}, RT={stats.rt_movies:,}, "
        f"candidatas={stats.candidates:,}, verificadas={stats.verified:,}, "
        f"sin candidato={stats.unmatched:,}"
    )
    for method, count in stats.methods.items():
        print(f"  {method}: {count:,}")
    print(f"Informe de no resueltas: {args.unmatched_report}")


if __name__ == "__main__":
    main()
