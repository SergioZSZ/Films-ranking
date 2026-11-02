"""Load processed Rotten Tomatoes movies into PostgreSQL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine

from src.db.session import create_db_engine
from src.db.tables import RtMovie

from .common import (
    PROJECT_ROOT,
    LoadStats,
    clean_text,
    deduplicate_records,
    optional_date,
    optional_float,
    optional_int,
    positive_int,
    print_stats,
    require_columns,
    required_text,
    write_batch,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "processed_datasets"
    / "rottentomatoes"
    / "rotten_tomatoes_movies_clean.csv"
)
REQUIRED_COLUMNS = (
    "rotten_tomatoes_link",
    "movie_title",
    "original_release_date",
    "directors",
    "tomatometer_rating",
    "tomatometer_count",
)


def transform_rt_movie_row(row: Mapping[str, object]) -> dict[str, object]:
    """Map and validate one processed Rotten Tomatoes movie row."""
    return {
        "rt_id": required_text(
            row.get("rotten_tomatoes_link"),
            "rotten_tomatoes_link",
        ),
        "movie_title": required_text(row.get("movie_title"), "movie_title"),
        "original_release_date": optional_date(
            row.get("original_release_date"),
            "original_release_date",
        ),
        "directors": clean_text(row.get("directors")),
        "tomatometer_rating": optional_float(
            row.get("tomatometer_rating"),
            "tomatometer_rating",
            minimum=0,
            maximum=100,
        ),
        "tomatometer_count": optional_int(
            row.get("tomatometer_count"),
            "tomatometer_count",
            minimum=0,
        ),
    }


def load_rt_movies(
    engine: Engine,
    input_path: Path = DEFAULT_INPUT,
    *,
    batch_size: int = 5_000,
) -> LoadStats:
    """Stream, validate and upsert Rotten Tomatoes movies."""
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el CSV de películas RT: {input_path}")
    header = pd.read_csv(input_path, nrows=0, encoding="utf-8")
    require_columns(header.columns, REQUIRED_COLUMNS)
    stats = LoadStats()
    for chunk in pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        chunksize=batch_size,
        encoding="utf-8",
    ):
        stats.rows_read += len(chunk)
        records = [transform_rt_movie_row(row) for row in chunk.to_dict("records")]
        records = deduplicate_records(records, ("rt_id",))
        stats.rows_written += write_batch(
            engine,
            RtMovie.__table__,
            records,
            conflict_columns=("rt_id",),
        )
        stats.batches += 1
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=positive_int, default=5_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    engine = create_db_engine()
    try:
        stats = load_rt_movies(engine, args.input, batch_size=args.batch_size)
        print_stats("rt_movies", stats)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

