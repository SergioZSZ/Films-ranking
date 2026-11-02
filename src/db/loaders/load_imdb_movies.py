"""Load processed IMDb movies and Bayesian ratings into PostgreSQL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine

from src.db.session import create_db_engine
from src.db.tables import ImdbMovie

from .common import (
    PROJECT_ROOT,
    LoadStats,
    clean_text,
    deduplicate_records,
    optional_int,
    positive_int,
    print_stats,
    require_columns,
    required_float,
    required_int,
    required_text,
    write_batch,
)


DEFAULT_INPUT = PROJECT_ROOT / "processed_datasets" / "imdb" / "imdb_peliculas_bayesiano.tsv"
REQUIRED_COLUMNS = (
    "tconst",
    "primaryTitle",
    "originalTitle",
    "startYear",
    "directors",
    "averageRating",
    "numVotes",
    "bayesianRating",
)


def transform_imdb_row(row: Mapping[str, object]) -> dict[str, object] | None:
    """Map one IMDb row, or skip it when both source titles are missing."""
    original_title = clean_text(row.get("originalTitle"))
    primary_title = clean_text(row.get("primaryTitle")) or original_title
    if primary_title is None:
        return None
    return {
        "tconst": required_text(row.get("tconst"), "tconst"),
        "primary_title": primary_title,
        "original_title": original_title,
        "start_year": optional_int(row.get("startYear"), "startYear", minimum=0),
        "directors": clean_text(row.get("directors")),
        "average_rating": required_float(
            row.get("averageRating"),
            "averageRating",
            minimum=0,
            maximum=10,
        ),
        "num_votes": required_int(row.get("numVotes"), "numVotes", minimum=0),
        "bayesian_rating": required_float(
            row.get("bayesianRating"),
            "bayesianRating",
            minimum=0,
            maximum=10,
        ),
    }


def load_imdb_movies(
    engine: Engine,
    input_path: Path = DEFAULT_INPUT,
    *,
    batch_size: int = 5_000,
) -> LoadStats:
    """Stream, validate and upsert the processed IMDb dataset."""
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el TSV de IMDb: {input_path}")
    header = pd.read_csv(input_path, sep="\t", nrows=0, encoding="utf-8")
    require_columns(header.columns, REQUIRED_COLUMNS)
    stats = LoadStats()
    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=batch_size,
        encoding="utf-8",
    ):
        stats.rows_read += len(chunk)
        transformed = [
            transform_imdb_row(row) for row in chunk.to_dict("records")
        ]
        records = [record for record in transformed if record is not None]
        stats.rows_skipped += len(transformed) - len(records)
        records = deduplicate_records(records, ("tconst",))
        stats.rows_written += write_batch(
            engine,
            ImdbMovie.__table__,
            records,
            conflict_columns=("tconst",),
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
        stats = load_imdb_movies(engine, args.input, batch_size=args.batch_size)
        print_stats("imdb_movies", stats)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

