"""Load scored Rotten Tomatoes critic reviews into PostgreSQL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine

from src.db.session import create_db_engine
from src.db.tables import RtReview

from .common import (
    PROJECT_ROOT,
    LoadStats,
    clean_text,
    deduplicate_records,
    optional_bool,
    optional_date,
    optional_float,
    positive_int,
    print_stats,
    require_columns,
    required_text,
    review_source_hash,
    write_batch,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "processed_datasets"
    / "rottentomatoes"
    / "rotten_tomatoes_critic_reviews_scored.csv"
)
REQUIRED_COLUMNS = (
    "rotten_tomatoes_link",
    "critic_name",
    "publisher_name",
    "review_date",
    "top_critic",
    "review_type",
    "review_score",
    "review_content",
    "normalized_score",
    "predicted_score",
    "final_score",
    "score_source",
    "prediction_model",
    "prediction_metadata",
)
SOURCE_MAPPING = {
    "normalized_original": ("observed", None),
    "roberta_prediction": ("predicted", None),
    "unavailable_unparseable": ("unavailable", "unparseable_score"),
    "unavailable_missing_text": ("unavailable", "missing_text"),
}


def transform_rt_review_row(
    row: Mapping[str, object],
    prediction_run_id: str,
) -> dict[str, object]:
    """Map one scored review while preserving score and model provenance."""
    source_value = required_text(row.get("score_source"), "score_source")
    try:
        score_source, unavailable_reason = SOURCE_MAPPING[source_value]
    except KeyError as exc:
        raise ValueError(f"score_source desconocido: {source_value!r}") from exc

    review_type = required_text(row.get("review_type"), "review_type")
    if review_type not in {"Fresh", "Rotten"}:
        raise ValueError(f"review_type desconocido: {review_type!r}")

    normalized_score = optional_float(
        row.get("normalized_score"),
        "normalized_score",
        minimum=0,
        maximum=10,
    )
    predicted_score = optional_float(
        row.get("predicted_score"),
        "predicted_score",
        minimum=0,
        maximum=10,
    )
    final_score = optional_float(
        row.get("final_score"),
        "final_score",
        minimum=0,
        maximum=10,
    )

    if score_source == "observed":
        if normalized_score is None or final_score is None:
            raise ValueError("Una reseña observada necesita normalized_score y final_score")
        if not math_isclose(normalized_score, final_score):
            raise ValueError("final_score debe coincidir con normalized_score")
    elif score_source == "predicted":
        if predicted_score is None or final_score is None:
            raise ValueError("Una reseña predicha necesita predicted_score y final_score")
        if not math_isclose(predicted_score, final_score):
            raise ValueError("final_score debe coincidir con predicted_score")
        if not prediction_run_id.strip():
            raise ValueError("prediction_run_id no puede estar vacío")
    elif final_score is not None:
        raise ValueError("Una reseña no disponible no puede tener final_score")

    is_prediction = score_source == "predicted"
    return {
        "source_hash": review_source_hash(row),
        "rotten_tomatoes_link": required_text(
            row.get("rotten_tomatoes_link"),
            "rotten_tomatoes_link",
        ),
        "critic_name": clean_text(row.get("critic_name")),
        "publisher_name": clean_text(row.get("publisher_name")),
        "review_date": optional_date(row.get("review_date"), "review_date"),
        "top_critic": optional_bool(row.get("top_critic"), "top_critic"),
        "review_type": review_type,
        "review_score": clean_text(row.get("review_score")),
        "normalized_score": normalized_score,
        "predicted_score": predicted_score,
        "final_score": final_score,
        "score_source": score_source,
        "unavailable_reason": unavailable_reason,
        "review_content": clean_text(row.get("review_content")),
        "model_name": clean_text(row.get("prediction_model")) if is_prediction else None,
        "model_version": (
            clean_text(row.get("prediction_metadata")) if is_prediction else None
        ),
        "prediction_run_id": prediction_run_id.strip() if is_prediction else None,
    }


def math_isclose(left: float, right: float) -> bool:
    """Compare serialized scores without hiding meaningful differences."""
    return abs(left - right) <= 1e-9


def load_rt_reviews(
    engine: Engine,
    input_path: Path = DEFAULT_INPUT,
    *,
    prediction_run_id: str,
    batch_size: int = 2_500,
) -> LoadStats:
    """Stream, validate and upsert scored critic reviews."""
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el CSV de reseñas RT: {input_path}")
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
        records = [
            transform_rt_review_row(row, prediction_run_id)
            for row in chunk.to_dict("records")
        ]
        records = deduplicate_records(records, ("source_hash",))
        stats.rows_written += write_batch(
            engine,
            RtReview.__table__,
            records,
            conflict_columns=("source_hash",),
        )
        stats.batches += 1
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prediction-run-id", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=2_500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    engine = create_db_engine()
    try:
        stats = load_rt_reviews(
            engine,
            args.input,
            prediction_run_id=args.prediction_run_id,
            batch_size=args.batch_size,
        )
        print_stats("rt_reviews", stats)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
