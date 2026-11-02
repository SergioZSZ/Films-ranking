"""Unit tests for processed-dataset database loaders."""

from __future__ import annotations

import subprocess
import sys
import unittest

from sqlalchemy.dialects import postgresql

from src.db.loaders.common import (
    build_upsert_statement,
    optional_int,
    review_source_hash,
)
from src.db.loaders.load_imdb_movies import transform_imdb_row
from src.db.loaders.load_rt_movies import transform_rt_movie_row
from src.db.loaders.load_rt_reviews import (
    parse_args as parse_rt_review_args,
    transform_rt_review_row,
)
from src.db.tables import ImdbMovie


class CommonLoaderTest(unittest.TestCase):
    def test_accepts_integer_values_serialized_as_decimals(self) -> None:
        self.assertEqual(optional_int("1994.0", "startYear"), 1994)
        with self.assertRaisesRegex(ValueError, "entero"):
            optional_int("1994.5", "startYear")

    def test_loader_modules_run_without_package_import_warnings(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "src.db.loaders.load_imdb_movies",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_review_hash_is_stable_and_ignores_model_outputs(self) -> None:
        original = {
            "rotten_tomatoes_link": "m/example",
            "critic_name": "Critic",
            "publisher_name": "Publisher",
            "review_date": "2020-01-02",
            "review_type": "Fresh",
            "review_score": "4/5",
            "review_content": "  Great film.  ",
        }
        with_prediction = {
            **original,
            "predicted_score": "8.2",
            "prediction_model": "new-model",
        }

        self.assertEqual(
            review_source_hash(original),
            review_source_hash(with_prediction),
        )
        self.assertEqual(len(review_source_hash(original)), 64)

    def test_upsert_targets_the_natural_key(self) -> None:
        statement = build_upsert_statement(
            ImdbMovie.__table__,
            [{"tconst": "tt0000001", "primary_title": "Example"}],
            conflict_columns=("tconst",),
        )
        sql = str(
            statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

        self.assertIn("ON CONFLICT (tconst) DO UPDATE", sql)


class ImdbLoaderTest(unittest.TestCase):
    def test_maps_processed_imdb_columns(self) -> None:
        record = transform_imdb_row(
            {
                "tconst": "tt0000001",
                "primaryTitle": "Primary",
                "originalTitle": "Original",
                "startYear": "1999",
                "directors": "Director One,Director Two",
                "averageRating": "7.5",
                "numVotes": "1200",
                "bayesianRating": "7.2",
            }
        )

        self.assertEqual(record["primary_title"], "Primary")
        self.assertEqual(record["start_year"], 1999)
        self.assertEqual(record["num_votes"], 1200)
        self.assertEqual(record["bayesian_rating"], 7.2)

    def test_uses_original_title_as_fallback_and_skips_untitled_rows(self) -> None:
        base = {
            "tconst": "tt0000001",
            "primaryTitle": "",
            "originalTitle": "Original",
            "startYear": "1999",
            "directors": "Director",
            "averageRating": "7.5",
            "numVotes": "1200",
            "bayesianRating": "7.2",
        }

        fallback = transform_imdb_row(base)
        skipped = transform_imdb_row({**base, "originalTitle": ""})

        self.assertEqual(fallback["primary_title"], "Original")
        self.assertIsNone(skipped)


class RottenTomatoesMovieLoaderTest(unittest.TestCase):
    def test_maps_nullable_movie_fields(self) -> None:
        record = transform_rt_movie_row(
            {
                "rotten_tomatoes_link": "m/example",
                "movie_title": "Example",
                "original_release_date": "",
                "directors": "",
                "tomatometer_rating": "85",
                "tomatometer_count": "100",
            }
        )

        self.assertEqual(record["rt_id"], "m/example")
        self.assertIsNone(record["original_release_date"])
        self.assertIsNone(record["directors"])
        self.assertEqual(record["tomatometer_rating"], 85.0)


class RottenTomatoesReviewLoaderTest(unittest.TestCase):
    def test_default_batch_stays_below_postgresql_parameter_limit(self) -> None:
        args = parse_rt_review_args(["--prediction-run-id", "run-1"])

        self.assertLessEqual(args.batch_size, 3_000)

    def review(self, source: str) -> dict[str, str]:
        return {
            "rotten_tomatoes_link": "m/example",
            "critic_name": "Critic",
            "publisher_name": "Publisher",
            "review_date": "2020-01-02",
            "top_critic": "True",
            "review_type": "Fresh",
            "review_score": "",
            "review_content": "Great film.",
            "normalized_score": "",
            "predicted_score": "",
            "final_score": "",
            "score_source": source,
            "prediction_model": "",
            "prediction_metadata": "",
        }

    def test_maps_observed_review(self) -> None:
        row = self.review("normalized_original")
        row.update(normalized_score="8", final_score="8")

        record = transform_rt_review_row(row, "run-1")

        self.assertEqual(record["score_source"], "observed")
        self.assertEqual(record["normalized_score"], 8.0)
        self.assertIsNone(record["unavailable_reason"])
        self.assertIsNone(record["prediction_run_id"])

    def test_maps_predicted_review_and_model_provenance(self) -> None:
        row = self.review("roberta_prediction")
        row.update(
            predicted_score="7.125",
            final_score="7.125",
            prediction_model="epac-roberta-residual-full",
            prediction_metadata="training_metadata.json",
        )

        record = transform_rt_review_row(row, "kaggle-full-2026-10-30")

        self.assertEqual(record["score_source"], "predicted")
        self.assertEqual(record["model_name"], "epac-roberta-residual-full")
        self.assertEqual(record["model_version"], "training_metadata.json")
        self.assertEqual(record["prediction_run_id"], "kaggle-full-2026-10-30")

    def test_maps_both_unavailable_reasons(self) -> None:
        missing_text = transform_rt_review_row(
            self.review("unavailable_missing_text"),
            "run-1",
        )
        unparseable = transform_rt_review_row(
            self.review("unavailable_unparseable"),
            "run-1",
        )

        self.assertEqual(missing_text["score_source"], "unavailable")
        self.assertEqual(missing_text["unavailable_reason"], "missing_text")
        self.assertEqual(unparseable["unavailable_reason"], "unparseable_score")
        self.assertIsNone(unparseable["final_score"])

    def test_rejects_unknown_score_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "score_source"):
            transform_rt_review_row(self.review("unknown"), "run-1")


if __name__ == "__main__":
    unittest.main()
