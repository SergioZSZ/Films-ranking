"""Schema-level tests for the SQLAlchemy ranking models."""

from __future__ import annotations

import unittest

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from src.db.tables import Base


class DatabaseTablesTest(unittest.TestCase):
    """Verify the schema contract without requiring a running database."""

    def test_metadata_contains_only_the_five_ranking_tables(self) -> None:
        self.assertEqual(
            set(Base.metadata.tables),
            {
                "imdb_movies",
                "rt_movies",
                "rt_reviews",
                "movie_matches",
                "movie_ranking",
            },
        )

    def test_reviews_keep_observed_predicted_and_provenance_fields(self) -> None:
        columns = Base.metadata.tables["rt_reviews"].columns

        self.assertTrue(
            {
                "review_score",
                "normalized_score",
                "predicted_score",
                "final_score",
                "score_source",
                "model_name",
                "model_version",
                "prediction_run_id",
            }.issubset(columns.keys())
        )
        self.assertTrue(columns["rotten_tomatoes_link"].index)

    def test_reviews_have_a_stable_unique_source_hash(self) -> None:
        table = Base.metadata.tables["rt_reviews"]
        source_hash = table.columns["source_hash"]
        unique_column_sets = {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertFalse(source_hash.nullable)
        self.assertEqual(source_hash.type.length, 64)
        self.assertIn(("source_hash",), unique_column_sets)

    def test_unavailable_reviews_keep_the_reason(self) -> None:
        table = Base.metadata.tables["rt_reviews"]
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }

        self.assertIn("unavailable_reason", table.columns)
        self.assertTrue(
            {
                "ck_rt_reviews_source_hash_length",
                "ck_rt_reviews_unavailable_reason_present",
                "ck_rt_reviews_available_reason_is_null",
            }.issubset(check_names)
        )

    def test_match_and_ranking_use_the_source_ids_as_composite_keys(self) -> None:
        matches = Base.metadata.tables["movie_matches"]
        ranking = Base.metadata.tables["movie_ranking"]
        self.assertEqual(
            {column.name for column in matches.primary_key.columns},
            {"imdb_id", "rt_id"},
        )
        self.assertEqual(
            {column.name for column in ranking.primary_key.columns},
            {"imdb_id"},
        )

        ranking_foreign_keys = {
            foreign_key.target_fullname
            for foreign_key in ranking.foreign_keys
        }
        self.assertEqual(
            ranking_foreign_keys,
            {"imdb_movies.tconst", "rt_movies.rt_id"},
        )
        self.assertFalse(ranking.columns["imdb_id"].nullable)
        self.assertFalse(ranking.columns["movie_title"].nullable)
        self.assertEqual(ranking.columns["movie_title"].type.length, 512)
        self.assertTrue(ranking.columns["rt_id"].nullable)
        self.assertTrue(ranking.columns["critic_score"].nullable)

    def test_only_one_verified_match_is_allowed_per_source_id(self) -> None:
        indexes = {
            index.name: index
            for index in Base.metadata.tables["movie_matches"].indexes
        }

        for name in (
            "uq_movie_matches_verified_imdb_id",
            "uq_movie_matches_verified_rt_id",
        ):
            self.assertIn(name, indexes)
            self.assertTrue(indexes[name].unique)
            self.assertIsNotNone(
                indexes[name].dialect_options["postgresql"].get("where")
            )

    def test_ranking_keeps_every_formula_input(self) -> None:
        columns = Base.metadata.tables["movie_ranking"].columns

        self.assertTrue(
            {
                "imdb_bayesian_rating",
                "critic_score",
                "observed_review_count",
                "predicted_review_count",
                "effective_critic_count",
                "alpha",
                "k",
                "prediction_weight",
                "final_score",
                "rank_position",
                "calculated_at",
            }.issubset(columns.keys())
        )

    def test_score_and_confidence_checks_are_present(self) -> None:
        expected_checks = {
            "ck_imdb_movies_average_rating_range",
            "ck_imdb_movies_bayesian_rating_range",
            "ck_rt_reviews_normalized_score_range",
            "ck_rt_reviews_predicted_score_range",
            "ck_rt_reviews_final_score_range",
            "ck_movie_matches_match_confidence_range",
            "ck_movie_ranking_final_score_range",
        }
        actual_checks = {
            constraint.name
            for table in Base.metadata.tables.values()
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }

        self.assertTrue(expected_checks.issubset(actual_checks))

    def test_all_tables_compile_for_postgresql(self) -> None:
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            self.assertIn(f"CREATE TABLE {table.name}", ddl)


if __name__ == "__main__":
    unittest.main()
