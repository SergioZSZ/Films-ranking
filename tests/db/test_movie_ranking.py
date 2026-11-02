"""Tests for final ranking calculation and persistence mapping."""

from __future__ import annotations

import unittest

from sqlalchemy.dialects import postgresql

from src.db.loaders.load_movie_ranking import (
    build_ranking_source_query,
    ranking_to_record,
)
from src.db.ranking.calculator import (
    RankingParameters,
    assign_rank_positions,
    calculate_movie_score,
)


class RankingCalculatorTest(unittest.TestCase):
    def test_imdb_only_movie_keeps_bayesian_score(self) -> None:
        score = calculate_movie_score(
            imdb_id="tt1",
            movie_title="The Godfather",
            imdb_bayesian_rating=7.25,
            num_votes=100,
        )

        self.assertEqual(score.movie_title, "The Godfather")
        self.assertIsNone(score.rt_id)
        self.assertIsNone(score.critic_score)
        self.assertEqual(score.effective_critic_count, 0.0)
        self.assertEqual(score.final_score, 7.25)

    def test_observed_and_predicted_reviews_use_configured_weight(self) -> None:
        score = calculate_movie_score(
            imdb_id="tt1",
            movie_title="The Godfather",
            imdb_bayesian_rating=7.0,
            num_votes=100,
            rt_id="m/example",
            observed_score_sum=16.0,
            observed_review_count=2,
            predicted_score_sum=6.0,
            predicted_review_count=1,
            parameters=RankingParameters(alpha=0.30, k=20.0, prediction_weight=0.8),
        )

        expected_effective_count = 2.8
        expected_critic_score = (16.0 + 0.8 * 6.0) / expected_effective_count
        expected_final = 7.0 + 0.30 * (
            expected_effective_count / (expected_effective_count + 20.0)
        ) * (expected_critic_score - 7.0)
        self.assertAlmostEqual(score.effective_critic_count, expected_effective_count)
        self.assertAlmostEqual(score.critic_score, expected_critic_score)
        self.assertAlmostEqual(score.final_score, expected_final)

    def test_rejects_invalid_parameters_and_score_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            RankingParameters(alpha=1.1)
        with self.assertRaisesRegex(ValueError, "bayesian"):
            calculate_movie_score(
                imdb_id="tt1",
                movie_title="The Godfather",
                imdb_bayesian_rating=11.0,
                num_votes=1,
            )
        with self.assertRaisesRegex(ValueError, "movie_title"):
            calculate_movie_score(
                imdb_id="tt1",
                movie_title="   ",
                imdb_bayesian_rating=7.0,
                num_votes=1,
            )

    def test_assigns_stable_unique_positions(self) -> None:
        lower_votes = calculate_movie_score(
            imdb_id="tt2",
            movie_title="Second film",
            imdb_bayesian_rating=8.0,
            num_votes=10,
        )
        higher_votes = calculate_movie_score(
            imdb_id="tt1",
            movie_title="First film",
            imdb_bayesian_rating=8.0,
            num_votes=100,
        )

        ranked = assign_rank_positions([lower_votes, higher_votes])

        self.assertEqual([movie.imdb_id for movie in ranked], ["tt1", "tt2"])
        self.assertEqual([movie.rank_position for movie in ranked], [1, 2])


class RankingLoaderTest(unittest.TestCase):
    def test_source_query_compiles_for_postgresql(self) -> None:
        sql = str(
            build_ranking_source_query().compile(dialect=postgresql.dialect())
        )

        self.assertIn("LEFT OUTER JOIN", sql)
        self.assertIn("rt_reviews", sql)
        self.assertIn("movie_matches", sql)
        self.assertIn("primary_title", sql)

    def test_maps_ranked_movie_to_database_columns(self) -> None:
        ranked = assign_rank_positions(
            [
                calculate_movie_score(
                    imdb_id="tt1",
                    movie_title="The Godfather",
                    imdb_bayesian_rating=7.25,
                    num_votes=100,
                )
            ]
        )[0]

        record = ranking_to_record(ranked)

        self.assertEqual(record["imdb_id"], "tt1")
        self.assertEqual(record["movie_title"], "The Godfather")
        self.assertIsNone(record["rt_id"])
        self.assertIsNone(record["critic_score"])
        self.assertEqual(record["rank_position"], 1)
        self.assertEqual(record["final_score"], 7.25)


if __name__ == "__main__":
    unittest.main()
