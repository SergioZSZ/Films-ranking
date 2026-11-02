"""Tests for deterministic IMDb–Rotten Tomatoes movie matching."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.db.loaders.load_movie_matches import (
    candidate_to_record,
    write_unmatched_report,
)
from src.db.matching.matcher import (
    ImdbFilm,
    MatchCandidate,
    RtFilm,
    UnmatchedFilm,
    match_movies,
)
from src.db.matching.normalization import normalize_title, title_variants


class NormalizationTest(unittest.TestCase):
    def test_normalizes_accents_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_title("Léon: The Professional"), "leontheprofessional")

    def test_extracts_parenthetical_title_variants(self) -> None:
        self.assertEqual(
            title_variants("12 Angry Men (Twelve Angry Men)"),
            {"12angrymen", "twelveangrymen", "12angrymentwelveangrymen"},
        )


class MovieMatcherTest(unittest.TestCase):
    def imdb(
        self,
        imdb_id: str,
        title: str,
        year: int | None,
        directors: str = "Director One",
        original_title: str | None = None,
    ) -> ImdbFilm:
        return ImdbFilm(imdb_id, title, original_title, year, directors)

    def rt(
        self,
        rt_id: str,
        title: str,
        year: int | None,
        directors: str = "Director One",
    ) -> RtFilm:
        return RtFilm(rt_id, title, year, directors)

    def test_exact_title_and_year_are_enough_for_unique_match(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Example", 2000, "Different Director")],
            [self.rt("m/example", "Example", 2000)],
        )

        self.assertEqual(result.candidates[0].imdb_id, "tt1")
        self.assertTrue(result.candidates[0].verified)
        self.assertEqual(result.candidates[0].match_method, "exact_title_year")

    def test_director_resolves_multiple_exact_candidates(self) -> None:
        result = match_movies(
            [
                self.imdb("tt1", "Example", 2000, "Director One"),
                self.imdb("tt2", "Example", 2000, "Director Two"),
            ],
            [self.rt("m/example", "Example", 2000, "Director Two")],
        )

        verified = [candidate for candidate in result.candidates if candidate.verified]
        self.assertEqual([candidate.imdb_id for candidate in verified], ["tt2"])
        self.assertEqual(verified[0].match_method, "exact_title_year_director")

    def test_accepts_exact_title_with_one_year_tolerance(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Example", 1999)],
            [self.rt("m/example", "Example", 2000)],
        )

        self.assertTrue(result.candidates[0].verified)
        self.assertEqual(result.candidates[0].match_method, "exact_title_year_tolerance")

    def test_accepts_alternate_title_with_one_year_tolerance(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Twelve Angry Men", 1957)],
            [self.rt("m/12-angry-men", "12 Angry Men (Twelve Angry Men)", 1958)],
        )

        self.assertTrue(result.candidates[0].verified)
        self.assertEqual(result.candidates[0].match_method, "alternate_title_year")

    def test_missing_year_requires_title_and_director(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Example", 2000, "Director One")],
            [self.rt("m/example", "Example", None, "Director One")],
        )

        self.assertTrue(result.candidates[0].verified)
        self.assertEqual(result.candidates[0].match_method, "title_director_no_year")

    def test_ambiguous_matches_are_not_verified(self) -> None:
        result = match_movies(
            [
                self.imdb("tt1", "Example", 2000, "Director One"),
                self.imdb("tt2", "Example", 2000, "Director Two"),
            ],
            [self.rt("m/example", "Example", 2000, "Unknown Director")],
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertFalse(any(candidate.verified for candidate in result.candidates))

    def test_same_imdb_cannot_be_verified_for_two_rt_movies(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Example", 2000)],
            [
                self.rt("m/example-one", "Example", 2000),
                self.rt("m/example-two", "Example", 2000),
            ],
        )

        self.assertFalse(any(candidate.verified for candidate in result.candidates))
        self.assertEqual(len(result.candidates), 2)

    def test_reports_movies_without_candidates(self) -> None:
        result = match_movies(
            [self.imdb("tt1", "Different", 2000)],
            [self.rt("m/example", "Example", 2000)],
        )

        self.assertEqual(result.candidates, [])
        self.assertEqual(result.unmatched[0].rt_id, "m/example")


class MovieMatchLoaderTest(unittest.TestCase):
    def test_maps_candidate_to_database_columns(self) -> None:
        record = candidate_to_record(
            MatchCandidate("tt1", "m/example", "exact_title_year", 0.98, True)
        )

        self.assertEqual(
            record,
            {
                "imdb_id": "tt1",
                "rt_id": "m/example",
                "match_method": "exact_title_year",
                "match_confidence": 0.98,
                "verified": True,
            },
        )

    def test_writes_unmatched_report(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "reports" / "unmatched.csv"
            write_unmatched_report(
                output_path,
                [UnmatchedFilm("m/missing", "Missing", 2000, "no_candidate")],
            )

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("rt_id,movie_title,release_year,reason", content)
        self.assertIn("m/missing,Missing,2000,no_candidate", content)


if __name__ == "__main__":
    unittest.main()
