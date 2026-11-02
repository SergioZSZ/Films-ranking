"""Pure deterministic matching between IMDb and Rotten Tomatoes movies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from .normalization import normalize_directors, normalize_title, title_variants


@dataclass(frozen=True)
class ImdbFilm:
    imdb_id: str
    primary_title: str
    original_title: str | None
    year: int | None
    directors: str | None


@dataclass(frozen=True)
class RtFilm:
    rt_id: str
    title: str
    year: int | None
    directors: str | None


@dataclass(frozen=True)
class MatchCandidate:
    imdb_id: str
    rt_id: str
    match_method: str
    match_confidence: float
    verified: bool


@dataclass(frozen=True)
class UnmatchedFilm:
    rt_id: str
    title: str
    year: int | None
    reason: str


@dataclass
class MatchResult:
    candidates: list[MatchCandidate]
    unmatched: list[UnmatchedFilm]


class MovieIndex:
    """In-memory IMDb index used by all matching stages."""

    def __init__(self, movies: Iterable[ImdbFilm]) -> None:
        self.movies: dict[str, ImdbFilm] = {}
        self.by_title: dict[str, set[str]] = defaultdict(set)
        for movie in movies:
            self.movies[movie.imdb_id] = movie
            for title in (movie.primary_title, movie.original_title):
                normalized = normalize_title(title)
                if normalized:
                    self.by_title[normalized].add(movie.imdb_id)

    def title_candidates(self, titles: Iterable[str]) -> set[str]:
        candidates: set[str] = set()
        for title in titles:
            candidates.update(self.by_title.get(title, set()))
        return candidates


def _director_matches(index: MovieIndex, imdb_id: str, rt_movie: RtFilm) -> bool:
    rt_directors = normalize_directors(rt_movie.directors)
    imdb_directors = normalize_directors(index.movies[imdb_id].directors)
    return bool(rt_directors and imdb_directors and rt_directors & imdb_directors)


def _resolve_candidates(
    index: MovieIndex,
    rt_movie: RtFilm,
    candidate_ids: set[str],
    *,
    method: str,
    confidence: float,
) -> list[MatchCandidate]:
    if not candidate_ids:
        return []
    ordered_ids = sorted(candidate_ids)
    if len(ordered_ids) == 1:
        return [
            MatchCandidate(
                ordered_ids[0],
                rt_movie.rt_id,
                method,
                confidence,
                True,
            )
        ]

    director_ids = [
        imdb_id
        for imdb_id in ordered_ids
        if _director_matches(index, imdb_id, rt_movie)
    ]
    if len(director_ids) == 1:
        selected = director_ids[0]
        return [
            MatchCandidate(
                imdb_id,
                rt_movie.rt_id,
                f"{method}_director" if imdb_id == selected else f"ambiguous_{method}",
                min(1.0, confidence + 0.02) if imdb_id == selected else confidence - 0.1,
                imdb_id == selected,
            )
            for imdb_id in ordered_ids
        ]

    return [
        MatchCandidate(
            imdb_id,
            rt_movie.rt_id,
            f"ambiguous_{method}",
            max(0.0, confidence - 0.1),
            False,
        )
        for imdb_id in ordered_ids
    ]


def _match_one(index: MovieIndex, rt_movie: RtFilm) -> list[MatchCandidate]:
    full_title = normalize_title(rt_movie.title)
    variants = title_variants(rt_movie.title)

    if rt_movie.year is None:
        title_ids = index.title_candidates(variants)
        director_ids = {
            imdb_id
            for imdb_id in title_ids
            if _director_matches(index, imdb_id, rt_movie)
        }
        if len(director_ids) == 1:
            imdb_id = next(iter(director_ids))
            return [
                MatchCandidate(
                    imdb_id,
                    rt_movie.rt_id,
                    "title_director_no_year",
                    0.90,
                    True,
                )
            ]
        if title_ids:
            return [
                MatchCandidate(
                    imdb_id,
                    rt_movie.rt_id,
                    "ambiguous_title_no_year",
                    0.60,
                    False,
                )
                for imdb_id in sorted(title_ids)
            ]
        return []

    exact_ids = {
        imdb_id
        for imdb_id in index.by_title.get(full_title, set())
        if index.movies[imdb_id].year == rt_movie.year
    }
    if exact_ids:
        return _resolve_candidates(
            index,
            rt_movie,
            exact_ids,
            method="exact_title_year",
            confidence=0.98,
        )

    tolerance_ids = {
        imdb_id
        for imdb_id in index.by_title.get(full_title, set())
        if index.movies[imdb_id].year is not None
        and abs(index.movies[imdb_id].year - rt_movie.year) == 1
    }
    if tolerance_ids:
        return _resolve_candidates(
            index,
            rt_movie,
            tolerance_ids,
            method="exact_title_year_tolerance",
            confidence=0.94,
        )

    alternate_ids = {
        imdb_id
        for imdb_id in index.title_candidates(variants - {full_title})
        if index.movies[imdb_id].year is not None
        and abs(index.movies[imdb_id].year - rt_movie.year) <= 1
    }
    if alternate_ids:
        return _resolve_candidates(
            index,
            rt_movie,
            alternate_ids,
            method="alternate_title_year",
            confidence=0.90,
        )
    return []


def _remove_duplicate_verified_imdb(
    candidates: list[MatchCandidate],
) -> list[MatchCandidate]:
    verified_by_imdb: dict[str, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        if candidate.verified:
            verified_by_imdb[candidate.imdb_id].append(position)

    result = list(candidates)
    for positions in verified_by_imdb.values():
        if len(positions) <= 1:
            continue
        for position in positions:
            candidate = result[position]
            result[position] = replace(
                candidate,
                match_method=f"{candidate.match_method}_duplicate_imdb",
                verified=False,
            )
    return result


def match_movies(
    imdb_movies: Iterable[ImdbFilm],
    rt_movies: Iterable[RtFilm],
) -> MatchResult:
    """Apply matching stages and enforce globally unique verified IMDb IDs."""
    index = MovieIndex(imdb_movies)
    candidates: list[MatchCandidate] = []
    unmatched: list[UnmatchedFilm] = []
    for rt_movie in rt_movies:
        movie_candidates = _match_one(index, rt_movie)
        if movie_candidates:
            candidates.extend(movie_candidates)
        else:
            unmatched.append(
                UnmatchedFilm(
                    rt_movie.rt_id,
                    rt_movie.title,
                    rt_movie.year,
                    "no_deterministic_candidate",
                )
            )
    return MatchResult(_remove_duplicate_verified_imdb(candidates), unmatched)
