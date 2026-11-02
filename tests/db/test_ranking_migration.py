"""Regression tests for the IMDb-only ranking schema migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


class OperationRecorder:
    """Record Alembic operations without connecting to PostgreSQL."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def f(self, name: str) -> str:
        return name

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return record


def _operation_names(recorder: OperationRecorder) -> list[str]:
    return [name for name, _args, _kwargs in recorder.calls]


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "src/db/migrations/versions/dd75da52fd4e_allow_imdb_only_movie_rankings.py"
    )
    spec = importlib.util.spec_from_file_location("ranking_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_title_migration() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "src/db/migrations/versions/4ad21a86c9b7_add_imdb_title_to_movie_ranking.py"
    )
    spec = importlib.util.spec_from_file_location("ranking_title_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_replaces_primary_key_before_making_rt_id_nullable() -> None:
    migration = _load_migration()
    recorder = OperationRecorder()
    migration.op = recorder

    migration.upgrade()

    operations = _operation_names(recorder)
    drop_pk = recorder.calls.index(
        (
            "drop_constraint",
            ("pk_movie_ranking", "movie_ranking"),
            {"type_": "primary"},
        )
    )
    nullable_rt_id = next(
        index
        for index, (name, args, kwargs) in enumerate(recorder.calls)
        if name == "alter_column"
        and args[:2] == ("movie_ranking", "rt_id")
        and kwargs["nullable"] is True
    )

    assert drop_pk < nullable_rt_id
    assert "create_primary_key" in operations


def test_downgrade_restores_composite_primary_key_after_rt_id_not_null() -> None:
    migration = _load_migration()
    recorder = OperationRecorder()
    migration.op = recorder

    migration.downgrade()

    not_null_rt_id = next(
        index
        for index, (name, args, kwargs) in enumerate(recorder.calls)
        if name == "alter_column"
        and args[:2] == ("movie_ranking", "rt_id")
        and kwargs["nullable"] is False
    )
    create_pk = recorder.calls.index(
        (
            "create_primary_key",
            ("pk_movie_ranking", "movie_ranking", ["imdb_id", "rt_id"]),
            {},
        )
    )

    assert not_null_rt_id < create_pk


def test_title_migration_backfills_before_enforcing_not_null() -> None:
    migration = _load_title_migration()
    recorder = OperationRecorder()
    migration.op = recorder

    migration.upgrade()

    add_column = next(
        index
        for index, (name, args, _kwargs) in enumerate(recorder.calls)
        if name == "add_column"
        and args[0] == "movie_ranking"
        and args[1].name == "movie_title"
        and args[1].nullable is True
    )
    backfill = next(
        index
        for index, (name, args, _kwargs) in enumerate(recorder.calls)
        if name == "execute"
        and "UPDATE movie_ranking" in str(args[0])
        and "imdb_movies" in str(args[0])
    )
    enforce_not_null = next(
        index
        for index, (name, args, kwargs) in enumerate(recorder.calls)
        if name == "alter_column"
        and args[:2] == ("movie_ranking", "movie_title")
        and kwargs["nullable"] is False
    )

    assert add_column < backfill < enforce_not_null

    downgrade_recorder = OperationRecorder()
    migration.op = downgrade_recorder
    migration.downgrade()
    assert (
        "drop_column",
        ("movie_ranking", "movie_title"),
        {},
    ) in downgrade_recorder.calls
