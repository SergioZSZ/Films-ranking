"""Shared validation and PostgreSQL upsert helpers for dataset loaders."""

from __future__ import annotations

import argparse
import hashlib
import math
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sqlalchemy import Connection, Engine, Table
from sqlalchemy.dialects.postgresql import Insert, insert


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEW_HASH_FIELDS = (
    "rotten_tomatoes_link",
    "critic_name",
    "publisher_name",
    "review_date",
    "review_type",
    "review_score",
    "review_content",
)


@dataclass
class LoadStats:
    """Progress counters returned by every loader."""

    rows_read: int = 0
    rows_written: int = 0
    rows_skipped: int = 0
    batches: int = 0


def positive_int(value: str) -> int:
    """Argparse type for positive batch sizes."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return parsed


def clean_text(value: object) -> str | None:
    """Convert dataset null markers and blank strings to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text == r"\N":
        return None
    return text


def required_text(value: object, field: str) -> str:
    """Return a non-empty string for a required source field."""
    text = clean_text(value)
    if text is None:
        raise ValueError(f"{field} no puede estar vacío")
    return text


def optional_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
) -> int | None:
    """Parse an optional integer and validate its lower bound."""
    text = clean_text(value)
    if text is None:
        return None
    try:
        decimal_value = Decimal(text)
        if decimal_value != decimal_value.to_integral_value():
            raise ValueError
        parsed = int(decimal_value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} debe ser un entero: {text!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} debe ser >= {minimum}: {parsed}")
    return parsed


def required_int(value: object, field: str, *, minimum: int | None = None) -> int:
    """Parse a required integer."""
    parsed = optional_int(value, field, minimum=minimum)
    if parsed is None:
        raise ValueError(f"{field} no puede estar vacío")
    return parsed


def optional_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Parse an optional finite float and validate its range."""
    text = clean_text(value)
    if text is None:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} debe ser numérico: {text!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} debe ser finito: {text!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} debe ser >= {minimum}: {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field} debe ser <= {maximum}: {parsed}")
    return parsed


def required_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a required finite float."""
    parsed = optional_float(
        value,
        field,
        minimum=minimum,
        maximum=maximum,
    )
    if parsed is None:
        raise ValueError(f"{field} no puede estar vacío")
    return parsed


def optional_date(value: object, field: str) -> date | None:
    """Parse an optional ISO-8601 date."""
    text = clean_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} debe tener formato YYYY-MM-DD: {text!r}") from exc


def optional_bool(value: object, field: str) -> bool | None:
    """Parse common textual boolean representations."""
    text = clean_text(value)
    if text is None:
        return None
    normalized = text.casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} debe ser booleano: {text!r}")


def require_columns(columns: Iterable[str], required: Sequence[str]) -> None:
    """Fail before loading when a processed dataset changed its contract."""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")


def review_source_hash(row: Mapping[str, object]) -> str:
    """Return a stable SHA-256 identity based only on original review fields."""
    canonical_values = []
    for field in REVIEW_HASH_FIELDS:
        value = clean_text(row.get(field)) or ""
        canonical_values.append(unicodedata.normalize("NFC", value))
    payload = "\x1f".join(canonical_values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deduplicate_records(
    records: Iterable[dict[str, object]],
    key_columns: Sequence[str],
) -> list[dict[str, object]]:
    """Keep the last occurrence of each key inside one database batch."""
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        key = tuple(record[column] for column in key_columns)
        unique[key] = record
    return list(unique.values())


def build_upsert_statement(
    table: Table,
    records: Sequence[Mapping[str, object]],
    *,
    conflict_columns: Sequence[str],
) -> Insert:
    """Build a PostgreSQL upsert that never mutates its natural key."""
    if not records:
        raise ValueError("No se puede construir un upsert sin registros")
    statement = insert(table).values(list(records))
    record_columns = set(records[0])
    primary_keys = {column.name for column in table.primary_key.columns}
    immutable = set(conflict_columns) | primary_keys
    update_columns = sorted(record_columns - immutable)
    if not update_columns:
        return statement.on_conflict_do_nothing(index_elements=list(conflict_columns))
    return statement.on_conflict_do_update(
        index_elements=list(conflict_columns),
        set_={column: getattr(statement.excluded, column) for column in update_columns},
    )


def execute_upsert(
    connection: Connection,
    table: Table,
    records: Sequence[Mapping[str, object]],
    *,
    conflict_columns: Sequence[str],
) -> int:
    """Execute one non-empty upsert batch and return its submitted row count."""
    if not records:
        return 0
    connection.execute(
        build_upsert_statement(
            table,
            records,
            conflict_columns=conflict_columns,
        )
    )
    return len(records)


def write_batch(
    engine: Engine,
    table: Table,
    records: Sequence[Mapping[str, object]],
    *,
    conflict_columns: Sequence[str],
) -> int:
    """Write one batch atomically, allowing a failed load to resume safely."""
    with engine.begin() as connection:
        return execute_upsert(
            connection,
            table,
            records,
            conflict_columns=conflict_columns,
        )


def print_stats(entity: str, stats: LoadStats) -> None:
    """Print a compact loader summary."""
    print(
        f"{entity}: leídas={stats.rows_read:,}, "
        f"escritas={stats.rows_written:,}, omitidas={stats.rows_skipped:,}, "
        f"lotes={stats.batches:,}"
    )

