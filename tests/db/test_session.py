"""Tests for database engine configuration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.db.session import create_db_engine, database_url_from_env


class DatabaseSessionTest(unittest.TestCase):
    def test_database_url_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                database_url_from_env()

    def test_engine_uses_an_explicit_postgresql_url(self) -> None:
        engine = create_db_engine(
            "postgresql+psycopg://user:password@localhost:5433/database"
        )

        self.assertEqual(engine.dialect.name, "postgresql")
        self.assertTrue(engine.pool._pre_ping)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()

