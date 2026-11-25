import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import loa_brain


class BrainSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "brain.sqlite3"
        self.path_patcher = patch(
            "loa_brain.brain_database_path",
            return_value=self.database_path,
        )
        self.path_patcher.start()

    def tearDown(self):
        self.path_patcher.stop()
        self.temp_dir.cleanup()

    def test_database_connection_closes_after_context_exit(self):
        with loa_brain._managed_connection() as connection:
            connection.execute("SELECT 1")

        with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
            connection.execute("SELECT 1")

    def test_empty_and_whitespace_queries_list_recent_entries(self):
        first = loa_brain.add_brain_entry("First saved fact")
        second = loa_brain.add_brain_entry("Second saved fact")

        for query in ("", "   \t"):
            with self.subTest(query=repr(query)):
                entries = loa_brain.list_brain_entries(query)
                self.assertEqual(
                    [entry["id"] for entry in entries],
                    [second["id"], first["id"]],
                )

    def test_brain_tool_lookup_skips_empty_and_stopword_only_queries(self):
        loa_brain.add_brain_entry("A saved fact about local settings")

        self.assertEqual(loa_brain.search_brain(""), [])
        self.assertEqual(loa_brain.search_brain("   "), [])
        self.assertEqual(loa_brain.search_brain("the and"), [])


if __name__ == "__main__":
    unittest.main()
