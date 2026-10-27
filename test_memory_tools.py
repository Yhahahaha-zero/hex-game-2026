# -*- coding: utf-8 -*-
"""V1 command-line tooling and benchmark wrapper tests."""
import tempfile
import unittest
from pathlib import Path

from ai.memory.memory_ai import MemoryAssistedAI
from ai.memory.store import ExperienceStore
from memory_test_support import collect_scripted_match
from tools import benchmark_ai, build_memory_features, validate_memory


class MemoryToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "tools.sqlite3"

    def test_validate_and_feature_tools(self):
        collect_scripted_match(self.db_path)
        report_path = Path(self.temp_dir.name) / "validation.json"
        code = validate_memory.main([
            "--db", str(self.db_path),
            "--promote",
            "--output", str(report_path),
        ])
        self.assertEqual(code, 0)
        self.assertTrue(report_path.is_file())
        code = build_memory_features.main([
            "--db", str(self.db_path),
            "--limit", "1000",
        ])
        self.assertEqual(code, 0)
        with ExperienceStore(self.db_path) as store:
            positions = store.all_positions()
            decisions = store.list_decisions(limit=1000)
        self.assertTrue(positions)
        self.assertTrue(all(row["features_json"] for row in positions))
        self.assertTrue(all(row["training_eligible"] == 1 for row in decisions))

    def test_benchmark_loads_memory_wrapper(self):
        factory = benchmark_ai.load_ai_factory("memory-shadow:basic")
        ai = factory()
        try:
            self.assertIsInstance(ai, MemoryAssistedAI)
            self.assertEqual(ai.settings.mode, "shadow")
        finally:
            ai.close()


if __name__ == "__main__":
    unittest.main()