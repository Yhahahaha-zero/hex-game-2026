# -*- coding: utf-8 -*-
"""V3 cycle and status command tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.memory.store import ExperienceStore
from memory_test_support import collect_scripted_match
from tools import generate_memory_selfplay, memory_cycle, memory_status


class MemoryV3ToolsTest(unittest.TestCase):
    def test_cycle_finalizes_and_exports_registered_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            dataset_dir = root / "dataset"
            collect_scripted_match(db_path)
            code = memory_cycle.main([
                "--db", str(db_path),
                "--dataset-output", str(dataset_dir),
                "--export-dataset",
                "--seed", "17",
            ])
            self.assertEqual(code, 0)
            self.assertTrue((dataset_dir / "manifest.json").is_file())
            with ExperienceStore(db_path) as store:
                self.assertEqual(len(store.list_dataset_versions()), 1)
                self.assertTrue(all(
                    row["decision_quality"] == "validated"
                    for row in store.list_decisions(limit=1000)
                ))

    def test_selfplay_report_creates_nested_output_directory(self):
        class FakeResult:
            def as_dict(self):
                return {"match_id": "match-1", "games": 2}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "nested" / "reports" / "selfplay.json"
            with patch.object(
                generate_memory_selfplay,
                "run_selfplay_series",
                return_value=[FakeResult()],
            ), patch.object(
                generate_memory_selfplay,
                "load_ai_factory",
                return_value=lambda: object(),
            ):
                code = generate_memory_selfplay.main([
                    "--db", str(root / "memory.sqlite3"),
                    "--matches", "1",
                    "--output", str(output),
                ])
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
    def test_status_command_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            models = root / "models"
            models.mkdir()
            output = root / "status.json"
            with ExperienceStore(db_path):
                pass
            code = memory_status.main([
                "--db", str(db_path),
                "--models-dir", str(models),
                "--output", str(output),
            ])
            self.assertEqual(code, 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("counts", document)
            self.assertEqual(document["rollback_available"], False)


if __name__ == "__main__":
    unittest.main()