# -*- coding: utf-8 -*-
"""V2 dataset export tool and package inclusion tests."""
import tempfile
import unittest
from pathlib import Path

from tools import export_memory_dataset
from memory_test_support import collect_scripted_match, promote_match


class MemoryV2ToolsTest(unittest.TestCase):
    def test_export_tool_creates_manifest_and_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            output = root / "dataset"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            code = export_memory_dataset.main([
                "--db", str(db_path),
                "--output", str(output),
                "--seed", "9",
            ])
            self.assertEqual(code, 0)
            self.assertTrue((output / "samples.npz").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()