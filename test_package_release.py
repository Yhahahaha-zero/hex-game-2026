# -*- coding: utf-8 -*-
"""发布包收集、排除规则和压缩内容测试。"""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools import package_release


class PackageReleaseTest(unittest.TestCase):
    def test_profile_collection_rules(self):
        root = package_release.WORKSPACE_ROOT
        dev = package_release.collect_release_files("dev", root)
        demo = package_release.collect_release_files("demo", root)
        archive = package_release.collect_release_files("archive", root)

        self.assertIn("hex_game/test_ai.py", dev)
        self.assertIn("hex_game/ai/train_v5.py", dev)
        self.assertIn("models/hex_model_v5_reference.pt", dev)

        self.assertIn("hex_game/hex_game.py", demo)
        self.assertIn("hex_game/ai/encode.py", demo)
        self.assertIn("models/selected_model.json", demo)
        self.assertIn("models/hex_model_v5.pt", demo)
        self.assertNotIn("hex_game/test_ai.py", demo)
        self.assertNotIn("models/hex_model_v5_reference.pt", demo)
        self.assertNotIn("models/hex_model_v5_trial.pt", demo)

        self.assertIn(
            "evaluation/benchmark_results/official_v5_1000.json", archive
        )
        self.assertIn("docs/SYSTEM_ARCHITECTURE.png", archive)
        self.assertNotIn("hex_game/hex_game.py", archive)

    def test_demo_archive_excludes_development_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hex_game" / "ai").mkdir(parents=True)
            (root / "hex_game" / "ui").mkdir(parents=True)
            (root / "hex_game" / "tools").mkdir(parents=True)
            (root / "models").mkdir()
            (root / "docs").mkdir()
            (root / "evaluation" / "training").mkdir(parents=True)
            (root / "evaluation" / "benchmark_results").mkdir(parents=True)

            for name in ("hex_game.py", "hex_board.py", "hex_engine.py"):
                (root / "hex_game" / name).write_text("# runtime\n", encoding="utf-8")
            (root / "hex_game" / "test_ai.py").write_text("# test\n", encoding="utf-8")
            (root / "hex_game" / "ai" / "__init__.py").write_text("", encoding="utf-8")
            (root / "models" / "hex_model_v5.pt").write_bytes(b"model-data")
            model_hash = hashlib.sha256(b"model-data").hexdigest()
            (root / "models" / "selected_model.json").write_text(json.dumps({
                "schema_version": 1,
                "model_file": "hex_model_v5.pt",
                "model_version": "5.0.0",
                "encoding_version": "3x11x11-v1",
                "sha256": model_hash,
                "selected_games": 2000,
            }), encoding="utf-8")
            (root / "models" / "hex_model_v5_trial.pt").write_bytes(b"trial")
            (root / "README.md").write_text("# demo\n", encoding="utf-8")

            archive_path, manifest = package_release.build_release_archive(
                "demo", output_dir=root / "dist", name="demo-test", workspace_root=root
            )
            self.assertTrue(archive_path.exists())
            self.assertEqual(manifest["profile"], "demo")
            with zipfile.ZipFile(archive_path) as zf:
                names = set(zf.namelist())
            self.assertIn("demo-test/hex_game/hex_game.py", names)
            self.assertIn("demo-test/models/hex_model_v5.pt", names)
            self.assertIn("demo-test/models/selected_model.json", names)
            self.assertIn("demo-test/release_manifest.json", names)
            self.assertIn("demo-test/ENVIRONMENT.json", names)
            self.assertNotIn("demo-test/hex_game/test_ai.py", names)
            self.assertNotIn("demo-test/models/hex_model_v5_trial.pt", names)

    def test_setup_files_exist(self):
        root = package_release.WORKSPACE_ROOT / "hex_game"
        self.assertTrue((root / "requirements-dev.txt").is_file())
        self.assertTrue((root / "setup_environment.bat").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
