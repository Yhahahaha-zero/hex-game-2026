# -*- coding: utf-8 -*-
"""V1 command-line tooling and benchmark wrapper tests."""
import tempfile
import unittest
from pathlib import Path

from ai.memory.memory_ai import MemoryAssistedAI
from ai.memory.store import ExperienceStore
from memory_test_support import collect_scripted_match
from tools import benchmark_ai, build_memory_features, create_baseline, learning_readiness, validate_memory


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

    def test_baseline_sqlite_backup_is_consistent(self):
        collect_scripted_match(self.db_path)
        backup = Path(self.temp_dir.name) / "baseline" / "experience.sqlite3"
        result = create_baseline.backup_sqlite(self.db_path, backup)
        self.assertEqual(result["integrity_check"], "ok")
        self.assertTrue(backup.is_file())
        self.assertEqual(result["table_counts"]["matches"], 1)
        self.assertEqual(result["table_counts"]["games"], 2)
        self.assertEqual(result["sha256"], create_baseline.file_sha256(backup))

    def test_baseline_source_archive_excludes_credentials(self):
        code_root = Path(self.temp_dir.name) / "code"
        (code_root / "tools").mkdir(parents=True)
        (code_root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (code_root / ".env.local").write_text("LLM_API_KEY=secret\n", encoding="utf-8")
        (code_root / ".env.example").write_text("LLM_API_KEY=\n", encoding="utf-8")
        (code_root / "__pycache__").mkdir()
        (code_root / "__pycache__" / "main.pyc").write_bytes(b"cache")
        archive = create_baseline._source_archive(
            code_root,
            Path(self.temp_dir.name) / "source.zip",
        )
        import zipfile
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
        self.assertIn("main.py", names)
        self.assertIn(".env.example", names)
        self.assertNotIn(".env.local", names)
        self.assertNotIn("__pycache__/main.pyc", names)

    def test_learning_readiness_requires_balanced_feedback_and_fresh_dataset(self):
        class FakeStore:
            def counts(self):
                return {"matches": 2, "retrieval_outcomes": 20}

            def list_matches(self, limit=100):
                return [{"status": "completed"}, {"status": "completed"}]

            def list_retrieval_outcomes(self, limit=100):
                return (
                    [{"applied": 1, "retrieval_reward": 1.0}] * 10
                    + [{"applied": 1, "retrieval_reward": -1.0}] * 10
                )

            def list_training_runs(self, limit=100):
                return [{"dataset_id": "dataset-sha"}]

            def list_trust_models(self, limit=100):
                return []

        dataset = Path(self.temp_dir.name) / "dataset"
        dataset.mkdir()
        (dataset / "manifest.json").write_text(
            '{"sample_count":20,"split_counts":{"train":14,"validation":3,"test":3},'
            '"match_ids":["m1","m2"],"samples_sha256":"dataset-sha"}',
            encoding="utf-8",
        )
        candidate = Path(self.temp_dir.name) / "candidate.pt"
        candidate.write_bytes(b"candidate")
        selected = Path(self.temp_dir.name) / "selected.json"
        selected.write_text(
            '{"sha256":"' + learning_readiness.file_sha256(candidate) + '"}',
            encoding="utf-8",
        )
        gate = Path(self.temp_dir.name) / "gate.json"
        gate.write_text('{"passed":true}', encoding="utf-8")
        report = learning_readiness.build_readiness_report(
            FakeStore(),
            dataset_dir=dataset,
            candidate_path=candidate,
            selected_model_path=selected,
            candidate_gate_path=gate,
        )
        self.assertTrue(report["ready_for_trust_training"])
        self.assertTrue(report["ready_for_candidate_training"])
        self.assertTrue(report["ready_for_formal_benchmark"])
        self.assertTrue(report["ready_for_promotion"])
        self.assertTrue(report["candidate"]["latest_training_uses_current_dataset"])
        self.assertEqual(report["blockers"], [])

    def test_learning_readiness_reports_imbalanced_feedback(self):
        class FakeStore:
            def counts(self):
                return {"matches": 1, "retrieval_outcomes": 8}

            def list_matches(self, limit=100):
                return [{"status": "completed"}]

            def list_retrieval_outcomes(self, limit=100):
                return [{"applied": 1, "retrieval_reward": 1.0}] * 8

            def list_training_runs(self, limit=100):
                return []

            def list_trust_models(self, limit=100):
                return []

        report = learning_readiness.build_readiness_report(
            FakeStore(),
            dataset_dir=Path(self.temp_dir.name) / "missing",
            candidate_path=Path(self.temp_dir.name) / "missing.pt",
            selected_model_path=Path(self.temp_dir.name) / "missing.json",
        )
        self.assertFalse(report["ready_for_trust_training"])
        gate_names = {item["gate"] for item in report["blockers"]}
        self.assertIn("negative_feedback", gate_names)
        self.assertIn("trust_samples", gate_names)
        self.assertIn("dataset", gate_names)
        self.assertEqual(report["database"]["positive_feedback"], 8)
        self.assertEqual(report["database"]["negative_feedback"], 0)

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