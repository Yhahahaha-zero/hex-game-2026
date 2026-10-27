# -*- coding: utf-8 -*-
"""V2 supervised memory-model candidate smoke test."""
import json
import tempfile
import unittest
from pathlib import Path

from ai.memory.dataset import build_memory_dataset
from ai.memory.store import ExperienceStore
from ai.train_memory import train_memory_model
from memory_test_support import collect_scripted_match, promote_match


class MemoryTrainingTest(unittest.TestCase):
    def test_candidate_checkpoint_contains_reproducible_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "training.sqlite3"
            dataset_dir = root / "dataset"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            with ExperienceStore(db_path) as store:
                build_memory_dataset(store, dataset_dir, seed=7)
            output = root / "memory_candidate.pt"
            summary_path = root / "training_summary.json"
            summary = train_memory_model(
                dataset_dir=dataset_dir,
                output_path=output,
                parent_checkpoint="none",
                epochs=1,
                batch_size=64,
                seed=7,
                device="cpu",
                summary_path=summary_path,
                lineage_db=db_path,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pt.sha256").is_file())
            self.assertTrue(summary_path.is_file())
            document = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(document["checkpoint_kind"], "memory_supervised_candidate")
            self.assertEqual(document["sha256"], summary["sha256"])
            self.assertEqual(document["dataset_seed"], 7)
            self.assertEqual(document["dataset_split_counts"]["train"], 42)
            self.assertIsNotNone(document["dataset_manifest_sha256"])
            with ExperienceStore(db_path) as store:
                runs = store.list_training_runs()
                datasets = store.list_dataset_versions()
            self.assertEqual(len(runs), 1)
            self.assertEqual(len(datasets), 1)
            self.assertEqual(runs[0]["candidate_model_sha256"], summary["sha256"])


if __name__ == "__main__":
    unittest.main()