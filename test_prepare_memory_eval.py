# -*- coding: utf-8 -*-
"""Training-only memory database preparation tests."""
import json
import tempfile
import unittest
from pathlib import Path

from ai.competition_config import file_sha256
from ai.memory.store import ExperienceStore
from memory_test_support import (
    collect_scripted_match,
    populate_all_features,
    promote_match,
)
from tools import memory_ablation, prepare_memory_eval


class PrepareMemoryEvalTest(unittest.TestCase):
    def test_builds_isolated_training_database_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.sqlite3"
            for _ in range(4):
                match_id = collect_scripted_match(source)
                promote_match(source, match_id)
            populate_all_features(source)
            source_sha = file_sha256(source)
            result = prepare_memory_eval.prepare_training_database(
                db_path=source,
                output_dir=root / "output",
                seed=20261025,
            )
            provenance = json.loads(
                result["provenance"].read_text(encoding="utf-8")
            )
            training = set(provenance["training_match_ids"])
            validation = set(provenance["validation_match_ids"])
            test = set(provenance["test_match_ids"])
            self.assertTrue(training)
            self.assertTrue(validation)
            self.assertTrue(test)
            self.assertTrue(training.isdisjoint(validation | test))
            self.assertTrue(validation.isdisjoint(test))
            self.assertEqual(provenance["db_sha256"], file_sha256(result["database"]))
            self.assertEqual(file_sha256(source), source_sha)

            with ExperienceStore(result["database"]) as store:
                output_matches = {
                    row["match_id"] for row in store.list_matches(limit=1000000)
                }
                counts = store.counts()
            self.assertEqual(output_matches, training)
            self.assertFalse(output_matches & (validation | test))
            self.assertGreater(counts["decisions"], 0)
            checked = memory_ablation.validate_training_provenance(
                result["database"], result["provenance"]
            )
            self.assertTrue(checked["verified"])
            self.assertEqual(checked["validation_match_count"], len(validation))

    def test_missing_position_features_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.sqlite3"
            for _ in range(3):
                match_id = collect_scripted_match(source)
                promote_match(source, match_id)
            with self.assertRaisesRegex(ValueError, "position features"):
                prepare_memory_eval.prepare_training_database(
                    db_path=source, output_dir=root / "output"
                )
    def test_existing_output_requires_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.sqlite3"
            for _ in range(3):
                match_id = collect_scripted_match(source)
                promote_match(source, match_id)
            populate_all_features(source)
            output = root / "output"
            prepare_memory_eval.prepare_training_database(
                db_path=source, output_dir=output
            )
            with self.assertRaisesRegex(ValueError, "--force"):
                prepare_memory_eval.prepare_training_database(
                    db_path=source, output_dir=output
                )


if __name__ == "__main__":
    unittest.main()