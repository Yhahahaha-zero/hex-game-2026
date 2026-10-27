# -*- coding: utf-8 -*-
"""V2 leak-free dataset export tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.memory.dataset import build_memory_dataset, load_memory_dataset, split_match_ids
from ai.memory.store import ExperienceStore
from memory_test_support import collect_scripted_match, promote_match


class MemoryDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "dataset.sqlite3"
        self.dataset_dir = Path(self.temp_dir.name) / "dataset"
        self.match_ids = []
        for _ in range(3):
            match_id = collect_scripted_match(self.db_path)
            promote_match(self.db_path, match_id)
            self.match_ids.append(match_id)
        with ExperienceStore(self.db_path) as store:
            self.export = build_memory_dataset(
                store,
                self.dataset_dir,
                seed=20261025,
            )

    def test_split_is_grouped_by_match_id_and_complete(self):
        arrays, manifest = load_memory_dataset(self.dataset_dir)
        splits = manifest["splits"]
        self.assertTrue(set(splits["train"]).isdisjoint(splits["validation"]))
        self.assertTrue(set(splits["train"]).isdisjoint(splits["test"]))
        self.assertTrue(set(splits["validation"]).isdisjoint(splits["test"]))
        self.assertEqual(set(manifest["match_ids"]), set(self.match_ids))
        all_split_matches = set().union(*(set(value) for value in splits.values()))
        self.assertEqual(all_split_matches, set(self.match_ids))
        self.assertEqual(sum(manifest["split_counts"].values()), manifest["sample_count"])
        self.assertEqual(arrays["states"].shape[1:], (3, 11, 11))
        self.assertEqual(arrays["legal_masks"].shape[1], 121)
        self.assertEqual(arrays["policy_targets"].shape[1], 121)

    def test_labels_and_policy_targets_are_valid(self):
        arrays, _manifest = load_memory_dataset(self.dataset_dir)
        self.assertTrue(set(arrays["value_targets"].tolist()) <= {-1.0, 1.0})
        sums = arrays["policy_targets"].sum(axis=1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-5))
        for row in arrays["policy_targets"]:
            self.assertGreater(float(row.max()), 0.5)
        actions = arrays["actions"]
        masks = arrays["legal_masks"]
        self.assertTrue(all(bool(masks[index, action]) for index, action in enumerate(actions)))

    def test_split_assignment_is_deterministic(self):
        first = split_match_ids(self.match_ids, seed=20261025)
        second = split_match_ids(reversed(self.match_ids), seed=20261025)
        self.assertEqual(first, second)
        third = split_match_ids(self.match_ids, seed=20261025)
        self.assertEqual(first, third)

    def test_dataset_hash_detects_tampering(self):
        self.export.samples_path.write_bytes(self.export.samples_path.read_bytes() + b"x")
        with self.assertRaises(ValueError):
            load_memory_dataset(self.dataset_dir)


if __name__ == "__main__":
    unittest.main()