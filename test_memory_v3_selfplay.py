# -*- coding: utf-8 -*-
"""V3 explicit self-play generation tests."""
import tempfile
import unittest
from pathlib import Path

from ai.memory.pipeline import MemoryLearningPipeline
from ai.memory.selfplay import run_selfplay_series
from ai.memory.store import ExperienceStore
from hex_ai import BasicAI


class MemorySelfPlayTest(unittest.TestCase):
    def test_selfplay_match_is_recorded_and_can_be_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "selfplay.sqlite3"
            results = run_selfplay_series(
                lambda: BasicAI(),
                lambda: BasicAI(),
                db_path=db_path,
                matches=1,
                seed=1234,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].games, 2)
            with ExperienceStore(db_path) as store:
                match = store.list_matches()[0]
                counts = store.counts()
                self.assertEqual(match["source_type"], "self_play")
                self.assertEqual(counts["games"], 2)
                self.assertGreater(counts["decisions"], 0)
            pipeline = MemoryLearningPipeline(
                lambda: ExperienceStore(db_path).open(), mode="full"
            )
            report = pipeline.process_match(results[0].match_id)
            self.assertEqual(report["status"], "completed")
            with ExperienceStore(db_path) as store:
                decisions = store.list_decisions(limit=1000)
                outcomes = store.list_retrieval_outcomes(limit=1000)
            self.assertTrue(decisions)
            self.assertTrue(all(row["training_eligible"] == 1 for row in decisions))
            self.assertEqual(outcomes, [])


if __name__ == "__main__":
    unittest.main()