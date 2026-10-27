# -*- coding: utf-8 -*-
"""Offline smoke tests for the fixed-position candidate evaluator."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_llm_candidates import run_evaluation


class LLMCandidateToolTest(unittest.TestCase):
    def test_mock_analysis_only_evaluation_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document, path = run_evaluation(
                positions=2,
                seed=20261025,
                mode="analysis_only",
                alpha=0.0,
                candidates=5,
                live=False,
                output_dir=Path(temp_dir),
                name="smoke",
            )
            self.assertTrue(path.is_file())
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(document["summary"]["positions_completed"], 2)
        self.assertEqual(document["summary"]["errors"], 0)
        self.assertEqual(document["summary"]["changed_decisions"], 0)
        self.assertFalse(document["summary"]["live"])
        self.assertEqual(loaded["schema_version"], 1)

    def test_mock_rerank_evaluation_produces_safe_valid_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document, _ = run_evaluation(
                positions=2,
                seed=20261026,
                mode="rerank",
                alpha=0.5,
                candidates=5,
                live=False,
                output_dir=Path(temp_dir),
                name="rerank",
            )

        self.assertEqual(document["summary"]["errors"], 0)
        for row in document["results"]:
            self.assertEqual(len(row["final_move"]), 2)
            self.assertIn("decision_source", row)


if __name__ == "__main__":
    unittest.main()