# -*- coding: utf-8 -*-
"""Isolated memory ablation provenance and execution tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.competition_config import file_sha256
from ai.memory.store import ExperienceStore
from memory_test_support import (
    collect_scripted_match,
    populate_all_features,
    promote_match,
)
from tools import memory_ablation


class MemoryAblationTest(unittest.TestCase):
    def test_paired_bootstrap_detects_improvement_and_regression(self):
        def games(winner):
            return [{
                "repeat": 1,
                "pair": index,
                "orientation": "AB",
                "seed": 1000 + index,
                "status": "completed",
                "winner_ai": winner,
            } for index in range(1, 41)]

        improved = memory_ablation.paired_bootstrap_comparison(
            games("B"), games("A"), resamples=200,
            minimum_pairs=20, seed=7,
        )
        self.assertEqual(improved["status"], "significant_improvement")
        self.assertEqual(improved["difference_pp"], 100.0)
        self.assertGreater(improved["ci95_low_pp"], 0.0)

        regressed = memory_ablation.paired_bootstrap_comparison(
            games("A"), games("B"), resamples=200,
            minimum_pairs=20, seed=7,
        )
        self.assertEqual(regressed["status"], "significant_regression")
        self.assertLess(regressed["ci95_high_pp"], 0.0)
    def test_provenance_rejects_test_match_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({
                "schema_version": 1,
                "db_sha256": file_sha256(db_path),
                "training_match_ids": [match_id],
                "test_match_ids": [match_id],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                memory_ablation.validate_training_provenance(
                    db_path, provenance
                )

    def test_require_clean_blocks_formal_ablation(self):
        with patch.object(
            memory_ablation, "require_clean_worktree",
            side_effect=ValueError("working tree is dirty"),
        ):
            with self.assertRaisesRegex(ValueError, "dirty"):
                memory_ablation.run_ablation(
                    db_path="missing.sqlite3", require_clean=True
                )

    def test_apply_profile_requires_clean_mode(self):
        with self.assertRaisesRegex(ValueError, "requires --require-clean"):
            memory_ablation.run_ablation(
                db_path="missing.sqlite3", apply_profile=True
            )
    def test_ablation_runs_three_isolated_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            populate_all_features(db_path)
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({
                "schema_version": 1,
                "db_sha256": file_sha256(db_path),
                "training_match_ids": [match_id],
                "test_match_ids": ["heldout-match"],
            }), encoding="utf-8")
            source_sha = file_sha256(db_path)
            profile = root / "competition_profile.json"
            profile.write_text(
                (memory_ablation.PROJECT_ROOT / "competition_profile.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            with patch.object(
                memory_ablation, "require_clean_worktree",
                return_value="e" * 40,
            ):
                code = memory_ablation.main([
                    "--db", str(db_path),
                    "--provenance", str(provenance),
                    "--profile", str(profile),
                    "--base-ai", "basic",
                    "--opponent", "random",
                    "--games", "2",
                    "--bootstrap-resamples", "100",
                    "--minimum-pairs", "2",
                    "--output-dir", str(root / "out"),
                    "--name", "smoke",
                    "--require-clean",
                    "--apply-profile",
                    "--quiet",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(
                (root / "out" / "smoke" / "ablation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(payload["results"]), {"off", "shadow", "assist"})
            self.assertTrue(payload["provenance"]["verified"])
            self.assertEqual(payload["provenance"]["db_sha256"], source_sha)
            self.assertEqual(file_sha256(db_path), source_sha)
            self.assertEqual(payload["ablation_schema_version"], 2)
            self.assertIn("deltas_vs_off", payload)
            self.assertEqual(
                set(payload["paired_comparisons"]), {"shadow", "assist"}
            )
            recommendation_status = payload["recommendation"]["status"]
            self.assertIn(
                recommendation_status,
                ("assist_supported", "off_recommended", "no_stable_evidence"),
            )
            self.assertTrue(payload["profile_update"]["applied"])
            updated_profile = json.loads(profile.read_text(encoding="utf-8"))
            expected_mode = (
                "assist" if recommendation_status == "assist_supported" else "off"
            )
            self.assertEqual(updated_profile["memory_mode"], expected_mode)
            self.assertIn("memory_selection", updated_profile)
            for mode in ("off", "shadow", "assist"):
                summary = payload["results"][mode]
                self.assertEqual(summary["games_completed"], 2)
                self.assertEqual(summary["errors"], 0)
                self.assertIn("ai_a_p99_move_ms", summary)


if __name__ == "__main__":
    unittest.main()