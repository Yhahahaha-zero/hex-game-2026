# -*- coding: utf-8 -*-
"""Competition baseline freeze and integrity verification tests."""
import json
import tempfile
import unittest
from pathlib import Path

from test_competition_artifacts import ablation_document, tournament_document
from tools import freeze_competition_baseline, verify_competition_baseline


COMMIT = "b" * 40


def write_final_check(path, *, dirty=False):
    path.write_text(
        "git_commit=" + COMMIT + "\n"
        "working_tree_modified=" + ("yes" if dirty else "no") + "\n"
        "test_count=123\n"
        "test_result=OK\n",
        encoding="utf-8",
    )


class CompetitionBaselineTest(unittest.TestCase):
    def _artifacts(self, root):
        tournament = root / "tournament.json"
        ablation = root / "ablation.json"
        final_check = root / "final_check.txt"
        tournament.write_text(
            json.dumps(tournament_document()), encoding="utf-8"
        )
        ablation.write_text(
            json.dumps(ablation_document()), encoding="utf-8"
        )
        write_final_check(final_check)
        return tournament, ablation, final_check

    def test_freeze_and_verify_complete_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tournament, ablation, final_check = self._artifacts(root)
            manifest = freeze_competition_baseline.freeze_baseline(
                tournament_path=tournament,
                ablation_path=ablation,
                final_check_path=final_check,
                output_dir=root / "baselines",
                baseline_id="baseline-test",
                clean_checker=lambda: COMMIT,
            )
            baseline_dir = Path(manifest["baseline_dir"])
            self.assertTrue((baseline_dir / "baseline_manifest.json").is_file())
            self.assertTrue(
                (baseline_dir / "competition_profile.frozen.json").is_file()
            )
            self.assertTrue((baseline_dir / "report" / "competition_report.md").is_file())
            report = verify_competition_baseline.verify_baseline(baseline_dir)
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["git_commit"], COMMIT)
            self.assertTrue(report["result_validations"]["tournament"]["passed"])
            self.assertTrue(
                report["result_validations"]["memory_ablation"]["passed"]
            )

    def test_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tournament, ablation, final_check = self._artifacts(root)
            manifest = freeze_competition_baseline.freeze_baseline(
                tournament_path=tournament,
                ablation_path=ablation,
                final_check_path=final_check,
                output_dir=root / "baselines",
                baseline_id="baseline-tamper",
                clean_checker=lambda: COMMIT,
            )
            baseline_dir = Path(manifest["baseline_dir"])
            tournament_copy = baseline_dir / "tournament.json"
            tournament_copy.write_text(
                tournament_copy.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            report = verify_competition_baseline.verify_baseline(baseline_dir)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["file_hashes"])

    def test_dirty_final_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tournament, ablation, final_check = self._artifacts(root)
            write_final_check(final_check, dirty=True)
            with self.assertRaisesRegex(ValueError, "dirty"):
                freeze_competition_baseline.freeze_baseline(
                    tournament_path=tournament,
                    ablation_path=ablation,
                    final_check_path=final_check,
                    output_dir=root / "baselines",
                    baseline_id="baseline-dirty",
                    clean_checker=lambda: COMMIT,
                )


if __name__ == "__main__":
    unittest.main()