# -*- coding: utf-8 -*-
"""Competition result validation and report generation tests."""
import json
import tempfile
import unittest
from pathlib import Path

from ai.competition_config import load_competition_settings
from tools import competition_report, verify_competition_results


PROFILE_SHA = load_competition_settings().profile_sha256


def reproducibility(dirty=False):
    return {
        "code_commit": "b" * 40,
        "working_tree_dirty": dirty,
        "competition_profile": {
            "profile_name": "test-profile",
            "profile_sha256": PROFILE_SHA,
            "ai_engine": "hybrid",
            "search_time_limit_ms": 200,
            "hard_move_limit_ms": 500,
            "memory_mode": "off",
            "random_seed": 7,
        },
    }


def benchmark_document(dirty=False):
    return {
        "benchmark_schema_version": 2,
        "reproducibility": reproducibility(dirty),
        "summary": {
            "games_requested": 4,
            "games_completed": 4,
            "errors": 0,
            "illegal_move_games": 0,
            "move_timeout_count": 0,
            "ai_a_red_games": 2,
            "ai_a_blue_games": 2,
            "ai_a_p50_move_ms": 100.0,
            "ai_a_p95_move_ms": 150.0,
            "ai_a_p99_move_ms": 190.0,
        },
        "games": [{}, {}, {}, {}],
    }


def tournament_document(dirty=False):
    return {
        "tournament_schema_version": 2,
        "config": {
            "opponents": ["hybrid", "basic"],
            "opponent_names": ["HybridAI", "BasicAI"],
            "games_per_pair": 4,
        },
        "reproducibility": reproducibility(dirty),
        "totals": {
            "matchups": 1,
            "games_requested": 4,
            "games_completed": 4,
            "errors": 0,
            "draws": 0,
            "illegal_move_rate": 0.0,
            "move_timeout_rate": 0.0,
        },
        "pairings": [{
            "ai_a": "hybrid",
            "ai_b": "basic",
            "ai_a_name": "HybridAI",
            "ai_b_name": "BasicAI",
            "games_requested": 4,
            "games_completed": 4,
            "errors": 0,
            "ai_a_win_rate": 75.0,
            "ai_a_red_win_rate": 100.0,
            "ai_a_blue_win_rate": 50.0,
            "ai_a_p95_move_ms": 180.0,
            "ai_a_p99_move_ms": 220.0,
            "ai_b_p95_move_ms": 1.0,
            "ai_b_p99_move_ms": 2.0,
            "move_timeout_rate": 0.0,
            "illegal_move_rate": 0.0,
        }],
        "matrix_win_rate": {
            "hybrid": {"hybrid": None, "basic": 75.0},
            "basic": {"hybrid": 25.0, "basic": None},
        },
        "standings": [
            {"rank": 1, "name": "HybridAI", "opponent_id": "hybrid",
             "games": 4, "wins": 3, "losses": 1, "win_rate": 75.0,
             "max_p95_move_ms": 180.0, "max_p99_move_ms": 220.0,
             "max_timeout_rate": 0.0},
            {"rank": 2, "name": "BasicAI", "opponent_id": "basic",
             "games": 4, "wins": 1, "losses": 3, "win_rate": 25.0,
             "max_p95_move_ms": 1.0, "max_p99_move_ms": 2.0,
             "max_timeout_rate": 0.0},
        ],
        "recommendation": {
            "eligible": True,
            "recommended_ai": "hybrid",
            "tied_ai": ["hybrid"],
            "reason": "complete",
        },
    }


def ablation_document(dirty=False):
    summary = {
        "games_completed": 2,
        "errors": 0,
        "move_timeout_count": 0,
        "ai_a_win_rate": 50.0,
        "ai_a_red_win_rate": 50.0,
        "ai_a_blue_win_rate": 50.0,
        "ai_a_p50_move_ms": 100.0,
        "ai_a_p95_move_ms": 150.0,
        "ai_a_p99_move_ms": 190.0,
    }
    return {
        "ablation_schema_version": 2,
        "config": {"games": 2, "minimum_pairs": 2, "base_ai": "hybrid", "opponent": "basic"},
        "provenance": {"verified": True, "db_sha256": "c" * 64},
        "reproducibility": reproducibility(dirty),
        "results": {
            "off": dict(summary),
            "shadow": dict(summary),
            "assist": dict(summary),
        },
        "paired_comparisons": {
            mode: {
                "method": "paired_bootstrap",
                "games_compared": 2,
                "difference_pp": 0.0,
                "ci95_low_pp": -50.0,
                "ci95_high_pp": 50.0,
                "status": "no_significant_difference",
            }
            for mode in ("shadow", "assist")
        },        "memory_configs": {
            "off": {"mode": "off"},
            "shadow": {"mode": "shadow"},
            "assist": {"mode": "assist"},
        },
        "memory_evidence": {
            "off": {"events": 0, "applied": 0, "coverage": 0.0},
            "shadow": {"events": 2, "applied": 0, "coverage": 0.5},
            "assist": {"events": 2, "applied": 1, "coverage": 0.5},
        },
        "deltas_vs_off": {
            "shadow": {"ai_a_win_rate": 0.0, "memory_events": 2,
                       "memory_applied": 0},
            "assist": {"ai_a_win_rate": 5.0, "memory_events": 2,
                       "memory_applied": 1},
        },
        "recommendation": {"status": "assist_positive"},
    }


class CompetitionArtifactsTest(unittest.TestCase):
    def test_valid_documents_pass(self):
        for document in (
            benchmark_document(), tournament_document(), ablation_document()
        ):
            result = verify_competition_results.verify_document(document)
            self.assertTrue(result["passed"], result["errors"])

    def test_dirty_or_incomplete_results_fail(self):
        dirty = verify_competition_results.verify_benchmark(
            benchmark_document(dirty=True)
        )
        self.assertFalse(dirty["passed"])
        mismatch = verify_competition_results.verify_document(
            benchmark_document(), expected_profile_sha256="0" * 64
        )
        self.assertFalse(mismatch["passed"])
        incomplete = benchmark_document()
        incomplete["summary"]["games_completed"] = 2
        self.assertFalse(
            verify_competition_results.verify_benchmark(incomplete)["passed"]
        )
        no_recommendation = tournament_document()
        no_recommendation["recommendation"]["recommended_ai"] = None
        self.assertFalse(
            verify_competition_results.verify_tournament(no_recommendation)["passed"]
        )

    def test_report_generation_and_cli_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tournament_path = root / "tournament.json"
            ablation_path = root / "ablation.json"
            tournament_path.write_text(
                json.dumps(tournament_document()), encoding="utf-8"
            )
            ablation_path.write_text(
                json.dumps(ablation_document()), encoding="utf-8"
            )
            paths = competition_report.generate_report(
                tournament_path=tournament_path,
                ablation_path=ablation_path,
                output_dir=root / "report",
            )
            report = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("# Hex Competition Evaluation Report", report)
            self.assertIn("HybridAI", report)
            self.assertIn("Memory Ablation", report)
            self.assertIn("assist", report)
            self.assertIn("Paired status", report)
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["validation"].is_file())

            validation_path = root / "validation.json"
            code = verify_competition_results.main([
                "--input", str(tournament_path), str(ablation_path),
                "--output", str(validation_path),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["document_count"], 2)


if __name__ == "__main__":
    unittest.main()