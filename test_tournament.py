# -*- coding: utf-8 -*-
"""固定对手池交叉评测工具测试。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import tournament


def fake_summary(ai_a, ai_b, index):
    a_rate = 40.0 + index * 5
    return {
        "ai_a": ai_a,
        "ai_b": ai_b,
        "games_requested": 2,
        "games_completed": 2,
        "errors": 0,
        "ai_a_wins": int(a_rate / 100 * 2),
        "ai_b_wins": 2 - int(a_rate / 100 * 2),
        "ai_a_win_rate": a_rate,
        "ai_b_win_rate": 100 - a_rate,
        "ai_a_red_win_rate": a_rate,
        "ai_a_blue_win_rate": a_rate,
        "ai_a_win_rate_ci95_low": 10.0,
        "ai_a_win_rate_ci95_high": 90.0,
        "average_moves": 20.0,
        "ai_a_p50_move_ms": 3.0,
        "ai_a_p95_move_ms": 5.0,
        "ai_a_p99_move_ms": 6.0,
        "ai_b_p50_move_ms": 4.0,
        "ai_b_p95_move_ms": 7.0,
        "ai_b_p99_move_ms": 8.0,
        "illegal_move_rate": 0.0,
        "move_timeout_rate": 0.0,
        "draws": 0,
        "error_types": {},
    }


class TournamentTest(unittest.TestCase):
    def test_cross_matrix_and_totals(self):
        calls = []
        def fake_run(ai_a, ai_b, **kwargs):
            calls.append((ai_a, ai_b))
            document = {"summary": fake_summary(ai_a, ai_b, len(calls) - 1)}
            return document, {}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(tournament.benchmark_ai, "run_benchmark",
                              side_effect=fake_run):
                payload, paths = tournament.run_tournament(
                    ["basic", "strategic", "hybrid"],
                    games=2,
                    seed=123,
                    output_dir=Path(temp_dir),
                    name="matrix",
                    quiet=True,
                )
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["matrix_csv"].exists())
            self.assertEqual(len(calls), 3)
            self.assertEqual(payload["totals"]["matchups"], 3)
            self.assertEqual(payload["totals"]["games_completed"], 6)
            self.assertTrue(payload["recommendation"]["eligible"])
            self.assertIn("hybrid", payload["recommendation"]["tied_ai"])
            self.assertEqual(payload["standings"][0]["opponent_id"], "hybrid")
            self.assertEqual(
                payload["reproducibility"]["competition_profile"]["ai_engine"],
                "hybrid",
            )
            for left in ("basic", "strategic", "hybrid"):
                for right in ("basic", "strategic", "hybrid"):
                    if left != right:
                        self.assertIsNotNone(payload["matrix_win_rate"][left][right])

            document = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(document["config"]["opponent_names"],
                             ["BasicAI", "StrategicAI", "HybridAI"])

            with paths["matrix_csv"].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0][0], "AI \\ 对手")

    def test_invalid_opponent_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            tournament.run_tournament(["basic"], games=2)

    def test_require_clean_blocks_formal_tournament(self):
        with patch.object(
            tournament, "require_clean_worktree",
            side_effect=ValueError("working tree is dirty"),
        ):
            with self.assertRaisesRegex(ValueError, "dirty"):
                tournament.run_tournament(
                    ["basic", "strategic"], games=2, require_clean=True
                )
    def test_default_pool_contains_official_and_reference_models(self):
        self.assertIn("neural", tournament.DEFAULT_OPPONENTS)
        self.assertIn(
            "neural:../models/hex_model_v5_reference.pt",
            tournament.DEFAULT_OPPONENTS,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
