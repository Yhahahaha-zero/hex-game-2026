# -*- coding: utf-8 -*-
"""AI 对比评测工具的回归测试。"""
import contextlib
import csv
import io
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.competition_config import load_competition_settings
from hex_ai import BasicAI
from hex_board import HexBoard, IllegalMoveError
from tools import benchmark_ai as benchmark


class BenchmarkAITest(unittest.TestCase):
    def test_benchmark_outputs_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            code = benchmark.main([
                "--ai-a", "basic",
                "--ai-b", "random",
                "--games", "4",
                "--repeats", "2",
                "--seed", "12345",
                "--output-dir", str(output_dir),
                "--name", "smoke",
                "--quiet",
            ])

            self.assertEqual(code, 0)
            json_path = output_dir / "smoke.json"
            summary_path = output_dir / "smoke_summary.csv"
            games_path = output_dir / "smoke_games.csv"
            self.assertTrue(json_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(games_path.exists())

            document = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(document["config"]["total_games"], 8)
            self.assertEqual(document["config"]["pairing"],
                             "AB/BA with colors swapped")
            self.assertEqual(document["benchmark_schema_version"], 2)
            self.assertEqual(document["config"]["ai_a_metadata"]["name"], BasicAI().name)
            self.assertIn("version", document["config"]["ai_b_metadata"])
            reproducibility = document["reproducibility"]
            self.assertIn("code_commit", reproducibility)
            self.assertIn("working_tree_dirty", reproducibility)
            self.assertEqual(
                reproducibility["competition_profile"]["ai_engine"], "hybrid"
            )
            self.assertEqual(len(
                reproducibility["competition_profile"]["profile_sha256"]
            ), 64)
            self.assertEqual(len(document["games"]), 8)
            self.assertEqual(document["summary"]["games_completed"], 8)
            self.assertEqual(document["summary"]["errors"], 0)
            self.assertEqual(document["summary"]["ai_a_red_games"], 4)
            self.assertEqual(document["summary"]["ai_a_blue_games"], 4)
            self.assertEqual(len(document["repeats"]), 2)
            self.assertGreaterEqual(document["summary"]["ai_a_average_load_ms"], 0)
            self.assertGreaterEqual(document["summary"]["ai_b_average_load_ms"], 0)
            self.assertGreaterEqual(document["repeats"][0]["ai_a_load_ms"], 0)

            orientations = [game["orientation"] for game in document["games"]]
            self.assertEqual(orientations.count("AB"), 4)
            self.assertEqual(orientations.count("BA"), 4)
            self.assertEqual(len({game["seed"] for game in document["games"]}), 4)

            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertEqual(len(summary_rows), 3)
            self.assertEqual(summary_rows[-1]["_scope"], "overall")

            with games_path.open(encoding="utf-8-sig", newline="") as stream:
                game_rows = list(csv.DictReader(stream))
            self.assertEqual(len(game_rows), 8)
            self.assertIn("a_move_count", game_rows[0])
            self.assertIn("b_move_count", game_rows[0])
            self.assertIn("a_move_times_ms", game_rows[0])
            self.assertIn("a_timeout_count", game_rows[0])
            for key in (
                "ai_a_p50_move_ms", "ai_a_p95_move_ms", "ai_a_p99_move_ms",
                "ai_b_p50_move_ms", "ai_b_p95_move_ms", "ai_b_p99_move_ms",
                "move_timeout_rate", "illegal_move_rate", "draws",
            ):
                self.assertIn(key, document["summary"])

    def test_require_clean_blocks_formal_benchmark(self):
        with patch.object(
            benchmark, "require_clean_worktree",
            side_effect=ValueError("working tree is dirty"),
        ):
            with self.assertRaisesRegex(ValueError, "dirty"):
                benchmark.run_benchmark(
                    "basic", "random", games=2, repeats=1, seed=7,
                    output_dir=Path("unused"), quiet=True, require_clean=True,
                )
    def test_odd_game_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main([
                    "--games", "3",
                    "--output-dir", str(Path(temp_dir)),
                    "--name", "invalid",
                    "--quiet",
                ])
        self.assertEqual(code, 2)

    def test_ai_errors_are_counted_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(benchmark, "play_game",
                              side_effect=RuntimeError("simulated failure")):
                document, paths = benchmark.run_benchmark(
                    "basic", "random", games=2, repeats=1, seed=7,
                    output_dir=Path(temp_dir), name="errors", quiet=True)

            self.assertTrue(paths["json"].exists())
            self.assertEqual(document["summary"]["games_completed"], 0)
            self.assertEqual(document["summary"]["errors"], 2)
            self.assertEqual(document["summary"]["error_types"],
                             {"RuntimeError": 2})
            self.assertEqual(document["games"][0]["status"], "error")
            self.assertEqual(document["games"][0]["error_message"],
                             "simulated failure")

    def test_percentile_and_illegal_move_rate(self):
        self.assertEqual(benchmark._percentile([1, 2, 3, 4], 0.50), 2.5)
        self.assertEqual(benchmark._percentile([1, 2, 3, 4], 0.95), 3.85)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                benchmark, "play_game",
                side_effect=IllegalMoveError("illegal"),
            ):
                document, _ = benchmark.run_benchmark(
                    "basic", "random", games=2, repeats=1, seed=7,
                    output_dir=Path(temp_dir), name="illegal", quiet=True,
                )
        self.assertEqual(document["summary"]["illegal_move_games"], 2)
        self.assertEqual(document["summary"]["illegal_move_rate"], 100.0)

    def test_strategic_ai_can_be_loaded(self):
        factory = benchmark.load_ai_factory("strategic")
        ai = factory()
        self.assertEqual(ai.name, "StrategicAI")
        move, reason = ai.choose_move(HexBoard(), 1)
        self.assertIsNotNone(move)
        self.assertIsInstance(reason, str)

    def test_strategic_ai_ablation_variants_can_be_loaded(self):
        for variant, flags in {
            "basic": (False, False, False),
            "distance": (True, False, False),
            "threat": (True, True, False),
            "full": (True, True, True),
        }.items():
            with self.subTest(variant=variant):
                ai = benchmark.load_ai_factory(f"strategic:{variant}")()
                self.assertEqual(ai.name, "StrategicAI")
                self.assertEqual(
                    (ai.use_distance, ai.use_double_threat, ai.use_bridges),
                    flags,
                )

    def test_competition_factory_uses_configured_time_limit(self):
        settings = replace(load_competition_settings(), memory_mode="off")
        ai = benchmark.load_ai_factory("competition", competition_settings=settings)()
        self.assertEqual(ai.time_limit_ms, 200)
        self.assertEqual(ai.competition_profile["ai_engine"], "hybrid")

    def test_unknown_ai_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.load_ai_factory("unknown-ai")

    def test_custom_ai_can_be_loaded_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "tiny_ai.py"
            module_path.write_text(
                "class TinyAI:\n"
                "    name = 'TinyAI'\n"
                "    def choose_move(self, board, player):\n"
                "        return board.legal_moves()[0], 'tiny'\n",
                encoding="utf-8")
            factory = benchmark.load_ai_factory(f"{module_path}:TinyAI")
            ai = factory()
            self.assertEqual(ai.name, "TinyAI")
            move, reason = ai.choose_move(HexBoard(), 1)
            self.assertEqual(move, (0, 0))
            self.assertEqual(reason, "tiny")


if __name__ == "__main__":
    unittest.main(verbosity=2)
