# -*- coding: utf-8 -*-
"""AI 对比评测工具的回归测试。"""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hex_board import HexBoard
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
            self.assertEqual(len(document["games"]), 8)
            self.assertEqual(document["summary"]["games_completed"], 8)
            self.assertEqual(document["summary"]["errors"], 0)
            self.assertEqual(document["summary"]["ai_a_red_games"], 4)
            self.assertEqual(document["summary"]["ai_a_blue_games"], 4)
            self.assertEqual(len(document["repeats"]), 2)

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
