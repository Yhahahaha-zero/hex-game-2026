# -*- coding: utf-8 -*-
"""hex_match 自动对局、记录和复盘流程的回归测试。"""
import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import hex_match as hm
from hex_ai import BasicAI, RandomAI
from hex_board import RED, BLUE, parse_cell


class HexMatchTest(unittest.TestCase):
    def setUp(self):
        random.seed(20261024)

    def test_play_game_record_schema_and_alternating_colors(self):
        record = hm.play_game(BasicAI(), RandomAI())

        self.assertIn(record["winner"], ("红", "蓝"))
        self.assertEqual(record["moves_count"], len(record["moves"]))
        self.assertGreater(record["moves_count"], 0)
        self.assertGreaterEqual(record["total_ms"], 0)

        for index, move in enumerate(record["moves"]):
            self.assertEqual(set(move), {"p", "cell", "reason", "ms"})
            expected = "红" if index % 2 == 0 else "蓝"
            self.assertEqual(move["p"], expected)
            self.assertIsNotNone(parse_cell(move["cell"]))
            self.assertGreaterEqual(move["ms"], 0)

    def test_same_named_ais_can_play_against_each_other(self):
        red_ai = BasicAI()
        blue_ai = BasicAI()
        self.assertEqual(red_ai.name, blue_ai.name)

        record = hm.play_game(red_ai, blue_ai)
        self.assertIn(record["winner"], ("红", "蓝"))
        self.assertGreater(record["moves_count"], 0)

    def test_run_match_builds_summary_and_saves_records(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.json"
            with contextlib.redirect_stdout(output):
                document = hm.run_match(2, BasicAI(), RandomAI(),
                                        quiet=True, path=path)

            self.assertTrue(path.exists())
            self.assertEqual(hm.load_records(path), document)

        self.assertEqual(document["game_count"], 2)
        self.assertEqual(len(document["games"]), 2)
        self.assertEqual(
            document["summary"]["red_wins"] + document["summary"]["blue_wins"], 2)
        self.assertIn("对局统计", output.getvalue())

    def test_replay_game_runs_through_engine(self):
        record = hm.play_game(BasicAI(), RandomAI())
        record["id"] = 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hm.replay_game(record, speed=0.0)

        text = output.getvalue()
        self.assertIn("本局结束", text)
        self.assertIn(f"{record['winner']}方获胜", text)

    def test_replay_rejects_invalid_coordinate(self):
        record = {
            "id": 1,
            "red_ai": "A",
            "blue_ai": "B",
            "winner": "红",
            "moves_count": 1,
            "moves": [{"p": "红", "cell": "ZZ", "reason": "", "ms": 0.0}],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                hm.replay_game(record, speed=0.0)

    def test_run_match_rejects_zero_games(self):
        with self.assertRaises(ValueError):
            hm.run_match(0, BasicAI(), BasicAI(), quiet=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
