# -*- coding: utf-8 -*-
"""GameMode、双计时器和 JSON 记录复盘测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from game_mode import GameMode
from game_records import ReplaySession, build_match_record, load_record, save_match_record
from game_timer import DualGameTimer, format_seconds
from hex_board import BLUE, RED
from hex_engine import HexEngine


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class P1Test(unittest.TestCase):
    def test_game_mode_metadata(self):
        self.assertEqual(GameMode.HUMAN_VS_HUMAN.label, "双人对战")
        self.assertEqual(GameMode.HUMAN_RED_VS_AI.human_player, 1)
        self.assertEqual(GameMode.HUMAN_BLUE_VS_AI.human_player, 2)
        self.assertFalse(GameMode.HUMAN_VS_HUMAN.uses_ai)
        self.assertTrue(GameMode.AI_VS_AI.uses_ai)

    def test_dual_timer_accumulates_and_pauses(self):
        clock = FakeClock()
        timer = DualGameTimer(clock)
        timer.start(RED)
        clock.value = 5
        timer.switch_to(BLUE)
        clock.value = 7
        timer.pause()
        clock.value = 10
        timer.resume()
        clock.value = 12
        timer.stop()
        self.assertAlmostEqual(timer.elapsed(RED), 5.0)
        self.assertAlmostEqual(timer.elapsed(BLUE), 4.0)
        self.assertEqual(format_seconds(65.9), "01:05")
        self.assertEqual(timer.formatted(), {RED: "00:05", BLUE: "00:04"})

    def test_json_record_save_load_and_replay(self):
        engine = HexEngine("玩家1", "玩家2")
        engine.play_move(5, 5, reason="first")
        engine.play_move(6, 6, reason="second")
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN,
                                      {"玩家1": "玩家1", "玩家2": "玩家2"})
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(len(document["games"][0]["moves"]), 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.json"
            save_match_record(
                engine,
                GameMode.HUMAN_VS_HUMAN,
                {"玩家1": "玩家1", "玩家2": "玩家2"},
                path=path,
            )
            loaded = load_record(path)
            replay = ReplaySession(loaded, game_no=1)
            self.assertEqual(replay.step, 0)
            replay.next()
            self.assertEqual(replay.step, 1)
            self.assertEqual(replay.board.grid[5][5], RED)
            replay.next()
            self.assertEqual(replay.board.grid[6][6], BLUE)
            self.assertTrue(replay.finished)
            replay.previous()
            self.assertEqual(replay.board.grid[6][6], 0)

    def test_record_contains_json_serializable_timer_values(self):
        engine = HexEngine()
        timer = DualGameTimer(FakeClock())
        timer.start(RED)
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN,
                                      {"玩家1": "玩家1", "玩家2": "玩家2"}, timer)
        json.dumps(document)


if __name__ == "__main__":
    unittest.main(verbosity=2)
