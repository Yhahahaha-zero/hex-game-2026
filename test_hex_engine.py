# -*- coding: utf-8 -*-
"""HexEngine 状态机、统一接口和两局制流程的单元测试。"""
import unittest

from hex_board import RED, BLUE, IllegalMoveError
from hex_engine import (HexEngine, STATE_PLAYING, STATE_GAME_OVER,
                        STATE_MATCH_OVER)


class HexEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = HexEngine("玩家A", "玩家B")

    def _win_as_red(self, engine):
        """让当前局红方沿 A 列连成上到下通路，蓝方只占 K 列。"""
        won = False
        for row in range(11):
            self.assertEqual(engine.current_color, RED)
            won = engine.play_move(0, row)
            if row < 10:
                self.assertEqual(engine.current_color, BLUE)
                self.assertFalse(engine.play_move(10, row))
        self.assertTrue(won)

    def test_initial_state_and_unified_properties(self):
        engine = self.engine
        self.assertEqual(engine.game_no, 1)
        self.assertEqual(engine.state, STATE_PLAYING)
        self.assertEqual(engine.current_player, "玩家A")
        self.assertEqual(engine.current_color, RED)
        self.assertIsNone(engine.winner)
        self.assertIsNone(engine.match_winner())
        self.assertEqual(engine.match_score, {"玩家A": 0, "玩家B": 0})
        self.assertEqual(engine.move_history, ())

    def test_play_move_records_and_alternates_turns(self):
        engine = self.engine
        self.assertFalse(engine.play_move(5, 5, ms=12.345, reason="开局"))

        move = engine.move_history[0]
        self.assertEqual(move, {
            "game": 1,
            "no": 1,
            "color": RED,
            "side": "玩家A",
            "cell": "6F",
            "q": 5,
            "r": 5,
            "ms": 12.35,
            "reason": "开局",
        })
        self.assertEqual(engine.current_player, "玩家B")
        self.assertEqual(engine.current_color, BLUE)
        self.assertEqual(len(engine.game_moves(1)), 1)

        self.assertFalse(engine.play_move(5, 6, reason="第二手"))
        self.assertEqual(engine.current_player, "玩家A")
        self.assertEqual(engine.current_color, RED)
        self.assertEqual(len(engine.move_history), 2)

    def test_history_and_score_are_safe_snapshots(self):
        engine = self.engine
        engine.play_move(5, 5)

        history = engine.move_history
        history[0]["cell"] = "被外部修改"
        self.assertEqual(engine.move_history[0]["cell"], "6F")

        score = engine.match_score
        score["玩家A"] = 99
        self.assertEqual(engine.match_score["玩家A"], 0)

    def test_invalid_moves_do_not_change_state(self):
        engine = self.engine
        with self.assertRaises(IllegalMoveError):
            engine.play_move(11, 0)
        self.assertEqual(engine.current_player, "玩家A")
        self.assertEqual(engine.move_history, ())

        engine.play_move(5, 5)
        with self.assertRaises(IllegalMoveError):
            engine.play_move(5, 5)
        self.assertEqual(len(engine.move_history), 1)
        self.assertEqual(engine.current_player, "玩家B")

    def test_red_win_ends_current_game(self):
        engine = self.engine
        self._win_as_red(engine)

        self.assertEqual(engine.state, STATE_GAME_OVER)
        self.assertEqual(engine.winner_color, RED)
        self.assertEqual(engine.winner_side, "玩家A")
        self.assertEqual(engine.winner, "玩家A")
        self.assertIsNone(engine.current_player)
        self.assertIsNone(engine.current_color)
        self.assertEqual(engine.match_score, {"玩家A": 1, "玩家B": 0})
        self.assertEqual(len(engine.game_moves(1)), 21)
        self.assertIsNotNone(engine.win_path)
        self.assertTrue(any(r == 0 for q, r in engine.win_path))
        self.assertTrue(any(r == 10 for q, r in engine.win_path))
        self.assertEqual(engine.end_reason, "connect")

        with self.assertRaises(IllegalMoveError):
            engine.play_move(4, 4)

    def test_next_game_swaps_colors_and_keeps_history(self):
        engine = self.engine
        self._win_as_red(engine)
        self.assertEqual(engine.next_game(), STATE_PLAYING)

        self.assertEqual(engine.game_no, 2)
        self.assertEqual(engine.red_side, "玩家B")
        self.assertEqual(engine.blue_side, "玩家A")
        self.assertEqual(engine.current_player, "玩家B")
        self.assertEqual(engine.current_color, RED)
        self.assertEqual(engine.match_score, {"玩家A": 1, "玩家B": 0})
        self.assertEqual(len(engine.move_history), 21)
        self.assertEqual(engine.game_moves(2), ())

        engine.play_move(5, 5)
        self.assertEqual(engine.move_history[-1]["game"], 2)
        self.assertEqual(engine.move_history[-1]["side"], "玩家B")
        self.assertEqual(engine.move_history[-1]["color"], RED)

    def test_two_games_end_in_match_over_with_1_to_1_score(self):
        engine = self.engine
        self._win_as_red(engine)
        engine.next_game()
        self._win_as_red(engine)

        self.assertEqual(engine.state, STATE_GAME_OVER)
        self.assertEqual(engine.winner, "玩家B")
        self.assertEqual(engine.match_score, {"玩家A": 1, "玩家B": 1})
        self.assertEqual(engine.next_game(), STATE_MATCH_OVER)
        self.assertEqual(engine.state, STATE_MATCH_OVER)
        self.assertIsNone(engine.current_player)
        self.assertIsNone(engine.current_color)
        self.assertIsNone(engine.match_winner())
        self.assertEqual(len(engine.game_moves(1)), 21)
        self.assertEqual(len(engine.game_moves(2)), 21)

        with self.assertRaises(IllegalMoveError):
            engine.play_move(4, 4)
        with self.assertRaises(IllegalMoveError):
            engine.next_game()

    def test_force_win_can_end_match_2_to_0(self):
        engine = self.engine
        engine.force_win(RED, "超时")
        self.assertEqual(engine.winner, "玩家A")
        engine.next_game()
        engine.force_win(BLUE, "非法落子")
        self.assertEqual(engine.winner, "玩家A")
        self.assertEqual(engine.match_score, {"玩家A": 2, "玩家B": 0})
        self.assertEqual(engine.next_game(), STATE_MATCH_OVER)
        self.assertEqual(engine.match_winner(), "玩家A")

    def test_force_win_and_transition_validation(self):
        engine = self.engine
        with self.assertRaises(IllegalMoveError):
            engine.next_game()
        with self.assertRaises(IllegalMoveError):
            engine.force_win(99, "非法颜色")

        engine.force_win(RED, "测试")
        with self.assertRaises(IllegalMoveError):
            engine.force_win(BLUE, "重复裁定")

    def test_start_match_and_reset_match_restore_initial_state(self):
        engine = self.engine
        engine.force_win(RED, "测试")
        engine.next_game()
        engine.force_win(BLUE, "测试")
        engine.next_game()
        self.assertEqual(engine.state, STATE_MATCH_OVER)

        engine.start_match()
        self.assertEqual(engine.game_no, 1)
        self.assertEqual(engine.state, STATE_PLAYING)
        self.assertEqual(engine.red_side, "玩家A")
        self.assertEqual(engine.blue_side, "玩家B")
        self.assertEqual(engine.current_player, "玩家A")
        self.assertEqual(engine.current_color, RED)
        self.assertEqual(engine.match_score, {"玩家A": 0, "玩家B": 0})
        self.assertEqual(engine.move_history, ())
        self.assertEqual(engine.results, [])

        engine.play(5, 5)
        self.assertEqual(len(engine.move_history), 1)
        engine.reset_match()
        self.assertEqual(engine.game_no, 1)
        self.assertEqual(engine.match_score, {"玩家A": 0, "玩家B": 0})
        self.assertEqual(engine.move_history, ())

    def test_side_and_color_validation(self):
        with self.assertRaises(ValueError):
            HexEngine("同名", "同名")
        with self.assertRaises(ValueError):
            HexEngine("", "玩家B")
        with self.assertRaises(IllegalMoveError):
            self.engine.side_of(99)
        with self.assertRaises(ValueError):
            self.engine.color_of("未知队伍")
        self.assertEqual(self.engine.color_of("玩家A"), RED)
        self.assertEqual(self.engine.color_of("玩家B"), BLUE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
