# -*- coding: utf-8 -*-
"""UI 渲染和引擎集成测试。"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from game_mode import GameMode
from ui.board_renderer import BOARD_W, cell_center, pixel_to_hex
from ui.history_panel import PANEL_W
from hex_engine import STATE_GAME_OVER


class UITest(unittest.TestCase):
    def test_board_geometry_round_trip(self):
        for q in range(11):
            for r in range(11):
                x, y = cell_center(q, r)
                self.assertEqual(pixel_to_hex(x, y), (q, r))

    def test_window_has_history_panel_width(self):
        import hex_game
        self.assertEqual(hex_game.WINDOW_W, BOARD_W + PANEL_W)

    def test_ui_json_save_and_replay(self):
        import hex_game as hg
        from game_records import save_match_record

        game = hg.HexGame()
        game.start_mode(1)
        game.apply_move(5, 5, reason="one")
        game.apply_move(6, 6, reason="two")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ui_record.json"
            saved = save_match_record(
                game.engine,
                game.mode,
                game._player_labels(),
                game.timer,
                path=path,
            )
            with patch("hex_game.latest_record", return_value=saved):
                game.start_replay()
            self.assertEqual(game.state, hg.S_REPLAY)
            self.assertEqual(game.replay.step, 0)
            game.on_space()
            self.assertEqual(game.replay.step, 1)
            game.exit_replay()
            self.assertEqual(game.state, hg.S_PLAY)

    def test_hex_game_uses_engine_for_moves_and_win(self):
        import hex_game as hg
        from hex_board import RED

        game = hg.HexGame()
        game.start_mode(1)
        self.assertEqual(game.engine.state, "playing")
        self.assertEqual(game.mode, GameMode.HUMAN_VS_HUMAN)
        self.assertEqual(game.timer.active_color, hg.RED)
        self.assertFalse(game.apply_move(5, 5, reason="engine-test"))
        self.assertEqual(len(game.engine.move_history), 1)
        self.assertEqual(game.engine.move_history[0]["reason"], "engine-test")
        self.assertEqual(game.engine.current_color, hg.BLUE)
        self.assertEqual(game.timer.active_color, hg.BLUE)

        column = hg.HexGame()
        column.start_mode(1)
        for row in range(11):
            won = column.apply_move(0, row)
            if row < 10:
                self.assertFalse(won)
                self.assertFalse(column.apply_move(10, row))
            else:
                self.assertTrue(won)
        self.assertEqual(column.engine.state, STATE_GAME_OVER)
        self.assertEqual(column.engine.winner_color, RED)
        self.assertEqual(column.winner, RED)
        self.assertEqual(column.match_scores, [1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
