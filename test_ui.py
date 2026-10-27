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

    def test_ai_explanation_panel_renders(self):
        import hex_game as hg

        game = hg.HexGame()
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        game.apply_move(5, 5, reason="test")
        game.last_ai_explanation = {
            "selected_move": (6, 5),
            "tactic": "double-threat",
            "score_type": "heuristic",
            "elapsed_ms": 12.5,
            "search_type": "strategy",
            "candidates": [
                {"move": (6, 5), "score": 8.2, "reason": "bridge"},
                {"move": (5, 6), "score": 6.1, "reason": "distance"},
            ],
        }
        game.draw()
        self.assertEqual(game.last_ai_explanation["selected_move"], (6, 5))

    def test_new_board_effects_and_colorblind_rendering(self):
        import hex_game as hg
        from hex_board import RED

        game = hg.HexGame()
        game.draw()
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        game.apply_move(5, 5, reason="effect-test")
        game.hover = (4, 5)
        game.draw()

        game.colorblind = True
        game.renderer.set_colorblind(True)
        game.history_panel.set_palette(game.renderer.CC)
        game.draw()

        self.assertTrue(game.renderer.colorblind)
        self.assertIsNotNone(game.renderer.static_surface)
        self.assertIn(RED, game.renderer.CC)
        self.assertGreater(len(game.ripples), 0)

    def test_win_overlay_generates_confetti(self):
        import hex_game as hg
        from hex_board import RED

        game = hg.HexGame()
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        for row in range(11):
            game.apply_move(0, row)
            if row < 10:
                game.apply_move(10, row)
        self.assertEqual(game.state, hg.S_WIN)
        self.assertEqual(game.engine.winner_color, RED)
        self.assertTrue(game.confetti)
        game.draw()

    def test_invalid_record_sets_record_error_message(self):
        import hex_game as hg

        game = hg.HexGame()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text('{"schema_version": 2, "games": [', encoding="utf-8")
            with patch("hex_game.latest_record", return_value=path):
                game.start_replay()
        self.assertNotEqual(game.state, hg.S_REPLAY)
        self.assertTrue(game.panel_message.startswith("记录异常"))

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


    def test_analysis_button_layout_and_panel_actions(self):
        import hex_game as hg

        game = hg.HexGame()
        self.assertGreaterEqual(
            game.history_panel.analysis_button.top,
            game.history_panel.replay_button.bottom + 4,
        )
        self.assertLess(game.history_panel.analysis_button.bottom, 264)

        game.analysis_panel.draw(
            game.screen,
            status="complete",
            analysis={
                "summary": "测试局面摘要",
                "priority": "defense",
                "threats": [{
                    "move": [4, 5],
                    "severity": "high",
                    "reason": "关键阻断点",
                }],
                "candidates": [{
                    "move": [5, 4],
                    "preference": "prefer",
                    "pros": "改善连接",
                    "cons": "防守压力",
                }],
                "warnings": ["主观值不是胜率"],
                "subjective_confidence": 0.7,
            },
            error="",
            model_name="mock-model",
            elapsed_ms=12.5,
            position_id="hex1-test-position",
        )
        self.assertTrue(game.analysis_panel.visible)
        self.assertGreater(game.analysis_panel.panel_rect.width, 0)
        self.assertEqual(
            game.analysis_panel.handle_click(game.analysis_panel.retry_button.center),
            "retry",
        )
        self.assertEqual(
            game.analysis_panel.handle_click(game.analysis_panel.close_button.center),
            "close",
        )
        self.assertFalse(game.analysis_panel.visible)

    def test_analysis_panel_renders_loading_error_and_disabled_states(self):
        import hex_game as hg

        game = hg.HexGame()
        for status in ("loading", "error", "disabled", "stale"):
            with self.subTest(status=status):
                game.analysis_panel.draw(
                    game.screen,
                    status=status,
                    analysis=None,
                    error="测试状态",
                    model_name="",
                    elapsed_ms=None,
                    position_id="hex1-test",
                )
                self.assertTrue(game.analysis_panel.visible)
                game.analysis_panel.hide()

    def test_analysis_button_click_starts_mock_analysis(self):
        import hex_game as hg
        from ai.analysis_result import AnalysisResult

        class Advisor:
            def analyze(self, observation):
                return AnalysisResult(
                    schema_version=1,
                    prompt_version="ui-test-v1",
                    position_id=observation["position_id"],
                    summary="按钮触发的分析",
                    priority="balanced",
                    model_name="ui-mock",
                    elapsed_ms=1.0,
                )

        game = hg.HexGame(analysis_advisor_factory=Advisor)
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        game.on_click(game.history_panel.analysis_button.center)
        worker = game.analysis_worker
        self.assertIsNotNone(worker)
        self.assertTrue(worker.join(1.0))
        self.assertTrue(game.update_analysis())
        self.assertEqual(game.analysis_status, "complete")
        self.assertEqual(game.last_analysis["summary"], "按钮触发的分析")
        game.draw()
        game.on_click(game.analysis_panel.close_button.center)
        self.assertFalse(game.analysis_panel.visible)

    def test_vision_button_layout_and_error_panel(self):
        import hex_game as hg

        game = hg.HexGame()
        analysis = game.history_panel.analysis_button
        vision = game.history_panel.vision_button
        self.assertEqual(analysis.y, vision.y)
        self.assertLess(analysis.right, vision.left)
        game.vision_panel.show(status="error", error="测试视觉错误")
        game.draw()
        self.assertTrue(game.vision_panel.visible)
        self.assertGreater(game.vision_panel.panel_rect.width, 0)
        game.on_click(game.vision_panel.close_button.center)
        self.assertFalse(game.vision_panel.visible)

if __name__ == "__main__":
    unittest.main(verbosity=2)