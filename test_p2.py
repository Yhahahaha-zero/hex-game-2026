# -*- coding: utf-8 -*-
"""SearchResult、报告导出和窗口适配测试。"""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from ai.search_result import ChildStat, SearchResult
from game_mode import GameMode
from game_timer import DualGameTimer
from hex_engine import HexEngine
from report_export import capture_screen, export_report_bundle


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class P2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_search_result_summary(self):
        result = SearchResult(
            move=(5, 5),
            visits=30,
            simulations=100,
            elapsed_ms=123.456,
            children=(
                ChildStat((5, 5), 30, 0.4, 0.71),
                ChildStat((6, 5), 20, 0.3, 0.55),
            ),
        )
        summary = result.summary()
        self.assertEqual(summary["move"], (5, 5))
        self.assertEqual(summary["simulations"], 100)
        self.assertEqual(len(summary["children"]), 2)
        self.assertEqual(result.best_move, (5, 5))

    def test_report_export_creates_all_artifacts(self):
        engine = HexEngine("玩家1", "玩家2")
        engine.play_move(5, 5, reason="first")
        engine.play_move(6, 6, reason="second")
        timer = DualGameTimer(FakeClock())
        timer.start(1)
        screen = pygame.Surface((320, 240))
        screen.fill((20, 30, 40))
        search_stats = [
            {"move": (6, 6), "visits": 42, "prior": 0.31, "value": 0.52},
            {"move": (7, 6), "visits": 28, "prior": 0.20, "value": 0.48},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_report_bundle(
                engine,
                GameMode.HUMAN_VS_HUMAN,
                {"玩家1": "玩家1", "玩家2": "玩家2"},
                timer=timer,
                screen=screen,
                search_stats=search_stats,
                output_dir=Path(temp_dir),
            )
            self.assertTrue(paths["match_json"].exists())
            self.assertTrue(paths["summary_json"].exists())
            self.assertTrue(paths["moves_csv"].exists())
            self.assertTrue(paths["mcts_csv"].exists())
            self.assertTrue(paths["screenshot"].exists())

    def test_window_size_and_mouse_coordinate_adapter(self):
        import hex_game as hg

        game = hg.HexGame()
        width, height = game.window.get_size()
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertEqual(game._to_canvas((0, 0)), (0, 0))
        canvas_x, canvas_y = game._to_canvas((width, height))
        self.assertLessEqual(abs(canvas_x - hg.WINDOW_W), 1)
        self.assertLessEqual(abs(canvas_y - hg.WINDOW_H), 1)
        game._resize_window(600, 650)
        resized_w, resized_h = game.window.get_size()
        self.assertLessEqual(resized_w, 600)
        self.assertLessEqual(resized_h, 650)

    def test_screenshot_capture(self):
        screen = pygame.Surface((160, 120))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = capture_screen(screen, output_dir=Path(temp_dir))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
