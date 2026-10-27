# -*- coding: utf-8 -*-
"""Offline tests for standard screenshot import and VLM refinement."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from vision.config import VisionConfigError, load_vision_config

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

VISION_DEPS = bool(importlib.util.find_spec("numpy")) and bool(
    importlib.util.find_spec("PIL")
)

if VISION_DEPS:
    import pygame
    from ai.llm_client import MockLLMClient
    from hex_board import BLUE, RED, HexBoard
    from ui import theme
    from ui.board_renderer import BoardRenderer, BOARD_W, WINDOW_H
    from ui.history_panel import PANEL_W
    from vision.board_validator import validate_cells
    from vision.image_loader import load_image
    from vision.import_session import import_standard_image
    from vision.types import VisionCell, VisionImageError
    from vision.vlm_board_reader import refine_uncertain_cells
    from vision.worker import VisionImportWorker


def make_empty_cells():
    return tuple(VisionCell(q, r, "EMPTY", 1.0) for r in range(11) for q in range(11))


@unittest.skipUnless(VISION_DEPS, "Pillow/numpy not installed")
class VisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.font = pygame.font.Font(None, 18)
        cls.renderer = BoardRenderer(cls.font)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def render_board(self, directory, *, full=False):
        board = HexBoard()
        for q, r, color in ((5, 5, RED), (5, 6, RED), (4, 5, BLUE)):
            board.place(q, r, color)
        width = BOARD_W + PANEL_W if full else BOARD_W
        surface = pygame.Surface((width, WINDOW_H)).convert()
        surface.fill(theme.SIDE_BG if full else theme.BG)
        self.renderer.draw(surface, board)
        path = Path(directory) / ("full_board.png" if full else "board.png")
        pygame.image.save(surface, str(path))
        return board, path

    def test_board_only_screenshot_recognizes_pieces_and_empty_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            board, path = self.render_board(temp_dir)
            session = import_standard_image(path)
        state = session.state
        mapping = state.cell_map()
        self.assertTrue(state.valid)
        self.assertEqual(state.uncertain_cells, ())
        self.assertEqual(mapping[(5, 5)], "RED")
        self.assertEqual(mapping[(5, 6)], "RED")
        self.assertEqual(mapping[(4, 5)], "BLUE")
        self.assertEqual(sum(value == "RED" for value in mapping.values()), 2)
        self.assertEqual(sum(value == "BLUE" for value in mapping.values()), 1)
        self.assertEqual(state.side_to_move, "BLUE")

    def test_full_app_screenshot_is_cropped_to_board(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir, full=True)
            session = import_standard_image(path)
        self.assertEqual(session.state.metadata["region_mode"], "full_app")
        self.assertEqual(session.state.cell_map()[(5, 5)], "RED")
        self.assertTrue(session.state.valid)

    def test_colorblind_palette_is_recognized(self):
        board = HexBoard()
        board.place(5, 5, RED)
        board.place(4, 5, BLUE)
        renderer = BoardRenderer(self.font, colorblind=True)
        surface = pygame.Surface((BOARD_W, WINDOW_H)).convert()
        renderer.draw(surface, board)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "colorblind.png"
            pygame.image.save(surface, str(path))
            session = import_standard_image(path)
        mapping = session.state.cell_map()
        self.assertEqual(mapping[(5, 5)], "RED")
        self.assertEqual(mapping[(4, 5)], "BLUE")
        self.assertTrue(session.state.valid)

    def test_unsupported_image_aspect_is_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "square.png"
            Image.new("RGB", (900, 300), "white").save(path)
            with self.assertRaises(VisionImageError):
                import_standard_image(path)

    def test_invalid_image_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(VisionImageError):
                load_image(path)

    def test_validator_rejects_impossible_piece_count(self):
        cells = list(make_empty_cells())
        cells[0] = VisionCell(0, 0, "RED", 1.0)
        cells[1] = VisionCell(1, 0, "RED", 1.0)
        result = validate_cells(tuple(cells))
        self.assertFalse(result.valid)
        self.assertTrue(result.warnings)

    def test_validator_accepts_empty_board(self):
        result = validate_cells(make_empty_cells())
        self.assertTrue(result.valid)
        self.assertEqual(result.side_to_move, "RED")
        self.assertIsNone(result.winner_color)

    def test_vlm_refines_only_uncertain_cells_and_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            info, _ = load_image(path)
            cells = (
                VisionCell(5, 5, "UNKNOWN", 0.2),
                VisionCell(5, 6, "EMPTY", 1.0),
            )
            response = json.dumps({
                "schema_version": 1,
                "cells": [{
                    "q": 5, "r": 5, "color": "RED",
                    "confidence": 0.96, "reason": "红色棋子",
                }],
                "warnings": [],
            }, ensure_ascii=False)
            client = MockLLMClient(response_text=response)
            updated, warnings, model, hit = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=client, model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
            self.assertFalse(hit)
            self.assertEqual(model, "mock-vlm")
            self.assertEqual(updated[0].color, "RED")
            self.assertEqual(updated[0].source, "vlm")
            self.assertEqual(updated[1].color, "EMPTY")

            second_client = MockLLMClient(response_text=response)
            updated2, _, _, hit2 = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=second_client, model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
        self.assertTrue(hit2)
        self.assertEqual(second_client.calls, [])
        self.assertEqual(updated2[0].color, "RED")

    def test_vlm_ignores_coordinates_outside_uncertain_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            info, _ = load_image(path)
            response = json.dumps({
                "schema_version": 1,
                "cells": [{
                    "q": 0, "r": 0, "color": "RED", "confidence": 0.9,
                }],
                "warnings": [],
            })
            cells = (VisionCell(5, 5, "UNKNOWN", 0.2),)
            updated, warnings, _, _ = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=MockLLMClient(response_text=response),
                model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
        self.assertEqual(updated[0].color, "UNKNOWN")
        self.assertTrue(warnings)

    def test_worker_import_and_cancel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            worker = VisionImportWorker(path, request_id=1, use_vlm=False)
            worker.start()
            self.assertTrue(worker.join(5.0))
            self.assertEqual(worker.result.status, "complete")
            self.assertTrue(worker.result.state.valid)
            with self.assertRaises(RuntimeError):
                worker.start()

            cancelled = VisionImportWorker(path, request_id=2, use_vlm=False)
            cancelled.cancel()
            cancelled.start()
            self.assertTrue(cancelled.join(2.0))
            self.assertEqual(cancelled.result.status, "cancelled")

    def test_snapshot_save_without_mutating_live_game(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            session = import_standard_image(path)
            output = Path(temp_dir) / "out"
            saved = session.save_snapshot(output, "recognized")
            document = json.loads(saved.read_text(encoding="utf-8"))
            grid_before = [row[:] for row in session.board.grid]
            _ = session.board.has_won(RED)
        self.assertEqual(document["schema_version"], 1)
        self.assertTrue(document["valid"])
        self.assertEqual(session.board.grid, grid_before)


    def test_hex_game_import_is_read_only_and_saves_snapshot(self):
        from unittest.mock import patch
        import hex_game as hg
        from game_mode import GameMode
        import vision.import_session as import_session_module

        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            game = hg.HexGame()
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            game.apply_move(5, 5)
            before_grid = [row[:] for row in game.board.grid]
            before_history = list(game.board.history)

            self.assertTrue(game.start_vision_import(path))
            worker = game.vision_worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.join(5.0))
            self.assertTrue(game.update_vision_import())
            self.assertEqual(game.vision_status, "complete")
            self.assertTrue(game.vision_panel.visible)
            self.assertEqual(game.board.grid, before_grid)
            self.assertEqual(game.board.history, before_history)
            game.draw()
            with patch.object(
                import_session_module, "DEFAULT_OUTPUT_DIR", Path(temp_dir) / "snapshots"
            ):
                self.assertTrue(game.save_vision_snapshot())
            saved = list((Path(temp_dir) / "snapshots").glob("*.json"))
            self.assertEqual(len(saved), 1)
            self.assertFalse(game.vision_panel.visible)
            self.assertIn("视觉快照已保存", game.panel_message)


    def test_cli_writes_read_only_snapshot(self):
        from tools.check_vision import main

        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            output = Path(temp_dir) / "json"
            code = main([
                "--image", str(path),
                "--output-dir", str(output),
                "--name", "cli_test",
            ])
            self.assertEqual(code, 0)
            saved = output / "cli_test.json"
            self.assertTrue(saved.is_file())
            document = json.loads(saved.read_text(encoding="utf-8"))
        self.assertTrue(document["valid"])
if __name__ == "__main__":
    unittest.main()
# -*- coding: utf-8 -*-
"""Offline tests for standard screenshot import and VLM refinement."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from vision.config import VisionConfigError, load_vision_config

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

VISION_DEPS = bool(importlib.util.find_spec("numpy")) and bool(
    importlib.util.find_spec("PIL")
)

if VISION_DEPS:
    import pygame
    from ai.llm_client import MockLLMClient
    from hex_board import BLUE, RED, HexBoard
    from ui import theme
    from ui.board_renderer import BoardRenderer, BOARD_W, WINDOW_H
    from ui.history_panel import PANEL_W
    from vision.board_validator import validate_cells
    from vision.image_loader import load_image
    from vision.import_session import import_standard_image
    from vision.types import VisionCell, VisionImageError
    from vision.vlm_board_reader import refine_uncertain_cells
    from vision.worker import VisionImportWorker


def make_empty_cells():
    return tuple(VisionCell(q, r, "EMPTY", 1.0) for r in range(11) for q in range(11))


@unittest.skipUnless(VISION_DEPS, "Pillow/numpy not installed")
class VisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.font = pygame.font.Font(None, 18)
        cls.renderer = BoardRenderer(cls.font)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def render_board(self, directory, *, full=False):
        board = HexBoard()
        for q, r, color in ((5, 5, RED), (5, 6, RED), (4, 5, BLUE)):
            board.place(q, r, color)
        width = BOARD_W + PANEL_W if full else BOARD_W
        surface = pygame.Surface((width, WINDOW_H)).convert()
        surface.fill(theme.SIDE_BG if full else theme.BG)
        self.renderer.draw(surface, board)
        path = Path(directory) / ("full_board.png" if full else "board.png")
        pygame.image.save(surface, str(path))
        return board, path

    def test_board_only_screenshot_recognizes_pieces_and_empty_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            board, path = self.render_board(temp_dir)
            session = import_standard_image(path)
        state = session.state
        mapping = state.cell_map()
        self.assertTrue(state.valid)
        self.assertEqual(state.uncertain_cells, ())
        self.assertEqual(mapping[(5, 5)], "RED")
        self.assertEqual(mapping[(5, 6)], "RED")
        self.assertEqual(mapping[(4, 5)], "BLUE")
        self.assertEqual(sum(value == "RED" for value in mapping.values()), 2)
        self.assertEqual(sum(value == "BLUE" for value in mapping.values()), 1)
        self.assertEqual(state.side_to_move, "BLUE")

    def test_full_app_screenshot_is_cropped_to_board(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir, full=True)
            session = import_standard_image(path)
        self.assertEqual(session.state.metadata["region_mode"], "full_app")
        self.assertEqual(session.state.cell_map()[(5, 5)], "RED")
        self.assertTrue(session.state.valid)

    def test_colorblind_palette_is_recognized(self):
        board = HexBoard()
        board.place(5, 5, RED)
        board.place(4, 5, BLUE)
        renderer = BoardRenderer(self.font, colorblind=True)
        surface = pygame.Surface((BOARD_W, WINDOW_H)).convert()
        renderer.draw(surface, board)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "colorblind.png"
            pygame.image.save(surface, str(path))
            session = import_standard_image(path)
        mapping = session.state.cell_map()
        self.assertEqual(mapping[(5, 5)], "RED")
        self.assertEqual(mapping[(4, 5)], "BLUE")
        self.assertTrue(session.state.valid)

    def test_unsupported_image_aspect_is_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "square.png"
            Image.new("RGB", (900, 300), "white").save(path)
            with self.assertRaises(VisionImageError):
                import_standard_image(path)

    def test_invalid_image_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(VisionImageError):
                load_image(path)

    def test_validator_rejects_impossible_piece_count(self):
        cells = list(make_empty_cells())
        cells[0] = VisionCell(0, 0, "RED", 1.0)
        cells[1] = VisionCell(1, 0, "RED", 1.0)
        result = validate_cells(tuple(cells))
        self.assertFalse(result.valid)
        self.assertTrue(result.warnings)

    def test_validator_accepts_empty_board(self):
        result = validate_cells(make_empty_cells())
        self.assertTrue(result.valid)
        self.assertEqual(result.side_to_move, "RED")
        self.assertIsNone(result.winner_color)

    def test_vlm_refines_only_uncertain_cells_and_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            info, _ = load_image(path)
            cells = (
                VisionCell(5, 5, "UNKNOWN", 0.2),
                VisionCell(5, 6, "EMPTY", 1.0),
            )
            response = json.dumps({
                "schema_version": 1,
                "cells": [{
                    "q": 5, "r": 5, "color": "RED",
                    "confidence": 0.96, "reason": "红色棋子",
                }],
                "warnings": [],
            }, ensure_ascii=False)
            client = MockLLMClient(response_text=response)
            updated, warnings, model, hit = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=client, model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
            self.assertFalse(hit)
            self.assertEqual(model, "mock-vlm")
            self.assertEqual(updated[0].color, "RED")
            self.assertEqual(updated[0].source, "vlm")
            self.assertEqual(updated[1].color, "EMPTY")

            second_client = MockLLMClient(response_text=response)
            updated2, _, _, hit2 = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=second_client, model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
        self.assertTrue(hit2)
        self.assertEqual(second_client.calls, [])
        self.assertEqual(updated2[0].color, "RED")

    def test_vlm_ignores_coordinates_outside_uncertain_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            info, _ = load_image(path)
            response = json.dumps({
                "schema_version": 1,
                "cells": [{
                    "q": 0, "r": 0, "color": "RED", "confidence": 0.9,
                }],
                "warnings": [],
            })
            cells = (VisionCell(5, 5, "UNKNOWN", 0.2),)
            updated, warnings, _, _ = refine_uncertain_cells(
                path, info.sha256, cells, ((5, 5),),
                client=MockLLMClient(response_text=response),
                model_name="mock-vlm",
                cache_dir=Path(temp_dir) / "cache",
            )
        self.assertEqual(updated[0].color, "UNKNOWN")
        self.assertTrue(warnings)

    def test_worker_import_and_cancel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            worker = VisionImportWorker(path, request_id=1, use_vlm=False)
            worker.start()
            self.assertTrue(worker.join(5.0))
            self.assertEqual(worker.result.status, "complete")
            self.assertTrue(worker.result.state.valid)
            with self.assertRaises(RuntimeError):
                worker.start()

            cancelled = VisionImportWorker(path, request_id=2, use_vlm=False)
            cancelled.cancel()
            cancelled.start()
            self.assertTrue(cancelled.join(2.0))
            self.assertEqual(cancelled.result.status, "cancelled")

    def test_snapshot_save_without_mutating_live_game(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            session = import_standard_image(path)
            output = Path(temp_dir) / "out"
            saved = session.save_snapshot(output, "recognized")
            document = json.loads(saved.read_text(encoding="utf-8"))
            grid_before = [row[:] for row in session.board.grid]
            _ = session.board.has_won(RED)
        self.assertEqual(document["schema_version"], 1)
        self.assertTrue(document["valid"])
        self.assertEqual(session.board.grid, grid_before)


    def test_hex_game_import_is_read_only_and_saves_snapshot(self):
        from unittest.mock import patch
        import hex_game as hg
        from game_mode import GameMode
        import vision.import_session as import_session_module

        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            game = hg.HexGame()
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            game.apply_move(5, 5)
            before_grid = [row[:] for row in game.board.grid]
            before_history = list(game.board.history)

            self.assertTrue(game.start_vision_import(path))
            worker = game.vision_worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.join(5.0))
            self.assertTrue(game.update_vision_import())
            self.assertEqual(game.vision_status, "complete")
            self.assertTrue(game.vision_panel.visible)
            self.assertEqual(game.board.grid, before_grid)
            self.assertEqual(game.board.history, before_history)
            game.draw()
            with patch.object(
                import_session_module, "DEFAULT_OUTPUT_DIR", Path(temp_dir) / "snapshots"
            ):
                self.assertTrue(game.save_vision_snapshot())
            saved = list((Path(temp_dir) / "snapshots").glob("*.json"))
            self.assertEqual(len(saved), 1)
            self.assertFalse(game.vision_panel.visible)
            self.assertIn("视觉快照已保存", game.panel_message)


    def test_cli_writes_read_only_snapshot(self):
        from tools.check_vision import main

        with tempfile.TemporaryDirectory() as temp_dir:
            _, path = self.render_board(temp_dir)
            output = Path(temp_dir) / "json"
            code = main([
                "--image", str(path),
                "--output-dir", str(output),
                "--name", "cli_test",
            ])
            self.assertEqual(code, 0)
            saved = output / "cli_test.json"
            self.assertTrue(saved.is_file())
            document = json.loads(saved.read_text(encoding="utf-8"))
        self.assertTrue(document["valid"])
class VisionConfigTest(unittest.TestCase):
    def test_vision_config_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env.local"
            path.write_text(
                "VISION_ENABLED=0\n"
                "VISION_MIN_CONFIDENCE=0.9\n"
                "VISION_MAX_IMAGE_MB=20\n"
                "VISION_CACHE_ENABLED=0\n"
                "VISION_VLM_FALLBACK=0\n"
                "VISION_OUTPUT_DIR=artifacts/custom\n",
                encoding="utf-8",
            )
            config = load_vision_config(path, environ={})
        self.assertFalse(config.enabled)
        self.assertEqual(config.min_confidence, 0.9)
        self.assertEqual(config.max_image_mb, 20)
        self.assertFalse(config.cache_enabled)
        self.assertFalse(config.vlm_fallback)
        self.assertEqual(config.output_dir, "artifacts/custom")

    def test_invalid_vision_confidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env.local"
            path.write_text("VISION_MIN_CONFIDENCE=2\n", encoding="utf-8")
            with self.assertRaises(VisionConfigError):
                load_vision_config(path, environ={})

if __name__ == "__main__":
    unittest.main()