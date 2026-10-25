# -*- coding: utf-8 -*-
"""完整对局流程、AI 取消和错误恢复的端到端测试。"""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from game_mode import GameMode
from hex_board import HexBoard
from hex_engine import STATE_GAME_OVER, STATE_MATCH_OVER


class BlockingAI:
    name = "BlockingAI"

    def __init__(self):
        self.started = threading.Event()
        self.cancel_event = None

    def set_cancel_event(self, cancel_event):
        self.cancel_event = cancel_event

    def choose_move(self, board, player):
        self.started.set()
        while not self.cancel_event.wait(0.01):
            continue
        raise RuntimeError("cancelled")


class ImmediateAI:
    name = "ImmediateAI"

    def __init__(self, move):
        self.move = move

    def choose_move(self, board, player):
        if board.is_empty(*self.move):
            return self.move, "immediate"
        return board.legal_moves()[0], "first legal"


class EndToEndTest(unittest.TestCase):
    @staticmethod
    def _play_red_path(game):
        for row in range(11):
            game.apply_move(0, row)
            if game.engine.state == STATE_GAME_OVER:
                break
            game.apply_move(10, row)

    def _start_blocking_ai(self, game):
        game.start_mode(GameMode.HUMAN_RED_VS_AI)
        game.apply_move(5, 5, reason="human")
        ai = BlockingAI()
        game.ai = ai
        game.update_ai()
        self.assertTrue(ai.started.wait(1.0))
        return game.ai_worker

    def test_full_match_save_replay_report_and_reset(self):
        import hex_game as hg
        from game_records import save_match_record
        from report_export import export_report_bundle

        game = hg.HexGame()
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        self._play_red_path(game)
        self.assertEqual(game.state, hg.S_WIN)
        game.advance_match()
        self.assertEqual(game.game_no, 2)
        self._play_red_path(game)
        self.assertEqual(game.engine.state, STATE_GAME_OVER)
        game.advance_match()
        self.assertEqual(game.engine.state, STATE_MATCH_OVER)
        self.assertEqual(game.state, hg.S_MATCH_END)
        self.assertEqual(game.match_scores, [1, 1])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record_dir = root / "records"
            with patch("game_records.RECORD_DIR", record_dir):
                game.save_record()
            self.assertIsNotNone(game.last_record_path)
            self.assertTrue(game.last_record_path.exists())

            with patch("hex_game.latest_record", return_value=game.last_record_path):
                game.start_replay()
            self.assertEqual(game.state, hg.S_REPLAY)
            self.assertEqual(game.replay.record_status, "正常结束")
            initial = game.replay.step
            game.replay_next()
            self.assertEqual(game.replay.step, initial + 1)
            game.replay_previous()
            self.assertEqual(game.replay.step, initial)
            game.exit_replay()

            def export_to_temp(*args, **kwargs):
                kwargs["output_dir"] = root / "reports"
                return export_report_bundle(*args, **kwargs)

            with patch("hex_game.export_report_bundle", side_effect=export_to_temp):
                game.export_report()
            self.assertTrue(game.panel_message.startswith("报告已导出"))
            self.assertIsNotNone(game.last_report_paths)
            self.assertTrue(game.last_report_paths["match_json"].exists())
            self.assertTrue(game.last_report_paths["summary_json"].exists())
            self.assertTrue(game.last_report_paths["moves_csv"].exists())
            self.assertTrue(game.last_report_paths["screenshot"].exists())

        game.reset_match()
        self.assertEqual(game.state, hg.S_PLAY)
        self.assertEqual(game.match_scores, [0, 0])
        self.assertEqual(game.engine.history, [])

    def test_reset_cancels_ai_and_new_game_can_start_another(self):
        import hex_game as hg

        game = hg.HexGame()
        old_worker = self._start_blocking_ai(game)
        old_result_id = old_worker.request_id
        game.reset_match()
        self.assertIsNone(game.ai_worker)
        self.assertTrue(old_worker.cancelled)
        self.assertTrue(old_worker.join(1.0))
        self.assertTrue(old_worker.result.cancelled)
        self.assertIsNone(old_worker.result.move)

        stale_result = old_worker.result
        game._consume_ai_result(stale_result)
        self.assertEqual(game.engine.history, [])
        self.assertIsNone(game.pending_move)
        self.assertNotEqual(game.ai_request_id, old_result_id)

        game.start_mode(GameMode.HUMAN_RED_VS_AI)
        game.apply_move(5, 5, reason="human")
        game.ai = ImmediateAI((4, 5))
        game.update_ai()
        new_worker = game.ai_worker
        self.assertIsNotNone(new_worker)
        self.assertTrue(new_worker.join(1.0))
        game.update_ai()
        self.assertEqual(game.pending_move, (4, 5))
        game.ai_act_at = 0
        game.update_ai()
        self.assertEqual(len(game.engine.history), 2)
        self.assertEqual(game.engine.history[-1]["q"], 4)

    def test_switch_mode_cancels_active_ai(self):
        import hex_game as hg

        game = hg.HexGame()
        worker = self._start_blocking_ai(game)
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        self.assertEqual(game.mode, GameMode.HUMAN_VS_HUMAN)
        self.assertTrue(worker.cancelled)
        self.assertTrue(worker.join(1.0))
        self.assertIsNone(game.ai_worker)

    def test_replay_cancels_active_ai(self):
        import hex_game as hg
        from game_records import save_match_record

        game = hg.HexGame()
        old_worker = self._start_blocking_ai(game)
        source = hg.HexGame()
        source.start_mode(GameMode.HUMAN_VS_HUMAN)
        source.apply_move(5, 5)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.json"
            save_match_record(
                source.engine,
                source.mode,
                source._player_labels(),
                source.timer,
                path=path,
            )
            with patch("hex_game.latest_record", return_value=path):
                game.start_replay()
        self.assertEqual(game.state, hg.S_REPLAY)
        self.assertEqual(game.replay.record_status, "进行中存档")
        self.assertTrue(old_worker.cancelled)
        self.assertTrue(old_worker.join(1.0))
        self.assertIsNone(game.ai_worker)

    def test_quit_cancels_active_ai(self):
        import hex_game as hg

        game = hg.HexGame()
        worker = self._start_blocking_ai(game)
        with self.assertRaises(SystemExit):
            game.quit()
        self.assertTrue(worker.cancelled)
        self.assertTrue(worker.join(1.0))
        self.assertTrue(worker.result.cancelled)

    def test_report_failure_is_reported_without_crash_and_resize_stays_valid(self):
        import hex_game as hg

        game = hg.HexGame()
        game.start_mode(GameMode.HUMAN_VS_HUMAN)
        game.apply_move(5, 5)
        with patch("hex_game.export_report_bundle", side_effect=OSError("permission denied")):
            game.export_report()
        self.assertIn("报告导出失败", game.panel_message)

        game._resize_window(600, 650)
        width, height = game.window.get_size()
        self.assertLessEqual(width, 600)
        self.assertLessEqual(height, 650)
        self.assertEqual(game._to_canvas((0, 0)), (0, 0))
        canvas_x, canvas_y = game._to_canvas((width, height))
        self.assertLessEqual(abs(canvas_x - hg.WINDOW_W), 1)
        self.assertLessEqual(abs(canvas_y - hg.WINDOW_H), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
