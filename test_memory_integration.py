# -*- coding: utf-8 -*-
"""P0-C end-to-end integration with HexGame shadow capture."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from game_mode import GameMode
from hex_engine import STATE_GAME_OVER, STATE_MATCH_OVER
from ai.memory.collector import CAPTURE_SHADOW, MemoryCaptureConfig, MemoryCollector
from ai.memory.store import ExperienceStore


class RecordingPipeline:
    enabled = True

    def __init__(self):
        self.submitted = []
        self.closed = False

    def submit(self, match_id):
        self.submitted.append(match_id)
        return True

    def close(self, timeout=3.0):
        self.closed = True


class MemoryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "integration.sqlite3"

    def _collector(self):
        return MemoryCollector(
            MemoryCaptureConfig(
                mode=CAPTURE_SHADOW,
                db_path=str(self.db_path),
                flush_timeout_seconds=3.0,
                fail_open=True,
            ),
            synchronous=True,
        )

    @staticmethod
    def _play_red_path(game):
        for row in range(11):
            game.apply_move(0, row)
            if game.engine.state == STATE_GAME_OVER:
                break
            game.apply_move(10, row)

    def test_complete_two_game_match_is_captured_through_hex_game(self):
        import hex_game as hg

        collector = self._collector()
        game = hg.HexGame(memory_collector=collector)
        try:
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            self._play_red_path(game)
            self.assertEqual(game.engine.state, STATE_GAME_OVER)
            self.assertFalse(self.db_path.exists())
            game.advance_match()
            self._play_red_path(game)
            self.assertEqual(game.engine.state, STATE_GAME_OVER)
            game.advance_match()
            self.assertEqual(game.engine.state, STATE_MATCH_OVER)
            self.assertFalse(self.db_path.exists())
            history_count = len(game.engine.history)
            game.new_game()
            self.assertEqual(collector.stats().written, 1)
            with ExperienceStore(self.db_path) as store:
                counts = store.counts()
                self.assertEqual(counts["matches"], 1)
                self.assertEqual(counts["games"], 2)
                self.assertEqual(counts["decisions"], history_count)
                match = store.list_matches()[0]
                self.assertEqual(match["status"], "completed")
                self.assertEqual(match["capture_mode"], "shadow")
                decisions = store.list_decisions(limit=history_count + 1)
                self.assertTrue(decisions)
                self.assertTrue(all(row["actor_type"] == "human" for row in decisions))
                self.assertTrue(all(row["candidate_json"] is None for row in decisions))
                self.assertTrue(all(row["training_eligible"] == 0 for row in decisions))
                self.assertEqual({row["result"] for row in decisions}, {0.0, 1.0})
        finally:
            with self.assertRaises(SystemExit):
                game.quit()

    def test_completed_match_is_submitted_to_v3_pipeline(self):
        import hex_game as hg

        collector = self._collector()
        pipeline = RecordingPipeline()
        game = hg.HexGame(memory_collector=collector, memory_pipeline=pipeline)
        try:
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            self._play_red_path(game)
            game.advance_match()
            self._play_red_path(game)
            game.advance_match()
            game.new_game()
            self.assertEqual(len(pipeline.submitted), 1)
        finally:
            with self.assertRaises(SystemExit):
                game.quit()
        self.assertTrue(pipeline.closed)

    def test_active_match_is_not_persisted_on_restart(self):
        import hex_game as hg

        collector = self._collector()
        game = hg.HexGame(memory_collector=collector)
        try:
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            game.apply_move(5, 5)
            game.reset_match()
            self.assertFalse(self.db_path.exists())
            self.assertEqual(collector.stats().written, 0)
        finally:
            with self.assertRaises(SystemExit):
                game.quit()

    def test_undo_after_win_then_replay_is_buffered_correctly(self):
        import hex_game as hg

        collector = self._collector()
        game = hg.HexGame(memory_collector=collector)
        try:
            game.start_mode(GameMode.HUMAN_VS_HUMAN)
            self._play_red_path(game)
            before = collector.stats().buffered_moves
            self.assertTrue(game.undo_move())
            self.assertLess(collector.stats().buffered_moves, before)
            self.assertEqual(game.engine.state, "playing")
            self.assertFalse(self.db_path.exists())
        finally:
            with self.assertRaises(SystemExit):
                game.quit()


if __name__ == "__main__":
    unittest.main()