# -*- coding: utf-8 -*-
"""P0-C end-to-end integration with HexGame shadow capture."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from game_mode import GameMode
from hex_engine import STATE_GAME_OVER, STATE_MATCH_OVER, STATE_PLAYING
from ai.memory.collector import CAPTURE_SHADOW, MemoryCaptureConfig, MemoryCollector
from ai.memory.store import ExperienceStore
from ai.worker import AIResult
from hex_board import BLUE, RED


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

    def _new_game(self, hg, collector, pipeline=None):
        return hg.HexGame(
            memory_collector=collector,
            memory_pipeline=pipeline or RecordingPipeline(),
        )

    @staticmethod
    def _play_red_path(game):
        for row in range(11):
            game.apply_move(0, row)
            if game.engine.state == STATE_GAME_OVER:
                break
            game.apply_move(10, row)

    def _inject_ai_move(self, game, move):
        import hex_game as hg

        ai = game.ai or type("ScriptedAI", (), {
            "name": "ScriptedAI",
            "version": "test-1.0",
        })()
        game.ai = ai
        result = AIResult(
            request_id=game.ai_request_id,
            color=game.engine.current_color,
            move=tuple(move),
            reason="scripted human-vs-ai move",
            ai=ai,
            ai_name=ai.name,
            ai_version=ai.version,
            elapsed_ms=0.5,
            explanation={
                "selected_move": tuple(move),
                "tactic": "scripted",
                "search_type": "test",
                "candidates": [{"move": tuple(move), "score": 1.0}],
            },
        )
        with patch.object(hg, "AI_DELAY_MS", 0):
            game._consume_ai_result(result)
            game.update_ai()

    def _play_scripted_human_vs_ai(self, game, *, undo_after_first_pair=False):
        human_color = RED if game.is_human_color(RED) else BLUE
        if human_color == RED:
            human_path = [(0, row) for row in range(11)]
            ai_path = [(10, row) for row in range(11)]
        else:
            human_path = [(q, 0) for q in range(11)]
            ai_path = [(q, 10) for q in range(11)]
        human_index = 0
        ai_index = 0
        first_pair_played = False
        while game.engine.state == STATE_PLAYING:
            if game.is_human_turn():
                move = human_path[human_index]
                human_index += 1
                before = len(game.engine.history)
                game.apply_move(*move, reason="scripted human move")
                self.assertEqual(len(game.engine.history), before + 1)
            else:
                move = ai_path[ai_index]
                ai_index += 1
                before = len(game.engine.history)
                self._inject_ai_move(game, move)
                self.assertEqual(len(game.engine.history), before + 1)
            if undo_after_first_pair and not first_pair_played and len(game.engine.history) == 2:
                first_pair_played = True
                self.assertEqual(game.memory_collector.stats().buffered_moves, 2)
                self.assertTrue(game.undo_move())
                self.assertEqual(game.memory_collector.stats().buffered_moves, 0)
                human_index = 0
                ai_index = 0

    def test_complete_two_game_match_is_captured_through_hex_game(self):
        import hex_game as hg

        collector = self._collector()
        game = self._new_game(hg, collector)
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
        game = self._new_game(hg, collector)
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
        game = self._new_game(hg, collector)
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

    def test_complete_human_vs_ai_match_captures_human_and_ai_moves(self):
        import hex_game as hg

        collector = self._collector()
        pipeline = RecordingPipeline()
        game = self._new_game(hg, collector, pipeline)
        try:
            game.start_mode(GameMode.HUMAN_RED_VS_AI)
            self._play_scripted_human_vs_ai(game)
            self.assertEqual(game.engine.state, STATE_GAME_OVER)
            game.advance_match()
            self._play_scripted_human_vs_ai(game)
            self.assertEqual(game.engine.state, STATE_GAME_OVER)
            game.advance_match()
            self.assertEqual(game.engine.state, STATE_MATCH_OVER)
            expected_moves = len(game.engine.history)
            game.new_game()
            self.assertEqual(collector.stats().written, 1)
            self.assertEqual(len(pipeline.submitted), 1)

            with ExperienceStore(self.db_path) as store:
                match = store.list_matches()[0]
                self.assertEqual(match["source_type"], "human_vs_ai")
                self.assertEqual(match["status"], "completed")
                self.assertEqual(store.counts()["games"], 2)
                decisions = store.list_decisions(limit=expected_moves + 1)
                self.assertEqual(len(decisions), expected_moves)
                self.assertEqual({row["actor_type"] for row in decisions}, {"human", "ai"})
                ai_rows = [row for row in decisions if row["actor_type"] == "ai"]
                self.assertTrue(ai_rows)
                self.assertTrue(all(row["ai_name"] == "ScriptedAI" for row in ai_rows))
                self.assertTrue(all(row["candidate_json"] for row in ai_rows))

            restarted = self._new_game(hg, self._collector(), RecordingPipeline())
            try:
                with ExperienceStore(self.db_path) as store:
                    self.assertEqual(store.counts()["matches"], 1)
                    self.assertEqual(store.counts()["decisions"], expected_moves)
            finally:
                with self.assertRaises(SystemExit):
                    restarted.quit()
        finally:
            with self.assertRaises(SystemExit):
                game.quit()

    def test_human_vs_ai_undo_removes_uncommitted_moves(self):
        import hex_game as hg

        collector = self._collector()
        game = self._new_game(hg, collector)
        try:
            game.start_mode(GameMode.HUMAN_RED_VS_AI)
            self._play_scripted_human_vs_ai(game, undo_after_first_pair=True)
            self.assertEqual(game.engine.state, STATE_GAME_OVER)
            game.advance_match()
            self._play_scripted_human_vs_ai(game)
            game.advance_match()
            expected_moves = len(game.engine.history)
            game.new_game()
            with ExperienceStore(self.db_path) as store:
                decisions = store.list_decisions(limit=expected_moves + 1)
                self.assertEqual(len(decisions), expected_moves)
                plies_by_game = {}
                for row in decisions:
                    plies_by_game.setdefault(row["game_id"], []).append(row["ply"])
                self.assertEqual(len(plies_by_game), 2)
                for plies in plies_by_game.values():
                    self.assertEqual(sorted(plies), list(range(1, len(plies) + 1)))
        finally:
            with self.assertRaises(SystemExit):
                game.quit()

    def test_active_human_vs_ai_match_is_not_persisted_on_restart(self):
        import hex_game as hg

        collector = self._collector()
        game = self._new_game(hg, collector)
        try:
            game.start_mode(GameMode.HUMAN_RED_VS_AI)
            game.apply_move(0, 0, reason="human before restart")
            if not game.is_human_turn():
                self._inject_ai_move(game, (10, 0))
            game.reset_match()
            self.assertFalse(self.db_path.exists())
            self.assertEqual(collector.stats().written, 0)
        finally:
            with self.assertRaises(SystemExit):
                game.quit()


if __name__ == "__main__":
    unittest.main()
