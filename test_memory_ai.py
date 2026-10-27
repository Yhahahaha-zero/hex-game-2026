# -*- coding: utf-8 -*-
"""V1 memory shadow/assist wrapper and retrieval logging tests."""
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from ai.explanation import AIExplanation
from ai.memory.memory_ai import (
    MemoryAssistedAI,
    MemoryRetrievalSettings,
    wrap_ai_with_memory,
)
from ai.memory.shadow import summarize_shadow_events
from ai.memory.store import ExperienceStore
from hex_board import RED, HexBoard
from memory_test_support import (
    collect_scripted_match,
    populate_all_features,
    promote_match,
)


class ScriptedAI:
    name = "ScriptedAI"
    version = "1.0.0"
    model_sha256 = "b" * 64

    def __init__(self, move=(5, 5)):
        self.move = move
        self.last_explanation = None

    def choose_move(self, board, player):
        self.last_explanation = AIExplanation(
            selected_move=self.move,
            tactic="scripted",
            candidates=(
                {"move": self.move, "score": 1.0},
                {"move": (0, 0), "score": 0.8},
            ),
            search_type="rule",
        )
        return self.move, "scripted"

    def analysis_candidates(self, limit=8):
        return [
            {"move": (5, 5), "score": 1.0},
            {"move": (0, 0), "score": 0.8},
        ][:limit]


class MemoryAssistedAITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "memory_ai.sqlite3"
        for _ in range(3):
            match_id = collect_scripted_match(self.db_path)
            promote_match(self.db_path, match_id)
        populate_all_features(self.db_path)

    def _settings(self, mode):
        return MemoryRetrievalSettings(
            mode=mode,
            db_path=str(self.db_path),
            top_k=5,
            minimum_similarity=0.70,
            minimum_support=3,
            minimum_confidence=0.50,
            alpha=0.15,
            ply_window=2,
        )

    def test_shadow_mode_never_changes_base_move_and_logs_event(self):
        with ExperienceStore(self.db_path) as store:
            ai = MemoryAssistedAI(
                ScriptedAI(),
                settings=self._settings("shadow"),
                store=store,
            )
            ai.set_memory_context(game_id="current-game", match_id="current-match")
            move, reason = ai.choose_move(HexBoard(), RED)
            events = store.list_retrieval_events(limit=10)
        self.assertEqual(move, (5, 5))
        self.assertEqual(reason, "scripted")
        self.assertIsNotNone(ai.last_memory_advice)
        self.assertEqual(ai.last_shadow_comparison.memory_move, (0, 0))
        self.assertFalse(ai.last_shadow_comparison.applied)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["retrieval_mode"], "shadow")
        self.assertEqual((events[0]["shadow_q"], events[0]["shadow_r"]), (0, 0))
        self.assertEqual((events[0]["final_q"], events[0]["final_r"]), (5, 5))
        self.assertEqual(events[0]["applied"], 0)
        self.assertEqual(events[0]["query_game_id"], "current-game")
        self.assertEqual(events[0]["query_match_id"], "current-match")

    def test_assist_mode_can_rerank_existing_candidate(self):
        with ExperienceStore(self.db_path) as store:
            ai = MemoryAssistedAI(
                ScriptedAI(),
                settings=self._settings("assist"),
                store=store,
            )
            move, _ = ai.choose_move(HexBoard(), RED)
            events = store.list_retrieval_events(limit=10)
        self.assertEqual(move, (0, 0))
        self.assertEqual((events[0]["final_q"], events[0]["final_r"]), (0, 0))
        self.assertEqual(events[0]["applied"], 1)
        self.assertEqual(ai.last_explanation["search_type"], "memory_assist")

    def test_forced_tactic_skips_memory(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        with ExperienceStore(self.db_path) as store:
            ai = MemoryAssistedAI(
                ScriptedAI((0, 10)),
                settings=self._settings("assist"),
                store=store,
            )
            move, _ = ai.choose_move(board, RED)
            events = store.list_retrieval_events(limit=10)
        self.assertEqual(move, (0, 10))
        self.assertIsNone(ai.last_memory_advice)
        self.assertEqual(events, [])

    def test_database_failure_falls_back_to_base_ai(self):
        def broken_store():
            raise RuntimeError("database unavailable")

        ai = MemoryAssistedAI(
            ScriptedAI(),
            settings=self._settings("assist"),
            store_factory=broken_store,
        )
        move, reason = ai.choose_move(HexBoard(), RED)
        self.assertEqual(move, (5, 5))
        self.assertEqual(reason, "scripted")
        self.assertIn("database unavailable", ai.last_memory_error)

    def test_retrieval_settings_load_from_env_file(self):
        env_file = Path(self.temp_dir.name) / ".env.local"
        env_file.write_text(
            "HEX_MEMORY_RETRIEVAL=assist\n"
            f"HEX_MEMORY_DB={self.db_path}\n"
            "HEX_MEMORY_TOP_K=7\n"
            "HEX_MEMORY_MIN_SUPPORT=2\n"
            "HEX_MEMORY_ALPHA=0.25\n",
            encoding="utf-8",
        )
        settings = MemoryRetrievalSettings.from_env(env_file=env_file, environ={})
        self.assertEqual(settings.mode, "assist")
        self.assertEqual(settings.top_k, 7)
        self.assertEqual(settings.minimum_support, 2)
        self.assertEqual(settings.alpha, 0.25)

    def test_shadow_event_summary(self):
        summary = summarize_shadow_events([
            {"memory_q": 0, "memory_r": 0, "base_q": 5, "base_r": 5, "applied": 0},
            {"memory_q": None, "memory_r": None, "base_q": 5, "base_r": 5, "applied": 0},
        ])
        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["with_advice"], 1)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["differs_from_base"], 1)

    def test_ai_factory_wraps_when_retrieval_enabled(self):
        with mock.patch.dict(os.environ, {
            "HEX_MEMORY_RETRIEVAL": "shadow",
            "HEX_MEMORY_DB": str(self.db_path),
        }, clear=False):
            from ai.ai_factory import create_competition_ai
            ai, error = create_competition_ai("basic")
        self.assertIsNone(error)
        self.assertIsInstance(ai, MemoryAssistedAI)
        ai.close()

    def test_off_mode_wrapper_is_noop(self):
        base = ScriptedAI()
        wrapped = wrap_ai_with_memory(base, settings=self._settings("off"))
        self.assertIs(wrapped, base)


if __name__ == "__main__":
    unittest.main()