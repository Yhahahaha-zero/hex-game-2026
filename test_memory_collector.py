# -*- coding: utf-8 -*-
"""P0-C tests for shadow collection, buffering and background persistence."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ai.memory.collector import (
    CAPTURE_SHADOW,
    MemoryCaptureConfig,
    MemoryCollector,
    MemoryCollectorError,
    create_memory_collector_from_env,
)
from ai.memory.store import ExperienceStore
from ai.memory.trace import ActorMetadata, DecisionTrace
from hex_board import BLUE, RED, HexBoard


class MemoryCollectorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "collector.sqlite3"
        self.timestamp = "2026-01-01T00:00:00+00:00"

    def _config(self, *, fail_open=True):
        return MemoryCaptureConfig(
            mode=CAPTURE_SHADOW,
            db_path=str(self.db_path),
            queue_size=2,
            flush_timeout_seconds=3.0,
            fail_open=fail_open,
        )

    def _collector(self, *, synchronous=True, fail_open=True, store_factory=None):
        collector = MemoryCollector(
            self._config(fail_open=fail_open),
            store_factory=store_factory,
            clock=lambda: self.timestamp,
            synchronous=synchronous,
        )
        self.addCleanup(collector.close)
        return collector

    def _start(self, collector):
        return collector.start_match(
            mode_key=4,
            mode_name="AI vs AI",
            red_team="AI-1",
            blue_team="AI-2",
            source_type="ai_vs_ai",
            actor_types={RED: "ai", BLUE: "ai"},
            started_at=self.timestamp,
        )

    def _trace(self, move=(5, 5), color=RED):
        actor = ActorMetadata(
            actor_type="ai",
            actor_id="HybridAI",
            ai_name="HybridAI",
            ai_version="6.0.0",
            model_sha256="a" * 64,
        )
        return DecisionTrace(
            move=move,
            color=color,
            reason="test",
            actor=actor,
            candidate_payload={"candidates": [{"move": list(move), "score": 1.0}]},
            search_config={"simulations": 100},
            elapsed_ms=12.5,
        )

    def _record_complete_match(self, collector):
        self._start(collector)
        collector.record_move(HexBoard(), self._trace())
        collector.mark_game_over(RED, "connect")
        collector.begin_game(
            2,
            sides={RED: "AI-2", BLUE: "AI-1"},
            actor_types={RED: "ai", BLUE: "ai"},
            started_at=self.timestamp,
        )
        collector.record_move(HexBoard(), self._trace((4, 4)))
        collector.mark_game_over(RED, "connect")

    def test_off_mode_returns_noop_collector(self):
        collector = create_memory_collector_from_env(
            env_file=Path(self.temp_dir.name) / "missing.env",
            environ={},
        )
        self.assertFalse(collector.enabled)
        self.assertFalse(collector.accept_match())

    def test_no_database_is_created_before_accept(self):
        collector = self._collector()
        self._start(collector)
        collector.record_move(HexBoard(), self._trace())
        collector.mark_game_over(RED, "connect")
        self.assertFalse(self.db_path.exists())
        collector.abort_match("match_restarted_before_finish")
        self.assertFalse(self.db_path.exists())

    def test_complete_match_is_written_atomically(self):
        collector = self._collector()
        self._record_complete_match(collector)
        self.assertTrue(collector.accept_match())
        with ExperienceStore(self.db_path) as store:
            self.assertEqual(store.counts(), {
                "matches": 1,
                "games": 2,
                "positions": 1,
                "decisions": 2,
                "retrieval_events": 0,
                "retrieval_outcomes": 0,
                "memory_jobs": 0,
                "dataset_versions": 0,
                "training_runs": 0,
                "trust_models": 0,
                "active_learning_queue": 0,
                "opponent_profiles": 0,
                "league_runs": 0,
                "drift_snapshots": 0,
            "trust_models": 0,
            "active_learning_queue": 0,
            "opponent_profiles": 0,
            "league_runs": 0,
            "drift_snapshots": 0,
            "retrieval_outcomes": 0,
            "memory_jobs": 0,
            "dataset_versions": 0,
            "training_runs": 0,
            "trust_models": 0,
            "active_learning_queue": 0,
            "opponent_profiles": 0,
            "league_runs": 0,
            "drift_snapshots": 0,
            })
            match = store.list_matches()[0]
            stored_match = store.get_match(match["match_id"])
            self.assertEqual(stored_match["capture_mode"], "shadow")
            self.assertEqual(stored_match["status"], "completed")
            decisions = store.list_decisions(limit=10)
            self.assertEqual([row["result"] for row in decisions], [1.0, 1.0])
            self.assertEqual(decisions[0]["training_eligible"], 0)
            self.assertEqual(decisions[0]["decision_quality"], "unverified")
            self.assertEqual(decisions[0]["model_sha256"], "a" * 64)

    def test_undo_removes_buffered_move_and_winner(self):
        collector = self._collector()
        self._start(collector)
        collector.record_move(HexBoard(), self._trace())
        collector.mark_game_over(RED, "connect")
        self.assertEqual(collector.undo_moves(1), 1)
        self.assertEqual(collector.stats().buffered_moves, 0)
        collector.record_move(HexBoard(), self._trace((4, 4)))
        collector.mark_game_over(RED, "connect")
        collector.begin_game(
            2,
            sides={RED: "AI-2", BLUE: "AI-1"},
            actor_types={RED: "ai", BLUE: "ai"},
        )
        collector.record_move(HexBoard(), self._trace())
        collector.mark_game_over(RED, "connect")
        self.assertTrue(collector.accept_match())
        with ExperienceStore(self.db_path) as store:
            decisions = store.list_decisions(limit=10)
            self.assertEqual(len(decisions), 2)
            self.assertEqual({row["chosen_q"] for row in decisions}, {4, 5})

    def test_accept_is_idempotent_after_buffer_is_cleared(self):
        collector = self._collector()
        self._record_complete_match(collector)
        self.assertTrue(collector.accept_match())
        self.assertFalse(collector.accept_match())
        with ExperienceStore(self.db_path) as store:
            self.assertEqual(store.counts()["matches"], 1)


    def test_incomplete_match_cannot_be_accepted(self):
        collector = self._collector()
        self._start(collector)
        collector.record_move(HexBoard(), self._trace())
        collector.mark_game_over(RED, "connect")
        with self.assertRaises(MemoryCollectorError):
            collector.accept_match()
        self.assertFalse(self.db_path.exists())

    def test_invalid_actor_side_is_rejected(self):
        collector = self._collector()
        self._start(collector)
        trace = DecisionTrace(
            move=(5, 5),
            color=RED,
            reason="bad",
            actor=ActorMetadata(actor_type="human", actor_id="player1"),
        )
        with self.assertRaises(MemoryCollectorError):
            collector.record_move(HexBoard(), trace)
    def test_background_writer_flushes_complete_match(self):
        collector = self._collector(synchronous=False)
        self._record_complete_match(collector)
        self.assertTrue(collector.accept_match())
        self.assertTrue(collector.flush(timeout=5.0))
        collector.close()
        with ExperienceStore(self.db_path) as store:
            self.assertEqual(store.counts()["decisions"], 2)
        self.assertEqual(collector.stats().written, 1)
        self.assertEqual(collector.stats().failed, 0)

    def test_fail_open_keeps_call_side_alive(self):
        def broken_store_factory():
            raise RuntimeError("disk unavailable")

        collector = self._collector(
            synchronous=True,
            fail_open=True,
            store_factory=broken_store_factory,
        )
        self._record_complete_match(collector)
        self.assertFalse(collector.accept_match())
        self.assertEqual(collector.stats().failed, 1)
        self.assertIn("disk unavailable", collector.last_error)

    def test_fail_closed_can_surface_store_error(self):
        def broken_store_factory():
            raise RuntimeError("disk unavailable")

        collector = self._collector(
            synchronous=True,
            fail_open=False,
            store_factory=broken_store_factory,
        )
        self._record_complete_match(collector)
        with self.assertRaises(RuntimeError):
            collector.accept_match()

    def test_abort_clears_current_match(self):
        collector = self._collector()
        self._start(collector)
        collector.record_move(HexBoard(), self._trace())
        collector.abort_match("mode_switched_before_finish")
        self.assertEqual(collector.stats().buffered_moves, 0)
        with self.assertRaises(MemoryCollectorError):
            collector.mark_game_over(RED, "connect")
        self.assertFalse(self.db_path.exists())

    def test_env_file_is_loaded_and_process_env_wins(self):
        env_file = Path(self.temp_dir.name) / ".env.local"
        env_file.write_text(
            "HEX_MEMORY_CAPTURE=shadow\n"
            f"HEX_MEMORY_DB={self.db_path}\n"
            "HEX_MEMORY_QUEUE_SIZE=7\n",
            encoding="utf-8",
        )
        collector = create_memory_collector_from_env(
            env_file=env_file,
            environ={},
        )
        self.addCleanup(collector.close)
        self.assertTrue(collector.enabled)
        self.assertEqual(collector.config.queue_size, 7)
        overridden = create_memory_collector_from_env(
            env_file=env_file,
            environ={"HEX_MEMORY_CAPTURE": "off"},
        )
        self.assertFalse(overridden.enabled)


    def test_environment_factory_builds_shadow_collector(self):
        with mock.patch.dict(
            "os.environ",
            {
                "HEX_MEMORY_CAPTURE": "shadow",
                "HEX_MEMORY_DB": str(self.db_path),
                "HEX_MEMORY_QUEUE_SIZE": "4",
                "HEX_MEMORY_FLUSH_TIMEOUT_SECONDS": "2",
                "HEX_MEMORY_FAIL_OPEN": "1",
            },
            clear=False,
        ):
            collector = create_memory_collector_from_env()
        self.addCleanup(collector.close)
        self.assertTrue(collector.enabled)
        self.assertEqual(collector.config.mode, CAPTURE_SHADOW)
        self.assertEqual(collector.config.queue_size, 4)


if __name__ == "__main__":
    unittest.main()
