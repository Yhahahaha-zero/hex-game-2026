# -*- coding: utf-8 -*-
"""P0-A contract tests for stable IDs and the memory SQLite schema."""
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from ai.board_observation import build_position_id
from ai.memory.ids import (
    MemoryIdError,
    build_decision_id,
    build_experience_key,
    build_game_id,
    build_memory_position_id,
    new_match_id,
)
from ai.memory.schema import (
    DEFAULT_MEMORY_DB,
    DO_NOT_WRITE_REASONS,
    MEMORY_SCHEMA_VERSION,
    SCHEMA_DDL,
    memory_db_path,
)
from hex_board import BLUE, RED, HexBoard


class MemoryIdContractTest(unittest.TestCase):
    def test_match_and_game_ids_are_stable(self):
        match_id = new_match_id()
        self.assertEqual(str(uuid.UUID(match_id)), match_id)
        game_one = build_game_id(match_id, 1)
        game_two = build_game_id(match_id, 2)
        self.assertEqual(game_one, build_game_id(match_id, 1))
        self.assertNotEqual(game_one, game_two)
        self.assertNotEqual(game_one, build_game_id(new_match_id(), 1))

    def test_invalid_game_identifier_inputs_are_rejected(self):
        match_id = new_match_id()
        for game_no in (0, 3, True, "1"):
            with self.subTest(game_no=game_no):
                with self.assertRaises(MemoryIdError):
                    build_game_id(match_id, game_no)
        with self.assertRaises(MemoryIdError):
            build_game_id("not-a-uuid", 1)

    def test_memory_position_key_ignores_history_and_match_context(self):
        first = HexBoard()
        first.place(5, 5, RED)
        first.place(4, 5, BLUE)
        second = HexBoard()
        second.place(4, 5, BLUE)
        second.place(5, 5, RED)
        key_a = build_memory_position_id(first, BLUE)
        key_b = build_memory_position_id(second, BLUE)
        self.assertEqual(key_a, key_b)
        self.assertEqual(key_a, build_experience_key(first, BLUE))
        self.assertNotEqual(key_a, build_memory_position_id(first, RED))

    def test_analysis_position_id_is_not_reused_as_memory_key(self):
        board = HexBoard()
        first_analysis = build_position_id(board, RED, game_no=1, state="playing")
        second_analysis = build_position_id(board, RED, game_no=2, state="playing")
        self.assertNotEqual(first_analysis, second_analysis)
        self.assertEqual(
            build_memory_position_id(board, RED),
            build_memory_position_id(board, RED),
        )

    def test_decision_id_is_idempotent_and_scoped(self):
        game_id = build_game_id(new_match_id(), 1)
        first = build_decision_id(game_id, 4, "ai", RED)
        self.assertEqual(first, build_decision_id(game_id, 4, "AI", RED))
        self.assertNotEqual(first, build_decision_id(game_id, 4, "human", RED))
        self.assertNotEqual(first, build_decision_id(game_id, 5, "ai", RED))
        self.assertNotEqual(first, build_decision_id(game_id, 4, "ai", BLUE))

    def test_memory_position_rejects_invalid_color(self):
        with self.assertRaises(MemoryIdError):
            build_memory_position_id(HexBoard(), 0)


class MemorySchemaContractTest(unittest.TestCase):
    def test_schema_creates_required_tables_and_indexes(self):
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(SCHEMA_DDL)
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            indexes = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
        finally:
            connection.close()
        self.assertTrue(
            {
                "matches", "games", "positions", "decisions", "retrieval_events",
                "retrieval_outcomes", "memory_jobs", "dataset_versions",
                "training_runs", "trust_models", "active_learning_queue",
                "opponent_profiles", "league_runs", "drift_snapshots",
            } <= tables
        )
        self.assertIn("idx_positions_canonical", indexes)
        self.assertIn("idx_decisions_quality", indexes)
        self.assertIn("idx_decisions_training", indexes)
        self.assertIn("idx_retrieval_query_game", indexes)
        self.assertIn("idx_outcomes_game", indexes)
        self.assertIn("idx_jobs_status", indexes)
        self.assertIn("idx_training_dataset", indexes)
        self.assertIn("idx_active_learning_priority", indexes)
        self.assertIn("idx_trust_models_status", indexes)
        self.assertIn("idx_league_runs_status", indexes)
        self.assertIn("idx_drift_created", indexes)

    def test_schema_accepts_one_consistent_case(self):
        match_id = new_match_id()
        game_id = build_game_id(match_id, 1)
        position_id = build_memory_position_id(HexBoard(), RED)
        decision_id = build_decision_id(game_id, 1, "ai", RED)
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(SCHEMA_DDL)
            connection.execute(
                """
                INSERT INTO matches (
                    match_id, schema_version, rules_version, mode_key, mode_name,
                    red_team, blue_team, source_type, started_at, ended_at,
                    status, record_id, capture_mode, collector_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (match_id, MEMORY_SCHEMA_VERSION, 1, 2, "人机对战",
                 "红队", "蓝队", "human_vs_ai", "2026-01-01T00:00:00",
                 None, "active", None, "shadow", "1.0.0"),
            )
            connection.execute(
                """
                INSERT INTO games (
                    game_id, match_id, game_no, red_actor_type, blue_actor_type,
                    red_actor_id, blue_actor_id, red_ai_name, blue_ai_name,
                    red_ai_version, blue_ai_version, winner_color,
                    termination_reason, status, started_at, ended_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (game_id, match_id, 1, "human", "ai", None, None,
                 None, "HybridAI", None, "6.0.0", None, None,
                 "active", "2026-01-01T00:00:00", None),
            )
            connection.execute(
                """INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (position_id, position_id, "can1-test", 1, 1,
                 '{"grid": []}', RED, 1, None, "2026-01-01T00:00:00"),
            )
            connection.execute(
                """
                INSERT INTO decisions (
                    decision_id, game_id, position_id, ply, actor_type,
                    actor_color, actor_id, ai_name, ai_version, candidate_json,
                    search_config_json, chosen_q, chosen_r, result,
                    decision_quality, training_eligible, integrity_status,
                    rule_status, result_status, validation_reason,
                    validator_version, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (decision_id, game_id, position_id, 1, "ai", RED, "hybrid",
                 "HybridAI", "test", "[]", "{}", 5, 5, 1.0,
                 "validated", 1, "valid", "valid", "known",
                 "test", "test-v1", "2026-01-01T00:00:00"),
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 1)
        finally:
            connection.close()

    def test_schema_and_write_rules_are_documented_in_constants(self):
        self.assertEqual(MEMORY_SCHEMA_VERSION, 5)
        self.assertIn("proposal_not_committed", DO_NOT_WRITE_REASONS)
        self.assertIn("move_undone", DO_NOT_WRITE_REASONS)
        self.assertIn("replay_session", DO_NOT_WRITE_REASONS)
        self.assertIn("invalid_record_hash", DO_NOT_WRITE_REASONS)

    def test_database_path_allows_environment_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "custom.sqlite3"
            self.assertEqual(
                memory_db_path({"HEX_MEMORY_DB": str(target)}),
                target.resolve(),
            )
        self.assertTrue(str(DEFAULT_MEMORY_DB).endswith("hex_experience.sqlite3"))


if __name__ == "__main__":
    unittest.main()