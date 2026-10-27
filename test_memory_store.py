# -*- coding: utf-8 -*-
"""P0-B tests for the SQLite-backed experience store."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ai.memory.canonical import canonicalize_position
from ai.memory.ids import (
    build_decision_id,
    build_game_id,
    build_memory_position_id,
    new_match_id,
)
from ai.memory.schema import MEMORY_SCHEMA_VERSION
from ai.memory.store import (
    DecisionRecord,
    ExperienceStore,
    GameRecord,
    MatchRecord,
    MemoryClosedError,
    MemoryConflictError,
    MemoryIntegrityError,
    MemorySchemaError,
    MemoryStoreError,
    MemoryValidationError,
    PositionRecord,
    recover_database,
)
from hex_board import BLUE, RED, HexBoard


class ExperienceStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "memory.sqlite3"

    def _bundle(self, *, match_id=None, game_no=1, board=None, color=RED,
                ply=1, candidate_json="[]", decision=None, match=None,
                game=None, position=None):
        board = board or HexBoard()
        match_id = match_id or new_match_id()
        game_id = build_game_id(match_id, game_no)
        position_id = build_memory_position_id(board, color)
        canonical = canonicalize_position(board, color)
        actor_type = "ai" if color == RED else "human"
        decision_id = build_decision_id(game_id, ply, actor_type, color)
        match = match or MatchRecord(
            match_id=match_id,
            mode_name="human_vs_ai",
            red_team="HybridAI",
            blue_team="Human",
            source_type="human_vs_ai",
            started_at="2026-01-01T00:00:00",
            mode_key=2,
        )
        game = game or GameRecord(
            game_id=game_id,
            match_id=match_id,
            game_no=game_no,
            red_actor_type="ai",
            blue_actor_type="human",
            started_at="2026-01-01T00:00:00",
            red_ai_name="HybridAI",
            red_ai_version="6.0.0",
        )
        position = position or PositionRecord(
            position_id=position_id,
            experience_key=position_id,
            canonical_key=canonical.key,
            board_json=json.dumps(
                {"rows": 11, "cols": 11, "grid": board.grid},
                sort_keys=True,
                separators=(",", ":"),
            ),
            current_color=color,
            ply=ply,
            created_at="2026-01-01T00:00:01",
        )
        decision = decision or DecisionRecord(
            decision_id=decision_id,
            game_id=game_id,
            position_id=position_id,
            ply=ply,
            actor_type=actor_type,
            actor_color=color,
            created_at="2026-01-01T00:00:02",
            actor_id="HybridAI",
            ai_name="HybridAI",
            ai_version="6.0.0",
            candidate_json=candidate_json,
            search_config_json='{"simulations":100}',
            chosen_q=5,
            chosen_r=5,
        )
        return match, game, position, decision

    def _open_store(self, *, recover_corrupt=False):
        store = ExperienceStore(
            self.db_path,
            busy_timeout_ms=750,
            recover_corrupt=recover_corrupt,
        )
        store.open()
        self.addCleanup(store.close)
        return store

    def test_constructor_does_not_create_database_and_open_migrates(self):
        store = ExperienceStore(self.db_path)
        self.assertFalse(self.db_path.exists())
        store.open()
        try:
            self.assertTrue(self.db_path.exists())
            health = store.health_check()
            self.assertEqual(health.schema_version, MEMORY_SCHEMA_VERSION)
            self.assertEqual(health.journal_mode, "wal")
            self.assertTrue(health.foreign_keys)
            self.assertEqual(health.busy_timeout_ms, 5000)
            self.assertEqual(health.counts, {
                "matches": 0,
                "games": 0,
                "positions": 0,
                "decisions": 0,
            })
        finally:
            store.close()

    def test_save_decision_bundle_round_trip(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        result = store.save_decision_bundle(match, game, position, decision)
        self.assertTrue(result.any_changed)
        self.assertEqual(store.get_match(match.match_id)["red_team"], "HybridAI")
        self.assertEqual(store.get_game(game.game_id)["game_no"], 1)
        self.assertEqual(store.get_position(position.position_id)["ply"], 1)
        stored = store.get_decision(decision.decision_id)
        self.assertEqual(stored["chosen_q"], 5)
        self.assertEqual(stored["chosen_r"], 5)
        self.assertEqual(stored["training_eligible"], 0)
        self.assertEqual(store.counts(), {
            "matches": 1,
            "games": 1,
            "positions": 1,
            "decisions": 1,
        })

    def test_repeated_bundle_is_idempotent(self):
        bundle = self._bundle()
        store = self._open_store()
        first = store.save_decision_bundle(*bundle)
        second = store.save_decision_bundle(*bundle)
        self.assertTrue(first.any_changed)
        self.assertFalse(second.any_changed)
        self.assertEqual(store.counts()["decisions"], 1)

    def test_validation_fields_can_be_finalized_without_duplicate(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(match, game, position, decision)
        finalized = replace(
            decision,
            result=1.0,
            decision_quality="validated",
            training_eligible=1,
            integrity_status="valid",
            rule_status="valid",
            result_status="known",
            validation_reason="replayed successfully",
            validator_version="1.0",
        )
        result = store.save_decision_bundle(match, game, position, finalized)
        self.assertFalse(result.match_changed)
        self.assertFalse(result.game_changed)
        self.assertFalse(result.position_changed)
        self.assertTrue(result.decision_changed)
        stored = store.get_decision(decision.decision_id)
        self.assertEqual(stored["result"], 1.0)
        self.assertEqual(stored["decision_quality"], "validated")
        self.assertEqual(stored["training_eligible"], 1)
        self.assertEqual(store.counts()["decisions"], 1)

    def test_conflicting_decision_rolls_back_other_table_updates(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(match, game, position, decision)
        changed_match = replace(
            match,
            status="completed",
            ended_at="2026-01-01T00:10:00",
        )
        changed_decision = replace(
            decision,
            candidate_json='[{"move":[4,4]}]',
        )
        with self.assertRaises(MemoryConflictError):
            store.save_decision_bundle(
                changed_match, game, position, changed_decision
            )
        self.assertEqual(store.get_match(match.match_id)["status"], "active")
        self.assertEqual(store.get_decision(decision.decision_id)["candidate_json"], "[]")

    def test_exact_position_is_reused_across_games(self):
        match, game_one, position, decision_one = self._bundle(game_no=1)
        store = self._open_store()
        first = store.save_decision_bundle(match, game_one, position, decision_one)
        _, game_two, _, decision_two = self._bundle(
            match_id=match.match_id,
            game_no=2,
            board=HexBoard(),
            color=RED,
            ply=1,
            match=match,
            position=position,
        )
        second = store.save_decision_bundle(
            match, game_two, position, decision_two
        )
        self.assertTrue(first.position_changed)
        self.assertFalse(second.position_changed)
        self.assertEqual(store.counts()["positions"], 1)
        self.assertEqual(store.counts()["games"], 2)
        self.assertEqual(store.counts()["decisions"], 2)
    def test_orphan_game_is_rejected(self):
        _, game, _, _ = self._bundle()
        store = self._open_store()
        with self.assertRaises(MemoryConflictError):
            store.save_game(game)
        self.assertEqual(store.counts()["games"], 0)

    def test_duplicate_game_slot_with_new_id_is_rejected(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(match, game, position, decision)
        other_game = replace(game, game_id=build_game_id(new_match_id(), 1))
        with self.assertRaises(MemoryConflictError):
            store.save_game(other_game)
        self.assertEqual(store.counts()["games"], 1)

    def test_position_identity_conflict_is_rejected(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(match, game, position, decision)
        changed = replace(
            position,
            board_json='{"rows":11,"cols":11,"grid":[[1]]}',
        )
        with self.assertRaises(MemoryConflictError):
            store.save_position(changed)
        self.assertEqual(store.counts()["positions"], 1)

    def test_invalid_records_are_rejected_before_writing(self):
        match, game, position, decision = self._bundle()
        store = self._open_store()
        cases = (
            replace(position, current_color=0),
            replace(position, board_json="not-json"),
            replace(position, canonical_key="invalid"),
            replace(decision, training_eligible=2),
            replace(decision, chosen_q=5, chosen_r=None),
            replace(decision, result=0.25),
        )
        for invalid in cases:
            with self.subTest(record=type(invalid).__name__):
                with self.assertRaises(MemoryValidationError):
                    if isinstance(invalid, PositionRecord):
                        store.save_position(invalid)
                    else:
                        store.save_decision(invalid)
        self.assertFalse(store.counts()["decisions"])

    def test_newer_database_schema_is_not_downgraded(self):
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(f"PRAGMA user_version = {MEMORY_SCHEMA_VERSION + 1}")
            connection.commit()
        finally:
            connection.close()
        store = ExperienceStore(self.db_path)
        with self.assertRaises(MemorySchemaError):
            store.open()
        self.assertFalse(store.is_open)

    def test_missing_index_is_detected_as_schema_damage(self):
        store = self._open_store()
        store.close()
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute("DROP INDEX idx_decisions_quality")
            connection.commit()
        finally:
            connection.close()
        damaged = ExperienceStore(self.db_path)
        with self.assertRaises(MemorySchemaError):
            damaged.open()
        self.assertFalse(damaged.is_open)
    def test_backup_is_consistent_and_reopenable(self):
        bundle = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(*bundle)
        backup_path = Path(self.temp_dir.name) / "backup.sqlite3"
        result = store.backup_to(backup_path)
        self.assertEqual(result, backup_path.resolve())
        with ExperienceStore(backup_path) as backup:
            self.assertEqual(backup.health_check().counts["decisions"], 1)
            self.assertEqual(
                backup.get_decision(bundle[3].decision_id)["chosen_q"],
                5,
            )

    def test_backup_rejects_source_path(self):
        store = self._open_store()
        with self.assertRaises(MemoryValidationError):
            store.backup_to(self.db_path)

    def test_corrupt_database_is_quarantined_and_rebuilt(self):
        payload = b"this is not a sqlite database"
        self.db_path.write_bytes(payload)
        store = ExperienceStore(self.db_path, recover_corrupt=True)
        store.open()
        try:
            self.assertIsNotNone(store.last_recovery_path)
            quarantine = store.last_recovery_path
            preserved = quarantine / self.db_path.name
            self.assertTrue(preserved.exists())
            self.assertEqual(preserved.read_bytes(), payload)
            self.assertEqual(store.health_check().integrity, "ok")
        finally:
            store.close()

    def test_explicit_recovery_creates_a_clean_database(self):
        self.db_path.write_bytes(b"broken database bytes")
        quarantine = recover_database(self.db_path)
        self.assertTrue((quarantine / self.db_path.name).exists())
        with ExperienceStore(self.db_path) as store:
            self.assertEqual(store.health_check().schema_version, MEMORY_SCHEMA_VERSION)

    def test_recovery_refuses_a_healthy_database(self):
        with ExperienceStore(self.db_path):
            pass
        with self.assertRaises(MemoryStoreError):
            recover_database(self.db_path)
        self.assertTrue(self.db_path.exists())

    def test_closed_store_rejects_operations(self):
        store = ExperienceStore(":memory:").open()
        store.close()
        self.assertFalse(store.is_open)
        with self.assertRaises(MemoryClosedError):
            store.counts()
        with self.assertRaises(MemoryClosedError):
            store.health_check()

    def test_context_manager_closes_store(self):
        with ExperienceStore(self.db_path) as store:
            self.assertTrue(store.is_open)
            self.assertEqual(store.health_check().integrity, "ok")
        self.assertFalse(store.is_open)

    def test_list_and_canonical_queries(self):
        match, game_one, position, decision_one = self._bundle(game_no=1)
        store = self._open_store()
        store.save_decision_bundle(match, game_one, position, decision_one)
        _, game_two, _, decision_two = self._bundle(
            match_id=match.match_id,
            game_no=2,
            match=match,
            position=position,
        )
        store.save_decision_bundle(match, game_two, position, decision_two)
        decisions = store.list_decisions(limit=10)
        self.assertEqual([row["ply"] for row in decisions], [1, 1])
        self.assertEqual(len(store.find_positions_by_canonical_key(position.canonical_key)), 1)
        self.assertEqual(
            len(store.list_decisions(training_eligible=0, limit=10)),
            2,
        )

    def test_two_store_instances_observe_committed_writes(self):
        bundle = self._bundle()
        first = self._open_store()
        second = ExperienceStore(self.db_path)
        second.open()
        self.addCleanup(second.close)
        first.save_decision_bundle(*bundle)
        self.assertEqual(second.counts()["decisions"], 1)
        self.assertEqual(
            second.get_decision(bundle[3].decision_id)["chosen_q"],
            5,
        )
    def test_foreign_key_corruption_is_detected(self):
        bundle = self._bundle()
        store = self._open_store()
        store.save_decision_bundle(*bundle)
        store.close()
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DELETE FROM matches")
            connection.commit()
        finally:
            connection.close()
        damaged = ExperienceStore(self.db_path)
        with self.assertRaises(MemoryIntegrityError):
            damaged.open()
        self.assertFalse(damaged.is_open)


if __name__ == "__main__":
    unittest.main()