# -*- coding: utf-8 -*-
"""V4 trust, active-learning, drift and orchestration tool tests."""
import tempfile
import unittest
from pathlib import Path
import uuid

from ai.memory.schema import RETRIEVAL_VERSION
from ai.memory.store import (
    ExperienceStore,
    RetrievalEventRecord,
    RetrievalOutcomeRecord,
)
from memory_test_support import collect_scripted_match, promote_match
from tools import build_active_dataset, memory_drift_report, train_memory_trust


class MemoryV4ToolsTest(unittest.TestCase):
    def test_v4_tools_create_trust_model_active_queue_and_drift_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            with ExperienceStore(db_path) as store:
                document = store.load_match_for_validation(match_id)
                game = document["games"][0]
                decision = store.list_decisions(game_id=game["game_id"], limit=10)[0]
                event_id = str(uuid.uuid4())
                store.save_retrieval_event(RetrievalEventRecord(
                    retrieval_event_id=event_id,
                    query_position_id=decision["position_id"],
                    query_game_id=game["game_id"],
                    query_match_id=match_id,
                    retrieval_mode="assist",
                    top_k=5,
                    selected_memory_decision_id=decision["decision_id"],
                    similarity=0.9,
                    memory_move=(decision["chosen_q"], decision["chosen_r"]),
                    base_move=(decision["chosen_q"], decision["chosen_r"]),
                    shadow_move=(decision["chosen_q"], decision["chosen_r"]),
                    final_move=(decision["chosen_q"], decision["chosen_r"]),
                    confidence=0.8,
                    applied=1,
                    feature_version=1,
                    retrieval_version=RETRIEVAL_VERSION,
                    elapsed_ms=2.0,
                    evidence_json=(
                        '{"support_count":3,"vote_share":1.0,"conflict":false,'
                        '"hits":[{"similarity_type":"exact"}]}'
                    ),
                    created_at="2026-01-01T00:00:00",
                    query_ply=decision["ply"],
                    selection_probability=0.8,
                    exploration_epsilon=0.0,
                    policy_version="test",
                ))
                store.save_retrieval_outcome(RetrievalOutcomeRecord(
                    retrieval_event_id=event_id,
                    game_id=game["game_id"],
                    decision_id=decision["decision_id"],
                    applied=1,
                    decision_result=1.0,
                    game_result=1.0,
                    retrieval_reward=1.0,
                    similarity=0.9,
                    confidence=0.8,
                    retrieval_mode="assist",
                    created_at="2026-01-01T00:00:01",
                ))
            trust_path = root / "memory_trust.json"
            self.assertEqual(train_memory_trust.main([
                "--db", str(db_path),
                "--output", str(trust_path),
            ]), 0)
            self.assertTrue(trust_path.is_file())
            self.assertEqual(build_active_dataset.main([
                "--db", str(db_path),
                "--quota", "1",
            ]), 0)
            self.assertEqual(memory_drift_report.main([
                "--db", str(db_path),
            ]), 0)
            with ExperienceStore(db_path) as store:
                self.assertEqual(len(store.list_trust_models()), 1)
                self.assertEqual(len(store.list_active_learning_candidates()), 1)
                self.assertEqual(len(store.list_drift_snapshots()), 1)


if __name__ == "__main__":
    unittest.main()