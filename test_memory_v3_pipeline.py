# -*- coding: utf-8 -*-
"""V3 outcome attribution and background pipeline tests."""
import tempfile
import time
import unittest
from pathlib import Path
import uuid

from ai.memory.outcomes import (
    attribute_match_outcomes,
    summarize_outcomes,
)
from ai.memory.pipeline import (
    MemoryLearningPipeline,
    MemoryPipelineWorker,
    NullMemoryPipelineWorker,
    create_memory_pipeline_from_env,
)
from ai.memory.schema import RETRIEVAL_VERSION
from ai.memory.store import ExperienceStore, RetrievalEventRecord
from memory_test_support import collect_scripted_match
from tools import tune_memory_retrieval


class MemoryV3PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "pipeline.sqlite3"

    def _pipeline(self):
        return MemoryLearningPipeline(
            lambda: ExperienceStore(self.db_path).open(), mode="full"
        )

    def _save_event(self, store, match_id, game_id, decision, applied=1):
        move = (int(decision["chosen_q"]), int(decision["chosen_r"]))
        record = RetrievalEventRecord(
            retrieval_event_id=str(uuid.uuid4()),
            query_position_id=decision["position_id"],
            query_game_id=game_id,
            query_match_id=match_id,
            retrieval_mode="assist",
            top_k=5,
            selected_memory_decision_id=decision["decision_id"],
            similarity=0.91,
            memory_move=move,
            base_move=move,
            shadow_move=move,
            final_move=move,
            confidence=0.88,
            applied=applied,
            feature_version=1,
            retrieval_version=RETRIEVAL_VERSION,
            elapsed_ms=3.2,
            evidence_json='{"support_count": 3}',
            created_at="2026-01-01T00:00:00+00:00",
            query_ply=int(decision["ply"]),
        )
        store.save_retrieval_event(record)

    def test_pipeline_validates_features_and_jobs_idempotently(self):
        match_id = collect_scripted_match(self.db_path)
        pipeline = self._pipeline()
        first = pipeline.process_match(match_id)
        second = pipeline.process_match(match_id)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "already_completed")
        self.assertGreater(first["features_updated"], 0)
        with ExperienceStore(self.db_path) as store:
            decisions = store.list_decisions(limit=1000)
            positions = store.all_positions()
            jobs = store.list_memory_jobs()
        self.assertTrue(all(row["training_eligible"] == 1 for row in decisions))
        self.assertTrue(all(row["features_json"] for row in positions))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["status"], "completed")

    def test_outcome_attribution_rewards_applied_win_and_loss(self):
        match_id = collect_scripted_match(self.db_path)
        self._pipeline().process_match(match_id)
        with ExperienceStore(self.db_path) as store:
            document = store.load_match_for_validation(match_id)
            game_one, game_two = document["games"]
            decisions_one = store.list_decisions(game_id=game_one["game_id"], limit=1000)
            decisions_two = store.list_decisions(game_id=game_two["game_id"], limit=1000)
            self._save_event(store, match_id, game_one["game_id"], decisions_one[0])
            self._save_event(store, match_id, game_two["game_id"], decisions_two[1])
            counts = attribute_match_outcomes(store, match_id)
            rows = store.list_retrieval_outcomes(limit=100)
        self.assertEqual(sum(counts.values()), 2)
        rewards = sorted(row["retrieval_reward"] for row in rows)
        self.assertEqual(rewards, [-1.0, 1.0])
        summary = summarize_outcomes(rows)
        self.assertEqual(summary["applied"], 2)
        self.assertEqual(summary["mean_reward"], 0.0)
        self.assertEqual(summary["positive_rate"], 0.5)
        self.assertEqual(summary["negative_rate"], 0.5)

    def test_tuning_tool_writes_versioned_policy_from_outcomes(self):
        match_id = collect_scripted_match(self.db_path)
        self._pipeline().process_match(match_id)
        with ExperienceStore(self.db_path) as store:
            document = store.load_match_for_validation(match_id)
            game_one = document["games"][0]
            decision = store.list_decisions(game_id=game_one["game_id"], limit=100)[0]
            self._save_event(store, match_id, game_one["game_id"], decision)
            attribute_match_outcomes(store, match_id)
        policy_path = Path(self.temp_dir.name) / "memory_policy.json"
        code = tune_memory_retrieval.main([
            "--db", str(self.db_path),
            "--policy-file", str(policy_path),
            "--apply",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(policy_path.is_file())

    def test_pipeline_reads_settings_from_env_file(self):
        env_file = Path(self.temp_dir.name) / ".env.local"
        env_file.write_text(
            "HEX_MEMORY_PIPELINE=full\n"
            "HEX_MEMORY_PIPELINE_QUEUE_SIZE=4\n"
            f"HEX_MEMORY_DB={self.db_path}\n",
            encoding="utf-8",
        )
        worker = create_memory_pipeline_from_env(env_file=env_file, environ={})
        try:
            self.assertIsInstance(worker, MemoryPipelineWorker)
            self.assertTrue(worker.enabled)
        finally:
            worker.close()
        env_file.write_text("HEX_MEMORY_PIPELINE=off\n", encoding="utf-8")
        disabled = create_memory_pipeline_from_env(env_file=env_file, environ={})
        self.assertIsInstance(disabled, NullMemoryPipelineWorker)

    def test_background_worker_processes_submitted_match(self):
        match_id = collect_scripted_match(self.db_path)
        worker = MemoryPipelineWorker(self._pipeline(), queue_size=4)
        worker.start()
        try:
            self.assertTrue(worker.submit(match_id))
            deadline = time.time() + 5
            while time.time() < deadline:
                with ExperienceStore(self.db_path) as store:
                    jobs = store.list_memory_jobs()
                if jobs and jobs[0]["status"] == "completed":
                    break
                time.sleep(0.02)
            self.assertTrue(jobs)
            self.assertEqual(jobs[0]["status"], "completed")
        finally:
            worker.close()
        self.assertGreaterEqual(worker.stats()["processed"], 1)


if __name__ == "__main__":
    unittest.main()