# -*- coding: utf-8 -*-
"""V3 lineage registry, retrieval policy and status tests."""
import json
import tempfile
import unittest
from pathlib import Path

from ai.memory.dataset import build_memory_dataset
from ai.memory.dataset_registry import register_dataset_version, register_training_run
from ai.memory.memory_ai import MemoryAssistedAI, MemoryRetrievalSettings
from ai.memory.model_registry import PromotionGate, evaluate_promotion_gate
from ai.memory.retrieval_policy import (
    RetrievalPolicy,
    load_retrieval_policy,
    save_retrieval_policy,
)
from ai.memory.status import build_memory_status
from ai.memory.store import ExperienceStore
from memory_test_support import collect_scripted_match, promote_match


class MemoryV3RegistryPolicyTest(unittest.TestCase):
    def test_dataset_and_training_lineage_are_registered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            dataset_dir = root / "dataset"
            match_id = collect_scripted_match(db_path)
            promote_match(db_path, match_id)
            with ExperienceStore(db_path) as store:
                export = build_memory_dataset(store, dataset_dir, seed=11)
                dataset = register_dataset_version(store, dataset_dir)
                created = register_training_run(
                    store,
                    training_run_id="run-1",
                    dataset_id=dataset["dataset_id"],
                    parent_model_sha256="a" * 64,
                    candidate_model_sha256="b" * 64,
                    status="completed",
                    metrics={"validation_loss": 0.25},
                )
                datasets = store.list_dataset_versions()
                runs = store.list_training_runs()
            self.assertTrue(created)
            self.assertEqual(export.sha256, dataset["dataset_id"])
            self.assertEqual(len(datasets), 1)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["training_run_id"], "run-1")

    def test_policy_file_round_trip_and_memory_ai_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "memory_policy.json"
            policy = RetrievalPolicy(
                top_k=7,
                minimum_similarity=0.81,
                minimum_support=4,
                minimum_confidence=0.67,
                alpha=0.22,
                retrieval_types=("exact", "canonical"),
            )
            save_retrieval_policy(policy_path, policy)
            loaded = load_retrieval_policy(policy_path)
            self.assertEqual(loaded.top_k, 7)
            settings = MemoryRetrievalSettings(
                mode="shadow",
                policy_file=str(policy_path),
            )
            ai = MemoryAssistedAI(object(), settings=settings)
            self.assertEqual(ai.settings.top_k, 7)
            self.assertEqual(ai.settings.minimum_similarity, 0.81)
            self.assertEqual(ai.settings.retrieval_types, ("exact", "canonical"))

    def test_status_report_contains_model_policy_and_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "memory.sqlite3"
            models = root / "models"
            models.mkdir()
            (models / "selected_model.json").write_text(json.dumps({
                "schema_version": 1,
                "model_file": "model.pt",
                "sha256": "a" * 64,
            }), encoding="utf-8")
            policy = RetrievalPolicy()
            save_retrieval_policy(models / "memory_policy.json", policy)
            with ExperienceStore(db_path) as store:
                status = build_memory_status(
                    store,
                    models_dir=models,
                )
            self.assertIn("retrieval_events", status["counts"])
            self.assertEqual(status["selected_model"]["model_file"], "model.pt")
            self.assertEqual(status["retrieval_policy"]["top_k"], 5)
            self.assertIn("jobs", status)

    def test_v3_promotion_gate_checks_ci_color_gap_and_p95(self):
        gate = PromotionGate(
            minimum_games=2000,
            minimum_win_rate=50.0,
            minimum_color_win_rate=40.0,
            maximum_errors=0,
            maximum_move_ms=500.0,
            minimum_ci95_low=48.0,
            maximum_color_gap=10.0,
            maximum_p95_move_ms=200.0,
        )
        passing = evaluate_promotion_gate({"summary": {
            "games_completed": 2000,
            "errors": 0,
            "ai_a_win_rate": 55.0,
            "ai_a_red_win_rate": 54.0,
            "ai_a_blue_win_rate": 56.0,
            "ai_a_win_rate_ci95_low": 52.0,
            "ai_a_max_move_ms": 210.0,
            "ai_a_p95_move_ms": 120.0,
        }}, gate)
        failing = evaluate_promotion_gate({"summary": {
            "games_completed": 2000,
            "errors": 0,
            "ai_a_win_rate": 55.0,
            "ai_a_red_win_rate": 65.0,
            "ai_a_blue_win_rate": 45.0,
            "ai_a_win_rate_ci95_low": 47.0,
            "ai_a_max_move_ms": 210.0,
            "ai_a_p95_move_ms": 220.0,
        }}, gate)
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["checks"]["confidence_lower_bound"])
        self.assertFalse(failing["checks"]["color_gap"])
        self.assertFalse(failing["checks"]["p95_latency"])


if __name__ == "__main__":
    unittest.main()