# -*- coding: utf-8 -*-
"""V4 trust, active learning, league, orchestrator, drift and deployment tests."""
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from ai.memory.active_learning import (
    persist_candidates,
    score_candidate,
    select_candidates,
)
from ai.memory.deployment import (
    activate_deployment,
    build_deployment_manifest,
    rollback_deployment,
    validate_deployment_manifest,
)
from ai.memory.drift import build_drift_snapshot
from ai.memory.league import build_league_table, expected_score, update_elo
from ai.memory.orchestrator import LearningOrchestrator, OrchestratorConfig
from ai.memory.store import ExperienceStore
from ai.memory.trust import (
    MemoryTrustFeatures,
    MemoryTrustModel,
    load_trust_model,
    save_trust_model,
)


class MemoryV4Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _features(self, confidence, support=4, conflict=False, type_name="similar"):
        return MemoryTrustFeatures(
            similarity=0.8,
            confidence=confidence,
            support_count=support,
            vote_share=0.8,
            conflict=conflict,
            ply=12,
            retrieval_type=type_name,
            model_agrees=False,
        )

    def test_trust_model_training_calibration_and_round_trip(self):
        features = []
        labels = []
        for index in range(30):
            good = index % 2 == 0
            features.append(self._features(0.9 if good else 0.2, conflict=not good))
            labels.append(1 if good else 0)
        model = MemoryTrustModel.fit(features, labels, epochs=300)
        good_probability = model.predict(features[0])
        bad_probability = model.predict(features[1])
        self.assertGreater(good_probability, bad_probability)
        self.assertIn("brier", model.metrics)
        path = save_trust_model(self.root / "trust.json", model)
        loaded = load_trust_model(path)
        self.assertAlmostEqual(loaded.predict(features[0]), good_probability, places=6)

    def test_active_learning_prefers_uncertain_conflicting_samples(self):
        uncertain = score_candidate(
            position_id="pos-uncertain",
            decision_id="dec-uncertain",
            trust_probability=0.51,
            confidence=0.7,
            support_count=3,
            conflict=True,
            retrieval_type="similar",
            ply=50,
        )
        certain = score_candidate(
            position_id="pos-certain",
            decision_id="dec-certain",
            trust_probability=0.98,
            confidence=0.95,
            support_count=8,
            conflict=False,
            retrieval_type="exact",
            ply=5,
        )
        self.assertGreater(uncertain.priority, certain.priority)
        selected = select_candidates([certain, uncertain], quota=1)
        self.assertEqual(selected[0].candidate_id, uncertain.candidate_id)
        with ExperienceStore(":memory:") as store:
            self.assertEqual(persist_candidates(store, selected), 1)
            rows = store.list_active_learning_candidates()
        self.assertEqual(rows[0]["position_id"], "pos-uncertain")

    def test_league_elo_and_table(self):
        self.assertLess(expected_score(1400, 1600), expected_score(1600, 1400))
        self.assertGreater(update_elo(1500, 1500, 1.0), 1500)
        table = build_league_table([
            {"opponent_id": "basic", "games_completed": 10, "ai_a_wins": 7,
             "ai_b_wins": 3, "opponent_elo": 1400},
            {"opponent_id": "strategic", "games_completed": 10, "ai_a_wins": 4,
             "ai_b_wins": 6, "opponent_elo": 1600},
        ])
        self.assertEqual(table[0].opponent_id, "basic")
        self.assertGreater(table[0].win_rate, table[1].win_rate)

    def test_drift_snapshot_persists(self):
        with ExperienceStore(self.root / "drift.sqlite3") as store:
            report = build_drift_snapshot(store)
            rows = store.list_drift_snapshots()
        self.assertEqual(rows[0]["snapshot_id"], report["snapshot_id"])
        self.assertEqual(report["metrics"]["validation_rate"], 0.0)

    def test_orchestrator_reports_not_ready_for_empty_db(self):
        db_path = self.root / "orchestrator.sqlite3"
        with ExperienceStore(db_path):
            pass
        orchestrator = LearningOrchestrator(
            OrchestratorConfig(minimum_new_matches=2, minimum_training_samples=10),
            db_path=db_path,
        )
        status = orchestrator.status()
        self.assertFalse(status["ready"])
        self.assertFalse(status["auto_promote"])

    def test_deployment_apply_validate_and_rollback(self):
        models = self.root / "models"
        models.mkdir()
        model = models / "candidate.pt"
        policy = models / "memory_policy.json"
        model.write_bytes(b"model")
        policy.write_text('{"schema_version":1}', encoding="utf-8")
        first = build_deployment_manifest(
            models_dir=models,
            model_path=model,
            policy_path=policy,
            dataset_id="data-1",
        )
        result = activate_deployment(models_dir=models, manifest=first, apply=True)
        self.assertTrue(result["applied"])
        selected = validate_deployment_manifest(
            models / "deployment_manifest.json", models_dir=models
        )
        self.assertEqual(selected["dataset_id"], "data-1")
        second = dict(first)
        second["dataset_id"] = "data-2"
        activate_deployment(models_dir=models, manifest=second, apply=True)
        rolled_back = rollback_deployment(models_dir=models)
        self.assertEqual(rolled_back["dataset_id"], "data-1")


if __name__ == "__main__":
    unittest.main()