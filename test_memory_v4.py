# -*- coding: utf-8 -*-
"""V4 trust, active learning, league, orchestrator, drift and deployment tests."""
from datetime import datetime, timezone
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai.memory.active_learning import (
    persist_candidates,
    score_candidate,
    select_candidates,
)
from ai.memory.deployment import (
    DeploymentError,
    activate_deployment,
    build_deployment_manifest,
    rollback_deployment,
    validate_deployment_manifest,
    validate_deployment_manifest_from_document,
)
from ai.memory.drift import build_drift_snapshot
from ai.memory.league import (
    build_league_table,
    expected_score,
    register_league_run,
    update_elo,
)
from ai.memory.orchestrator import LearningOrchestrator, OrchestratorConfig
from ai.memory.store import (
    ActiveLearningCandidateRecord,
    DatasetVersionRecord,
    DriftSnapshotRecord,
    ExperienceStore,
    MemoryValidationError,
    TrustModelRecord,
)
from ai.memory.trust import (
    MemoryTrustFeatures,
    MemoryTrustModel,
    load_trust_model,
    save_trust_model,
)
from memory_test_support import collect_scripted_match, promote_match


PINNED_TIME = "2026-01-01T00:00:00+00:00"


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

    def _write_checkpoint(self, path, **overrides):
        checkpoint = {
            "model_state_dict": {"test": torch.zeros(1)},
            "model_name": "HexNetV5",
            "model_version": "5.0.0",
            "encoding_version": "3x11x11-v1",
        }
        checkpoint.update(overrides)
        torch.save(checkpoint, path)
        return path

    def _register_dataset(self, store, *, dataset_id="dataset-1"):
        matches = len(store.list_matches(limit=1000000))
        samples = len(store.query_training_samples(limit=1000000))
        self.assertTrue(store.save_dataset_version(DatasetVersionRecord(
            dataset_id=dataset_id,
            dataset_dir=str(self.root / dataset_id),
            manifest_sha256="a" * 64,
            samples_sha256="b" * 64,
            feature_version=1,
            encoding_version="3x11x11-v1",
            sample_count=samples,
            split_counts_json=json.dumps({"train": samples}),
            source_match_count=matches,
            created_at=PINNED_TIME,
        )))
        return matches, samples

    def test_trust_model_training_round_trip_and_platt_calibration(self):
        features = []
        labels = []
        for index in range(40):
            good = index % 2 == 0
            features.append(self._features(0.9 if good else 0.2, conflict=not good))
            labels.append(1 if good else 0)
        model = MemoryTrustModel.fit(
            features[:20],
            labels[:20],
            sample_weights=[2.0 if label else 1.0 for label in labels[:20]],
            epochs=300,
        )
        model.calibrate(features[20:], labels[20:], epochs=200)
        self.assertEqual(model.calibration_method, "platt_scaling")
        self.assertGreater(model.calibration_samples, 0)
        self.assertGreater(model.predict(features[0]), model.predict(features[1]))
        self.assertIn("brier", model.metrics)
        path = save_trust_model(self.root / "trust.json", model)
        loaded = load_trust_model(path)
        self.assertEqual(loaded.calibration_method, "platt_scaling")
        self.assertAlmostEqual(loaded.predict(features[0]), model.predict(features[0]), places=6)

    def test_active_learning_prefers_uncertain_conflicting_samples(self):
        db_path = self.root / "active-learning.sqlite3"
        match_id = collect_scripted_match(db_path)
        promote_match(db_path, match_id)
        with ExperienceStore(db_path) as store:
            decisions = store.list_decisions(limit=2)
        uncertain_decision, certain_decision = decisions
        uncertain = score_candidate(
            position_id=uncertain_decision["position_id"],
            decision_id=uncertain_decision["decision_id"],
            trust_probability=0.51,
            confidence=0.7,
            support_count=3,
            conflict=True,
            retrieval_type="similar",
            ply=50,
        )
        certain = score_candidate(
            position_id=certain_decision["position_id"],
            decision_id=certain_decision["decision_id"],
            trust_probability=0.98,
            confidence=0.95,
            support_count=8,
            conflict=False,
            retrieval_type="exact",
            ply=5,
        )
        duplicate = score_candidate(
            position_id=uncertain_decision["position_id"],
            decision_id=uncertain_decision["decision_id"],
            trust_probability=0.51, confidence=0.7, support_count=3,
            conflict=True, retrieval_type="similar", ply=50,
        )
        self.assertEqual(uncertain.candidate_id, duplicate.candidate_id)
        self.assertGreater(uncertain.priority, certain.priority)
        selected = select_candidates([certain, uncertain], quota=1)
        self.assertEqual(selected[0].candidate_id, uncertain.candidate_id)
        with ExperienceStore(db_path) as store:
            self.assertEqual(persist_candidates(store, selected), 1)
            updated = score_candidate(
                position_id=uncertain_decision["position_id"],
                decision_id=uncertain_decision["decision_id"],
                trust_probability=0.9,
                confidence=0.95,
                support_count=8,
                conflict=False,
                retrieval_type="exact",
                ply=5,
            )
            self.assertEqual(updated.candidate_id, uncertain.candidate_id)
            self.assertEqual(persist_candidates(store, [updated]), 1)
            rows = store.list_active_learning_candidates()
        self.assertEqual(rows[0]["position_id"], uncertain_decision["position_id"])
        self.assertEqual(rows[0]["priority"], updated.priority)

    def test_store_rejects_invalid_v4_records(self):
        with ExperienceStore(self.root / "validation.sqlite3") as store:
            with self.assertRaises(MemoryValidationError):
                store.save_active_learning_candidate(ActiveLearningCandidateRecord(
                    candidate_id="candidate",
                    position_id="position",
                    decision_id="decision",
                    priority=1.5,
                    created_at=PINNED_TIME,
                ))
            with self.assertRaises(MemoryValidationError):
                store.save_trust_model(TrustModelRecord(
                    trust_model_id="trust",
                    model_path="trust.json",
                    sha256="not-a-hash",
                    feature_version=1,
                    metrics_json="{}",
                    status="candidate",
                    created_at=PINNED_TIME,
                ))
            with self.assertRaises(MemoryValidationError):
                store.save_drift_snapshot(DriftSnapshotRecord(
                    snapshot_id="drift",
                    window_start=PINNED_TIME,
                    window_end=PINNED_TIME,
                    metrics_json="{}",
                    alerts_json="{}",
                    created_at=PINNED_TIME,
                ))
            with self.assertRaises(MemoryValidationError):
                store.save_active_learning_candidate(ActiveLearningCandidateRecord(
                    candidate_id="missing-reference",
                    position_id="position",
                    decision_id="decision",
                    priority=0.5,
                    created_at=PINNED_TIME,
                ))

    def test_league_elo_table_is_order_independent(self):
        self.assertLess(expected_score(1400, 1600), expected_score(1600, 1400))
        self.assertGreater(update_elo(1500, 1500, 1.0), 1500)
        rows = [
            {"opponent_id": "basic", "games_completed": 10, "ai_a_wins": 7,
             "ai_b_wins": 3, "opponent_elo": 1400},
            {"opponent_id": "strategic", "games_completed": 10, "ai_a_wins": 4,
             "ai_b_wins": 6, "opponent_elo": 1600},
        ]
        table = build_league_table(rows)
        reversed_table = build_league_table(list(reversed(rows)))
        self.assertEqual(
            [entry.__dict__ for entry in table],
            [entry.__dict__ for entry in reversed_table],
        )
        self.assertEqual(table[0].opponent_id, "basic")
        self.assertGreater(table[0].win_rate, table[1].win_rate)

    def test_league_registration_records_candidate_hash_and_updates_profiles(self):
        candidate = self.root / "candidate.pt"
        candidate.write_bytes(b"candidate-model")
        document = {
            "candidate": f"neural:{candidate}",
            "results": [{
                "opponent_id": "basic",
                "ai_b": "BasicAI",
                "opponent_elo": 1500.0,
                "games_completed": 10,
                "ai_a_wins": 6,
                "ai_b_wins": 4,
            }],
        }
        with ExperienceStore(self.root / "league.sqlite3") as store:
            first_id = register_league_run(store, document)
            document["results"][0]["opponent_elo"] = 1540.0
            second_id = register_league_run(store, document)
            runs = store.list_league_runs()
            profiles = store.list_opponent_profiles()
        expected_hash = hashlib.sha256(b"candidate-model").hexdigest()
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(row["candidate_sha256"] == expected_hash for row in runs))
        self.assertEqual(profiles[0]["elo"], 1540.0)

    def test_drift_snapshot_uses_ordered_window_and_baseline_alert(self):
        db_path = self.root / "drift.sqlite3"
        match_id = collect_scripted_match(db_path)
        promote_match(db_path, match_id)
        with ExperienceStore(db_path) as store:
            store.save_drift_snapshot(DriftSnapshotRecord(
                snapshot_id="previous",
                window_start="2025-12-31T00:00:00+00:00",
                window_end="2025-12-31T01:00:00+00:00",
                metrics_json=json.dumps({
                    "validation_rate": 1.0,
                    "mean_reward": 0.5,
                }),
                alerts_json="[]",
                created_at="2025-12-31T01:00:00+00:00",
            ))
            report = build_drift_snapshot(store)
            rows = store.list_drift_snapshots()
        self.assertEqual(report["snapshot_id"], rows[0]["snapshot_id"])
        self.assertLessEqual(rows[0]["window_start"], rows[0]["window_end"])
        self.assertIn("memory_reward_drop", report["alerts"])

    def test_drift_reports_low_validation_rate(self):
        db_path = self.root / "drift-low.sqlite3"
        collect_scripted_match(db_path)
        with ExperienceStore(db_path) as store:
            report = build_drift_snapshot(store)
        self.assertEqual(report["metrics"]["validation_rate"], 0.0)
        self.assertIn("validation_rate_low", report["alerts"])

    def test_orchestrator_rejects_no_new_data(self):
        db_path = self.root / "orchestrator.sqlite3"
        match_id = collect_scripted_match(db_path)
        promote_match(db_path, match_id)
        with ExperienceStore(db_path) as store:
            self._register_dataset(store)
        orchestrator = LearningOrchestrator(
            OrchestratorConfig(minimum_new_matches=1, minimum_training_samples=1),
            db_path=db_path,
        )
        status = orchestrator.status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["new_matches_since_dataset"], 0)
        self.assertFalse(status["auto_promote"])

    def test_orchestrator_accepts_new_data_and_rejects_drift_alerts(self):
        db_path = self.root / "orchestrator-new.sqlite3"
        first_match = collect_scripted_match(db_path)
        promote_match(db_path, first_match)
        with ExperienceStore(db_path) as store:
            self._register_dataset(store)
        second_match = collect_scripted_match(db_path)
        promote_match(db_path, second_match)
        orchestrator = LearningOrchestrator(
            OrchestratorConfig(minimum_new_matches=1, minimum_training_samples=1),
            db_path=db_path,
        )
        self.assertTrue(orchestrator.status()["ready"])
        with ExperienceStore(db_path) as store:
            store.save_drift_snapshot(DriftSnapshotRecord(
                snapshot_id="alert",
                window_start=PINNED_TIME,
                window_end=PINNED_TIME,
                metrics_json="{}",
                alerts_json='["validation_rate_drop"]',
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ))
        blocked = orchestrator.status()
        self.assertFalse(blocked["ready"])
        self.assertEqual(blocked["drift_alerts"], ["validation_rate_drop"])

    def test_orchestrator_learning_lock_rejects_overlap_and_releases(self):
        db_path = self.root / "lock.sqlite3"
        with ExperienceStore(db_path):
            pass
        orchestrator = LearningOrchestrator(
            OrchestratorConfig(minimum_new_matches=1, minimum_training_samples=1),
            db_path=db_path,
        )
        with orchestrator._cycle_lock():
            with self.assertRaises(RuntimeError):
                with orchestrator._cycle_lock():
                    pass
        self.assertFalse(Path(db_path).with_suffix(".learning.lock").exists())
        with orchestrator._cycle_lock():
            pass

    def test_deployment_apply_validate_and_rollback(self):
        models = self.root / "models"
        models.mkdir()
        model = self._write_checkpoint(models / "candidate.pt")
        policy = models / "memory_policy.json"
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

    def test_deployment_rejects_incompatible_checkpoint(self):
        models = self.root / "invalid-checkpoint"
        models.mkdir()
        model = self._write_checkpoint(models / "candidate.pt", model_version="6.0.0")
        policy = models / "memory_policy.json"
        policy.write_text('{"schema_version":1}', encoding="utf-8")
        with self.assertRaises(DeploymentError):
            build_deployment_manifest(
                models_dir=models, model_path=model, policy_path=policy
            )

    def test_deployment_rejects_hash_and_selected_model_mismatch(self):
        models = self.root / "hash-check"
        models.mkdir()
        model = self._write_checkpoint(models / "candidate.pt")
        policy = models / "memory_policy.json"
        policy.write_text('{"schema_version":1}', encoding="utf-8")
        manifest = build_deployment_manifest(
            models_dir=models, model_path=model, policy_path=policy
        )
        (models / "selected_model.json").write_text(json.dumps({
            "model_file": "other.pt",
            "sha256": manifest["model_sha256"],
        }), encoding="utf-8")
        with self.assertRaises(DeploymentError):
            validate_deployment_manifest_from_document(manifest, models)
        (models / "selected_model.json").write_text(json.dumps({
            "model_file": model.name,
            "sha256": "c" * 64,
        }), encoding="utf-8")
        with self.assertRaises(DeploymentError):
            validate_deployment_manifest_from_document(manifest, models)
        (models / "selected_model.json").unlink()
        policy.write_text('{"schema_version":2}', encoding="utf-8")
        with self.assertRaises(DeploymentError):
            validate_deployment_manifest_from_document(manifest, models)


if __name__ == "__main__":
    unittest.main()