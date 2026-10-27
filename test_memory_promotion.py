# -*- coding: utf-8 -*-
"""V2 candidate promotion, backup and rollback tests."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai.memory.model_registry import (
    PromotionGate,
    evaluate_promotion_gate,
    file_sha256,
    promote_candidate,
    rollback_selected_model,
)


class MemoryPromotionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.models = Path(self.temp_dir.name) / "models"
        self.models.mkdir()
        self.current = self.models / "current.pt"
        self.current.write_bytes(b"current-model")
        self.current_config = self.models / "current_config.json"
        self.current_config.write_text(json.dumps({
            "model_version": "5.0.0",
            "encoding_version": "3x11x11-v1",
        }), encoding="utf-8")
        self.selected = self.models / "selected_model.json"
        self.selected.write_text(json.dumps({
            "schema_version": 1,
            "model_file": "current.pt",
            "model_version": "5.0.0",
            "encoding_version": "3x11x11-v1",
            "sha256": file_sha256(self.current),
            "checkpoint_kind": "official",
        }), encoding="utf-8")
        self.candidate = self.models / "candidate.pt"
        torch.save({
            "model_state_dict": {"test": torch.zeros(1)},
            "model_name": "HexNetV5",
            "model_version": "5.0.0",
            "encoding_version": "3x11x11-v1",
        }, self.candidate)
        self.candidate_config = self.models / "candidate_config.json"
        self.candidate_config.write_text(json.dumps({
            "model_version": "5.0.0",
            "encoding_version": "3x11x11-v1",
            "checkpoint_kind": "memory_supervised_candidate",
            "training_samples": 100,
        }), encoding="utf-8")
        self.benchmark = Path(self.temp_dir.name) / "benchmark.json"
        self.benchmark.write_text(json.dumps({
            "summary": {
                "games_completed": 1000,
                "errors": 0,
                "ai_a_win_rate": 56.0,
                "ai_a_red_win_rate": 53.0,
                "ai_a_blue_win_rate": 59.0,
                "ai_a_max_move_ms": 200.0,
            }
        }), encoding="utf-8")
        self.gate = PromotionGate(
            minimum_games=1000,
            minimum_win_rate=50.0,
            minimum_color_win_rate=40.0,
            maximum_errors=0,
            maximum_move_ms=500.0,
        )

    def test_gate_rejects_low_win_rate(self):
        report = evaluate_promotion_gate({
            "summary": {
                "games_completed": 1000,
                "errors": 0,
                "ai_a_win_rate": 49.0,
                "ai_a_red_win_rate": 50.0,
                "ai_a_blue_win_rate": 48.0,
                "ai_a_max_move_ms": 100.0,
            }
        }, self.gate)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["win_rate"])

    def test_dry_run_does_not_modify_manifest_or_create_backup(self):
        before = self.selected.read_bytes()
        report = promote_candidate(
            candidate_path=self.candidate,
            benchmark_path=self.benchmark,
            models_dir=self.models,
            gate=self.gate,
            apply=False,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(self.selected.read_bytes(), before)
        self.assertFalse((self.models / "backups").exists())

    def test_apply_and_rollback_restore_previous_model(self):
        old_hash = file_sha256(self.current)
        report = promote_candidate(
            candidate_path=self.candidate,
            benchmark_path=self.benchmark,
            models_dir=self.models,
            gate=self.gate,
            apply=True,
        )
        self.assertTrue(report["passed"])
        selected = json.loads(self.selected.read_text(encoding="utf-8"))
        self.assertEqual(selected["model_file"], self.candidate.name)
        self.assertEqual(selected["sha256"], file_sha256(self.candidate))
        self.assertTrue((self.models / "selected_model_history.jsonl").is_file())
        rollback = rollback_selected_model(models_dir=self.models)
        self.assertEqual(rollback["model_file"], self.current.name)
        self.assertEqual(rollback["sha256"], old_hash)
        restored = json.loads(self.selected.read_text(encoding="utf-8"))
        self.assertEqual(restored["model_file"], self.current.name)


if __name__ == "__main__":
    unittest.main()