# -*- coding: utf-8 -*-
"""AI 契约、模型回退和只读行为测试。"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ai.ai_factory import create_competition_ai
from ai.competition_config import load_competition_settings
from ai.hybrid_ai import HybridAI
from ai.neural_ai import (NeuralMCTSAI, choose_tactical_move,
                         create_ai_with_fallback, load_selected_model,
                         resolve_model_path)
from hex_ai import BasicAI
from hex_board import BLUE, HexBoard, RED


class AIContractTest(unittest.TestCase):
    def test_basic_ai_returns_unified_format_without_mutating_board(self):
        board = HexBoard()
        before_grid = [row[:] for row in board.grid]
        before_history = list(board.history)
        move, reason = BasicAI().choose_move(board, RED)
        self.assertEqual(len(move), 2)
        self.assertIsInstance(reason, str)
        self.assertTrue(board.is_empty(*move))
        self.assertEqual(board.grid, before_grid)
        self.assertEqual(board.history, before_history)

    def test_selected_model_manifest_is_valid(self):
        model_path, metadata = load_selected_model()
        self.assertTrue(model_path.is_file())
        self.assertEqual(metadata["model_file"], "hex_model_v5.pt")
        self.assertEqual(metadata["model_version"], "5.0.0")
        self.assertEqual(metadata["selected_games"], 2000)
        resolved_path, resolved_metadata = resolve_model_path()
        self.assertEqual(resolved_path, model_path)
        self.assertEqual(resolved_metadata["sha256"], metadata["sha256"])

    def test_selected_model_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "fake.pt").write_bytes(b"fake model")
            manifest = root / "selected_model.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "model_file": "fake.pt",
                "model_version": "5.0.0",
                "encoding_version": "3x11x11-v1",
                "sha256": "0" * 64,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                load_selected_model(manifest)

    def test_missing_model_falls_back_to_basic_ai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.pt"
            ai, error = create_ai_with_fallback(model_path=missing)
        self.assertIsInstance(ai, BasicAI)
        self.assertIsNotNone(error)
        self.assertIn("模型不可用", error)

    def test_basic_ai_exposes_explanation(self):
        board = HexBoard()
        ai = BasicAI()
        move, reason = ai.choose_move(board, RED)
        explanation = ai.last_explanation.as_dict()
        self.assertEqual(explanation["selected_move"], move)
        self.assertEqual(explanation["tactic"], reason)
        self.assertTrue(explanation["candidates"])
        self.assertEqual(explanation["search_type"], "rule")

    def test_competition_ai_defaults_to_hybrid(self):
        settings = replace(load_competition_settings(), memory_mode="off")
        ai, error = create_competition_ai("hybrid", settings=settings)
        self.assertIsInstance(ai, HybridAI)
        self.assertIsNone(error)

    def test_tactical_move_prefers_direct_win(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        move, reason = choose_tactical_move(board, RED)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "直接获胜")

    def test_tactical_move_blocks_multiple_immediate_threats(self):
        board = HexBoard()
        for col in range(1, 11):
            board.place(col, 5, BLUE)
        move, reason = choose_tactical_move(board, RED)
        self.assertIn(move, {(0, 5), (0, 6)})
        self.assertEqual(reason, "多威胁最佳防守")


    def test_neural_ai_metadata_and_missing_model_error(self):
        self.assertEqual(NeuralMCTSAI.version, "5.0.0")
        self.assertEqual(NeuralMCTSAI.time_limit_ms, 200)
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                NeuralMCTSAI(model_path=missing)


    def test_hybrid_llm_factory_falls_back_when_disabled(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"LLM_ENABLED": "0"}):
            settings = replace(load_competition_settings(), memory_mode="off")
            ai, error = create_competition_ai("hybrid_llm", settings=settings)
        self.assertIsInstance(ai, HybridAI)
        self.assertIn("LLM_ENABLED=0", error)

    def test_hybrid_llm_factory_creates_experimental_wrapper(self):
        import os
        from unittest.mock import patch
        from ai.ai_coordinator import LLMAssistedAI

        env = {
            "LLM_ENABLED": "1",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.test/v1",
        }
        with patch.dict(os.environ, env):
            settings = replace(load_competition_settings(), memory_mode="off")
            ai, error = create_competition_ai("hybrid_llm", settings=settings)
        self.assertIsInstance(ai, LLMAssistedAI)
        self.assertIsNone(error)

if __name__ == "__main__":
    unittest.main(verbosity=2)