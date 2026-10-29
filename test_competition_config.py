# -*- coding: utf-8 -*-
"""Competition profile and reproducibility contract tests."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.ai_factory import create_competition_ai
from ai.competition_config import (
    CompetitionConfigError,
    load_competition_settings,
    require_clean_worktree,
)
from ai.hybrid_ai import HybridAI
from ai.memory.memory_ai import MemoryAssistedAI


class CompetitionConfigTest(unittest.TestCase):
    def test_default_profile_loads_and_matches_selected_model(self):
        settings = load_competition_settings(
            Path(__file__).with_name("competition_profile.json")
        )
        self.assertEqual(settings.ai_engine, "hybrid")
        self.assertEqual(settings.search_time_limit_ms, 200.0)
        self.assertEqual(settings.hard_move_limit_ms, 500.0)
        self.assertEqual(settings.random_seed, 20261025)
        self.assertEqual(settings.memory_mode, "shadow")
        self.assertEqual(len(settings.profile_sha256), 64)
        self.assertTrue(settings.neural_model_path.is_file())

    def test_environment_engine_cannot_silently_override_profile(self):
        with patch.dict(os.environ, {"HEX_AI_ENGINE": "basic"}):
            settings = load_competition_settings(
                Path(__file__).with_name("competition_profile.json")
            )
            ai, warning = create_competition_ai(settings=settings)
        self.assertIsInstance(ai, MemoryAssistedAI)
        self.assertIsInstance(ai.base_ai, HybridAI)
        self.assertIn("ignored", warning)
        self.assertEqual(ai.competition_profile["ai_engine"], "hybrid")
        ai.close()

    def test_dirty_worktree_is_rejected_for_formal_runs(self):
        with patch("ai.competition_config._git_output", return_value=" M file.py"):
            with self.assertRaisesRegex(CompetitionConfigError, "dirty"):
                require_clean_worktree()

    def test_invalid_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "competition_profile.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "profile_name": "invalid",
                "profile_version": "1.0.0",
                "ai_engine": "neural",
                "search_time_limit_ms": 200,
                "hard_move_limit_ms": 500,
                "random_seed": 1,
                "memory_mode": "off",
                "tournament_games": 200,
                "tournament_final_games": 1000,
            }), encoding="utf-8")
            with self.assertRaisesRegex(CompetitionConfigError, "neural engine"):
                load_competition_settings(path)


if __name__ == "__main__":
    unittest.main()