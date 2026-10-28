# -*- coding: utf-8 -*-
"""Competition AI selection pipeline tests."""
import json
import tempfile
import unittest
from pathlib import Path

from ai.competition_config import load_competition_settings
from tools import select_competition_ai


def screening_document():
    return {
        "standings": [
            {"opponent_id": "hybrid", "name": "HybridAI", "win_rate": 75.0},
            {"opponent_id": "strategic", "name": "StrategicAI", "win_rate": 68.0},
            {"opponent_id": "basic", "name": "BasicAI", "win_rate": 10.0},
        ]
    }


def final_document():
    return {
        "config": {"opponents": ["hybrid", "strategic"]},
        "standings": [
            {"opponent_id": "hybrid", "name": "HybridAI", "win_rate": 60.0},
            {"opponent_id": "strategic", "name": "StrategicAI", "win_rate": 40.0},
        ],
        "recommendation": {
            "eligible": True,
            "recommended_ai": "hybrid",
            "tied_ai": ["hybrid"],
        },
    }


class CompetitionSelectionTest(unittest.TestCase):
    def test_select_finalists_and_reject_tied_cutoff(self):
        self.assertEqual(
            select_competition_ai.select_finalists(screening_document()),
            ["hybrid", "strategic"],
        )
        tied = screening_document()
        tied["standings"][2]["win_rate"] = 68.0
        with self.assertRaisesRegex(ValueError, "tied"):
            select_competition_ai.select_finalists(tied)

    def test_pipeline_runs_screening_and_final_with_injected_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def runner(opponents, games, **kwargs):
                calls.append((list(opponents), games, kwargs))
                index = len(calls)
                document = screening_document() if index == 1 else final_document()
                path = root / f"result_{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                return document, {"json": path}

            def reporter(**kwargs):
                return {
                    "markdown": root / "report.md",
                    "csv": root / "report.csv",
                    "validation": root / "validation.json",
                }

            def validator(document, settings, *, require_recommendation=True):
                return {"passed": True, "errors": []}

            summary = select_competition_ai.run_selection(
                screening_games=20,
                final_games=100,
                screening_dir=root / "screening",
                final_dir=root / "final",
                report_dir=root / "report",
                name="test-selection",
                quiet=True,
                clean_checker=lambda: "c" * 40,
                tournament_runner=runner,
                report_generator=reporter,
                validator=validator,
            )
            self.assertEqual(summary["finalists"], ["hybrid", "strategic"])
            self.assertEqual(summary["winner"], "hybrid")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], 20)
            self.assertEqual(calls[1][0], ["hybrid", "strategic"])
            self.assertEqual(calls[1][1], 100)
            self.assertTrue(
                (root / "report" / "test-selection" / "selection_summary.json").is_file()
            )

    def test_profile_selection_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = root / "competition_profile.json"
            source = select_competition_ai.PROJECT_ROOT / "competition_profile.json"
            profile.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            settings = select_competition_ai.update_profile_selection(
                profile,
                winner="strategic",
                benchmark_file="evaluation/tournament_results/final.json",
                freeze_commit="d" * 40,
                selected_at="2026-10-28T00:00:00+08:00",
            )
            self.assertEqual(settings.ai_engine, "strategic")
            self.assertEqual(settings.freeze_commit, "d" * 40)
            document = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(document["benchmark_file"],
                             "evaluation/tournament_results/final.json")


if __name__ == "__main__":
    unittest.main()