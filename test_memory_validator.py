# -*- coding: utf-8 -*-
"""V1 replay-validation and quality-promotion tests."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai.memory.store import ExperienceStore
from ai.memory.validator import validate_all, validate_match
from memory_test_support import collect_scripted_match


class MemoryValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "validator.sqlite3"

    def test_dry_run_validates_without_promoting(self):
        match_id = collect_scripted_match(self.db_path)
        with ExperienceStore(self.db_path) as store:
            before = store.list_decisions(limit=1000)
            report = validate_match(store, match_id, promote=False)
            after = store.list_decisions(limit=1000)
        self.assertTrue(report.valid)
        self.assertEqual(report.games_checked, 2)
        self.assertEqual(report.decisions_checked, 42)
        self.assertEqual(report.ai_decisions_eligible, 42)
        self.assertTrue(all(row["training_eligible"] == 0 for row in before))
        self.assertTrue(all(row["training_eligible"] == 0 for row in after))

    def test_promote_marks_only_valid_ai_decisions_eligible(self):
        match_id = collect_scripted_match(self.db_path)
        with ExperienceStore(self.db_path) as store:
            report = validate_match(store, match_id, promote=True)
            decisions = store.list_decisions(limit=1000)
        self.assertTrue(report.valid)
        self.assertEqual(len(decisions), 42)
        self.assertTrue(all(row["decision_quality"] == "validated" for row in decisions))
        self.assertTrue(all(row["training_eligible"] == 1 for row in decisions))
        self.assertTrue(all(row["integrity_status"] == "valid" for row in decisions))
        self.assertTrue(all(row["result_status"] == "known" for row in decisions))

    def test_tampered_board_is_rejected_and_demoted(self):
        match_id = collect_scripted_match(self.db_path)
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                "UPDATE positions SET board_json = ? WHERE rowid = "
                "(SELECT MIN(rowid) FROM positions)",
                ('{"rows":11,"cols":11,"grid":[[1,0]]}',),
            )
            connection.commit()
        finally:
            connection.close()
        with ExperienceStore(self.db_path) as store:
            report = validate_match(store, match_id, promote=True)
            decisions = store.list_decisions(limit=1000)
            all_report = validate_all(store, promote=False)
        self.assertFalse(report.valid)
        self.assertGreater(report.rejected_decisions, 0)
        self.assertTrue(all(row["decision_quality"] == "rejected" for row in decisions))
        self.assertTrue(all(row["training_eligible"] == 0 for row in decisions))
        self.assertFalse(all_report[0].valid)


if __name__ == "__main__":
    unittest.main()