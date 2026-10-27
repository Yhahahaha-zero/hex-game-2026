# -*- coding: utf-8 -*-
"""V1 exact, canonical, similar retrieval and leakage tests."""
import tempfile
import unittest
from pathlib import Path

from ai.memory.retrieval import MemoryRetriever
from ai.memory.store import ExperienceStore
from hex_board import BLUE, RED, HexBoard
from memory_test_support import (
    collect_scripted_match,
    populate_all_features,
    promote_match,
)


class MemoryRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "retrieval.sqlite3"
        self.match_ids = []
        for _ in range(3):
            match_id = collect_scripted_match(self.db_path)
            promote_match(self.db_path, match_id)
            self.match_ids.append(match_id)
        populate_all_features(self.db_path)

    def _retriever(self, store, **kwargs):
        defaults = {
            "top_k": 8,
            "minimum_similarity": 0.70,
            "minimum_support": 3,
            "ply_window": 2,
        }
        defaults.update(kwargs)
        return MemoryRetriever(store, **defaults)

    def test_exact_position_aggregates_independent_matches(self):
        with ExperienceStore(self.db_path) as store:
            advice = self._retriever(store).query(
                HexBoard(), RED, ply=1
            )
        self.assertTrue(advice.available)
        self.assertEqual(advice.recommended_move, (0, 0))
        self.assertEqual(advice.support_count, 3)
        self.assertTrue(advice.hits)
        self.assertEqual(advice.hits[0].similarity_type, "exact")
        self.assertEqual(advice.win_rate, 1.0)

    def test_current_match_is_excluded(self):
        with ExperienceStore(self.db_path) as store:
            advice = self._retriever(store).query(
                HexBoard(), RED, ply=1, match_id=self.match_ids[0]
            )
        self.assertEqual(advice.support_count, 2)
        self.assertFalse(advice.available)
        self.assertTrue(all(hit.match_id != self.match_ids[0] for hit in advice.hits))

    def test_canonical_symmetry_maps_move_back_to_current_board(self):
        board = HexBoard()
        board.place(10, 0, RED)
        with ExperienceStore(self.db_path) as store:
            advice = self._retriever(store, minimum_support=1).query(
                board, BLUE, ply=2
            )
        canonical_hits = [
            hit for hit in advice.hits if hit.similarity_type == "canonical"
        ]
        self.assertTrue(canonical_hits)
        self.assertIn((0, 0), {hit.mapped_move for hit in canonical_hits})
        self.assertEqual(advice.recommended_move, (0, 0))

    def test_retrieval_type_ablation_filters_exact_hits(self):
        with ExperienceStore(self.db_path) as store:
            exact = self._retriever(store).query(HexBoard(), RED, ply=1)
            similar_only = self._retriever(
                store,
                allowed_types=("similar",),
                minimum_similarity=0.99,
            ).query(HexBoard(), RED, ply=1)
        self.assertTrue(exact.available)
        self.assertTrue(any(hit.similarity_type == "exact" for hit in exact.hits))
        self.assertEqual(similar_only.hits, ())
        self.assertFalse(similar_only.available)

    def test_similar_positions_are_returned_deterministically(self):
        board = HexBoard()
        board.place(1, 1, RED)
        with ExperienceStore(self.db_path) as store:
            first = self._retriever(
                store, minimum_similarity=0.30, minimum_support=1
            ).query(board, BLUE, ply=2)
            second = self._retriever(
                store, minimum_similarity=0.30, minimum_support=1
            ).query(board, BLUE, ply=2)
        self.assertTrue(any(
            hit.similarity_type == "similar" for hit in first.hits
        ))
        self.assertEqual(
            [hit.decision_id for hit in first.hits],
            [hit.decision_id for hit in second.hits],
        )


if __name__ == "__main__":
    unittest.main()