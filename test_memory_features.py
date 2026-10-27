# -*- coding: utf-8 -*-
"""V1 explainable position feature tests."""
import json
import unittest

from ai.memory.features import (
    FEATURE_VERSION,
    extract_features,
    feature_similarity,
    features_json,
    parse_features,
)
from hex_board import BLUE, RED, HexBoard


class MemoryFeaturesTest(unittest.TestCase):
    def test_empty_board_features_are_symmetric(self):
        features = extract_features(HexBoard(), RED, 1)
        self.assertEqual(features.feature_version, FEATURE_VERSION)
        self.assertEqual(features.red_stones, 0)
        self.assertEqual(features.blue_stones, 0)
        self.assertEqual(features.red_distance, features.blue_distance)
        self.assertEqual(features.red_winning_moves, 0)
        self.assertEqual(features.blue_winning_moves, 0)
        self.assertEqual(features.red_cells, ())
        self.assertEqual(features.blue_cells, ())

    def test_feature_json_round_trip(self):
        board = HexBoard()
        board.place(5, 5, RED)
        board.place(0, 0, BLUE)
        raw = features_json(board, BLUE, 3)
        parsed = parse_features(raw)
        self.assertEqual(parsed["feature_version"], FEATURE_VERSION)
        self.assertEqual(parsed["current_color"], BLUE)
        self.assertEqual(parsed["red_stones"], 1)
        self.assertEqual(parsed["blue_stones"], 1)
        self.assertEqual(parsed["red_cells"], [[5, 5]])
        self.assertEqual(parsed["blue_cells"], [[0, 0]])

    def test_similar_positions_score_higher_than_different_positions(self):
        base = HexBoard()
        base.place(5, 5, RED)
        base.place(4, 5, BLUE)
        near = HexBoard()
        near.place(5, 5, RED)
        near.place(4, 6, BLUE)
        far = HexBoard()
        far.place(0, 0, RED)
        far.place(10, 10, BLUE)
        base_features = extract_features(base, RED, 3).as_dict()
        near_features = extract_features(near, RED, 3).as_dict()
        far_features = extract_features(far, RED, 3).as_dict()
        self.assertGreater(
            feature_similarity(base_features, near_features),
            feature_similarity(base_features, far_features),
        )

    def test_invalid_feature_version_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_features(json.dumps({"feature_version": 999}))


if __name__ == "__main__":
    unittest.main()