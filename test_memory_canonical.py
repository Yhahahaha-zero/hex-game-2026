# -*- coding: utf-8 -*-
"""P0-A symmetry and canonical-position tests."""
import unittest

from ai.memory.canonical import (
    BoardTransform,
    MemoryCanonicalError,
    canonicalize_position,
    inverse_transform_move,
    transform_board,
    transform_color,
    transform_move,
    transform_position,
)
from hex_board import BLUE, EMPTY, RED, HexBoard


class MemoryCanonicalTest(unittest.TestCase):
    def test_all_enabled_transform_moves_round_trip(self):
        for transform in BoardTransform:
            for q in range(11):
                for r in range(11):
                    with self.subTest(transform=transform, move=(q, r)):
                        transformed = transform_move((q, r), transform)
                        restored = inverse_transform_move(transformed, transform)
                        self.assertEqual(restored, (q, r))

    def test_transforms_preserve_colors(self):
        for transform in BoardTransform:
            self.assertEqual(transform_color(RED, transform), RED)
            self.assertEqual(transform_color(BLUE, transform), BLUE)
            self.assertEqual(transform_color(EMPTY, transform), EMPTY)

    def test_mirror_board_positions_and_history(self):
        board = HexBoard()
        board.place(2, 3, RED)
        board.place(7, 8, BLUE)
        mirrored = transform_board(board, BoardTransform.MIRROR_Q)
        self.assertEqual(mirrored.grid[3][8], RED)
        self.assertEqual(mirrored.grid[8][3], BLUE)
        self.assertIn((8, 3, RED), mirrored.history)
        self.assertIn((3, 8, BLUE), mirrored.history)

    def test_source_board_is_not_mutated(self):
        board = HexBoard()
        board.place(2, 3, RED)
        grid_before = [row[:] for row in board.grid]
        history_before = list(board.history)
        transform_board(board, BoardTransform.ROTATE_180)
        self.assertEqual(board.grid, grid_before)
        self.assertEqual(board.history, history_before)

    def test_canonical_key_merges_supported_symmetries(self):
        board = HexBoard()
        board.place(2, 3, RED)
        board.place(7, 8, BLUE)
        board.place(5, 5, RED)
        variants = [
            transform_board(board, BoardTransform.IDENTITY),
            transform_board(board, BoardTransform.MIRROR_Q),
            transform_board(board, BoardTransform.MIRROR_R),
            transform_board(board, BoardTransform.ROTATE_180),
        ]
        keys = {
            canonicalize_position(item, BLUE).key for item in variants
        }
        self.assertEqual(len(keys), 1)

    def test_current_color_changes_canonical_key(self):
        board = HexBoard()
        board.place(5, 5, RED)
        red_key = canonicalize_position(board, RED).key
        blue_key = canonicalize_position(board, BLUE).key
        self.assertNotEqual(red_key, blue_key)

    def test_canonical_transform_can_map_recommendation_back(self):
        board = HexBoard()
        board.place(2, 3, RED)
        canonical = canonicalize_position(board, BLUE)
        recommendation = (7, 4)
        mapped = transform_move(recommendation, canonical.transform)
        restored = inverse_transform_move(mapped, canonical.transform)
        self.assertEqual(restored, recommendation)

    def test_transform_position_does_not_alias_board(self):
        board = HexBoard()
        board.place(5, 5, RED)
        transformed = transform_position(board, BLUE, BoardTransform.MIRROR_R)
        self.assertIsNot(transformed.board, board)
        self.assertEqual(transformed.current_color, BLUE)

    def test_invalid_transform_inputs_are_rejected(self):
        board = HexBoard()
        with self.assertRaises(MemoryCanonicalError):
            transform_move((0, 0), "unknown")
        with self.assertRaises(MemoryCanonicalError):
            transform_move((-1, 0), BoardTransform.IDENTITY)
        with self.assertRaises(MemoryCanonicalError):
            canonicalize_position(board, 0)
        with self.assertRaises(MemoryCanonicalError):
            canonicalize_position(board, RED, rules_version=0)


if __name__ == "__main__":
    unittest.main()