# -*- coding: utf-8 -*-
"""桥型、双威胁、多威胁防守和关键路径测试。"""
import unittest

from ai.tactics import (block_multiple_threats, bridge_move_bonus,
                        creates_double_threat, find_bridges,
                        find_double_threat_moves, hex_distance,
                        key_path_cells, opponent_double_threat_replies,
                        opponent_threats, shortest_path_key_cells,
                        shortest_path_key_weights)
from hex_ai import winning_moves
from hex_board import BLUE, HexBoard, RED


class TacticsTest(unittest.TestCase):
    @staticmethod
    def _true_double_threat_board():
        board = HexBoard()
        for cell in ((2, 10), (8, 0), (7, 1), (6, 3), (6, 5),
                     (6, 6), (5, 7), (3, 8), (4, 8), (2, 9)):
            board.place(cell[0], cell[1], RED)
        return board

    @staticmethod
    def _red_double_column_board():
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
            board.place(10, row, RED)
        return board

    def test_bridge_detection_and_bonus(self):
        board = HexBoard()
        board.place(0, 0, RED)
        board.place(1, 1, RED)
        bridges = find_bridges(board, RED)
        self.assertEqual(len(bridges), 1)
        cells = set(bridges[0][2])
        self.assertEqual(cells, {(1, 0), (0, 1)})
        bonus = bridge_move_bonus(board, RED)
        self.assertEqual(bonus[(1, 0)], 1)
        self.assertEqual(bonus[(0, 1)], 1)

    def test_hex_distance(self):
        self.assertEqual(hex_distance((0, 0), (1, 1)), 2)
        self.assertEqual(hex_distance((0, 0), (1, 0)), 1)

    def test_opponent_single_threat(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        self.assertEqual(opponent_threats(board, BLUE), ((0, 10),))

    def test_double_threat_detection(self):
        board = self._true_double_threat_board()
        self.assertEqual(winning_moves(board, RED), [])
        self.assertTrue(creates_double_threat(board, 6, 4, RED))
        self.assertIn((6, 4), find_double_threat_moves(board, RED))
        self.assertFalse(creates_double_threat(board, 0, 0, RED))

    def test_multiple_threat_best_defense_is_deterministic(self):
        board = self._red_double_column_board()
        threats = opponent_threats(board, BLUE)
        self.assertGreaterEqual(len(threats), 2)
        first = block_multiple_threats(board, BLUE)
        second = block_multiple_threats(board, BLUE)
        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_opponent_double_threat_replies_are_detected(self):
        board = self._true_double_threat_board()
        first = opponent_double_threat_replies(board, BLUE, limit=30)
        second = opponent_double_threat_replies(board, BLUE, limit=30)
        self.assertEqual(first, ((6, 4),))
        self.assertEqual(first, second)
        self.assertTrue(all(board.is_empty(*cell) for cell in first))

    def test_key_path_cells_and_weights(self):
        board = HexBoard()
        cells = shortest_path_key_cells(board, RED)
        weights = shortest_path_key_weights(board, RED)
        self.assertEqual(cells, key_path_cells(board, RED))
        self.assertTrue(cells)
        self.assertEqual(weights[(0, 0)], 2)
        self.assertTrue(all(board.is_empty(*cell) for cell in cells))


if __name__ == "__main__":
    unittest.main(verbosity=2)
