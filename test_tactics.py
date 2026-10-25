# -*- coding: utf-8 -*-
"""桥型、双威胁和关键路径测试。"""
import unittest

from ai.tactics import (bridge_move_bonus, creates_double_threat,
                        find_bridges, hex_distance, key_path_cells)
from hex_board import HexBoard, RED


class TacticsTest(unittest.TestCase):
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

    def test_double_threat_detection(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
            board.place(10, row, RED)
        self.assertTrue(creates_double_threat(board, (5, 5), RED))

    def test_key_path_cells_are_empty(self):
        board = HexBoard()
        cells = key_path_cells(board, RED)
        self.assertTrue(cells)
        self.assertTrue(all(board.is_empty(*cell) for cell in cells))


if __name__ == "__main__":
    unittest.main(verbosity=2)
