# -*- coding: utf-8 -*-
"""HybridAI 战术顺序、合法性和只读行为测试。"""
import unittest

from ai.hybrid_ai import HybridAI
from hex_board import BLUE, HexBoard, RED


class HybridAITest(unittest.TestCase):
    @staticmethod
    def _double_threat_board():
        board = HexBoard()
        for cell in ((2, 10), (8, 0), (7, 1), (6, 3), (6, 5),
                     (6, 6), (5, 7), (3, 8), (4, 8), (2, 9)):
            board.place(cell[0], cell[1], RED)
        return board

    def test_center_opening(self):
        move, reason = HybridAI(time_limit=0.08, seed=1).choose_move(HexBoard(), RED)
        self.assertEqual(move, (5, 5))
        self.assertEqual(reason, "中心开局")

    def test_immediate_win(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        move, reason = HybridAI(time_limit=0.08, seed=1).choose_move(board, RED)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "直接获胜")

    def test_immediate_defence(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        move, reason = HybridAI(time_limit=0.08, seed=1).choose_move(board, BLUE)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "紧急防守")

    def test_multiple_threats_use_deterministic_best_defence(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
            board.place(10, row, RED)
        first_ai = HybridAI(time_limit=0.08, seed=3)
        second_ai = HybridAI(time_limit=0.08, seed=3)
        first = first_ai.choose_move(board, BLUE)
        second = second_ai.choose_move(board, BLUE)
        self.assertEqual(first, second)
        self.assertEqual(first[1], "多威胁最佳防守")

    def test_creative_double_threat_has_priority(self):
        board = self._double_threat_board()
        move, reason = HybridAI(time_limit=0.08, seed=1).choose_move(board, RED)
        self.assertEqual(move, (6, 4))
        self.assertEqual(reason, "创造双威胁")

    def test_avoids_opponent_double_threat_key_cell(self):
        board = self._double_threat_board()
        move, reason = HybridAI(time_limit=0.08, seed=1).choose_move(board, BLUE)
        self.assertEqual(move, (6, 4))
        self.assertIn("双威胁规避", reason)

    def test_choose_move_does_not_mutate_formal_board(self):
        board = HexBoard()
        board.place(5, 5, RED)
        before_grid = [row[:] for row in board.grid]
        before_history = list(board.history)
        move, reason = HybridAI(time_limit=0.08, seed=2).choose_move(
            board, 2
        )
        self.assertTrue(board.is_empty(*move))
        self.assertIsInstance(reason, str)
        self.assertEqual(board.grid, before_grid)
        self.assertEqual(board.history, before_history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
