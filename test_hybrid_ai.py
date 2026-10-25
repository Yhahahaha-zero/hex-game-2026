# -*- coding: utf-8 -*-
"""HybridAI 合法性和只读行为测试。"""
import unittest

from ai.hybrid_ai import HybridAI
from hex_board import HexBoard, RED


class HybridAITest(unittest.TestCase):
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
