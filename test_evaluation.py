# -*- coding: utf-8 -*-
"""Dijkstra 连通距离与候选点评分测试。"""
import unittest

from ai.evaluation import (INF, candidate_distance_score, clone_board,
                           connection_cost, heuristic_score_map,
                           shortest_path_cells)
from hex_board import BLUE, EMPTY, HexBoard, RED


class EvaluationTest(unittest.TestCase):
    def test_empty_board_distance(self):
        board = HexBoard()
        self.assertEqual(connection_cost(board, RED), 11)
        self.assertEqual(connection_cost(board, BLUE), 11)
        path = shortest_path_cells(board, RED)
        self.assertEqual(len(path), 11)
        self.assertTrue(all(board.grid[r][q] == EMPTY for q, r in path))

    def test_connected_path_has_zero_cost(self):
        board = HexBoard()
        for row in range(11):
            board.place(0, row, RED)
        self.assertEqual(connection_cost(board, RED), 0)
        self.assertEqual(len(shortest_path_cells(board, RED)), 11)

    def test_candidate_score_and_no_board_mutation(self):
        board = HexBoard()
        before = [row[:] for row in board.grid]
        score = candidate_distance_score(board, (5, 5), RED)
        self.assertGreater(score, 0)
        self.assertEqual(board.grid, before)

    def test_immediate_win_has_maximum_score(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        self.assertEqual(candidate_distance_score(board, (0, 10), RED), 100000.0)

    def test_clone_is_independent(self):
        board = HexBoard()
        copied = clone_board(board)
        copied.place(3, 4, BLUE)
        self.assertTrue(board.is_empty(3, 4))
        self.assertFalse(copied.is_empty(3, 4))

    def test_heuristic_score_map_and_invalid_color(self):
        board = HexBoard()
        scores = heuristic_score_map(board, RED, [(5, 5), (6, 5)])
        self.assertEqual(set(scores), {(5, 5), (6, 5)})
        with self.assertRaises(ValueError):
            connection_cost(board, 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
