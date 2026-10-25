# -*- coding: utf-8 -*-
"""连通距离、0-1 BFS、候选点评分和性能测试。"""
import time
import unittest

from ai.evaluation import (BLOCK_WEIGHT, INF, OWN_WEIGHT,
                           candidate_distance_score, clone_board,
                           connection_cost, heuristic_score_map, score_move,
                           score_moves, shortest_path_cells)
from hex_board import BLUE, EMPTY, HexBoard, RED


class GridBoard:
    """供 5×5、7×7 手工局面测试的轻量棋盘。"""

    def __init__(self, grid):
        self.grid = grid


class EvaluationTest(unittest.TestCase):
    @staticmethod
    def _empty_grid(size):
        return [[EMPTY] * size for _ in range(size)]

    def test_empty_board_distance(self):
        board = HexBoard()
        self.assertEqual(INF, float("inf"))
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

    def test_small_board_handcrafted_scenarios(self):
        for size in (5, 7):
            with self.subTest(size=size):
                empty = GridBoard(self._empty_grid(size))
                self.assertEqual(connection_cost(empty, RED), size)
                self.assertEqual(connection_cost(empty, BLUE), size)
                self.assertEqual(len(shortest_path_cells(empty, RED)), size)
                self.assertEqual(len(shortest_path_cells(empty, BLUE)), size)

                red = GridBoard(self._empty_grid(size))
                for row in range(size):
                    red.grid[row][0] = RED
                self.assertEqual(connection_cost(red, RED), 0)

                red.grid[2][0] = EMPTY
                self.assertEqual(connection_cost(red, RED), 1)

                red.grid[2][0] = BLUE
                self.assertEqual(connection_cost(red, RED), 2)

                blue = GridBoard(self._empty_grid(size))
                middle = size // 2
                for col in range(size):
                    blue.grid[middle][col] = BLUE
                self.assertEqual(connection_cost(blue, BLUE), 0)

                blue.grid[middle][2] = EMPTY
                self.assertEqual(connection_cost(blue, BLUE), 1)

                blue.grid[middle][2] = RED
                self.assertEqual(connection_cost(blue, BLUE), 2)

    def test_candidate_score_contract_and_no_board_mutation(self):
        board = HexBoard()
        before = [row[:] for row in board.grid]
        score = score_move(board, 5, 5, RED)
        legacy = candidate_distance_score(board, (5, 5), RED)
        batch = score_moves(board, RED, [(5, 5), (6, 5)])
        mapped = heuristic_score_map(board, RED, [(5, 5)])

        self.assertGreater(score, 0)
        self.assertAlmostEqual(score, legacy)
        self.assertAlmostEqual(batch[(5, 5)], score)
        self.assertIn((6, 5), batch)
        self.assertAlmostEqual(mapped[(5, 5)], score)
        self.assertEqual(board.grid, before)
        self.assertEqual((OWN_WEIGHT, BLOCK_WEIGHT), (3.0, 2.0))

    def test_immediate_win_has_maximum_score(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        self.assertEqual(score_move(board, 0, 10, RED), 100000.0)
        self.assertEqual(candidate_distance_score(board, (0, 10), RED), 100000.0)

    def test_clone_is_independent(self):
        board = HexBoard()
        copied = clone_board(board)
        copied.place(3, 4, BLUE)
        self.assertTrue(board.is_empty(3, 4))
        self.assertFalse(copied.is_empty(3, 4))

    def test_invalid_color_and_candidate(self):
        board = HexBoard()
        with self.assertRaises(ValueError):
            connection_cost(board, 99)
        with self.assertRaises(ValueError):
            score_move(board, -1, 0, RED)
        with self.assertRaises(ValueError):
            score_move(board, 0, 0, 99)

    def test_121_candidates_under_20ms(self):
        board = HexBoard()
        moves = [(q, r) for r in range(11) for q in range(11)]
        score_moves(board, RED, moves[:1])

        timings = []
        for _ in range(3):
            started = time.perf_counter()
            scores = score_moves(board, RED, moves)
            timings.append((time.perf_counter() - started) * 1000)

        self.assertEqual(len(scores), 121)
        self.assertLess(min(timings), 20.0, f"121 点评分耗时: {timings}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
