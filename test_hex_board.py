# -*- coding: utf-8 -*-
"""HexBoard 边界校验、兼容接口和虚拟落子的单元测试。"""
import unittest

from hex_board import HexBoard, IllegalMoveError, RED, BLUE, EMPTY, cell_name


class HexBoardInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.board = HexBoard()

    def test_in_range_accepts_valid_coordinates_only(self):
        self.assertTrue(self.board.in_range(0, 0))
        self.assertTrue(self.board.in_range(10, 10))
        for q, r in ((-1, 0), (0, -1), (11, 0), (0, 11),
                     ("0", 0), (0, 1.5), (True, 0)):
            self.assertFalse(self.board.in_range(q, r), (q, r))

    def test_neighbors_reject_invalid_coordinates(self):
        self.assertEqual(len(list(self.board.neighbors(0, 0))), 2)
        self.assertEqual(len(list(self.board.neighbors(5, 5))), 6)
        for q, r in ((-1, 0), (11, 0), (0, -1), (0, 11), ("1", 2)):
            with self.assertRaises(IllegalMoveError):
                list(self.board.neighbors(q, r))

    def test_is_empty_rejects_invalid_coordinates(self):
        self.assertTrue(self.board.is_empty(0, 0))
        for q, r in ((-1, 0), (11, 0), (0, -1), (0, 11), (None, 0)):
            with self.assertRaises(IllegalMoveError):
                self.board.is_empty(q, r)

    def test_cell_name_rejects_invalid_coordinates(self):
        self.assertEqual(cell_name(2, 4), "5C")
        for q, r in ((-1, 0), (11, 0), (0, -1), (0, 11), (True, 0)):
            with self.assertRaises(IllegalMoveError):
                cell_name(q, r)

    def test_place_rejects_invalid_move_without_mutating_board(self):
        for q, r, color in ((-1, 0, RED), (11, 0, RED),
                            (0, -1, BLUE), (0, 11, BLUE),
                            ("0", 0, RED), (0, 0, 99), (0, 0, True)):
            with self.assertRaises(IllegalMoveError):
                self.board.place(q, r, color)
        self.assertEqual(self.board.history, [])
        self.assertEqual(self.board.grid[0][0], EMPTY)

        self.board.place(0, 0, RED)
        with self.assertRaises(IllegalMoveError):
            self.board.place(0, 0, BLUE)
        self.assertEqual(self.board.grid[0][0], RED)
        self.assertEqual(len(self.board.history), 1)

    def test_candidate_moves_and_legal_moves_compatibility(self):
        self.assertEqual(len(self.board.candidate_moves()), 121)
        self.assertEqual(len(self.board.legal_moves()), 121)

        self.board.place(5, 5, RED)
        candidates = set(self.board.candidate_moves(RED))
        self.assertEqual(len(candidates), 6)
        self.assertNotIn((5, 5), candidates)
        self.assertEqual(candidates, set(self.board.legal_moves(RED)))
        self.assertEqual(len(self.board.candidate_moves()), 120)

        with self.assertRaises(IllegalMoveError):
            self.board.candidate_moves(99)
        with self.assertRaises(IllegalMoveError):
            self.board.legal_moves(True)

    def test_would_win_validates_and_detects_winning_move(self):
        for r in range(10):
            self.board.place(0, r, RED)
        self.assertTrue(self.board.would_win(0, 10, RED))

        for args in ((-1, 0, RED), (11, 0, RED), (0, -1, RED),
                     (0, 11, RED), (1, 1, 99), (0, 0, RED)):
            with self.assertRaises(IllegalMoveError):
                self.board.would_win(*args)

    def test_has_won_and_winning_path_validate_arguments(self):
        for r in range(11):
            self.board.place(0, r, RED)
        self.assertTrue(self.board.has_won(RED))
        self.assertFalse(self.board.has_won(BLUE))
        for color in (0, 3, True, "red"):
            with self.assertRaises(IllegalMoveError):
                self.board.has_won(color)
            with self.assertRaises(IllegalMoveError):
                self.board.winning_path(color)

        path = self.board.winning_path(RED)
        self.assertIsNotNone(path)
        self.assertTrue(any(r == 0 for q, r in path))
        self.assertTrue(any(r == 10 for q, r in path))
        self.assertIsNotNone(self.board.winning_path(RED, "dfs"))
        with self.assertRaises(ValueError):
            self.board.winning_path(RED, "unknown")

    def test_render_text_and_undo(self):
        self.board.place(5, 5, RED)
        self.assertIn("R", self.board.render_text())
        with self.assertRaises(IllegalMoveError):
            self.board.render_text([(11, 0)])

        self.assertTrue(self.board.undo())
        self.assertEqual(self.board.grid[5][5], EMPTY)
        self.assertFalse(self.board.undo())


if __name__ == "__main__":
    unittest.main(verbosity=2)
