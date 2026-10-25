# -*- coding: utf-8 -*-
"""StrategicAI 的确定性战术、桥关键点和消融开关测试。"""
import unittest

from ai.ai_factory import create_competition_ai
from ai.strategic_ai import StrategicAI
from hex_board import BLUE, HexBoard, RED


class StrategicAITest(unittest.TestCase):
    @staticmethod
    def _mustplay_board():
        board = HexBoard()
        board.place(0, 0, RED)
        board.place(1, 1, RED)
        board.place(1, 0, BLUE)
        return board

    @staticmethod
    def _double_bridge_board():
        board = HexBoard()
        for cell in ((10, 0), (6, 4), (9, 4), (10, 5),
                     (7, 6), (8, 6), (5, 7), (10, 7), (2, 8)):
            board.place(cell[0], cell[1], RED)
        return board

    def test_center_opening(self):
        move, _ = StrategicAI().choose_move(HexBoard(), RED)
        self.assertEqual(move, (5, 5))

    def test_strategic_ai_exposes_ranked_explanation(self):
        ai = StrategicAI()
        move, reason = ai.choose_move(HexBoard(), RED)
        explanation = ai.last_explanation.as_dict()
        self.assertEqual(explanation["selected_move"], move)
        self.assertEqual(explanation["tactic"], reason)
        self.assertEqual(explanation["score_type"], "??????")
        self.assertTrue(explanation["candidates"])

    def test_strategic_ai_exposes_ranked_explanation(self):
        ai = StrategicAI()
        move, reason = ai.choose_move(HexBoard(), RED)
        explanation = ai.last_explanation.as_dict()
        self.assertEqual(explanation["selected_move"], move)
        self.assertEqual(explanation["tactic"], reason)
        self.assertEqual(explanation["score_type"], "启发式综合分")
        self.assertTrue(explanation["candidates"])

    def test_immediate_win(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        move, reason = StrategicAI().choose_move(board, RED)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "直接获胜")

    def test_immediate_defence(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        move, reason = StrategicAI().choose_move(board, BLUE)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "紧急防守")

    def test_bridge_mustplay_has_priority(self):
        board = self._mustplay_board()
        move, reason = StrategicAI().choose_move(board, RED)
        self.assertEqual(move, (0, 1))
        self.assertEqual(reason, "桥关键点")

    def test_double_bridge_candidate_is_preferred(self):
        board = self._double_bridge_board()
        move, reason = StrategicAI().choose_move(board, RED)
        self.assertEqual(move, (9, 5))
        self.assertEqual(reason, "桥型评分")

    def test_bridge_ablation_switch(self):
        board = self._mustplay_board()
        _, reason = StrategicAI(use_bridges=False).choose_move(board, RED)
        self.assertNotEqual(reason, "桥关键点")

    def test_factory_and_board_immutability(self):
        ai, error = create_competition_ai("strategic")
        self.assertIsInstance(ai, StrategicAI)
        self.assertIsNone(error)
        board = self._double_bridge_board()
        before = [row[:] for row in board.grid]
        ai.choose_move(board, RED)
        self.assertEqual(board.grid, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
