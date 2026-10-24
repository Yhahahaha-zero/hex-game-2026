# -*- coding: utf-8 -*-
"""AIWorker 和 AIResult 的单元测试。"""
import unittest

from ai.worker import AIResult, AIWorker
from hex_board import BLUE, HexBoard, RED


class StaticAI:
    name = "静态AI"

    def __init__(self, move=(5, 5), reason="测试"):
        self.move = move
        self.reason = reason

    def choose_move(self, board, player):
        return self.move, self.reason


class ProgressAI:
    name = "ProgressAI"

    def __init__(self):
        self.callback = None

    def set_progress_callback(self, callback):
        self.callback = callback

    def choose_move(self, board, player):
        if self.callback:
            self.callback({"completed": 3, "total": 10, "best_move": (5, 5)})
        return (5, 5), "progress"


class FailingAI:
    name = "失败AI"

    def choose_move(self, board, player):
        raise RuntimeError("模拟推理失败")


class AIWorkerTest(unittest.TestCase):
    def test_worker_returns_legal_move_without_mutating_board(self):
        board = HexBoard()
        before_grid = [row[:] for row in board.grid]
        before_history = list(board.history)
        worker = AIWorker(board, RED, 7, ai=StaticAI())
        worker.start()
        worker.join(2.0)

        result = worker.result
        self.assertIsInstance(result, AIResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.request_id, 7)
        self.assertEqual(result.color, RED)
        self.assertEqual(result.move, (5, 5))
        self.assertEqual(result.reason, "测试")
        self.assertEqual(board.grid, before_grid)
        self.assertEqual(board.history, before_history)

    def test_worker_falls_back_when_ai_raises(self):
        worker = AIWorker(HexBoard(), BLUE, 8, ai=FailingAI())
        worker.start()
        worker.join(2.0)

        result = worker.result
        self.assertTrue(result.ok)
        self.assertTrue(result.used_fallback)
        self.assertIn("模拟推理失败", result.inference_error)
        self.assertEqual(result.ai.name, "基础AI")

    def test_worker_exposes_ai_search_progress(self):
        worker = AIWorker(HexBoard(), RED, 10, ai=ProgressAI())
        worker.start()
        worker.join(2.0)
        progress = worker.progress
        self.assertEqual(progress["completed"], 3)
        self.assertEqual(progress["total"], 10)
        self.assertEqual(progress["best_move"], (5, 5))
        self.assertEqual(progress["phase"], "done")

    def test_invalid_ai_move_triggers_fallback(self):
        worker = AIWorker(HexBoard(), RED, 9, ai=StaticAI(move=(99, 99)))
        worker.start()
        worker.join(2.0)

        result = worker.result
        self.assertTrue(result.ok)
        self.assertIn("非法着法", result.inference_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
