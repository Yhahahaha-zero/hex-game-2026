# -*- coding: utf-8 -*-
"""MCTS、编码和网络输出测试；未安装 torch 时自动跳过。"""
import importlib.util
import unittest

from hex_board import BLUE, HexBoard, RED

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class MCTSTest(unittest.TestCase):
    def test_encoding_shape_and_channels(self):
        import torch
        from ai.encode import encode_board

        board = HexBoard()
        board.place(0, 0, RED)
        board.place(10, 10, BLUE)
        state = encode_board(board, RED)
        self.assertEqual(tuple(state.shape), (3, 11, 11))
        self.assertEqual(float(state[0, 0, 0]), 1.0)
        self.assertEqual(float(state[1, 10, 10]), 1.0)
        self.assertTrue(torch.all(state[2] == 1.0))

    def test_mcts_search_result_contains_candidate_stats(self):
        import torch
        from ai.mcts import MCTS
        from ai.search_result import SearchResult

        class UniformModel(torch.nn.Module):
            def forward(self, state):
                return torch.zeros(state.shape[0], 121), torch.zeros(state.shape[0])

        result = MCTS(UniformModel(), simulations=10, time_limit_ms=1000).search_result(HexBoard(), RED)
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.simulations, 10)
        self.assertGreater(len(result.children), 0)
        self.assertEqual(result.move, result.best_move)

    def test_mcts_reports_progress(self):
        import torch
        from ai.mcts import MCTS

        class UniformModel(torch.nn.Module):
            def forward(self, state):
                return torch.zeros(state.shape[0], 121), torch.zeros(state.shape[0])

        events = []
        mcts = MCTS(UniformModel(), simulations=5, time_limit_ms=1000)
        mcts.search(HexBoard(), RED, on_progress=events.append)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["completed"], 0)
        self.assertEqual(events[-1]["completed"], 5)

    def test_network_output_shapes(self):
        import torch
        from ai.model_v5 import HexNetV5

        model = HexNetV5().eval()
        with torch.no_grad():
            policy, value = model(torch.zeros(1, 3, 11, 11))
        self.assertEqual(tuple(policy.shape), (1, 121))
        self.assertEqual(tuple(value.shape), (1,))

    def test_mcts_returns_legal_move_without_mutating_board(self):
        import torch
        from ai.mcts import MCTS
        from ai.model_v5 import HexNetV5

        class UniformModel(torch.nn.Module):
            def forward(self, state):
                return torch.zeros(state.shape[0], 121), torch.zeros(state.shape[0])

        board = HexBoard()
        board.place(5, 5, RED)
        before_grid = [row[:] for row in board.grid]
        before_history = list(board.history)
        mcts = MCTS(UniformModel(), simulations=5, time_limit_ms=1000)
        move, visits, _ = mcts.search(board, BLUE)
        self.assertTrue(board.is_empty(*move))
        self.assertGreaterEqual(visits, 0)
        self.assertEqual(board.grid, before_grid)
        self.assertEqual(board.history, before_history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
