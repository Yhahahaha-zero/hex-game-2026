# -*- coding: utf-8 -*-
"""PUCT MCTS 搜索实现。"""
import math
import time

import torch

from ai.encode import encode_board
from hex_board import BLUE, COLS, EMPTY, RED, ROWS


class MCTSNode:
    """MCTS 节点：保存局面、策略先验和访问统计。"""

    def __init__(self, board, player, parent=None, move=None, prior=0.0):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.priors = {}
        self.children = {}

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_expanded(self):
        return bool(self.priors)


class MCTS:
    """神经网络评估、PUCT 选择和有限时间搜索。"""

    def __init__(self, model, device="cpu", simulations=100, c_puct=1.5,
                 time_limit_ms=200):
        if simulations <= 0:
            raise ValueError("simulations 必须大于 0")
        if time_limit_ms <= 0:
            raise ValueError("time_limit_ms 必须大于 0")
        if c_puct <= 0:
            raise ValueError("c_puct 必须大于 0")

        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.time_limit_ms = time_limit_ms
        self.last_elapsed_ms = 0.0
        self.last_simulations = 0
        self.model.eval()

    @staticmethod
    def get_legal_moves(board):
        return [
            (q, r)
            for r in range(ROWS)
            for q in range(COLS)
            if board.grid[r][q] == EMPTY
        ]

    @staticmethod
    def copy_board(board):
        """复制局面，搜索过程不修改正式棋盘。"""
        new_board = type(board)()
        new_board.grid = [row[:] for row in board.grid]
        new_board.history = list(board.history)
        return new_board

    def evaluate(self, board, player):
        state = encode_board(board, player).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(state)

        policy_logits = policy_logits[0]
        value = float(value.item())
        legal_moves = self.get_legal_moves(board)
        if not legal_moves:
            return {}, value

        legal_indices = [r * COLS + q for q, r in legal_moves]
        probabilities = torch.softmax(policy_logits[legal_indices], dim=0)
        priors = {
            move: float(probabilities[index].item())
            for index, move in enumerate(legal_moves)
        }
        return priors, value

    def expand(self, node):
        """只计算策略先验，子棋盘在真正选中时再复制。"""
        priors, value = self.evaluate(node.board, node.player)
        node.priors = priors
        return value

    def _make_child(self, node, move, prior):
        child_board = self.copy_board(node.board)
        child_board.place(move[0], move[1], node.player)
        next_player = BLUE if node.player == RED else RED
        return MCTSNode(
            board=child_board,
            player=next_player,
            parent=node,
            move=move,
            prior=prior,
        )

    def select_child(self, node):
        parent_visits = max(node.visit_count, 1)
        best_score = float("-inf")
        best_child = None
        for move, prior in node.priors.items():
            child = node.children.get(move)
            if child is None:
                child = self._make_child(node, move, prior)
                node.children[move] = child
            exploration = (
                self.c_puct * prior * math.sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            score = -child.value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def get_terminal_value(board, player):
        opponent = BLUE if player == RED else RED
        if board.has_won(opponent):
            return -1.0
        if board.has_won(player):
            return 1.0
        return None

    def simulate(self, root):
        node = root
        while node.is_expanded():
            terminal_value = self.get_terminal_value(node.board, node.player)
            if terminal_value is not None:
                value = terminal_value
                break
            node = self.select_child(node)
        else:
            terminal_value = self.get_terminal_value(node.board, node.player)
            value = terminal_value if terminal_value is not None else self.expand(node)

        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value
            current = current.parent

    def search(self, board, player, on_progress=None):
        started = time.perf_counter()
        deadline = started + self.time_limit_ms / 1000.0
        root = MCTSNode(board=self.copy_board(board), player=player)
        self.expand(root)
        if not root.priors:
            raise ValueError("没有合法落子")

        def emit_progress(completed):
            if on_progress is None:
                return
            best = None
            if root.children:
                best = max(root.children, key=lambda move: root.children[move].visit_count)
            on_progress({
                "phase": "searching",
                "completed": completed,
                "total": self.simulations,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "best_move": best,
            })

        emit_progress(0)
        completed = 0
        while completed < self.simulations:
            if completed > 0 and time.perf_counter() >= deadline:
                break
            self.simulate(root)
            completed += 1
            if completed == 1 or completed % 10 == 0:
                emit_progress(completed)

        best_move = max(
            root.children,
            key=lambda move: root.children[move].visit_count,
        )
        self.last_simulations = completed
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000
        emit_progress(completed)
        return best_move, root.children[best_move].visit_count, root
