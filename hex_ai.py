# -*- coding: utf-8 -*-
"""
海克斯棋 AI。

基础 AI 决策顺序：
  1. 直接获胜；
  2. 紧急防守；
  3. 靠近棋盘中心的加权随机落子。
"""
import random
import time

from ai.explanation import AIExplanation
from hex_board import EMPTY, RED, BLUE, HexBoard


def winning_moves(board, player):
    """返回 player 一步取胜的全部空位。"""
    return [(q, r) for (q, r) in board.legal_moves(player)
            if board.would_win(q, r, player)]


class HexAI:
    """AI 基类：choose_move 返回 ((q, r), 原因说明)。"""

    name = "AI"
    last_explanation = None

    def choose_move(self, board, player):
        raise NotImplementedError


class BasicAI(HexAI):
    """基础 AI：直接获胜 > 紧急防守 > 中心偏好随机。"""

    name = "基础AI"

    def choose_move(self, board, player):
        started = time.perf_counter()
        opponent = 3 - player
        wins = winning_moves(board, player)
        if wins:
            move = random.choice(wins)
            return self._finish(move, "直接获胜", wins, started)
        threats = winning_moves(board, opponent)
        if threats:
            mine = set(board.legal_moves(player))
            dual = [cell for cell in threats if cell in mine]
            move = random.choice(dual or threats)
            return self._finish(move, "紧急防守", threats, started)
        move = self._center_random(board, player)
        return self._finish(move, "常规落子", [move], started)

    def _finish(self, move, reason, candidates, started):
        self.last_explanation = AIExplanation(
            selected_move=move,
            tactic=reason,
            candidates=tuple(
                {"move": cell, "reason": reason, "score": None}
                for cell in candidates[:5]
            ),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            search_type="rule",
            score_type="规则优先级",
        )
        return move, reason

    @staticmethod
    def _center_random(board, player):
        moves = board.legal_moves(player) or board.legal_moves()
        weights = [max(1, 12 - (abs(q - 5) + abs(r - 5))) for q, r in moves]
        return random.choices(moves, weights=weights, k=1)[0]


class RandomAI(HexAI):
    """纯随机 AI（对照基准）。"""

    name = "随机AI"

    def choose_move(self, board, player):
        started = time.perf_counter()
        move = random.choice(board.legal_moves())
        self.last_explanation = AIExplanation(
            selected_move=move,
            tactic="随机",
            candidates=({"move": move, "reason": "随机", "score": None},),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            search_type="random",
            score_type="无评分",
        )
        return move, "随机"


def create_ai(name):
    """创建基础 AI：basic / random / hybrid / strategic。"""
    key = str(name).lower()
    if key == "basic":
        return BasicAI()
    if key == "random":
        return RandomAI()
    if key in ("hybrid", "smart"):
        from ai.hybrid_ai import HybridAI
        return HybridAI()
    if key == "strategic":
        from ai.strategic_ai import StrategicAI
        return StrategicAI()
    raise ValueError(f"未知 AI 类型: {name}")
