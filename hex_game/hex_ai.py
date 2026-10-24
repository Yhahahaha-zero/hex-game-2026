# -*- coding: utf-8 -*-
"""
海克斯棋 AI。

当前实现（基础 AI），按优先级：
  1. 发现直接获胜机会：己方存在一步连通两边界的着法 → 立即取胜；
  2. 识别紧急防守机会：对手存在一步取胜的着法 → 抢占该点（优先选同时
     能帮己方连通的点；对手同时有多个取胜点时只能封住其一，此时基本败局已定）；
  3. 兜底：在合法位置中按“靠近棋盘中心优先”的权重随机选择。

扩展点（后续路径评估）：把第 3 步换成对每个候选点做“双端距离评估”，例如
用 Dijkstra 分别计算红方距上下边界的距离和、蓝方距左右边界的距离和，
再结合桥型（bridge）结构识别，即把 evaluate(board, q, r, player) 的得分
接入 choose_move 的兜底分支即可。
"""
import random

from hex_board import EMPTY, RED, BLUE, HexBoard


def winning_moves(board, player):
    """返回 player 一步取胜的全部空位（虚拟落子 + 并查集判定，每点 O(邻居数)）。"""
    return [(q, r) for (q, r) in board.legal_moves(player)
            if board.would_win(q, r, player)]


class HexAI:
    """AI 基类：choose_move 返回 ((q, r), 原因说明)。"""

    name = "AI"

    def choose_move(self, board, player):
        raise NotImplementedError


class BasicAI(HexAI):
    """基础 AI：直接获胜 > 紧急防守 > 中心偏好随机。"""

    name = "基础AI"

    def choose_move(self, board, player):
        opp = 3 - player
        wins = winning_moves(board, player)
        if wins:
            return random.choice(wins), "直接获胜"
        threats = winning_moves(board, opp)
        if threats:
            # 优先占既堵对手又能帮己方的点
            mine = set(board.legal_moves(player))
            dual = [t for t in threats if t in mine]
            return random.choice(dual or threats), "紧急防守"
        return self._center_random(board, player), "常规落子"

    @staticmethod
    def _center_random(board, player):
        moves = board.legal_moves(player) or board.legal_moves()
        # 距中心曼哈顿距离越近权重越高
        weights = [max(1, 12 - (abs(q - 5) + abs(r - 5))) for q, r in moves]
        return random.choices(moves, weights=weights, k=1)[0]


class RandomAI(HexAI):
    """纯随机 AI（对照基准）。"""

    name = "随机AI"

    def choose_move(self, board, player):
        return random.choice(board.legal_moves()), "随机"


def create_ai(name):
    """按名称创建 AI：basic / random。"""
    return {"basic": BasicAI(), "random": RandomAI()}[name]
