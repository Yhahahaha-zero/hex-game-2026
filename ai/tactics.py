# -*- coding: utf-8 -*-
"""桥型、双威胁和关键点检测。"""
from collections import Counter

from ai.evaluation import clone_board
from hex_ai import winning_moves
from hex_board import BLUE, RED


def hex_distance(a, b):
    """两个轴向坐标之间的六边形距离。"""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def find_bridges(board, player):
    """找出两个己方棋子之间具有两个共同空邻居的桥。"""
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")
    stones = [
        (q, r)
        for r in range(len(board.grid))
        for q in range(len(board.grid[0]))
        if board.grid[r][q] == player
    ]
    bridges = []
    for index, first in enumerate(stones):
        first_neighbors = set(board.neighbors(*first))
        for second in stones[index + 1:]:
            if hex_distance(first, second) != 2:
                continue
            common = sorted(first_neighbors & set(board.neighbors(*second)))
            empty = tuple(cell for cell in common if board.is_empty(*cell))
            if len(common) == 2 and len(empty) == 2:
                bridges.append((first, second, empty))
    return bridges


def bridge_move_bonus(board, player):
    """统计每个候选点参与的桥数量。"""
    bonus = Counter()
    for _, _, cells in find_bridges(board, player):
        for cell in cells:
            bonus[cell] += 1
    return bonus


def creates_double_threat(board, move, player):
    """试下后是否会出现两个以上下一步获胜点。"""
    trial = clone_board(board)
    trial.place(move[0], move[1], player)
    return len(winning_moves(trial, player)) >= 2


def find_double_threat_moves(board, player, candidates=None):
    """返回所有能创造双威胁的候选着法。"""
    moves = candidates if candidates is not None else board.legal_moves()
    return [move for move in moves if creates_double_threat(board, move, player)]


def key_path_cells(board, player):
    """返回己方和对手最短路径上的空位集合。"""
    from ai.evaluation import shortest_path_cells

    opponent = BLUE if player == RED else RED
    cells = set()
    for color in (player, opponent):
        for q, r in shortest_path_cells(board, color):
            if board.is_empty(q, r):
                cells.add((q, r))
    return cells
