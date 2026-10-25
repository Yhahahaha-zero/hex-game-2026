# -*- coding: utf-8 -*-
"""桥型、双威胁、多威胁防守和最短路径关键点检测。"""
from collections import Counter

from ai.evaluation import clone_board, score_moves, shortest_path_cells
from hex_ai import winning_moves
from hex_board import BLUE, RED


def _require_color(player):
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")


def _opponent(player):
    return BLUE if player == RED else RED


def hex_distance(a, b):
    """两个轴向坐标之间的六边形距离。"""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def find_bridges(board, player):
    """找出两个己方棋子之间具有两个共同空邻居的桥。"""
    _require_color(player)
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


def opponent_threats(board, player):
    """返回对手下一步可以立即获胜的全部空位，顺序固定。"""
    _require_color(player)
    return tuple(sorted(winning_moves(board, _opponent(player))))


def creates_double_threat(board, q, r=None, player=None):
    """落下 (q, r) 后，己方是否出现两个以上不同的下一步获胜点。

    兼容旧调用 creates_double_threat(board, move, player)。
    """
    if player is None:
        player = r
        q, r = q
    _require_color(player)
    if not board.in_range(q, r) or not board.is_empty(q, r):
        return False

    trial = clone_board(board)
    trial.place(q, r, player)
    return len(winning_moves(trial, player)) >= 2


def creates_double_threat_move(board, move, player):
    """元组坐标版本的兼容接口。"""
    return creates_double_threat(board, move[0], move[1], player)


def find_double_threat_moves(board, player, candidates=None):
    """返回能创造双威胁的候选点，按坐标稳定排序。"""
    _require_color(player)
    moves = board.legal_moves() if candidates is None else candidates
    return sorted(
        (q, r)
        for q, r in moves
        if board.in_range(q, r)
        and board.is_empty(q, r)
        and creates_double_threat(board, q, r, player)
    )


def block_multiple_threats(board, player):
    """对手存在多个一步获胜点时，选择防守价值最高的单步。"""
    _require_color(player)
    threats = opponent_threats(board, player)
    if not threats:
        return None
    if len(threats) == 1:
        return threats[0]

    opponent = _opponent(player)
    best_move = None
    best_key = None
    for move in sorted(board.legal_moves()):
        trial = clone_board(board)
        trial.place(move[0], move[1], player)
        if trial.has_won(player):
            return move

        remaining = len(opponent_threats(trial, player))
        own_followups = len(winning_moves(trial, player))
        blocked = len(threats) - remaining
        score = 100.0 * blocked + 20.0 * own_followups - 30.0 * remaining
        key = (score, -hex_distance(move, (5, 5)), -move[0], -move[1])
        if best_key is None or key > best_key:
            best_key = key
            best_move = move

    return best_move


def opponent_double_threat_replies(board, player, replies=None, limit=20):
    """返回对手下一步能形成双威胁的高分候选，用于有限两层检查。"""
    _require_color(player)
    opponent = _opponent(player)
    immediate = winning_moves(board, opponent)
    if immediate:
        return tuple(sorted(immediate))

    moves = board.legal_moves(opponent) if replies is None else replies
    moves = [
        (q, r)
        for q, r in moves
        if board.in_range(q, r) and board.is_empty(q, r)
    ]
    if not moves:
        return ()

    scores = score_moves(board, opponent, moves)
    ordered = sorted(
        scores,
        key=lambda move: (-scores[move], move[0], move[1]),
    )[:max(1, int(limit))]
    return tuple(
        move for move in ordered
        if creates_double_threat(board, move[0], move[1], opponent)
    )


def shortest_path_key_weights(board, player):
    """返回空位作为双方最短路径关键点的权重，2 表示同时涉及攻防。"""
    _require_color(player)
    weights = Counter()
    for color in (player, _opponent(player)):
        for q, r in shortest_path_cells(board, color):
            if board.is_empty(q, r):
                weights[(q, r)] += 1
    return dict(weights)


def shortest_path_key_cells(board, player):
    """返回己方和对手最短路径上的空位集合。"""
    return set(shortest_path_key_weights(board, player))


def key_path_cells(board, player):
    """兼容旧接口。"""
    return shortest_path_key_cells(board, player)
