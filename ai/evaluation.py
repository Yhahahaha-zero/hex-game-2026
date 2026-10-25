# -*- coding: utf-8 -*-
"""Hex 连通距离、最短路径和候选点评分。"""
from collections import deque

from hex_board import BLUE, COLS, EMPTY, HexBoard, RED, ROWS


INF = 10_000


def clone_board(board):
    """轻量复制棋盘，供 AI 试下使用。"""
    copied = HexBoard()
    copied.grid = [row[:] for row in board.grid]
    copied.history = list(board.history)
    return copied


def _require_color(player):
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")


def _boundary_sets(player):
    if player == RED:
        return [(q, 0) for q in range(COLS)], lambda q, r: r == ROWS - 1
    return [(0, r) for r in range(ROWS)], lambda q, r: q == COLS - 1


def connection_cost(board, player):
    """返回连接己方两边所需的最少空位代价，0表示已经连通。"""
    _require_color(player)
    starts, is_goal = _boundary_sets(player)
    opponent = BLUE if player == RED else RED
    dist = [[INF] * COLS for _ in range(ROWS)]
    queue = deque()

    for q, r in starts:
        value = board.grid[r][q]
        if value == opponent:
            continue
        cost = 0 if value == player else 1
        if cost < dist[r][q]:
            dist[r][q] = cost
            if cost == 0:
                queue.appendleft((q, r))
            else:
                queue.append((q, r))

    while queue:
        q, r = queue.popleft()
        current = dist[r][q]
        if is_goal(q, r):
            return current
        for nq, nr in board.neighbors(q, r):
            value = board.grid[nr][nq]
            if value == opponent:
                continue
            weight = 0 if value == player else 1
            candidate = current + weight
            if candidate < dist[nr][nq]:
                dist[nr][nq] = candidate
                if weight == 0:
                    queue.appendleft((nq, nr))
                else:
                    queue.append((nq, nr))
    return INF


def shortest_path_cells(board, player):
    """返回一条最短连接路径上的单元格，未连通时包含需要填补的空位。"""
    _require_color(player)
    starts, is_goal = _boundary_sets(player)
    opponent = BLUE if player == RED else RED
    dist = [[INF] * COLS for _ in range(ROWS)]
    parent = {}
    queue = deque()

    for q, r in starts:
        value = board.grid[r][q]
        if value == opponent:
            continue
        cost = 0 if value == player else 1
        if cost < dist[r][q]:
            dist[r][q] = cost
            if cost == 0:
                queue.appendleft((q, r))
            else:
                queue.append((q, r))

    goal = None
    while queue:
        q, r = queue.popleft()
        if is_goal(q, r):
            goal = (q, r)
            break
        current = dist[r][q]
        for nq, nr in board.neighbors(q, r):
            value = board.grid[nr][nq]
            if value == opponent:
                continue
            weight = 0 if value == player else 1
            candidate = current + weight
            if candidate < dist[nr][nq]:
                dist[nr][nq] = candidate
                parent[(nq, nr)] = (q, r)
                if weight == 0:
                    queue.appendleft((nq, nr))
                else:
                    queue.append((nq, nr))

    if goal is None:
        return []
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path


def candidate_distance_score(board, move, player):
    """计算候选着法对己方连通和对手阻延的综合分数。"""
    _require_color(player)
    q, r = move
    if not board.is_empty(q, r):
        raise ValueError(f"候选位置非空: {move!r}")
    if board.would_win(q, r, player):
        return 100000.0

    opponent = BLUE if player == RED else RED
    own_before = connection_cost(board, player)
    opp_before = connection_cost(board, opponent)
    trial = clone_board(board)
    trial.place(q, r, player)
    own_after = connection_cost(trial, player)
    opp_after = connection_cost(trial, opponent)

    own_gain = max(0, own_before - own_after)
    opp_delay = max(0, opp_after - opp_before)
    return 3.0 * own_gain + 2.0 * opp_delay


def heuristic_score_map(board, player, moves=None):
    """为全部或指定候选位置生成距离评分。"""
    moves = list(moves if moves is not None else board.legal_moves())
    return {move: candidate_distance_score(board, move, player) for move in moves}
