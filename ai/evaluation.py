# -*- coding: utf-8 -*-
"""Hex 连通距离、最短路径和候选点评分。

距离采用 0-1 BFS：己方棋子代价为 0，空位代价为 1，对手棋子不可通行。
所有计算只读取棋盘；候选点评分在独立网格副本上试下。
"""
from collections import deque
from functools import lru_cache

from hex_board import BLUE, EMPTY, HexBoard, RED


INF = float("inf")
OWN_WEIGHT = 3.0
BLOCK_WEIGHT = 2.0
WIN_SCORE = 100_000.0
DIRECTIONS = ((1, 0), (-1, 0), (1, -1), (0, -1), (0, 1), (-1, 1))


def clone_board(board):
    """轻量复制 HexBoard，供 AI 试下使用。"""
    copied = HexBoard()
    copied.grid = [row[:] for row in board.grid]
    copied.history = list(board.history)
    return copied


def _require_color(player):
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")


def _grid_size(grid):
    if not grid or not grid[0]:
        raise ValueError("棋盘不能为空")
    rows = len(grid)
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("棋盘行长度不一致")
    return rows, cols


@lru_cache(maxsize=None)
def _neighbor_table(rows, cols):
    """返回扁平索引到邻居扁平索引的缓存表。"""
    table = []
    for row in range(rows):
        for col in range(cols):
            neighbors = []
            for delta_col, delta_row in DIRECTIONS:
                next_col = col + delta_col
                next_row = row + delta_row
                if 0 <= next_col < cols and 0 <= next_row < rows:
                    neighbors.append(next_row * cols + next_col)
            table.append(tuple(neighbors))
    return tuple(table)


def _connection_cost_grid(grid, player):
    """在二维网格上执行 0-1 BFS，返回最低空位代价。"""
    rows, cols = _grid_size(grid)
    total = rows * cols
    opponent = BLUE if player == RED else RED
    flat = [grid[row][col] for row in range(rows) for col in range(cols)]
    distances = [INF] * total
    queue = deque()

    starts = range(cols) if player == RED else range(0, total, cols)
    for index in starts:
        value = flat[index]
        if value == opponent:
            continue
        weight = 0 if value == player else 1
        distances[index] = weight
        if weight == 0:
            queue.appendleft(index)
        else:
            queue.append(index)

    neighbors = _neighbor_table(rows, cols)
    while queue:
        index = queue.popleft()
        current = distances[index]
        row, col = divmod(index, cols)
        if ((player == RED and row == rows - 1)
                or (player == BLUE and col == cols - 1)):
            return current

        for next_index in neighbors[index]:
            value = flat[next_index]
            if value == opponent:
                continue
            weight = 0 if value == player else 1
            candidate = current + weight
            if candidate < distances[next_index]:
                distances[next_index] = candidate
                if weight == 0:
                    queue.appendleft(next_index)
                else:
                    queue.append(next_index)

    return INF


def connection_cost(board, player):
    """返回连接己方两侧边界所需的最低空位代价；已连通为 0。"""
    _require_color(player)
    return _connection_cost_grid(board.grid, player)


def _shortest_connection_grid(grid, player):
    rows, cols = _grid_size(grid)
    total = rows * cols
    opponent = BLUE if player == RED else RED
    flat = [grid[row][col] for row in range(rows) for col in range(cols)]
    distances = [INF] * total
    parent = {}
    queue = deque()

    starts = range(cols) if player == RED else range(0, total, cols)
    for index in starts:
        value = flat[index]
        if value == opponent:
            continue
        weight = 0 if value == player else 1
        distances[index] = weight
        if weight == 0:
            queue.appendleft(index)
        else:
            queue.append(index)

    neighbors = _neighbor_table(rows, cols)
    goal = None
    while queue:
        index = queue.popleft()
        row, col = divmod(index, cols)
        if ((player == RED and row == rows - 1)
                or (player == BLUE and col == cols - 1)):
            goal = index
            break

        current = distances[index]
        for next_index in neighbors[index]:
            value = flat[next_index]
            if value == opponent:
                continue
            weight = 0 if value == player else 1
            candidate = current + weight
            if candidate < distances[next_index]:
                distances[next_index] = candidate
                parent[next_index] = index
                if weight == 0:
                    queue.appendleft(next_index)
                else:
                    queue.append(next_index)

    if goal is None:
        return []

    path = []
    current = goal
    while current is not None:
        row, col = divmod(current, cols)
        path.append((col, row))
        current = parent.get(current)
    path.reverse()
    return path


def shortest_path_cells(board, player):
    """返回一条最低代价连接路径；不可达时返回空列表。"""
    _require_color(player)
    return _shortest_connection_grid(board.grid, player)


def _candidate_is_empty(grid, q, r):
    rows, cols = _grid_size(grid)
    return 0 <= q < cols and 0 <= r < rows and grid[r][q] == EMPTY


def _score_move_with_baseline(board, q, r, player, own_before, opp_before):
    opponent = BLUE if player == RED else RED
    trial_grid = [row[:] for row in board.grid]
    trial_grid[r][q] = player

    own_after = _connection_cost_grid(trial_grid, player)
    opp_after = _connection_cost_grid(trial_grid, opponent)

    if own_before > 0 and own_after == 0:
        return WIN_SCORE

    own_gain = max(0.0, own_before - own_after)
    opp_delay = max(0.0, opp_after - opp_before)
    return OWN_WEIGHT * own_gain + BLOCK_WEIGHT * opp_delay


def score_move(board, q, r, player):
    """对单个候选点评分，返回己方连通收益和对手延迟收益的加权和。"""
    _require_color(player)
    if not _candidate_is_empty(board.grid, q, r):
        raise ValueError(f"候选位置非法或非空: {(q, r)!r}")

    opponent = BLUE if player == RED else RED
    own_before = _connection_cost_grid(board.grid, player)
    opp_before = _connection_cost_grid(board.grid, opponent)
    return _score_move_with_baseline(
        board, q, r, player, own_before=own_before, opp_before=opp_before
    )


def candidate_distance_score(board, move, player):
    """兼容旧接口：等价于 score_move(board, q, r, player)。"""
    if len(move) != 2:
        raise ValueError(f"候选坐标格式非法: {move!r}")
    return score_move(board, move[0], move[1], player)


def score_moves(board, player, moves=None):
    """批量评分；己方和对手的基准距离各只计算一次。"""
    _require_color(player)
    if moves is None:
        moves = board.legal_moves()
    moves = list(moves)
    opponent = BLUE if player == RED else RED
    own_before = _connection_cost_grid(board.grid, player)
    opp_before = _connection_cost_grid(board.grid, opponent)

    scores = {}
    for q, r in moves:
        if not _candidate_is_empty(board.grid, q, r):
            continue
        scores[(q, r)] = _score_move_with_baseline(
            board, q, r, player,
            own_before=own_before,
            opp_before=opp_before,
        )
    return scores


def heuristic_score_map(board, player, moves=None):
    """兼容旧接口：返回候选点到距离评分的映射。"""
    return score_moves(board, player, moves)
