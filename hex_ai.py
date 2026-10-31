# -*- coding: utf-8 -*-
"""
强化版海克斯棋 AI。

融合了三个开源思路：
1. AI-for-Hex：基于最短路径/Dijkstra 的棋力评估、关键路径候选点；
2. hexai-ai：MCTS（蒙特卡洛树搜索）+ UCT 思想；
3. 原项目：直接胜利/紧急防守的战术搜索。

对外接口保持不变：choose_move(board, player) -> ((q, r), reason)
因此无需修改游戏规则和 pygame 界面。
"""
from __future__ import annotations

import math
import random
import time
from collections import deque

from hex_board import EMPTY, RED, BLUE, COLS, ROWS, HexBoard


INF = 10_000
DIRECTIONS = ((1, 0), (-1, 0), (1, -1), (0, -1), (0, 1), (-1, 1))


def winning_moves(board, player):
    """返回一步直接获胜的全部位置。"""
    return [(q, r) for q, r in board.legal_moves(player)
            if board.would_win(q, r, player)]


def _clone(board):
    """比 deepcopy 更轻的棋盘复制。"""
    b = HexBoard()
    b.grid = [row[:] for row in board.grid]
    b.history = list(board.history)
    b._dsu_cache = {RED: None, BLUE: None}
    return b


def _neighbors(q, r):
    for dq, dr in DIRECTIONS:
        nq, nr = q + dq, r + dr
        if 0 <= nq < COLS and 0 <= nr < ROWS:
            yield nq, nr


def _path_distance(board, player):
    """
    AI-for-Hex 风格的最短连接距离：
    己方棋子代价 0、空格代价 1、对手棋子不可通过。
    返回连接两端至少还需填的空格数量。
    """
    import heapq

    dist = [[INF] * COLS for _ in range(ROWS)]
    heap = []

    if player == RED:
        starts = [(q, 0) for q in range(COLS)]
        goal = lambda q, r: r == ROWS - 1
    else:
        starts = [(0, r) for r in range(ROWS)]
        goal = lambda q, r: q == COLS - 1

    for q, r in starts:
        v = board.grid[r][q]
        if v == 3 - player:
            continue
        cost = 0 if v == player else 1
        if cost < dist[r][q]:
            dist[r][q] = cost
            heapq.heappush(heap, (cost, q, r))

    while heap:
        d, q, r = heapq.heappop(heap)
        if d != dist[r][q]:
            continue
        if goal(q, r):
            return d
        for nq, nr in _neighbors(q, r):
            if board.grid[nr][nq] == 3 - player:
                continue
            nd = d + (0 if board.grid[nr][nq] == player else 1)
            if nd < dist[nr][nq]:
                dist[nr][nq] = nd
                heapq.heappush(heap, (nd, nq, nr))
    return INF


def _path_cells(board, player):
    """
    找出最短路径上的空格，作为搜索候选点。
    使用 Dijkstra 的距离场做一次反向恢复。
    """
    import heapq

    dist = [[INF] * COLS for _ in range(ROWS)]
    prev = {}
    heap = []

    if player == RED:
        starts = [(q, 0) for q in range(COLS)]
        is_goal = lambda q, r: r == ROWS - 1
    else:
        starts = [(0, r) for r in range(ROWS)]
        is_goal = lambda q, r: q == COLS - 1

    for q, r in starts:
        if board.grid[r][q] == 3 - player:
            continue
        c = 0 if board.grid[r][q] == player else 1
        if c < dist[r][q]:
            dist[r][q] = c
            heapq.heappush(heap, (c, q, r))

    goal = None
    while heap:
        d, q, r = heapq.heappop(heap)
        if d != dist[r][q]:
            continue
        if is_goal(q, r):
            goal = (q, r)
            break
        for nq, nr in _neighbors(q, r):
            if board.grid[nr][nq] == 3 - player:
                continue
            nd = d + (0 if board.grid[nr][nq] == player else 1)
            if nd < dist[nr][nq]:
                dist[nr][nq] = nd
                prev[(nq, nr)] = (q, r)
                heapq.heappush(heap, (nd, nq, nr))

    cells = []
    if goal is not None:
        cur = goal
        while cur is not None:
            q, r = cur
            if board.grid[r][q] == EMPTY:
                cells.append(cur)
            cur = prev.get(cur)
    return cells


def _center_score(q, r):
    # Hex 棋盘中心偏好，只作为很小的先验。
    return 6.0 - (abs(q - 5) + abs(r - 5)) * 0.55


def _local_score(board, q, r, player):
    """局部连接/桥接倾向评分。"""
    opp = 3 - player
    own = opp_n = 0
    for nq, nr in _neighbors(q, r):
        v = board.grid[nr][nq]
        if v == player:
            own += 1
        elif v == opp:
            opp_n += 1

    score = own * 2.4 - opp_n * 0.8
    # 接近自己的目标边界时略加分。
    if player == RED:
        score += (1.0 if r in (0, ROWS - 1) else 0.0)
    else:
        score += (1.0 if q in (0, COLS - 1) else 0.0)
    return score


def evaluate(board, player):
    """综合 Dijkstra + 局部连接 + 中心控制的静态评估，越大越有利于 player。"""
    opp = 3 - player
    my_d = _path_distance(board, player)
    op_d = _path_distance(board, opp)

    if my_d == 0:
        return 100000.0
    if op_d == 0:
        return -100000.0

    score = (op_d - my_d) * 18.0

    # 棋子数量不是主要指标，只作很小的稳定项。
    my_count = sum(row.count(player) for row in board.grid)
    op_count = sum(row.count(opp) for row in board.grid)
    score += (my_count - op_count) * 0.12

    return score


def _candidate_moves(board, player, limit=18):
    """融合最短路径、邻接区域、中心区域，压缩 121 个动作到少量高质量候选。"""
    moves = set()

    # AI-for-Hex 的关键路径候选。
    for m in _path_cells(board, player):
        moves.add(m)
    for m in _path_cells(board, 3 - player):
        moves.add(m)

    # 当前棋子周围的活跃区域。
    for q, r in board.legal_moves(player):
        for nq, nr in _neighbors(q, r):
            if board.grid[nr][nq] != EMPTY:
                moves.add((q, r))
                break

    if not moves:
        moves.update(board.legal_moves())

    # 用一步后的静态评估排序。
    scored = []
    for q, r in moves:
        if not board.is_empty(q, r):
            continue
        board.place(q, r, player)
        if board.has_won(player):
            s = 1000000.0
        else:
            s = evaluate(board, player)
        board.undo()
        s += _local_score(board, q, r, player) * 1.5
        s += _center_score(q, r) * 0.25
        scored.append((s, (q, r)))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored[:limit]]


def _rollout_winner(board, to_move, rng):
    """
    MCTS 快速模拟：使用轻量启发式加权随机，而不是纯随机。
    这样保留 MCTS 的探索性，同时避免大量明显劣着导致搜索浪费。
    """
    current = to_move
    moves = board.legal_moves()
    rng.shuffle(moves)

    while moves:
        # 每轮只在当前前沿中挑少量候选。
        sample_n = min(8, len(moves))
        sample = moves[:sample_n]
        best = None
        best_score = -float("inf")

        for m in sample:
            q, r = m
            if board.would_win(q, r, current):
                best = m
                break
            s = _local_score(board, q, r, current) + _center_score(q, r) * 0.12
            if s > best_score:
                best_score = s
                best = m

        # 从候选中加入一点随机性，避免模拟完全确定。
        if sample and rng.random() < 0.28:
            best = rng.choice(sample)

        try:
            moves.remove(best)
        except ValueError:
            pass
        board.place(best[0], best[1], current)

        if board.has_won(current):
            return current
        current = 3 - current

    # Hex 理论上不会走满而未分胜负；安全兜底。
    return RED if board.has_won(RED) else BLUE



def _minimax_preview(board, player, candidates, max_candidates=8, reply_limit=6):
    """
    吸收 AI-for-Hex/hexai-ai 的浅层 Minimax 思路：
    对候选着法假设对手也会选择最强回应，避免只看“自己走完后的静态分数”。
    这是搜索前置剪枝，不取代后面的 MCTS。
    """
    opp = 3 - player
    ranked = []

    for move in candidates[:max_candidates]:
        b = _clone(board)
        b.place(move[0], move[1], player)
        if b.has_won(player):
            ranked.append((1000000.0, move))
            continue

        replies = b.legal_moves(opp)
        replies.sort(key=lambda m: _local_score(b, m[0], m[1], opp), reverse=True)
        replies = replies[:reply_limit]

        worst_case = evaluate(b, player)
        for reply in replies:
            bb = _clone(b)
            bb.place(reply[0], reply[1], opp)
            if bb.has_won(opp):
                value = -100000.0
            else:
                value = evaluate(bb, player)
            worst_case = min(worst_case, value)

        ranked.append((worst_case, move))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


def _mcts(board, player, candidates, time_limit, rng):
    """
    根节点 MCTS。对每个候选维护 N/W/UCB，不建立完整 Python 树，
    在 11×11 棋盘上更适合桌面 GUI 的短时间思考。
    """
    if not candidates:
        return None, 0

    stats = {m: [0, 0.0] for m in candidates}  # visits, wins from player view
    start = time.perf_counter()
    iterations = 0

    # 每个候选先跑一次，保证 UCB 分母非零。
    seed_count = min(len(candidates), 8)
    for m in candidates[:seed_count]:
        b = _clone(board)
        b.place(m[0], m[1], player)
        if b.has_won(player):
            stats[m][0] += 1
            stats[m][1] += 1
        else:
            w = _rollout_winner(b, 3 - player, rng)
            stats[m][0] += 1
            if w == player:
                stats[m][1] += 1
        iterations += 1
        if time.perf_counter() - start >= time_limit:
            break

    c = 1.15
    while time.perf_counter() - start < time_limit:
        total = max(1, sum(v[0] for v in stats.values()))
        unvisited = [m for m, (n, _) in stats.items() if n == 0]
        if unvisited:
            move = rng.choice(unvisited)
        else:
            move = max(
                candidates,
                key=lambda m: (
                    stats[m][1] / stats[m][0]
                    + c * math.sqrt(math.log(total) / stats[m][0])
                ),
            )

        b = _clone(board)
        b.place(move[0], move[1], player)
        if b.has_won(player):
            winner = player
        else:
            winner = _rollout_winner(b, 3 - player, rng)

        n, w = stats[move]
        stats[move] = [n + 1, w + (1.0 if winner == player else 0.0)]
        iterations += 1

    # MCTS 胜率 + Dijkstra 静态评估混合，降低短时间随机波动。
    best = max(
        candidates,
        key=lambda m: (
            0.78 * (stats[m][1] / max(1, stats[m][0]))
            + 0.22 * _candidate_static(board, m, player)
        ),
    )
    return best, iterations


def _candidate_static(board, move, player):
    b = _clone(board)
    b.place(move[0], move[1], player)
    if b.has_won(player):
        return 1.0
    # 映射到 0~1 左右的稳定先验。
    e = evaluate(b, player)
    return 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, e / 20.0))))


class HybridAI:
    """
    强化 AI：
      直接获胜 → 必挡 → Dijkstra 关键点 → MCTS → 启发式兜底。
    """

    name = "强化混合AI"

    def __init__(self, time_limit=0.32, seed=None, candidate_limit=16):
        self.time_limit = max(0.08, float(time_limit))
        self.candidate_limit = max(6, int(candidate_limit))
        self.rng = random.Random(seed)

    def choose_move(self, board, player):
        legal = board.legal_moves()
        if not legal:
            raise ValueError("没有合法落子")

        # 0. 开局先占中心：Hex 的中心控制能同时影响多条潜在路径。
        if not board.history and board.is_empty(5, 5):
            return (5, 5), "中心开局"

        # 1. 战术必胜。
        wins = winning_moves(board, player)
        if wins:
            return self.rng.choice(wins), "必胜落子"

        # 2. 对手必胜点必须封。
        opp = 3 - player
        threats = winning_moves(board, opp)
        if threats:
            # 如果有多个威胁，优先选择还能降低己方路径距离的点。
            scored = []
            for m in threats:
                b = _clone(board)
                b.place(m[0], m[1], player)
                scored.append((evaluate(b, player), m))
            scored.sort(reverse=True)
            return scored[0][1], "紧急防守"

        # 3. 候选动作：两方最短路径 + 活跃区域。
        candidates = _candidate_moves(board, player, self.candidate_limit)
        if not candidates:
            return self.rng.choice(legal), "兜底落子"

        # 4. 先做一层静态搜索，优先保留最有希望的候选。
        scored = []
        for m in candidates:
            b = _clone(board)
            b.place(m[0], m[1], player)
            if b.has_won(player):
                return m, "路径搜索取胜"
            own = evaluate(b, player)
            # 反制对手下一步最强威胁。
            opp_wins = winning_moves(b, opp)
            penalty = 80.0 if opp_wins else 0.0
            scored.append((own - penalty, m))
        scored.sort(reverse=True)
        candidates = [m for _, m in scored[:min(12, len(scored))]]

        # 5. 浅层 Minimax 预览：假设对手会优先选择强回应。
        preview = _minimax_preview(board, player, candidates)
        if preview:
            candidates = [m for _, m in preview[:min(8, len(preview))]]

        # 6. MCTS 短时搜索。
        move, iterations = _mcts(
            board, player, candidates, self.time_limit, self.rng
        )
        if move is not None:
            return move, f"MCTS+路径评估（{iterations}次模拟）"

        return candidates[0], "路径评估"


class BasicAI(HybridAI):
    """兼容旧代码：默认 BasicAI 名称，但实际使用强化算法。"""

    name = "强化混合AI"


class RandomAI:
    """保留原项目随机 AI，方便做胜率基准测试。"""

    name = "随机AI"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, board, player):
        return self.rng.choice(board.legal_moves()), "随机"


def create_ai(name):
    """
    支持：
      basic / hybrid / smart -> 强化混合AI
      random -> 随机AI
    """
    name = str(name).lower()
    if name in ("basic", "hybrid", "smart", "mcts"):
        return HybridAI()
    if name == "random":
        return RandomAI()
    raise ValueError(f"未知 AI 类型: {name}")
