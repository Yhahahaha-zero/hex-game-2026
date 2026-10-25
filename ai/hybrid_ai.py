# -*- coding: utf-8 -*-
"""路径评估 + 双威胁 + 浅层预览 + 轻量 MCTS 的 HybridAI。"""
import math
import random
import time

from ai.evaluation import candidate_distance_score, connection_cost, shortest_path_cells
from ai.tactics import bridge_move_bonus, creates_double_threat
from hex_ai import winning_moves
from hex_board import BLUE, COLS, EMPTY, RED, ROWS


INF = 10_000.0


def _neighbors(q, r):
    for dq, dr in ((1, 0), (-1, 0), (1, -1), (0, -1), (0, 1), (-1, 1)):
        nq, nr = q + dq, r + dr
        if 0 <= nq < COLS and 0 <= nr < ROWS:
            yield nq, nr


def _local_score(board, q, r, player):
    opponent = BLUE if player == RED else RED
    own = enemy = 0
    for nq, nr in _neighbors(q, r):
        value = board.grid[nr][nq]
        if value == player:
            own += 1
        elif value == opponent:
            enemy += 1
    score = own * 2.4 - enemy * 0.8
    if player == RED:
        score += 1.0 if r in (0, ROWS - 1) else 0.0
    else:
        score += 1.0 if q in (0, COLS - 1) else 0.0
    return score


def _center_score(q, r):
    return 6.0 - (abs(q - 5) + abs(r - 5)) * 0.55


def evaluate(board, player):
    my_distance = connection_cost(board, player)
    opponent = BLUE if player == RED else RED
    opponent_distance = connection_cost(board, opponent)
    if my_distance == 0:
        return 100000.0
    if opponent_distance == 0:
        return -100000.0
    own_count = sum(row.count(player) for row in board.grid)
    opponent_count = sum(row.count(opponent) for row in board.grid)
    return (opponent_distance - my_distance) * 18.0 + (own_count - opponent_count) * 0.12


def _candidate_moves(board, player, limit=16):
    candidates = set(shortest_path_cells(board, player))
    candidates.update(shortest_path_cells(board, player=BLUE if player == RED else RED))
    candidates.update(board.legal_moves(player))
    bridge_bonus = bridge_move_bonus(board, player)
    scored = []
    for move in candidates:
        if not board.is_empty(*move):
            continue
        score = candidate_distance_score(board, move, player)
        score += _local_score(board, move[0], move[1], player) * 1.5
        score += _center_score(*move) * 0.25
        score += bridge_bonus.get(move, 0) * 4.0
        scored.append((score, move))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [move for _, move in scored[:limit]]


def _rollout_winner(board, to_move, rng):
    current = to_move
    moves = board.legal_moves()
    rng.shuffle(moves)
    while moves:
        sample = moves[:min(8, len(moves))]
        best = None
        best_score = -INF
        for move in sample:
            if board.would_win(move[0], move[1], current):
                best = move
                break
            score = _local_score(board, move[0], move[1], current) + _center_score(*move) * 0.12
            if score > best_score:
                best_score = score
                best = move
        if sample and rng.random() < 0.28:
            best = rng.choice(sample)
        try:
            moves.remove(best)
        except ValueError:
            pass
        board.place(best[0], best[1], current)
        if board.has_won(current):
            return current
        current = BLUE if current == RED else RED
    return RED if board.has_won(RED) else BLUE


def _minimax_preview(board, player, candidates, max_candidates=8, reply_limit=6):
    from ai.evaluation import clone_board

    opponent = BLUE if player == RED else RED
    ranked = []
    for move in candidates[:max_candidates]:
        trial = clone_board(board)
        trial.place(move[0], move[1], player)
        if trial.has_won(player):
            ranked.append((1000000.0, move))
            continue
        replies = trial.legal_moves(opponent)
        replies.sort(key=lambda cell: _local_score(trial, cell[0], cell[1], opponent), reverse=True)
        worst_case = evaluate(trial, player)
        for reply in replies[:reply_limit]:
            reply_board = clone_board(trial)
            reply_board.place(reply[0], reply[1], opponent)
            value = -100000.0 if reply_board.has_won(opponent) else evaluate(reply_board, player)
            worst_case = min(worst_case, value)
        ranked.append((worst_case, move))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked


def _mcts(board, player, candidates, time_limit, rng):
    from ai.evaluation import clone_board

    if not candidates:
        return None, 0
    stats = {move: [0, 0.0] for move in candidates}
    started = time.perf_counter()
    iterations = 0
    for move in candidates[:min(8, len(candidates))]:
        trial = clone_board(board)
        trial.place(move[0], move[1], player)
        winner = player if trial.has_won(player) else _rollout_winner(trial, BLUE if player == RED else RED, rng)
        stats[move][0] += 1
        stats[move][1] += 1.0 if winner == player else 0.0
        iterations += 1
        if time.perf_counter() - started >= time_limit:
            break

    exploration = 1.15
    while time.perf_counter() - started < time_limit:
        total = max(1, sum(value[0] for value in stats.values()))
        unvisited = [move for move, (visits, _) in stats.items() if visits == 0]
        if unvisited:
            move = rng.choice(unvisited)
        else:
            move = max(
                candidates,
                key=lambda cell: stats[cell][1] / stats[cell][0]
                + exploration * math.sqrt(math.log(total) / stats[cell][0]),
            )
        trial = clone_board(board)
        trial.place(move[0], move[1], player)
        winner = player if trial.has_won(player) else _rollout_winner(
            trial, BLUE if player == RED else RED, rng
        )
        stats[move][0] += 1
        stats[move][1] += 1.0 if winner == player else 0.0
        iterations += 1

    best = max(
        candidates,
        key=lambda move: 0.78 * (stats[move][1] / max(1, stats[move][0]))
        + 0.22 * _candidate_static(board, move, player),
    )
    return best, iterations


def _candidate_static(board, move, player):
    from ai.evaluation import clone_board

    trial = clone_board(board)
    trial.place(move[0], move[1], player)
    if trial.has_won(player):
        return 1.0
    value = evaluate(trial, player)
    return 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, value / 20.0))))


class HybridAI:
    name = "HybridAI"
    time_limit_ms = 200

    def __init__(self, time_limit=0.20, seed=None, candidate_limit=16):
        self.time_limit = max(0.08, float(time_limit))
        self.time_limit_ms = int(self.time_limit * 1000)
        self.candidate_limit = max(6, int(candidate_limit))
        self.rng = random.Random(seed)

    def set_seed(self, seed):
        self.rng.seed(seed)

    def choose_move(self, board, player):
        legal = board.legal_moves()
        if not legal:
            raise ValueError("没有合法落子")
        if not board.history and board.is_empty(5, 5):
            return (5, 5), "中心开局"

        wins = winning_moves(board, player)
        if wins:
            return self.rng.choice(wins), "直接获胜"
        opponent = BLUE if player == RED else RED
        threats = winning_moves(board, opponent)
        if threats:
            return self.rng.choice(threats), "紧急防守"

        candidates = _candidate_moves(board, player, self.candidate_limit)
        if not candidates:
            return self.rng.choice(legal), "兜底落子"

        for move in candidates[:8]:
            if creates_double_threat(board, move, player):
                return move, "创造双威胁"

        preview = _minimax_preview(board, player, candidates)
        if preview:
            candidates = [move for _, move in preview[:min(8, len(preview))]]

        move, iterations = _mcts(board, player, candidates, self.time_limit, self.rng)
        if move is not None:
            return move, f"Hybrid MCTS {iterations}次"
        return candidates[0], "路径评估"
