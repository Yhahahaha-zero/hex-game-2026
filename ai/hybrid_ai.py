# -*- coding: utf-8 -*-
"""路径评估 + 双威胁 + 浅层预览 + 轻量 MCTS 的 HybridAI。"""
import math
import random
import time

from ai.evaluation import (clone_board, connection_cost, score_moves,
                           shortest_path_cells)
from ai.explanation import AIExplanation
from ai.tactics import (block_multiple_threats, boundary_bridge_bonus,
                        bridge_score_map, find_double_threat_moves,
                        opponent_double_threat_replies, opponent_threats,
                        shortest_path_key_weights)
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


def _ranked_candidates(board, player, limit=24):
    opponent = BLUE if player == RED else RED
    candidates = set(shortest_path_cells(board, player))
    candidates.update(shortest_path_cells(board, opponent))
    candidates.update(board.legal_moves(player))
    candidates = [move for move in candidates if board.is_empty(*move)]
    distance_scores = score_moves(board, player, candidates)
    bridge_scores = bridge_score_map(board, player)
    key_weights = shortest_path_key_weights(board, player)
    scored = []
    for move in candidates:
        score = distance_scores.get(move, 0.0)
        score += _local_score(board, move[0], move[1], player) * 1.5
        score += _center_score(*move) * 0.25
        score += bridge_scores.get(move, 0.0)
        score += boundary_bridge_bonus(board, move[0], move[1], player)
        score += key_weights.get(move, 0) * 1.5
        scored.append((score, move))
    scored.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return scored[:limit]


def _candidate_moves(board, player, limit=24):
    return [move for _, move in _ranked_candidates(board, player, limit)]


def _rerank_with_threat_safety(board, player, ranked, reply_limit=16,
                               budget_seconds=0.05):
    """Limited two-ply check: penalize moves allowing opponent double threats."""
    started = time.perf_counter()
    adjusted = []
    danger_found = False
    for index, (base_score, move) in enumerate(ranked):
        if time.perf_counter() - started >= budget_seconds:
            adjusted.extend(ranked[index:])
            break

        trial = clone_board(board)
        trial.place(move[0], move[1], player)
        dangerous = opponent_double_threat_replies(
            trial, player, limit=reply_limit
        )
        if dangerous:
            danger_found = True
        adjusted.append((base_score - 45.0 * len(dangerous), move))

    adjusted.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return adjusted, danger_found


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


def _mcts(board, player, candidates, time_limit, rng, cancel_event=None):
    from ai.evaluation import clone_board

    if not candidates:
        return None, 0, {}
    stats = {move: [0, 0.0] for move in candidates}
    started = time.perf_counter()
    iterations = 0
    for move in candidates[:min(8, len(candidates))]:
        if cancel_event is not None and cancel_event.is_set():
            break
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
        if cancel_event is not None and cancel_event.is_set():
            break
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
    return best, iterations, stats


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

    def __init__(self, time_limit=0.20, seed=None, candidate_limit=24):
        self.time_limit = max(0.08, float(time_limit))
        self.time_limit_ms = int(self.time_limit * 1000)
        self.candidate_limit = max(20, int(candidate_limit))
        self.rng = random.Random(seed)
        self._cancel_event = None
        self.last_explanation = None
        self.analysis_candidate_limit = 8

    def set_seed(self, seed):
        self.rng.seed(seed)

    def set_cancel_event(self, cancel_event):
        self._cancel_event = cancel_event


    def _publish_explanation(self, move, tactic, started, ranked=None,
                             stats=None, metadata=None):
        candidate_rows = []
        if stats:
            ordered = sorted(
                stats.items(),
                key=lambda item: (item[1][0], item[1][1], item[0]),
                reverse=True,
            )
            moves = [item[0] for item in ordered]
            if move in moves:
                moves.remove(move)
                moves.insert(0, move)
            for cell in moves[:8]:
                visits, wins = stats[cell]
                candidate_rows.append({
                    "move": cell,
                    "visits": visits,
                    "score": round(wins / visits, 4) if visits else 0.0,
                    "reason": tactic,
                })
            score_type = "蒙特卡洛胜率"
        elif ranked:
            ordered_ranked = list(ranked)
            for index, (_, cell) in enumerate(ordered_ranked):
                if cell == move:
                    ordered_ranked.insert(0, ordered_ranked.pop(index))
                    break
            for score, cell in ordered_ranked[:8]:
                candidate_rows.append({
                    "move": cell,
                    "score": round(float(score), 4),
                    "reason": tactic,
                })
            score_type = "启发式综合分"
        else:
            candidate_rows.append({"move": move, "score": None, "reason": tactic})
            score_type = "无评分"
        self.last_explanation = AIExplanation(
            selected_move=move,
            tactic=tactic,
            candidates=tuple(candidate_rows),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            search_type="hybrid-monte-carlo",
            score_type=score_type,
            metadata=dict(metadata or {}),
        )

    def analysis_candidates(self, limit=None):
        """Return a read-only candidate snapshot for optional LLM ranking."""
        explanation = self.last_explanation
        if explanation is None:
            return ()
        requested = self.analysis_candidate_limit if limit is None else int(limit)
        requested = max(1, min(8, requested))
        rows = []
        for rank, item in enumerate(explanation.candidates[:requested], 1):
            move = item.get("move")
            if not isinstance(move, tuple) or len(move) != 2:
                try:
                    move = tuple(move)
                except TypeError:
                    continue
            rows.append({
                "move": move,
                "base_rank": rank,
                "base_score": item.get("score"),
                "base_score_type": explanation.score_type,
                "reason": item.get("reason", explanation.tactic),
                "visits": item.get("visits"),
            })
        return tuple(rows)

    def choose_move(self, board, player):
        started = time.perf_counter()
        legal = board.legal_moves()
        if not legal:
            raise ValueError("没有合法落子")
        if not board.history and board.is_empty(5, 5):
            move = (5, 5)
            tactic = "中心开局"
            self._publish_explanation(move, tactic, started)
            return move, tactic

        wins = winning_moves(board, player)
        if wins:
            move = sorted(wins)[0]
            tactic = "直接获胜"
            self._publish_explanation(move, tactic, started,
                                      ranked=[(0.0, cell) for cell in wins])
            return move, tactic

        threats = opponent_threats(board, player)
        if len(threats) == 1:
            move = threats[0]
            tactic = "紧急防守"
            self._publish_explanation(move, tactic, started)
            return move, tactic
        if len(threats) >= 2:
            move = block_multiple_threats(board, player)
            if move is not None:
                tactic = "多威胁最佳防守"
                self._publish_explanation(move, tactic, started)
                return move, tactic

        ranked = _ranked_candidates(board, player, self.candidate_limit)
        candidates = [move for _, move in ranked]
        if not candidates:
            move = sorted(legal)[0]
            tactic = "兜底落子"
            self._publish_explanation(move, tactic, started)
            return move, tactic

        double_moves = set(find_double_threat_moves(board, player, candidates))
        for move in candidates:
            if move in double_moves:
                tactic = "创造双威胁"
                self._publish_explanation(move, tactic, started, ranked=ranked)
                return move, tactic

        ranked, danger_found = _rerank_with_threat_safety(
            board,
            player,
            ranked,
            reply_limit=16,
            budget_seconds=min(0.05, self.time_limit * 0.3),
        )
        candidates = [move for _, move in ranked]

        preview = _minimax_preview(board, player, candidates)
        if preview:
            candidates = [move for _, move in preview[:min(8, len(preview))]]

        move, iterations, stats = _mcts(
            board, player, candidates, self.time_limit, self.rng,
            cancel_event=self._cancel_event,
        )
        if move is not None:
            reason = f"Hybrid MCTS {iterations}次"
            if danger_found:
                reason += " · 双威胁规避"
            self._publish_explanation(
                move,
                reason,
                started,
                ranked=ranked,
                stats=stats,
                metadata={"iterations": iterations, "danger_found": danger_found},
            )
            return move, reason
        move = candidates[0]
        reason = "路径评估"
        self._publish_explanation(move, reason, started, ranked=ranked)
        return move, reason