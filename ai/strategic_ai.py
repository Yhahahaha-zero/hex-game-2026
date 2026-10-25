# -*- coding: utf-8 -*-
"""Deterministic tactical AI with optional distance, threat and bridge scoring."""
import time

from ai.evaluation import score_moves
from ai.explanation import AIExplanation
from ai.tactics import (block_multiple_threats, boundary_bridge_bonus,
                        bridge_mustplay_cells, bridge_score_map,
                        find_double_threat_moves,
                        opponent_double_threat_replies, opponent_threats,
                        shortest_path_key_weights)
from hex_ai import winning_moves
from hex_board import BLUE, RED


class StrategicAI:
    """Rule-based AI used as a stable baseline and ablation target."""

    name = "StrategicAI"
    time_limit_ms = 50

    def __init__(self, use_distance=True, use_double_threat=True,
                 use_bridges=True, candidate_limit=30, safety_reply_limit=16):
        self.use_distance = bool(use_distance)
        self.use_double_threat = bool(use_double_threat)
        self.use_bridges = bool(use_bridges)
        self.candidate_limit = max(12, int(candidate_limit))
        self.safety_reply_limit = max(4, int(safety_reply_limit))
        self._cancel_event = None
        self.last_explanation = None

    def set_cancel_event(self, cancel_event):
        self._cancel_event = cancel_event

    @staticmethod
    def _center_score(q, r):
        return 4.0 - 0.5 * (abs(q - 5) + abs(r - 5))

    @staticmethod
    def _opponent(player):
        return BLUE if player == RED else RED

    def _score_candidates(self, board, player, candidates):
        distance_scores = {}
        key_weights = {}
        if self.use_distance:
            distance_scores = score_moves(board, player, candidates)
            key_weights = shortest_path_key_weights(board, player)

        bridge_scores = bridge_score_map(board, player) if self.use_bridges else {}

        scored = []
        for q, r in candidates:
            score = 0.0
            if self.use_distance:
                score += distance_scores.get((q, r), 0.0)
                score += 0.15 * key_weights.get((q, r), 0)
            if self.use_bridges:
                score += bridge_scores.get((q, r), 0.0)
                score += boundary_bridge_bonus(board, q, r, player)
            score += self._center_score(q, r)
            scored.append((score, (q, r)))
        scored.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return scored

    def _rerank_safety(self, board, player, ranked):
        opponent = self._opponent(player)
        adjusted = []
        danger_found = False
        for base_score, move in ranked[:20]:
            if self._cancel_event is not None and self._cancel_event.is_set():
                break
            trial = type(board)()
            trial.grid = [row[:] for row in board.grid]
            trial.history = list(board.history)
            trial.place(move[0], move[1], player)
            replies = trial.legal_moves(opponent)
            dangerous = opponent_double_threat_replies(
                trial, player, replies=replies,
                limit=self.safety_reply_limit,
            )
            if dangerous:
                danger_found = True
            adjusted.append((base_score - 45.0 * len(dangerous), move))

        adjusted.extend(ranked[20:])
        adjusted.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return adjusted, danger_found

    def _finish(self, move, tactic, started, ranked=None, candidates=None,
                metadata=None):
        if ranked:
            items = ranked[:5]
            candidate_rows = tuple({
                "move": cell,
                "score": round(float(score), 4),
                "reason": tactic,
            } for score, cell in items)
        else:
            candidate_rows = tuple({
                "move": cell,
                "score": None,
                "reason": tactic,
            } for cell in (candidates or [move])[:5])
        self.last_explanation = AIExplanation(
            selected_move=move,
            tactic=tactic,
            candidates=candidate_rows,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            search_type="bridge" if self.use_bridges else "strategy",
            score_type="启发式综合分",
            metadata=dict(metadata or {}),
        )
        return move, tactic

    def choose_move(self, board, player):
        started = time.perf_counter()
        if player not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {player!r}")
        legal = board.legal_moves()
        if not legal:
            raise ValueError("没有合法落子")

        wins = winning_moves(board, player)
        if wins:
            move = sorted(wins)[0]
            return self._finish(move, "直接获胜", started, candidates=wins)

        threats = opponent_threats(board, player)
        if len(threats) == 1:
            return self._finish(
                threats[0], "紧急防守", started, candidates=threats
            )
        if len(threats) >= 2:
            move = block_multiple_threats(board, player)
            if move is not None:
                return self._finish(
                    move, "多威胁最佳防守", started, candidates=threats
                )

        candidates = [
            (q, r) for q, r in legal
            if board.in_range(q, r) and board.is_empty(q, r)
        ]
        ranked = self._score_candidates(board, player, candidates)
        top = ranked[:self.candidate_limit]

        if self.use_double_threat:
            double_moves = set(
                find_double_threat_moves(board, player, [move for _, move in top])
            )
            for _, move in top:
                if move in double_moves:
                    return self._finish(move, "创造双威胁", started, ranked=ranked)

        if self.use_bridges:
            mustplay = bridge_mustplay_cells(board, player)
            if mustplay:
                score_map = dict((move, score) for score, move in ranked)
                move = min(
                    mustplay,
                    key=lambda cell: (-score_map.get(cell, float("-inf")),
                                      cell[0], cell[1]),
                )
                return self._finish(move, "桥关键点", started, ranked=ranked)

        danger_found = False
        if self.use_double_threat:
            ranked, danger_found = self._rerank_safety(board, player, ranked)

        move = ranked[0][1]
        if danger_found:
            tactic = "双威胁规避"
        elif self.use_bridges and bridge_score_map(board, player).get(move, 0) > 0:
            tactic = "桥型评分"
        else:
            tactic = "最短距离评分" if self.use_distance else "中心偏好"
        return self._finish(
            move, tactic, started, ranked=ranked,
            metadata={"danger_found": danger_found},
        )
