# -*- coding: utf-8 -*-
"""桥型、双威胁、多威胁防守和最短路径关键点检测。"""
from collections import Counter

from ai.evaluation import clone_board, score_moves, shortest_path_cells
from hex_ai import winning_moves
from hex_board import BLUE, RED


BRIDGE_WEIGHT = 0.5
DOUBLE_BRIDGE_WEIGHT = 1.0
MUSTPLAY_WEIGHT = 2.0
BOUNDARY_BRIDGE_WEIGHT = 0.3


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


def _stones(board, player):
    return [
        (q, r)
        for r in range(len(board.grid))
        for q in range(len(board.grid[0]))
        if board.grid[r][q] == player
    ]


def find_bridge_states(board, player):
    """Return standard bridges and one-blocked mustplay states."""
    _require_color(player)
    opponent = _opponent(player)
    stones = _stones(board, player)
    states = []
    for index, first in enumerate(stones):
        first_neighbors = set(board.neighbors(*first))
        for second in stones[index + 1:]:
            if hex_distance(first, second) != 2:
                continue
            common = sorted(first_neighbors & set(board.neighbors(*second)))
            if len(common) != 2:
                continue
            empty = tuple(cell for cell in common if board.is_empty(*cell))
            blocked = tuple(
                cell for cell in common
                if board.grid[cell[1]][cell[0]] == opponent
            )
            if len(empty) == 2:
                kind = "bridge"
            elif len(empty) == 1 and len(blocked) == 1:
                kind = "mustplay"
            else:
                continue
            states.append({
                "anchors": (first, second),
                "cells": tuple(common),
                "empty": empty,
                "blocked": blocked,
                "kind": kind,
            })
    return states


def find_bridges(board, player):
    """Return standard two-empty bridges as (anchor, anchor, empty_cells)."""
    return [
        (state["anchors"][0], state["anchors"][1], state["empty"])
        for state in find_bridge_states(board, player)
        if state["kind"] == "bridge"
    ]


def bridge_mustplay_cells(board, player):
    """Return the remaining bridge cell after the opponent takes its partner."""
    return {
        state["empty"][0]
        for state in find_bridge_states(board, player)
        if state["kind"] == "mustplay" and len(state["empty"]) == 1
    }


def bridge_participation(board, player):
    """Count how many bridges and mustplay states each empty cell involves."""
    participation = Counter()
    for state in find_bridge_states(board, player):
        for cell in state["empty"]:
            participation[cell] += 1
    return participation


def double_bridge_cells(board, player):
    """Simplified double bridge: a candidate participating in two or more bridges."""
    return {
        cell for cell, count in bridge_participation(board, player).items()
        if count >= 2
    }


def boundary_bridge_bonus(board, q, r, player):
    """Small bonus for a bridge point next to an own stone near a target edge."""
    rows = len(board.grid)
    cols = len(board.grid[0])
    if player == RED:
        near_target = r in (0, 1, rows - 2, rows - 1)
    else:
        near_target = q in (0, 1, cols - 2, cols - 1)
    if not near_target:
        return 0.0
    has_own_neighbor = any(
        board.grid[nr][nq] == player
        for nq, nr in board.neighbors(q, r)
    )
    return BOUNDARY_BRIDGE_WEIGHT if has_own_neighbor else 0.0


def bridge_score_map(board, player):
    """Weighted bridge scores; tactical decisions remain higher priority."""
    scores = Counter()
    participation = Counter()
    for state in find_bridge_states(board, player):
        weight = BRIDGE_WEIGHT if state["kind"] == "bridge" else MUSTPLAY_WEIGHT
        for cell in state["empty"]:
            scores[cell] += weight
            participation[cell] += 1

    for cell, count in participation.items():
        if count >= 2:
            scores[cell] += DOUBLE_BRIDGE_WEIGHT * (count - 1)
    return dict(scores)


def bridge_move_bonus(board, player):
    """Compatibility wrapper returning weighted bridge scores."""
    return Counter(bridge_score_map(board, player))


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
