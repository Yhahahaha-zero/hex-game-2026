# -*- coding: utf-8 -*-
"""只读棋局观察层：为分析、复盘和大模型提供稳定棋盘快照。

本模块只读取 HexBoard/HexEngine，不执行落子、悔棋或胜负裁定。
"""
from __future__ import annotations

import hashlib
import json
import math

from ai.evaluation import connection_cost, score_moves, shortest_path_cells
from ai.tactics import (
    bridge_mustplay_cells,
    bridge_score_map,
    find_double_threat_moves,
    opponent_threats,
    shortest_path_key_weights,
)
from hex_ai import winning_moves
from hex_board import BLUE, COLS, EMPTY, RED, ROWS, cell_name


SCHEMA_VERSION = 1
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_CANDIDATE_LIMIT = 8
_COLOR_NAMES = {RED: "RED", BLUE: "BLUE"}
_DEFAULT_SIDES = {RED: "红方", BLUE: "蓝方"}
_UNSET = object()


def _require_color(player):
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")


def _validate_integer(name, value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")


def _json_number(value):
    """Convert finite numbers for JSON; unreachable infinity becomes null."""
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 6)


def _move_object(move):
    q, r = move
    return {"move": [q, r], "cell": cell_name(q, r)}


def _move_objects(moves):
    return [_move_object(move) for move in moves]


def _path_pairs(path):
    return [[q, r] for q, r in path]


def _board_rows(board):
    return [
        "".join(
            "R" if board.grid[r][q] == RED
            else "B" if board.grid[r][q] == BLUE
            else "."
            for q in range(COLS)
        )
        for r in range(ROWS)
    ]


def _stones(board, color):
    return [
        [q, r]
        for r in range(ROWS)
        for q in range(COLS)
        if board.grid[r][q] == color
    ]


def _normalize_sides(sides):
    sides = sides or {}
    result = {}
    for color, key in ((RED, "RED"), (BLUE, "BLUE")):
        value = sides.get(color, sides.get(key, _DEFAULT_SIDES[color]))
        result[key] = str(value)
    return result


def _normalize_color_argument(color):
    return None if color is None else _COLOR_NAMES[color]


def _history_records(board, history, game_no, sides):
    """Normalize engine dict history or raw board tuple history."""
    records = []
    source = list(history or ())
    if source:
        for item in source:
            if isinstance(item, dict):
                if int(item.get("game", game_no)) != game_no:
                    continue
                q = item.get("q")
                r = item.get("r")
                color = item.get("color")
                no = item.get("no")
                reason = str(item.get("reason", ""))
            else:
                if len(item) != 3:
                    raise ValueError("棋局历史记录格式错误")
                q, r, color = item
                no = len(records) + 1
                reason = ""
            _require_color(color)
            if not board.in_range(q, r):
                raise ValueError(f"棋局历史包含非法坐标: {(q, r)!r}")
            records.append({
                "no": no,
                "color": _COLOR_NAMES[color],
                "side": sides[_COLOR_NAMES[color]],
                "move": [q, r],
                "cell": cell_name(q, r),
                "reason": reason,
            })
    else:
        for no, (q, r, color) in enumerate(board.history, 1):
            _require_color(color)
            records.append({
                "no": no,
                "color": _COLOR_NAMES[color],
                "side": sides[_COLOR_NAMES[color]],
                "move": [q, r],
                "cell": cell_name(q, r),
                "reason": "",
            })
    return records


def _center_distance(move):
    center = (ROWS - 1) / 2
    return abs(move[0] - center) + abs(move[1] - center)


def _normalize_candidates(board, player, candidates):
    if candidates is None:
        candidates = board.candidate_moves(player)
    result = []
    seen = set()
    for move in candidates:
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            raise ValueError(f"候选着法格式错误: {move!r}")
        q, r = move
        if (q, r) in seen:
            continue
        if not board.in_range(q, r) or not board.is_empty(q, r):
            raise ValueError(f"候选着法不是合法空位: {(q, r)!r}")
        seen.add((q, r))
        result.append((q, r))
    return result


def _candidate_rows(board, player, candidates, candidate_limit):
    candidates = _normalize_candidates(board, player, candidates)
    scores = score_moves(board, player, candidates)
    ordered = sorted(
        scores,
        key=lambda move: (
            -float(scores[move]),
            _center_distance(move),
            move[0],
            move[1],
        ),
    )[:candidate_limit]
    return [
        {
            "rank": index,
            "move": [q, r],
            "cell": cell_name(q, r),
            "score": _json_number(scores[(q, r)]),
            "source": "distance_heuristic",
            "reason": "距离评估",
        }
        for index, (q, r) in enumerate(ordered, 1)
    ]


def _tactics(board, player, candidates):
    opposite = BLUE if player == RED else RED
    candidates = board.candidate_moves(player)
    bridge_scores = {}
    bridge_mustplay = {}
    for color, name in ((RED, "RED"), (BLUE, "BLUE")):
        scored = sorted(
            bridge_score_map(board, color).items(),
            key=lambda item: (-float(item[1]), item[0][0], item[0][1]),
        )[:8]
        bridge_scores[name] = [
            {
                **_move_object(move),
                "score": _json_number(score),
            }
            for move, score in scored
        ]
        bridge_mustplay[name] = _move_objects(
            sorted(bridge_mustplay_cells(board, color))
        )

    key_weights = sorted(
        shortest_path_key_weights(board, player).items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return {
        "winning_moves": _move_objects(sorted(winning_moves(board, player))),
        "opponent_winning_moves": _move_objects(opponent_threats(board, player)),
        "connection_cost": {
            "RED": _json_number(connection_cost(board, RED)),
            "BLUE": _json_number(connection_cost(board, BLUE)),
        },
        "shortest_paths": {
            "RED": _path_pairs(shortest_path_cells(board, RED)),
            "BLUE": _path_pairs(shortest_path_cells(board, BLUE)),
        },
        "bridge_scores": bridge_scores,
        "bridge_mustplay": bridge_mustplay,
        "double_threat_moves": _move_objects(
            find_double_threat_moves(board, player, candidates)
        ),
        "key_path_weights": [
            {
                **_move_object(move),
                "weight": int(weight),
            }
            for move, weight in key_weights
        ],
        "opponent_color": _COLOR_NAMES[opposite],
    }


def build_position_id(board, player, game_no=1, state="playing",
                      current_color=_UNSET):
    """Build a deterministic fingerprint for the visible position."""
    _require_color(player)
    _validate_integer("game_no", game_no, 1, 1_000_000)
    state = str(state).strip()
    if not state:
        raise ValueError("state 不能为空")
    if current_color is _UNSET:
        current_color = player
    if current_color is not None:
        _require_color(current_color)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "game_no": game_no,
        "state": state,
        "analysis_color": _COLOR_NAMES[player],
        "current_color": _normalize_color_argument(current_color),
        "grid": [list(row) for row in board.grid],
        "history": [list(item) for item in board.history],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "hex1-" + hashlib.sha256(encoded).hexdigest()[:24]


def build_board_observation(
    board,
    player,
    *,
    game_no=1,
    state="playing",
    current_color=_UNSET,
    sides=None,
    history=(),
    candidates=None,
    history_limit=DEFAULT_HISTORY_LIMIT,
    candidate_limit=DEFAULT_CANDIDATE_LIMIT,
    winner_color=None,
    win_path=None,
    winner_side=None,
):
    """Return a read-only, JSON-safe observation of the current position."""
    _require_color(player)
    _validate_integer("game_no", game_no, 1, 1_000_000)
    _validate_integer("history_limit", history_limit, 0, 50)
    _validate_integer("candidate_limit", candidate_limit, 0, 20)
    state = str(state).strip()
    if not state:
        raise ValueError("state 不能为空")
    if current_color is _UNSET:
        current_color = player
    if current_color is not None:
        _require_color(current_color)
    if winner_color is not None:
        _require_color(winner_color)
    sides = _normalize_sides(sides)
    records = _history_records(board, history, game_no, sides)
    recent = records[-history_limit:] if history_limit else []
    move_count = len(board.history)
    current_key = _normalize_color_argument(current_color)
    winner_key = _normalize_color_argument(winner_color)
    terminal = state in ("game_over", "match_over")
    candidate_rows = (
        [] if terminal else
        _candidate_rows(board, player, candidates, candidate_limit)
    )
    empty_count = sum(
        1 for row in board.grid for value in row if value == EMPTY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "position_id": build_position_id(
            board, player, game_no, state, current_color=current_color
        ),
        "source": "HexEngine",
        "game_no": game_no,
        "move_count": move_count,
        "next_move_no": move_count + 1,
        "state": state,
        "is_terminal": terminal,
        "analysis_color": _COLOR_NAMES[player],
        "current_color": current_key,
        "current_player": sides.get(current_key) if current_key else None,
        "sides": dict(sides),
        "rules": {
            "RED": "连接 r=0 与 r=10",
            "BLUE": "连接 q=0 与 q=10",
        },
        "legal_move_count": empty_count,
        "last_move": records[-1] if records else None,
        "board_rows": _board_rows(board),
        "stones": {
            "RED": _stones(board, RED),
            "BLUE": _stones(board, BLUE),
        },
        "recent_history": recent,
        "winner_color": winner_key,
        "winner_side": str(winner_side) if winner_side is not None else None,
        "win_path": _path_pairs(win_path or ()),
        "candidate_score_type": "distance_heuristic",
        "candidates": candidate_rows,
        "tactics": _tactics(
            board,
            player,
            [tuple(row["move"]) for row in candidate_rows],
        ),
    }


def build_engine_observation(
    engine,
    *,
    candidates=None,
    history_limit=DEFAULT_HISTORY_LIMIT,
    candidate_limit=DEFAULT_CANDIDATE_LIMIT,
):
    """Build an observation directly from the public HexEngine interface."""
    board = engine.board
    current_color = engine.current_color
    player = current_color or engine.winner_color
    if player is None:
        player = board.history[-1][2] if board.history else RED
    sides = {RED: engine.side_of(RED), BLUE: engine.side_of(BLUE)}
    return build_board_observation(
        board,
        player,
        game_no=engine.game_no,
        state=engine.state,
        current_color=current_color,
        sides=sides,
        history=engine.move_history,
        candidates=candidates,
        history_limit=history_limit,
        candidate_limit=candidate_limit,
        winner_color=engine.winner_color,
        win_path=engine.win_path,
        winner_side=engine.winner,
    )


def observation_json(observation):
    """Serialize an observation as strict, deterministic UTF-8 JSON text."""
    return json.dumps(
        observation,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )