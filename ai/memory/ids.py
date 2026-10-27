# -*- coding: utf-8 -*-
"""Stable identifiers shared by experience collection and retrieval."""
from __future__ import annotations

import hashlib
import json
import uuid

from hex_board import BLUE, COLS, EMPTY, RED, ROWS


ID_SCHEMA_VERSION = 1
GAME_NAMESPACE = uuid.UUID("eef5c43c-3f33-5e4d-8bce-2e24c61c6834")
COLOR_NAMES = {RED: "RED", BLUE: "BLUE"}


class MemoryIdError(ValueError):
    """Raised when an identifier input violates the memory contract."""


def _require_color(color):
    if color not in (RED, BLUE):
        raise MemoryIdError(f"非法颜色编号: {color!r}")


def _validate_grid(grid):
    if len(grid) != ROWS or any(len(row) != COLS for row in grid):
        raise MemoryIdError("棋盘尺寸必须是 11×11")
    for row in grid:
        for value in row:
            if value not in (EMPTY, RED, BLUE):
                raise MemoryIdError(f"非法棋盘值: {value!r}")


def new_match_id():
    """Return a fresh UUID4 for one two-game match."""
    return str(uuid.uuid4())


def build_game_id(match_id, game_no):
    """Derive a stable UUID5 for game 1 or 2 of a match."""
    try:
        canonical_match = str(uuid.UUID(str(match_id)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise MemoryIdError("match_id 必须是合法 UUID") from exc
    if isinstance(game_no, bool) or not isinstance(game_no, int) or game_no not in (1, 2):
        raise MemoryIdError("game_no 必须是 1 或 2")
    return str(uuid.uuid5(GAME_NAMESPACE, f"{canonical_match}:{game_no}"))


def build_memory_position_id(board, current_color, rules_version=1):
    """Build an exact, cross-game position key independent of history/UI state."""
    _require_color(current_color)
    if isinstance(rules_version, bool) or not isinstance(rules_version, int) or rules_version < 1:
        raise MemoryIdError("rules_version 必须是正整数")
    _validate_grid(board.grid)
    payload = {
        "id_schema_version": ID_SCHEMA_VERSION,
        "rules_version": rules_version,
        "board_size": [ROWS, COLS],
        "grid": [list(row) for row in board.grid],
        "current_color": COLOR_NAMES[current_color],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "pos1-" + hashlib.sha256(encoded).hexdigest()[:32]


def build_experience_key(board, current_color, rules_version=1):
    """Compatibility alias: exact experience key and memory position ID are equal."""
    return build_memory_position_id(board, current_color, rules_version)


def build_decision_id(game_id, ply, actor_type, actor_color):
    """Build an idempotent decision ID for one game/ply/actor slot."""
    try:
        canonical_game = str(uuid.UUID(str(game_id)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise MemoryIdError("game_id 必须是合法 UUID") from exc
    if isinstance(ply, bool) or not isinstance(ply, int) or ply < 1:
        raise MemoryIdError("ply 必须是正整数")
    actor = str(actor_type).lower()
    if actor not in {"ai", "human"}:
        raise MemoryIdError("actor_type 必须是 ai 或 human")
    _require_color(actor_color)
    payload = {
        "id_schema_version": ID_SCHEMA_VERSION,
        "game_id": canonical_game,
        "ply": ply,
        "actor_type": actor,
        "actor_color": COLOR_NAMES[actor_color],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "dec1-" + hashlib.sha256(encoded).hexdigest()[:32]