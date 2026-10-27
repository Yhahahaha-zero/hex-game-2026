# -*- coding: utf-8 -*-
"""Validated board symmetries and canonical position keys.

Only four transformations that preserve RED top/bottom and BLUE left/right
roles are enabled in the first version. Axis-swapping transforms are deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from hex_board import BLUE, COLS, EMPTY, RED, ROWS, HexBoard


RULES_VERSION = 1
CANONICAL_SCHEMA_VERSION = 1


class MemoryCanonicalError(ValueError):
    """Raised when a transform or coordinate violates the contract."""


class BoardTransform(Enum):
    IDENTITY = "identity"
    MIRROR_Q = "mirror_q"
    MIRROR_R = "mirror_r"
    ROTATE_180 = "rotate_180"


@dataclass(frozen=True)
class TransformedPosition:
    board: HexBoard
    current_color: int
    transform: BoardTransform


@dataclass(frozen=True)
class CanonicalPosition:
    key: str
    transform: BoardTransform
    board: HexBoard
    current_color: int


def _coerce_transform(transform):
    if isinstance(transform, BoardTransform):
        return transform
    try:
        return BoardTransform(str(transform))
    except ValueError as exc:
        raise MemoryCanonicalError(f"未知局面变换: {transform!r}") from exc


def _validate_move(move):
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        raise MemoryCanonicalError(f"非法坐标: {move!r}")
    q, r = move
    if (isinstance(q, bool) or isinstance(r, bool)
            or not isinstance(q, int) or not isinstance(r, int)
            or not 0 <= q < COLS or not 0 <= r < ROWS):
        raise MemoryCanonicalError(f"坐标越界: {move!r}")
    return q, r


def transform_move(move, transform):
    q, r = _validate_move(move)
    transform = _coerce_transform(transform)
    if transform == BoardTransform.IDENTITY:
        return q, r
    if transform == BoardTransform.MIRROR_Q:
        return COLS - 1 - q, r
    if transform == BoardTransform.MIRROR_R:
        return q, ROWS - 1 - r
    if transform == BoardTransform.ROTATE_180:
        return COLS - 1 - q, ROWS - 1 - r
    raise MemoryCanonicalError(f"未实现变换: {transform!r}")


def inverse_transform_move(move, transform):
    # All currently enabled transforms are involutions.
    return transform_move(move, transform)


def transform_color(color, transform):
    transform = _coerce_transform(transform)
    if color not in (EMPTY, RED, BLUE):
        raise MemoryCanonicalError(f"非法颜色编号: {color!r}")
    # Identity, vertical/horizontal mirrors and 180-degree rotation preserve
    # the meaning of RED and BLUE because both objectives remain unchanged.
    return color


def transform_board(board, transform):
    transform = _coerce_transform(transform)
    copied = HexBoard()
    copied.grid = [[EMPTY] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for q in range(COLS):
            nq, nr = transform_move((q, r), transform)
            copied.grid[nr][nq] = transform_color(board.grid[r][q], transform)
    copied.history = [
        (*transform_move((q, r), transform), transform_color(color, transform))
        for q, r, color in board.history
    ]
    return copied


def transform_position(board, current_color, transform):
    if current_color not in (RED, BLUE):
        raise MemoryCanonicalError(f"非法颜色编号: {current_color!r}")
    transform = _coerce_transform(transform)
    return TransformedPosition(
        board=transform_board(board, transform),
        current_color=transform_color(current_color, transform),
        transform=transform,
    )


def _serialize_position(board, current_color, rules_version):
    payload = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "rules_version": int(rules_version),
        "board_size": [ROWS, COLS],
        "grid": [list(row) for row in board.grid],
        "current_color": current_color,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonicalize_position(board, current_color, rules_version=RULES_VERSION):
    if current_color not in (RED, BLUE):
        raise MemoryCanonicalError(f"非法颜色编号: {current_color!r}")
    if (isinstance(rules_version, bool) or not isinstance(rules_version, int)
            or rules_version < 1):
        raise MemoryCanonicalError("rules_version 必须是正整数")
    candidates = []
    for transform in BoardTransform:
        transformed = transform_position(board, current_color, transform)
        serialized = _serialize_position(
            transformed.board, transformed.current_color, rules_version
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
        candidates.append((digest, transform.value, transform,
                           transformed.board, transformed.current_color))
    digest, _, transform, canonical_board, canonical_color = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return CanonicalPosition(
        key="can1-" + digest,
        transform=transform,
        board=canonical_board,
        current_color=canonical_color,
    )