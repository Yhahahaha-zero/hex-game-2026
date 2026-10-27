# -*- coding: utf-8 -*-
"""Validate an imported Hex board without mutating any live game."""
from __future__ import annotations

from hex_board import BLUE, RED, HexBoard
from vision.types import ValidationResult, VisionValidationError


def _count_and_map(cells):
    mapping = {}
    red_count = 0
    blue_count = 0
    unknown = []
    for cell in cells:
        move = (cell.q, cell.r)
        if move in mapping:
            raise VisionValidationError(f"重复格子坐标: {move!r}")
        mapping[move] = cell.color
        if cell.color == "RED":
            red_count += 1
        elif cell.color == "BLUE":
            blue_count += 1
        elif cell.color != "EMPTY":
            unknown.append(move)
    if len(mapping) != 121:
        raise VisionValidationError(f"识别格子数量应为 121，实际为 {len(mapping)}")
    return mapping, red_count, blue_count, tuple(unknown)


def validate_cells(cells):
    mapping, red_count, blue_count, unknown = _count_and_map(cells)
    warnings = []
    valid = True
    if unknown:
        valid = False
        warnings.append(f"仍有 {len(unknown)} 个格子未确定")
    difference = red_count - blue_count
    if difference not in (0, 1):
        valid = False
        warnings.append("红蓝棋子数量不符合红方先行的标准进程")

    board = HexBoard()
    for r in range(11):
        for q in range(11):
            color = mapping[(q, r)]
            if color == "RED":
                board.place(q, r, RED)
            elif color == "BLUE":
                board.place(q, r, BLUE)
    red_won = board.has_won(RED)
    blue_won = board.has_won(BLUE)
    if red_won and blue_won:
        valid = False
        warnings.append("双方不可能同时连通边界")
        winner = None
    elif red_won:
        winner = "RED"
    elif blue_won:
        winner = "BLUE"
    else:
        winner = None

    if winner:
        side_to_move = None
    elif difference == 0:
        side_to_move = "RED"
    else:
        side_to_move = "BLUE"
    return ValidationResult(
        valid=valid,
        side_to_move=side_to_move,
        winner_color=winner,
        red_count=red_count,
        blue_count=blue_count,
        warnings=tuple(warnings),
    )