# -*- coding: utf-8 -*-
"""Stable contracts for the optional Hex experience-memory system."""

from ai.memory.ids import (
    build_decision_id,
    build_experience_key,
    build_game_id,
    build_memory_position_id,
    new_match_id,
)
from ai.memory.canonical import (
    BoardTransform,
    CanonicalPosition,
    TransformedPosition,
    canonicalize_position,
    inverse_transform_move,
    transform_board,
    transform_color,
    transform_move,
)

__all__ = [
    "BoardTransform",
    "CanonicalPosition",
    "TransformedPosition",
    "build_decision_id",
    "build_experience_key",
    "build_game_id",
    "build_memory_position_id",
    "canonicalize_position",
    "inverse_transform_move",
    "new_match_id",
    "transform_board",
    "transform_color",
    "transform_move",
]