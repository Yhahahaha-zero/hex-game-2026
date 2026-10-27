# -*- coding: utf-8 -*-
"""Explainable position features for V1 similarity retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from ai.evaluation import connection_cost, shortest_path_cells
from ai.tactics import find_bridges
from hex_ai import winning_moves
from hex_board import BLUE, EMPTY, RED


FEATURE_VERSION = 1
UNREACHABLE_DISTANCE = 121


@dataclass(frozen=True)
class PositionFeatures:
    feature_version: int
    current_color: int
    ply: int
    red_stones: int
    blue_stones: int
    empty_count: int
    red_distance: int
    blue_distance: int
    distance_gap: int
    red_winning_moves: int
    blue_winning_moves: int
    red_key_cells: tuple[tuple[int, int], ...]
    blue_key_cells: tuple[tuple[int, int], ...]
    red_bridges: int
    blue_bridges: int
    red_center_count: int
    blue_center_count: int
    red_edge_count: int
    blue_edge_count: int
    red_cells: tuple[tuple[int, int], ...]
    blue_cells: tuple[tuple[int, int], ...]

    def as_dict(self):
        data = asdict(self)
        data["red_key_cells"] = [list(cell) for cell in self.red_key_cells]
        data["blue_key_cells"] = [list(cell) for cell in self.blue_key_cells]
        data["red_cells"] = [list(cell) for cell in self.red_cells]
        data["blue_cells"] = [list(cell) for cell in self.blue_cells]
        return data


def _cells(board, color):
    return tuple(
        (q, r)
        for r, row in enumerate(board.grid)
        for q, value in enumerate(row)
        if value == color
    )


def _distance(value):
    return UNREACHABLE_DISTANCE if math.isinf(value) else int(value)


def _center_count(cells):
    return sum(1 for q, r in cells if abs(q - 5) + abs(r - 5) <= 3)


def _edge_count(cells):
    return sum(1 for q, r in cells if q in (0, 10) or r in (0, 10))


def extract_features(board, current_color, ply):
    if current_color not in (RED, BLUE):
        raise ValueError(f"Invalid current_color: {current_color!r}")
    if isinstance(ply, bool) or not isinstance(ply, int) or ply < 1:
        raise ValueError("ply must be a positive integer")
    red_cells = _cells(board, RED)
    blue_cells = _cells(board, BLUE)
    red_distance = _distance(connection_cost(board, RED))
    blue_distance = _distance(connection_cost(board, BLUE))
    red_path = tuple(shortest_path_cells(board, RED))
    blue_path = tuple(shortest_path_cells(board, BLUE))
    return PositionFeatures(
        feature_version=FEATURE_VERSION,
        current_color=current_color,
        ply=ply,
        red_stones=len(red_cells),
        blue_stones=len(blue_cells),
        empty_count=sum(1 for row in board.grid for value in row if value == EMPTY),
        red_distance=red_distance,
        blue_distance=blue_distance,
        distance_gap=red_distance - blue_distance,
        red_winning_moves=len(winning_moves(board, RED)),
        blue_winning_moves=len(winning_moves(board, BLUE)),
        red_key_cells=red_path,
        blue_key_cells=blue_path,
        red_bridges=len(find_bridges(board, RED)),
        blue_bridges=len(find_bridges(board, BLUE)),
        red_center_count=_center_count(red_cells),
        blue_center_count=_center_count(blue_cells),
        red_edge_count=_edge_count(red_cells),
        blue_edge_count=_edge_count(blue_cells),
        red_cells=red_cells,
        blue_cells=blue_cells,
    )


def features_json(board, current_color, ply):
    return json.dumps(
        extract_features(board, current_color, ply).as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_features(raw_value):
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("features_json is empty")
    data = json.loads(raw_value)
    if not isinstance(data, dict):
        raise ValueError("features_json must contain an object")
    if int(data.get("feature_version", 0)) != FEATURE_VERSION:
        raise ValueError("unsupported feature_version")
    return data


def _numeric_similarity(left, right, scale):
    return max(0.0, 1.0 - abs(float(left) - float(right)) / max(1.0, scale))


def feature_similarity(left, right):
    left = parse_features(left)
    right = parse_features(right)
    if int(left.get("current_color")) != int(right.get("current_color")):
        return 0.0
    components = [
        (_numeric_similarity(left["red_distance"], right["red_distance"], 121), 0.22),
        (_numeric_similarity(left["blue_distance"], right["blue_distance"], 121), 0.22),
        (_numeric_similarity(left["distance_gap"], right["distance_gap"], 121), 0.12),
        (_numeric_similarity(left["red_stones"], right["red_stones"], 121), 0.08),
        (_numeric_similarity(left["blue_stones"], right["blue_stones"], 121), 0.08),
        (_numeric_similarity(left["red_winning_moves"], right["red_winning_moves"], 8), 0.10),
        (_numeric_similarity(left["blue_winning_moves"], right["blue_winning_moves"], 8), 0.10),
        (_numeric_similarity(left["red_bridges"], right["red_bridges"], 30), 0.03),
        (_numeric_similarity(left["blue_bridges"], right["blue_bridges"], 30), 0.03),
        (_numeric_similarity(left["red_center_count"], right["red_center_count"], 20), 0.01),
        (_numeric_similarity(left["blue_center_count"], right["blue_center_count"], 20), 0.01),
    ]
    return max(0.0, min(1.0, sum(value * weight for value, weight in components)))