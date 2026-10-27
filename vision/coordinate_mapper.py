# -*- coding: utf-8 -*-
"""Map an 11x11 normalized board image to axial cell centers."""
from __future__ import annotations

from dataclasses import dataclass

from hex_board import COLS, ROWS
from ui.board_renderer import SIZE, cell_center


@dataclass(frozen=True)
class CellGeometry:
    q: int
    r: int
    cx: int
    cy: int
    radius: int


def cell_geometries(radius=None):
    sample_radius = max(6, int(radius or SIZE * 0.38))
    result = []
    for r in range(ROWS):
        for q in range(COLS):
            cx, cy = cell_center(q, r)
            result.append(CellGeometry(
                q=q, r=r, cx=int(round(cx)), cy=int(round(cy)),
                radius=sample_radius,
            ))
    return tuple(result)