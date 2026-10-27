# -*- coding: utf-8 -*-
"""Deterministic color classification for standard Hex screenshots."""
from __future__ import annotations

import math
import time

from ui import theme
from vision.coordinate_mapper import cell_geometries
from vision.types import VisionCell, VisionDependencyError


EMPTY_COLORS = (theme.CELL_A, theme.CELL_B, theme.CELL_HI)
RED_COLORS = (theme.RED_CLR, theme.RED_LIGHT, theme.RED_DARK,
              theme.CB_ORANGE, theme.CB_ORANGE_LIGHT, theme.CB_ORANGE_DARK)
BLUE_COLORS = (theme.BLUE_CLR, theme.BLUE_LIGHT, theme.BLUE_DARK,
               theme.CB_TEAL, theme.CB_TEAL_LIGHT, theme.CB_TEAL_DARK)


def _numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise VisionDependencyError("视觉功能需要 numpy") from exc


def _distance(color, references):
    return min(
        math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(color, ref)))
        for ref in references
    )


def _sample_ring(pixels, geometry):
    np = _numpy()
    radius = geometry.radius
    inner = max(2, radius // 2)
    y0 = max(0, geometry.cy - radius)
    y1 = min(pixels.shape[0], geometry.cy + radius + 1)
    x0 = max(0, geometry.cx - radius)
    x1 = min(pixels.shape[1], geometry.cx + radius + 1)
    patch = pixels[y0:y1, x0:x1]
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cy = geometry.cy - y0
    cx = geometry.cx - x0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2 >= inner ** 2) & (
        (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    )
    selected = patch[mask]
    if selected.size == 0:
        return pixels[geometry.cy, geometry.cx]
    return selected.mean(axis=0)


def classify_board(image, min_confidence=0.78):
    """Classify all 121 cells in a normalized board image."""
    np = _numpy()
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    started = time.perf_counter()
    cells = []
    uncertain = []
    for geometry in cell_geometries():
        average = _sample_ring(pixels, geometry)
        d_empty = _distance(average, EMPTY_COLORS)
        d_red = _distance(average, RED_COLORS)
        d_blue = _distance(average, BLUE_COLORS)
        ranked = sorted(((d_empty, "EMPTY"), (d_red, "RED"), (d_blue, "BLUE")))
        best_distance, color = ranked[0]
        second_distance = ranked[1][0]
        margin = max(0.0, second_distance - best_distance)
        confidence = min(1.0, 0.45 + margin / 100.0)
        if confidence < float(min_confidence):
            color = "UNKNOWN"
            uncertain.append((geometry.q, geometry.r))
        cells.append(VisionCell(
            q=geometry.q,
            r=geometry.r,
            color=color,
            confidence=round(confidence, 4),
            source="color",
        ))
    return tuple(cells), tuple(uncertain), round((time.perf_counter() - started) * 1000, 3)