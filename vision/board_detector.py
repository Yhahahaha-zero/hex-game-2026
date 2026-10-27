# -*- coding: utf-8 -*-
"""Normalize a standard full-app or board-only Hex screenshot."""
from __future__ import annotations

from ui.board_renderer import BOARD_W, WINDOW_H
from vision.types import BoardRegion, VisionImageError


FULL_ASPECT = WINDOW_H > 0 and (BOARD_W + 250) / WINDOW_H
BOARD_ASPECT = BOARD_W / WINDOW_H


def _resize(image, size):
    from PIL import Image
    return image.resize(size, Image.Resampling.LANCZOS)


def normalize_standard_board(image):
    """Return board-only image and the logical region it came from."""
    width, height = image.size
    aspect = width / max(1, height)
    if abs(aspect - FULL_ASPECT) <= 0.10:
        from ui.history_panel import PANEL_W
        normalized = _resize(image, (BOARD_W + PANEL_W, WINDOW_H))
        board = normalized.crop((0, 0, BOARD_W, WINDOW_H))
        return board, BoardRegion(0, 0, BOARD_W, WINDOW_H, "full_app")
    if abs(aspect - BOARD_ASPECT) <= 0.12:
        board = _resize(image, (BOARD_W, WINDOW_H))
        return board, BoardRegion(0, 0, BOARD_W, WINDOW_H, "board_only")
    raise VisionImageError(
        "暂不支持该截图比例；请裁剪为标准 11×11 棋盘截图后重试"
    )