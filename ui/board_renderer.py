# -*- coding: utf-8 -*-
"""Hex 棋盘几何与 pygame 绘制。"""
import math

import pygame

from hex_board import BLUE, COLS, EMPTY, RED, ROWS


SIZE = 28
MARGIN = 68
SQ3 = math.sqrt(3.0)
ORIGIN_X = MARGIN + SIZE
ORIGIN_Y = MARGIN + SQ3 * SIZE
BOARD_W = int(ORIGIN_X + 16 * SIZE + MARGIN)
WINDOW_H = int(ORIGIN_Y + SQ3 * SIZE * 15.5 + MARGIN)

BG = (28, 30, 38)
CELL_FILL = (242, 238, 224)
CELL_LINE = (112, 108, 98)
RED_CLR = (214, 62, 50)
RED_DARK = (148, 40, 32)
BLUE_CLR = (56, 104, 214)
BLUE_DARK = (36, 70, 150)
GOLD = (255, 205, 60)
WHITE = (255, 255, 255)
TEXT_CLR = (232, 232, 232)
DIM_CLR = (158, 160, 168)
PIECE_COLORS = {RED: RED_CLR, BLUE: BLUE_CLR}


def hex_corners(cx, cy):
    """返回平顶六边形的六个顶点。"""
    return [
        (cx + SIZE * math.cos(math.radians(60 * i)),
         cy + SIZE * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def cell_center(q, r):
    """轴向坐标 ``(q, r)`` 转换为像素中心。"""
    return (
        ORIGIN_X + 1.5 * SIZE * q,
        ORIGIN_Y + SQ3 * SIZE * (r + q / 2),
    )


def pixel_to_hex(x, y):
    """像素坐标转换为轴向坐标。"""
    qf = 2 / 3 * (x - ORIGIN_X) / SIZE
    rf = (-1 / 3 * (x - ORIGIN_X) + SQ3 / 3 * (y - ORIGIN_Y)) / SIZE
    sf = -qf - rf
    q, r, s = round(qf), round(rf), round(sf)
    dq, dr, ds = abs(q - qf), abs(r - rf), abs(s - sf)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


class BoardRenderer:
    """绘制棋盘、棋子、坐标、悬停和胜利路径。"""

    def __init__(self, font_s):
        self.font_s = font_s

    def draw(self, screen, board, hover=None, win_path=None, last_move=None,
             can_hover=False):
        self.draw_cells(screen)
        self.draw_borders(screen)
        self.draw_labels(screen)
        self.draw_pieces(screen, board)
        if can_hover and hover and board.is_empty(*hover):
            self.draw_hover_highlight(screen, hover)
        self.draw_win_highlight(screen, win_path)
        self.draw_last_marker(screen, last_move)

    @staticmethod
    def draw_cells(screen):
        for q in range(COLS):
            for r in range(ROWS):
                cx, cy = cell_center(q, r)
                points = hex_corners(cx, cy)
                pygame.draw.polygon(screen, CELL_FILL, points)
                pygame.draw.polygon(screen, CELL_LINE, points, 1)

    @staticmethod
    def draw_borders(screen):
        for q in range(COLS):
            cx, cy = cell_center(q, 0)
            points = hex_corners(cx, cy)
            pygame.draw.lines(screen, RED_CLR, False, [points[4], points[5], points[0]], 4)
            cx, cy = cell_center(q, ROWS - 1)
            points = hex_corners(cx, cy)
            pygame.draw.lines(screen, RED_CLR, False, [points[1], points[2], points[3]], 4)

        for r in range(ROWS):
            cx, cy = cell_center(0, r)
            points = hex_corners(cx, cy)
            pygame.draw.lines(screen, BLUE_CLR, False, [points[2], points[3], points[4]], 4)
            cx, cy = cell_center(COLS - 1, r)
            points = hex_corners(cx, cy)
            pygame.draw.lines(screen, BLUE_CLR, False, [points[5], points[0], points[1]], 4)

    def draw_labels(self, screen):
        for q in range(COLS):
            cx, cy = cell_center(q, 0)
            self._blit_center(screen, chr(65 + q),
                              (cx + SIZE / 2, cy - SQ3 * SIZE / 2 - 16), RED_CLR)
            cx, cy = cell_center(q, ROWS - 1)
            self._blit_center(screen, chr(65 + q),
                              (cx + SIZE / 2, cy + SQ3 * SIZE / 2 + 16), RED_CLR)

        for r in range(ROWS):
            cx, cy = cell_center(0, r)
            self._blit_center(screen, str(r + 1), (cx - SIZE - 18, cy), BLUE_CLR)
            cx, cy = cell_center(COLS - 1, r)
            self._blit_center(screen, str(r + 1), (cx + SIZE + 18, cy), BLUE_CLR)

    def _blit_center(self, screen, text, pos, color):
        surface = self.font_s.render(text, True, color)
        screen.blit(surface, surface.get_rect(center=pos))

    @staticmethod
    def draw_pieces(screen, board):
        radius = int(SIZE * 0.46)
        for q in range(COLS):
            for r in range(ROWS):
                value = board.grid[r][q]
                if value == EMPTY:
                    continue
                cx, cy = cell_center(q, r)
                pygame.draw.circle(
                    screen,
                    RED_DARK if value == RED else BLUE_DARK,
                    (cx, cy),
                    radius + 2,
                )
                pygame.draw.circle(screen, PIECE_COLORS[value], (cx, cy), radius)

    @staticmethod
    def draw_hover_highlight(screen, hover):
        cx, cy = cell_center(*hover)
        pygame.draw.polygon(screen, WHITE, hex_corners(cx, cy), 2)

    @staticmethod
    def draw_win_highlight(screen, win_path):
        if not win_path:
            return
        for q, r in win_path:
            cx, cy = cell_center(q, r)
            pygame.draw.circle(screen, GOLD, (cx, cy), int(SIZE * 0.46) + 4, 3)

    @staticmethod
    def draw_last_marker(screen, last_move):
        if not last_move:
            return
        q, r = last_move[:2]
        cx, cy = cell_center(q, r)
        pygame.draw.circle(screen, WHITE, (cx, cy), 3)
