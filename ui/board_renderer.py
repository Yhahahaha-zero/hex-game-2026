# -*- coding: utf-8 -*-
"""Hex 棋盘几何、静态底板、棋子与操作动效绘制。"""
import math

import pygame

from hex_board import BLUE, COLS, EMPTY, RED, ROWS
from ui import theme


SIZE = 28
MARGIN = 68
SQ3 = math.sqrt(3.0)
ORIGIN_X = MARGIN + SIZE
ORIGIN_Y = MARGIN + SQ3 * SIZE
BOARD_W = int(ORIGIN_X + 16 * SIZE + MARGIN)
WINDOW_H = int(ORIGIN_Y + SQ3 * SIZE * 15.5 + MARGIN)

# 兼容旧调用方的颜色名
BG = theme.BG
CELL_FILL = theme.CELL_A
CELL_LINE = theme.CELL_EDGE
RED_CLR = theme.RED_CLR
RED_DARK = theme.RED_DARK
BLUE_CLR = theme.BLUE_CLR
BLUE_DARK = theme.BLUE_DARK
GOLD = theme.GOLD
WHITE = theme.WHITE
TEXT_CLR = theme.TEXT_CLR
DIM_CLR = theme.DIM_CLR
PIECE_COLORS = dict(theme.PIECE_COLORS)


def hex_corners(cx, cy, size=SIZE):
    """返回平顶六边形的六个顶点，保留 size 参数供装饰图形复用。"""
    return [
        (
            cx + size * math.cos(math.radians(60 * index)),
            cy + size * math.sin(math.radians(60 * index)),
        )
        for index in range(6)
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
    """绘制棋盘、棋子、坐标、悬停、落子涟漪和胜利通路。"""

    def __init__(self, font_s, colorblind=False):
        self.font_s = font_s
        self.colorblind = bool(colorblind)
        self.CC = {}
        self.CL = {}
        self.CD = {}
        self.LBL = {}
        self._rebuild()

    def set_colorblind(self, enabled):
        """切换普通配色与色盲友好配色，并重建预渲染贴图。"""
        enabled = bool(enabled)
        if enabled == self.colorblind:
            return
        self.colorblind = enabled
        self._rebuild()

    def main_color(self, color):
        return self.CC.get(color, theme.WHITE)

    def _rebuild(self):
        palette = theme.get_palette(self.colorblind)
        self.CC = palette["main"]
        self.CL = palette["light"]
        self.CD = palette["dark"]
        self.LBL = palette["label"]
        self._build_sprites()
        self._build_static()

    def _build_sprites(self):
        self.piece_surf = {}
        self.ghost_surf = {}
        for color in (RED, BLUE):
            mark = None
            if self.colorblind:
                mark = "ring" if color == RED else "dot"
            sprite = self._make_piece(
                self.CC[color], self.CL[color], self.CD[color], mark
            )
            self.piece_surf[color] = sprite
            ghost = sprite.copy()
            ghost.set_alpha(115)
            self.ghost_surf[color] = ghost

        width, height = 2 * SIZE + 8, int(SQ3 * SIZE) + 8
        glow = pygame.Surface((width, height), pygame.SRCALPHA)
        points = hex_corners(width / 2, height / 2, SIZE + 1)
        pygame.draw.polygon(glow, (255, 254, 248, 38), points)
        pygame.draw.polygon(glow, (255, 254, 248, 175), points, 2)
        self.hover_surf = glow
        self.win_glow = pygame.transform.smoothscale(
            theme.radial_sprite(72, (228, 188, 108), 120), (68, 68)
        )
        self._hl_layer = pygame.Surface((BOARD_W, WINDOW_H), pygame.SRCALPHA)
        self._fx_layer = pygame.Surface((BOARD_W, WINDOW_H), pygame.SRCALPHA)

    @staticmethod
    def _make_piece(base, light, dark, mark=None):
        """以 4 倍超采样预渲染带投影和球面渐变的棋子。"""
        scale = 4
        radius = 15 * scale
        pad = 10 * scale
        width = 2 * (radius + pad)
        surface = pygame.Surface((width, width), pygame.SRCALPHA)
        center = width / 2
        for shadow_radius, alpha in (
            (radius + 7 * scale, 22),
            (radius + 3 * scale, 38),
            (radius - scale, 58),
        ):
            rect = pygame.Rect(0, 0, shadow_radius * 2, int(shadow_radius * 1.5))
            rect.center = (center + scale, center + 11 * scale)
            pygame.draw.ellipse(surface, (94, 82, 66, alpha), rect)
        for current_radius in range(radius, 0, -1):
            ratio = current_radius / radius
            if ratio > 0.55:
                color = theme.lerp_color(dark, base, (1 - ratio) / 0.45)
            else:
                color = theme.lerp_color(base, light, (0.55 - ratio) / 0.55)
            offset = 0.5 * (radius - current_radius)
            pygame.draw.circle(
                surface,
                color,
                (center - offset * 0.8, center - offset),
                current_radius,
            )
        pygame.draw.circle(surface, dark, (center, center), radius, scale)
        if mark == "ring":
            pygame.draw.circle(
                surface,
                (255, 255, 255, 235),
                (center, center),
                int(radius * 0.42),
                2 * scale,
            )
        elif mark == "dot":
            pygame.draw.circle(
                surface,
                (255, 255, 255, 235),
                (center, center),
                int(radius * 0.24),
            )
        return pygame.transform.smoothscale(surface, (width // scale, width // scale))

    def _build_static(self):
        """预渲染背景、底板、单元格、边界条与坐标标签。"""
        surface = pygame.Surface((BOARD_W, WINDOW_H))
        for y in range(WINDOW_H):
            color = theme.lerp_color(theme.INK_TOP, theme.INK_BOT,
                                     y / max(1, WINDOW_H - 1))
            surface.fill(color, (0, y, BOARD_W, 1))

        top_light = pygame.transform.smoothscale(
            theme.radial_sprite(96, (255, 250, 238), 44),
            (BOARD_W * 2, 360),
        )
        surface.blit(top_light, (-BOARD_W // 2, -140))

        pad_x, pad_top, pad_bottom = 30, 24, 30
        x0 = ORIGIN_X - SIZE - pad_x
        x1 = ORIGIN_X + 1.5 * SIZE * (COLS - 1) + SIZE + pad_x
        y0 = ORIGIN_Y - SQ3 * SIZE / 2 - pad_top
        y1 = (
            ORIGIN_Y
            + SQ3 * SIZE * (ROWS - 1 + (COLS - 1) / 2)
            + SQ3 * SIZE / 2
            + pad_bottom
        )
        board_rect = pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        shadow = pygame.Surface((BOARD_W, WINDOW_H), pygame.SRCALPHA)
        for offset_y, shrink, alpha in ((3, 3, 26), (7, 1, 20), (12, -2, 13)):
            shadow_rect = board_rect.inflate(-shrink * 2, -shrink * 2)
            shadow_rect.y += offset_y
            theme.rrect(shadow, shadow_rect, (104, 92, 74, alpha), 16)
        surface.blit(shadow, (0, 0))
        theme.rrect(surface, board_rect, theme.PANEL, 16)
        pygame.draw.line(
            surface,
            theme.PANEL_HI,
            (board_rect.x + 14, board_rect.y + 1),
            (board_rect.right - 14, board_rect.y + 1),
        )
        pygame.draw.line(
            surface,
            (192, 182, 160),
            (board_rect.x + 14, board_rect.bottom - 2),
            (board_rect.right - 14, board_rect.bottom - 2),
        )
        pygame.draw.rect(surface, theme.PANEL_EDGE, board_rect, 1, border_radius=16)

        cell_a = self._make_cell(theme.CELL_A)
        cell_b = self._make_cell(theme.CELL_B)
        for q in range(COLS):
            for r in range(ROWS):
                cx, cy = cell_center(q, r)
                cell = cell_a if (q + r) % 2 == 0 else cell_b
                surface.blit(
                    cell,
                    (cx - cell.get_width() / 2, cy - cell.get_height() / 2),
                )

        for q in range(COLS):
            points = hex_corners(*cell_center(q, 0))
            self._bar(surface, [points[4], points[5], points[0]],
                      self.CD[RED], self.CC[RED], self.CL[RED])
            points = hex_corners(*cell_center(q, ROWS - 1))
            self._bar(surface, [points[1], points[2], points[3]],
                      self.CD[RED], self.CC[RED], self.CL[RED])

        for r in range(ROWS):
            points = hex_corners(*cell_center(0, r))
            self._bar(surface, [points[2], points[3], points[4]],
                      self.CD[BLUE], self.CC[BLUE], self.CL[BLUE])
            points = hex_corners(*cell_center(COLS - 1, r))
            self._bar(surface, [points[5], points[0], points[1]],
                      self.CD[BLUE], self.CC[BLUE], self.CL[BLUE])

        for q in range(COLS):
            letter = chr(65 + q)
            cx, cy = cell_center(q, 0)
            self._label(
                surface,
                letter,
                (cx + SIZE / 2, cy - SQ3 * SIZE / 2 - 15),
                self.LBL[RED],
            )
            cx, cy = cell_center(q, ROWS - 1)
            self._label(
                surface,
                letter,
                (cx + SIZE / 2, cy + SQ3 * SIZE / 2 + 15),
                self.LBL[RED],
            )

        for r in range(ROWS):
            cx, cy = cell_center(0, r)
            self._label(surface, str(r + 1), (cx - SIZE - 17, cy), self.LBL[BLUE])
            cx, cy = cell_center(COLS - 1, r)
            self._label(surface, str(r + 1), (cx + SIZE + 17, cy), self.LBL[BLUE])

        surface.blit(theme.edge_shade_surf(BOARD_W, WINDOW_H), (0, 0))
        surface.blit(theme.grain_surface(BOARD_W, WINDOW_H), (0, 0))
        self.static_surface = surface

    @staticmethod
    def _make_cell(fill):
        width, height = 2 * SIZE + 2, int(SQ3 * SIZE) + 2
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        points = hex_corners(width / 2, height / 2)
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.lines(
            surface, theme.CELL_HI, False, [points[3], points[4], points[5], points[0]], 2
        )
        pygame.draw.lines(
            surface, theme.CELL_LO, False, [points[0], points[1], points[2], points[3]], 2
        )
        pygame.draw.polygon(surface, theme.CELL_EDGE, points, 1)
        return surface

    @staticmethod
    def _bar(surface, points, dark, base, light):
        pygame.draw.lines(surface, dark, False, points, 10)
        for point in points:
            pygame.draw.circle(surface, dark, point, 5)
        pygame.draw.lines(surface, base, False, points, 5)
        for point in points:
            pygame.draw.circle(surface, base, point, 2)
        pygame.draw.lines(surface, light, False, points, 1)

    def _label(self, surface, text, position, color):
        shadow = self.font_s.render(text, True, (255, 252, 244))
        surface.blit(shadow, shadow.get_rect(center=(position[0] + 1, position[1] + 2)))
        rendered = self.font_s.render(text, True, color)
        surface.blit(rendered, rendered.get_rect(center=position))

    def draw(self, screen, board, hover=None, preview_color=None, win_path=None,
             last_move=None, can_hover=False, now_ms=None, last_move_at=None,
             win_started_at=None, ripples=()):
        """绘制完整棋盘；所有游戏状态均由调用方提供。"""
        now = pygame.time.get_ticks() if now_ms is None else int(now_ms)
        screen.blit(self.static_surface, (0, 0))
        if can_hover and hover and board.in_range(*hover) and board.is_empty(*hover):
            self.draw_hover_highlight(screen, hover, preview_color)
        self.draw_pieces(screen, board, last_move, now, last_move_at)
        self.draw_ripples(screen, ripples, now)
        self.draw_last_marker(screen, last_move)
        self.draw_win_highlight(screen, win_path, now, win_started_at)

    def draw_pieces(self, screen, board, last_move=None, now_ms=0, last_move_at=None):
        animation = None
        if last_move and last_move_at is not None:
            elapsed = now_ms - int(last_move_at)
            if 0 <= elapsed < 220:
                animation = (
                    last_move[0],
                    last_move[1],
                    theme.ease_out_back(elapsed / 220),
                )
        for q in range(COLS):
            for r in range(ROWS):
                value = board.grid[r][q]
                if value == EMPTY:
                    continue
                sprite = self.piece_surf[value]
                if animation and (q, r) == animation[:2]:
                    scale = max(0.08, animation[2])
                    sprite = pygame.transform.smoothscale(
                        sprite,
                        (
                            max(2, int(sprite.get_width() * scale)),
                            max(2, int(sprite.get_height() * scale)),
                        ),
                    )
                cx, cy = cell_center(q, r)
                screen.blit(
                    sprite,
                    (cx - sprite.get_width() / 2, cy - sprite.get_height() / 2),
                )

    def draw_hover_highlight(self, screen, hover, preview_color=None):
        cx, cy = cell_center(*hover)
        self.hover_surf.set_alpha(int(175 + 55 * self._pulse()))
        screen.blit(
            self.hover_surf,
            (cx - self.hover_surf.get_width() / 2,
             cy - self.hover_surf.get_height() / 2),
        )
        if preview_color in self.ghost_surf:
            ghost = self.ghost_surf[preview_color]
            screen.blit(
                ghost,
                (cx - ghost.get_width() / 2, cy - ghost.get_height() / 2),
            )

    def draw_ripples(self, screen, ripples, now_ms):
        if not ripples:
            return
        layer = self._fx_layer
        layer.fill((0, 0, 0, 0))
        for q, r, color, started_at in ripples:
            if color not in self.CC:
                continue
            age = now_ms - int(started_at)
            if age < 0 or age > 640:
                continue
            cx, cy = cell_center(q, r)
            if age < 200:
                alpha = int(48 * (1 - age / 200))
                pygame.draw.polygon(
                    layer,
                    (*self.CL[color], alpha),
                    hex_corners(cx, cy, SIZE - 2),
                )
            for delay, width in ((0, 3), (90, 2)):
                ripple_age = age - delay
                if not 0 <= ripple_age <= 540:
                    continue
                progress = ripple_age / 540
                eased = 1 - (1 - progress) ** 2
                alpha = int(150 * (1 - progress))
                pygame.draw.polygon(
                    layer,
                    (*self.CC[color], alpha),
                    hex_corners(cx, cy, 16 + eased * 34),
                    max(1, int(round(width * (1 - progress)))),
                )
        screen.blit(layer, (0, 0))

    def draw_win_highlight(self, screen, win_path, now_ms=0, win_started_at=None):
        if not win_path:
            return
        cells = list(reversed(win_path))
        centers = [cell_center(q, r) for q, r in cells]
        count = len(centers)
        step = 70
        elapsed = 10 ** 9 if win_started_at is None else now_ms - int(win_started_at)
        layer = self._hl_layer
        layer.fill((0, 0, 0, 0))
        half = self.win_glow.get_width() / 2

        if elapsed < step * max(0, count - 1) + 260:
            lit = []
            for index, center in enumerate(centers):
                local = elapsed - index * step
                if local < 0:
                    break
                lit.append((center, min(1.0, local / 200)))
            points = [center for center, amount in lit if amount > 0.25]
            if len(points) > 1:
                pygame.draw.lines(layer, (*theme.GOLD_DEEP, 220), False, points, 7)
                for point in points:
                    pygame.draw.circle(layer, (*theme.GOLD_DEEP, 220), point, 3)
                pygame.draw.lines(layer, (*theme.GOLD, 255), False, points, 3)
            for center, amount in lit:
                self.win_glow.set_alpha(int(130 * amount))
                layer.blit(self.win_glow, (center[0] - half, center[1] - half))
                if amount > 0.4:
                    pygame.draw.circle(
                        layer,
                        (*theme.GOLD, int(255 * amount)),
                        center,
                        int(19 * amount) + 5,
                        2,
                    )
        else:
            pulse = self._pulse()
            self.win_glow.set_alpha(int(70 + 60 * pulse))
            for center in centers:
                layer.blit(self.win_glow, (center[0] - half, center[1] - half))
            if count > 1:
                pygame.draw.lines(layer, (*theme.GOLD_DEEP, 220), False, centers, 7)
                for point in centers:
                    pygame.draw.circle(layer, (*theme.GOLD_DEEP, 220), point, 3)
                pygame.draw.lines(layer, (*theme.GOLD, 255), False, centers, 3)
            ring_radius = 14 + int(2 * pulse)
            for center in centers:
                pygame.draw.circle(
                    layer, (*theme.GOLD, 255), center, ring_radius + 5, 2
                )
            spark_index = int(max(0, elapsed - step * count) / 55) % count
            self.win_glow.set_alpha(200)
            spark = centers[spark_index]
            layer.blit(self.win_glow, (spark[0] - half, spark[1] - half))
        screen.blit(layer, (0, 0))

    @staticmethod
    def draw_last_marker(screen, last_move):
        if not last_move or len(last_move) < 2:
            return
        q, r = last_move[:2]
        cx, cy = cell_center(q, r)
        pygame.draw.circle(screen, (122, 98, 60), (cx, cy), 5)
        pygame.draw.circle(screen, theme.GOLD, (cx, cy), 3)

    @staticmethod
    def _pulse():
        return 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.004)
