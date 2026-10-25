# -*- coding: utf-8 -*-
"""统一视觉主题、调色板与可复用的 Pygame 绘制辅助。"""
import random

import pygame

from hex_board import BLUE, RED


# 暖灰纸张主题
INK_TOP = (236, 231, 221)
INK_BOT = (212, 205, 191)
BG = INK_TOP
PANEL = (245, 240, 229)
PANEL_EDGE = (198, 188, 166)
PANEL_HI = (255, 252, 245)
BAR = (242, 237, 226)
BAR_LINE = (210, 201, 184)
CHIP = (231, 224, 209)
CHIP_EDGE = (202, 192, 171)

SIDE_BG = (235, 229, 217)
SIDE_CARD = (247, 242, 232)
SIDE_ALT = (241, 235, 224)
SIDE_LINE = (201, 191, 171)
SIDE_BUTTON = (225, 217, 202)
SIDE_BUTTON_HOVER = (214, 204, 186)

CELL_A = (233, 226, 210)
CELL_B = (226, 218, 200)
CELL_HI = (247, 242, 229)
CELL_LO = (206, 196, 175)
CELL_EDGE = (190, 179, 157)

GOLD = (207, 169, 98)
GOLD_DEEP = (180, 140, 74)
WHITE = (255, 255, 255)
TEXT_CLR = (90, 82, 71)
DIM_CLR = (148, 139, 124)
FAINT_CLR = (176, 167, 151)
SAGE = (158, 174, 148)

RED_CLR = (199, 110, 103)
RED_LIGHT = (225, 154, 145)
RED_DARK = (152, 76, 71)
BLUE_CLR = (118, 146, 180)
BLUE_LIGHT = (162, 188, 217)
BLUE_DARK = (80, 106, 139)

CB_ORANGE = (206, 120, 72)
CB_ORANGE_LIGHT = (228, 168, 126)
CB_ORANGE_DARK = (158, 88, 44)
CB_TEAL = (88, 144, 162)
CB_TEAL_LIGHT = (138, 188, 204)
CB_TEAL_DARK = (54, 104, 124)

LABEL_RED = (204, 130, 121)
LABEL_BLUE = (130, 157, 190)

PAL_STD = {
    RED: (RED_CLR, RED_LIGHT, RED_DARK),
    BLUE: (BLUE_CLR, BLUE_LIGHT, BLUE_DARK),
}
PAL_CB = {
    RED: (CB_ORANGE, CB_ORANGE_LIGHT, CB_ORANGE_DARK),
    BLUE: (CB_TEAL, CB_TEAL_LIGHT, CB_TEAL_DARK),
}
LABEL_STD = {RED: LABEL_RED, BLUE: LABEL_BLUE}
LABEL_CB = {RED: (208, 138, 92), BLUE: (108, 158, 176)}
PIECE_COLORS = {RED: RED_CLR, BLUE: BLUE_CLR}


def get_palette(colorblind=False):
    """返回当前主题下红蓝双方的主色、亮色、深色和标签色。"""
    source = PAL_CB if colorblind else PAL_STD
    return {
        "main": {color: source[color][0] for color in (RED, BLUE)},
        "light": {color: source[color][1] for color in (RED, BLUE)},
        "dark": {color: source[color][2] for color in (RED, BLUE)},
        "label": dict(LABEL_CB if colorblind else LABEL_STD),
    }


def lerp_color(color_a, color_b, amount):
    """按 amount 在线性 RGB 空间插值。"""
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round(a + (b - a) * amount))
        for a, b in zip(color_a, color_b)
    )


def ease_out_back(value):
    """带回弹的缓出曲线，用于落子动画。"""
    value = max(0.0, min(1.0, float(value)))
    c1 = 1.70158
    return 1 + (c1 + 1) * (value - 1) ** 3 + c1 * (value - 1) ** 2


def radial_sprite(size, color, max_alpha):
    """创建中心亮、边缘透明的柔光贴图。"""
    size = max(2, int(size))
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size / 2
    for radius in range(int(center), 0, -1):
        alpha = int(max_alpha * (1 - radius / center) ** 1.6)
        pygame.draw.circle(surface, (*color, alpha), (center, center), radius)
    return surface


def edge_shade_surf(width, height, max_alpha=30, depth=76):
    """创建只在窗口边缘生效的渐隐压暗层。"""
    width, height = max(1, int(width)), max(1, int(height))
    depth = max(2, min(int(depth), width // 2, height // 2))
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    strip = pygame.Surface((1, depth), pygame.SRCALPHA)
    for index in range(depth):
        alpha = int(max_alpha * (1 - index / (depth - 1)) ** 2.0)
        strip.set_at((0, index), (96, 84, 66, alpha))
    top = pygame.transform.smoothscale(strip, (width, depth))
    surface.blit(top, (0, 0))
    surface.blit(pygame.transform.flip(top, False, True), (0, height - depth))
    side = pygame.transform.rotate(
        pygame.transform.smoothscale(strip, (height, depth)), -90
    )
    surface.blit(pygame.transform.flip(side, True, False), (0, 0))
    surface.blit(side, (width - depth, 0))
    return surface


def rrect(surface, rect, color, radius, width=0):
    """圆角矩形简写。"""
    pygame.draw.rect(surface, color, rect, width, border_radius=int(radius))


def veil_surface(width, height, color=INK_TOP, alpha=205):
    """创建整页半透明纱罩。"""
    surface = pygame.Surface((int(width), int(height)), pygame.SRCALPHA)
    surface.fill((*color, int(alpha)))
    return surface


def panel_surface(rect, accent, fill=PANEL, radius=16):
    """生成带外晕、描边和受光边的覆盖层面板。"""
    width, height = rect.w + 28, rect.h + 28
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    for pad, alpha in ((12, 22), (8, 38), (4, 60)):
        pygame.draw.rect(
            surface,
            (*accent, alpha),
            pygame.Rect(
                14 - pad,
                14 - pad,
                rect.w + pad * 2,
                rect.h + pad * 2,
            ),
            2,
            border_radius=radius + pad,
        )
    body = pygame.Rect(14, 14, rect.w, rect.h)
    rrect(surface, body, (*fill, 246), radius)
    rrect(surface, body, (*accent, 255), radius, 2)
    return surface


def grain_surface(width, height, alpha=11, seed=20261025):
    """生成低对比度纸张颗粒，使用固定种子保证截图稳定。"""
    width, height = max(1, int(width)), max(1, int(height))
    small_w, small_h = max(1, width // 2), max(1, height // 2)
    rng = random.Random(seed)
    surface = pygame.Surface((small_w, small_h))
    for y in range(small_h):
        for x in range(small_w):
            value = rng.randrange(180, 256)
            surface.set_at((x, y), (value, value, value))
    surface = pygame.transform.smoothscale(surface, (width, height))
    surface.set_alpha(alpha)
    return surface
