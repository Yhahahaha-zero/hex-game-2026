# -*- coding: utf-8 -*-
"""右侧下棋记录、双计时器和复盘状态面板。"""
import pygame

from hex_board import BLUE, RED
from ui.board_renderer import BOARD_W, DIM_CLR, GOLD, TEXT_CLR, WINDOW_H


PANEL_W = 250
PANEL_BG = (22, 24, 30)
PANEL_LINE = (62, 67, 80)
ROW_ALT = (31, 34, 43)
BUTTON_BG = (52, 57, 70)
BUTTON_HOVER = (68, 75, 92)
PLAYER_NAMES = {RED: "红", BLUE: "蓝"}


class HistoryPanel:
    """显示计时、最近手数、AI 搜索进度和操作按钮。"""

    def __init__(self):
        self.title_font = None
        self.body_font = None
        self.save_button = pygame.Rect(BOARD_W + 12, 132, 106, 32)
        self.replay_button = pygame.Rect(BOARD_W + 130, 132, 106, 32)

    def set_fonts(self, title_font, body_font):
        self.title_font = title_font
        self.body_font = body_font

    def draw(self, screen, moves, game_no, timers=None, ai_status="",
             replay_enabled=True, replay_mode=False):
        pygame.draw.rect(screen, PANEL_BG, (BOARD_W, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W, 0), (BOARD_W, WINDOW_H), 2)
        self._text(screen, "下棋记录", (BOARD_W + 16, 14), self.title_font, TEXT_CLR)
        self._text(screen, f"第 {game_no}/2 局", (BOARD_W + 16, 46),
                   self.body_font, DIM_CLR)
        self._draw_timers(screen, timers or {})
        self._draw_buttons(screen, replay_enabled, replay_mode)

        self._text(screen, "手数  方  坐标  原因", (BOARD_W + 16, 178),
                   self.body_font, DIM_CLR)
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, 200),
                         (BOARD_W + PANEL_W - 12, 200), 1)

        visible = max(1, (WINDOW_H - 330) // 24)
        rows = list(moves)[-visible:]
        y = 208
        for index, move in enumerate(rows):
            if index % 2 == 0:
                pygame.draw.rect(screen, ROW_ALT, (BOARD_W + 8, y - 2, PANEL_W - 16, 23))
            reason = str(move.get("reason", ""))[:11]
            line = (
                f"{move.get('no', 0):>3}  "
                f"{PLAYER_NAMES.get(move.get('color'), '?'):<2} "
                f"{move.get('cell', '?'):<4} {reason}"
            )
            self._text(screen, line, (BOARD_W + 16, y), self.body_font, TEXT_CLR)
            y += 24

        if not rows:
            self._text(screen, "尚无落子", (BOARD_W + 16, 212), self.body_font, DIM_CLR)
        if ai_status:
            self._draw_status(screen, ai_status)

    def _draw_timers(self, screen, timers):
        red = timers.get(RED, "00:00")
        blue = timers.get(BLUE, "00:00")
        self._text(screen, f"红方 {red}", (BOARD_W + 16, 76), self.body_font, (236, 91, 82))
        self._text(screen, f"蓝方 {blue}", (BOARD_W + 132, 76), self.body_font, (95, 143, 240))

    def _draw_buttons(self, screen, replay_enabled, replay_mode):
        mouse = pygame.mouse.get_pos()
        for rect, label, enabled in (
            (self.save_button, "保存 JSON", True),
            (self.replay_button, "返回对局" if replay_mode else "复盘上一局", replay_enabled),
        ):
            color = BUTTON_HOVER if enabled and rect.collidepoint(mouse) else BUTTON_BG
            if not enabled:
                color = (38, 41, 49)
            pygame.draw.rect(screen, color, rect, border_radius=6)
            text_color = TEXT_CLR if enabled else DIM_CLR
            surface = self.body_font.render(label, True, text_color)
            screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_status(self, screen, status):
        status_y = WINDOW_H - 94
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, status_y - 14),
                         (BOARD_W + PANEL_W - 12, status_y - 14), 1)
        self._text(screen, "AI 状态", (BOARD_W + 16, status_y), self.body_font, GOLD)
        for line_no, part in enumerate(str(status).splitlines()[:3]):
            self._text(screen, part[:20], (BOARD_W + 16, status_y + 24 + line_no * 20),
                       self.body_font, DIM_CLR)

    @staticmethod
    def _text(screen, text, pos, font, color):
        if font is None:
            return
        screen.blit(font.render(str(text), True, color), pos)
