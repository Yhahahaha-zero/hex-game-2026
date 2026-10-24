# -*- coding: utf-8 -*-
"""右侧下棋记录面板。"""
import pygame

from hex_board import BLUE, RED, cell_name
from ui.board_renderer import BOARD_W, DIM_CLR, GOLD, TEXT_CLR, WINDOW_H


PANEL_W = 250
PANEL_BG = (22, 24, 30)
PANEL_LINE = (62, 67, 80)
ROW_ALT = (31, 34, 43)
PLAYER_NAMES = {RED: "红", BLUE: "蓝"}


class HistoryPanel:
    """显示当前局最近的手数、耗时和 AI 状态。"""

    def __init__(self):
        self.title_font = None
        self.body_font = None

    def set_fonts(self, title_font, body_font):
        self.title_font = title_font
        self.body_font = body_font

    def draw(self, screen, moves, game_no, ai_status=""):
        pygame.draw.rect(screen, PANEL_BG, (BOARD_W, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W, 0), (BOARD_W, WINDOW_H), 2)
        self._text(screen, "下棋记录", (BOARD_W + 16, 16), self.title_font, TEXT_CLR)
        self._text(screen, f"第 {game_no}/2 局", (BOARD_W + 16, 48),
                   self.body_font, DIM_CLR)
        self._text(screen, "手数  方  坐标  原因", (BOARD_W + 16, 78),
                   self.body_font, DIM_CLR)

        visible = max(1, (WINDOW_H - 180) // 24)
        rows = list(moves)[-visible:]
        y = 108
        for index, move in enumerate(rows):
            if index % 2 == 0:
                pygame.draw.rect(screen, ROW_ALT, (BOARD_W + 8, y - 2, PANEL_W - 16, 23))
            reason = str(move.get("reason", ""))[:12]
            line = (
                f"{move.get('no', 0):>3}  "
                f"{PLAYER_NAMES.get(move.get('color'), '?'):<2} "
                f"{move.get('cell', '?'):<4} {reason}"
            )
            self._text(screen, line, (BOARD_W + 16, y), self.body_font, TEXT_CLR)
            y += 24

        if not rows:
            self._text(screen, "尚无落子", (BOARD_W + 16, 112), self.body_font, DIM_CLR)

        if ai_status:
            status_y = WINDOW_H - 92
            pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, status_y - 14),
                             (BOARD_W + PANEL_W - 12, status_y - 14), 1)
            self._text(screen, "AI 状态", (BOARD_W + 16, status_y), self.body_font, GOLD)
            words = str(ai_status)
            for line_no in range(3):
                part = words[line_no * 18:(line_no + 1) * 18]
                if not part:
                    break
                self._text(screen, part, (BOARD_W + 16, status_y + 24 + line_no * 20),
                           self.body_font, DIM_CLR)

    @staticmethod
    def _text(screen, text, pos, font, color):
        if font is None:
            return
        surface = font.render(str(text), True, color)
        screen.blit(surface, pos)
