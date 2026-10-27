# -*- coding: utf-8 -*-
"""右侧下棋记录、双计时器、AI 进度和复盘状态面板。"""
import pygame

from hex_board import BLUE, RED, cell_name
from ui import theme
from ui.board_renderer import BOARD_W, WINDOW_H


PANEL_W = 250
PANEL_BG = theme.SIDE_BG
PANEL_LINE = theme.SIDE_LINE
ROW_ALT = theme.SIDE_ALT
BUTTON_BG = theme.SIDE_BUTTON
BUTTON_HOVER = theme.SIDE_BUTTON_HOVER
PLAYER_NAMES = {RED: "红", BLUE: "蓝"}


class HistoryPanel:
    """显示计时、最近手数、AI 搜索进度和操作按钮。"""

    def __init__(self):
        self.title_font = None
        self.body_font = None
        self.red_color = theme.RED_CLR
        self.blue_color = theme.BLUE_CLR
        self.undo_button = pygame.Rect(BOARD_W + 10, 132, 108, 38)
        self.save_button = pygame.Rect(BOARD_W + 132, 132, 108, 38)
        self.export_button = pygame.Rect(BOARD_W + 10, 176, 108, 38)
        self.replay_button = pygame.Rect(BOARD_W + 132, 176, 108, 38)
        self.analysis_button = pygame.Rect(BOARD_W + 10, 220, 230, 38)

    def set_fonts(self, title_font, body_font):
        self.title_font = title_font
        self.body_font = body_font

    def set_palette(self, main_colors):
        """同步色盲模式的双方主色。"""
        self.red_color = main_colors.get(RED, theme.RED_CLR)
        self.blue_color = main_colors.get(BLUE, theme.BLUE_CLR)

    def draw(self, screen, moves, game_no, timers=None, ai_status="",
             replay_enabled=True, replay_mode=False, search_stats=None,
             ai_explanation=None, undo_enabled=False,
             analysis_active=False, analysis_available=True):
        pygame.draw.rect(screen, PANEL_BG, (BOARD_W, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W, 0), (BOARD_W, WINDOW_H), 2)

        pygame.draw.polygon(
            screen,
            theme.GOLD,
            [(BOARD_W + 16, 18), (BOARD_W + 23, 25),
             (BOARD_W + 16, 32), (BOARD_W + 9, 25)],
        )
        self._text(screen, "下棋记录", (BOARD_W + 32, 15),
                   self.title_font, theme.TEXT_CLR)
        self._text(screen, f"第 {game_no}/2 局", (BOARD_W + 16, 45),
                   self.body_font, theme.DIM_CLR)
        self._draw_timers(screen, timers or {})
        self._draw_buttons(
            screen, replay_enabled, replay_mode, undo_enabled=undo_enabled,
            analysis_active=analysis_active,
            analysis_available=analysis_available,
        )

        self._text(screen, "手数  方  坐标  原因", (BOARD_W + 16, 264),
                   self.body_font, theme.DIM_CLR)
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, 286),
                         (BOARD_W + PANEL_W - 12, 286), 1)

        visible = max(1, (WINDOW_H - 538) // 24)
        rows = list(moves)[-visible:]
        y = 294
        for index, move in enumerate(rows):
            if index % 2 == 0:
                theme.rrect(
                    screen,
                    (BOARD_W + 8, y - 2, PANEL_W - 16, 23),
                    ROW_ALT,
                    5,
                )
            reason = str(move.get("reason", ""))[:11]
            line = (
                f"{move.get('no', 0):>3}  "
                f"{PLAYER_NAMES.get(move.get('color'), '?'):<2} "
                f"{move.get('cell', '?'):<4} {reason}"
            )
            self._text(screen, line, (BOARD_W + 16, y),
                       self.body_font, theme.TEXT_CLR)
            y += 24

        if not rows:
            self._text(screen, "尚无落子", (BOARD_W + 16, 298),
                       self.body_font, theme.DIM_CLR)
        if ai_explanation:
            self._draw_ai_explanation(screen, ai_explanation)
        elif search_stats:
            self._draw_mcts_chart(screen, search_stats)
        if ai_status:
            self._draw_status(screen, ai_status)

    def _draw_timers(self, screen, timers):
        red = timers.get(RED, "00:00")
        blue = timers.get(BLUE, "00:00")
        cards = (
            (pygame.Rect(BOARD_W + 12, 68, 108, 50),
             RED, self.red_color, red),
            (pygame.Rect(BOARD_W + 130, 68, 108, 50),
             BLUE, self.blue_color, blue),
        )
        for rect, color, accent, value in cards:
            theme.rrect(screen, rect, theme.SIDE_CARD, 9)
            pygame.draw.rect(screen, theme.SIDE_LINE, rect, 1, border_radius=9)
            pygame.draw.circle(screen, accent, (rect.x + 14, rect.y + 16), 5)
            self._text(screen, f"{PLAYER_NAMES[color]}方",
                       (rect.x + 25, rect.y + 8), self.body_font, theme.DIM_CLR)
            self._text(screen, value, (rect.x + 13, rect.y + 26),
                       self.body_font, theme.TEXT_CLR)

    def _draw_buttons(self, screen, replay_enabled, replay_mode,
                      undo_enabled=False, analysis_active=False,
                      analysis_available=True):
        mouse = pygame.mouse.get_pos()
        analysis_label = (
            "分析中…" if analysis_active
            else ("AI 分析" if analysis_available else "AI 未启用")
        )
        analysis_enabled = not replay_mode and analysis_available
        for rect, label, enabled in (
            (self.undo_button, "悔棋", undo_enabled),
            (self.save_button, "保存", True),
            (self.export_button, "报告", True),
            (self.replay_button, "返回" if replay_mode else "复盘", replay_enabled),
            (self.analysis_button, analysis_label, analysis_enabled),
        ):
            color = BUTTON_HOVER if enabled and rect.collidepoint(mouse) else BUTTON_BG
            if not enabled:
                color = (222, 216, 204)
            theme.rrect(screen, rect, color, 7)
            pygame.draw.rect(screen, theme.SIDE_LINE, rect, 1, border_radius=7)
            text_color = theme.TEXT_CLR if enabled else theme.FAINT_CLR
            surface = self.body_font.render(label, True, text_color)
            screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_mcts_chart(self, screen, search_stats):
        top = WINDOW_H - 285
        self._text(screen, "MCTS 候选统计", (BOARD_W + 16, top),
                   self.body_font, theme.GOLD_DEEP)
        max_visits = max((int(item.get("visits", 0)) for item in search_stats), default=1)
        max_visits = max(1, max_visits)
        y = top + 28
        for item in list(search_stats)[:5]:
            move = item.get("move")
            label = cell_name(*move) if move else "-"
            visits = int(item.get("visits", 0))
            self._text(screen, f"{label} {visits:>3}",
                       (BOARD_W + 16, y), self.body_font, theme.TEXT_CLR)
            theme.rrect(screen, (BOARD_W + 92, y + 3, 122, 12),
                        (216, 208, 192), 3)
            width = int(122 * visits / max_visits)
            if width:
                theme.rrect(screen, (BOARD_W + 92, y + 3, width, 12),
                            theme.GOLD_DEEP, 3)
            y += 24

    def _draw_ai_explanation(self, screen, explanation):
        top = WINDOW_H - 285
        self._text(screen, "AI \u51b3\u7b56\u5206\u6790", (BOARD_W + 16, top),
                   self.body_font, theme.GOLD_DEEP)
        selected = explanation.get("selected_move")
        tactic = str(explanation.get("tactic", ""))[:12]
        selected_text = cell_name(*selected) if selected else "-"
        self._text(screen, f"\u9009\u62e9 {selected_text} \u00b7 {tactic}",
                   (BOARD_W + 16, top + 26), self.body_font, theme.TEXT_CLR)
        score_type = str(explanation.get("score_type", "\u65e0\u8bc4\u5206"))[:18]
        elapsed = explanation.get("elapsed_ms")
        elapsed_text = f"{float(elapsed):.1f} ms" if elapsed is not None else "-"
        self._text(screen, f"{score_type} \u00b7 {elapsed_text}",
                   (BOARD_W + 16, top + 48), self.body_font, theme.DIM_CLR)
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, top + 70),
                         (BOARD_W + PANEL_W - 12, top + 70), 1)

        y = top + 78
        for item in list(explanation.get("candidates", ()))[:5]:
            move = item.get("move")
            label = cell_name(*move) if move else "-"
            visits = item.get("visits")
            score = item.get("score")
            if visits is not None:
                metric = f"{int(visits):>3}\u6b21"
            elif score is not None:
                metric = f"{float(score):>6.2f}"
            else:
                metric = "  -   "
            reason = str(item.get("reason", ""))[:6]
            self._text(screen, f"{label:<4} {metric} {reason}",
                       (BOARD_W + 16, y), self.body_font, theme.TEXT_CLR)
            y += 23

    def _draw_status(self, screen, status):
        status_y = WINDOW_H - 94
        pygame.draw.line(screen, PANEL_LINE, (BOARD_W + 12, status_y - 14),
                         (BOARD_W + PANEL_W - 12, status_y - 14), 1)
        self._text(screen, "AI 状态", (BOARD_W + 16, status_y),
                   self.body_font, theme.GOLD_DEEP)
        for line_no, part in enumerate(str(status).splitlines()[:3]):
            self._text(screen, part[:20],
                       (BOARD_W + 16, status_y + 24 + line_no * 20),
                       self.body_font, theme.DIM_CLR)

    @staticmethod
    def _text(screen, text, pos, font, color):
        if font is None:
            return
        screen.blit(font.render(str(text), True, color), pos)