# -*- coding: utf-8 -*-
"""Modal panel for displaying structured LLM analysis."""
from __future__ import annotations

import pygame

from hex_board import cell_name
from ui import theme
from ui.board_renderer import BOARD_W, WINDOW_H
from ui.history_panel import PANEL_W


WINDOW_W = BOARD_W + PANEL_W
PRIORITY_LABELS = {
    "win": "立即取胜",
    "block_win": "阻止获胜",
    "attack": "进攻",
    "defense": "防守",
    "connect": "连通",
    "develop": "发展",
    "balanced": "均衡",
}
PREFERENCE_LABELS = {
    "prefer": "优先",
    "neutral": "中性",
    "avoid": "回避",
}
SEVERITY_LABELS = {
    "immediate": "立即",
    "high": "高",
    "medium": "中",
}
STATUS_LABELS = {
    "idle": "待分析",
    "loading": "分析中",
    "complete": "分析完成",
    "error": "分析失败",
    "disabled": "AI 未启用",
    "stale": "结果已过期",
}


class AnalysisPanel:
    """Draw a modal analysis result and expose close/reanalyse actions."""

    def __init__(self):
        self.visible = False
        self.title_font = None
        self.body_font = None
        self.small_font = None
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.close_button = pygame.Rect(0, 0, 0, 0)
        self.retry_button = pygame.Rect(0, 0, 0, 0)
        self._status = "idle"
        self._analysis = None
        self._error = ""
        self._model_name = ""
        self._elapsed_ms = None
        self._position_id = ""

    def set_fonts(self, title_font, body_font, small_font):
        self.title_font = title_font
        self.body_font = body_font
        self.small_font = small_font

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def handle_click(self, pos):
        if not self.visible:
            return None
        if self.close_button.collidepoint(pos):
            self.hide()
            return "close"
        if self.retry_button.collidepoint(pos):
            return "retry"
        return None

    def draw(self, screen, *, status, analysis=None, error="",
             model_name="", elapsed_ms=None, position_id=""):
        self._status = str(status)
        self._analysis = analysis
        self._error = str(error or "")
        self._model_name = str(model_name or "")
        self._elapsed_ms = elapsed_ms
        self._position_id = str(position_id or "")
        self.visible = True

        screen.blit(theme.veil_surface(*screen.get_size(), alpha=178), (0, 0))
        width = min(760, WINDOW_W - 80)
        height = min(800, WINDOW_H - 72)
        self.panel_rect = pygame.Rect(0, 0, width, height)
        self.panel_rect.center = (WINDOW_W // 2, WINDOW_H // 2)
        panel = theme.panel_surface(self.panel_rect, theme.GOLD_DEEP)
        screen.blit(panel, (self.panel_rect.x - 14, self.panel_rect.y - 14))

        rect = self.panel_rect
        left = rect.x + 28
        right = rect.right - 28
        content_width = right - left
        self.close_button = pygame.Rect(right - 78, rect.y + 18, 78, 36)
        self.retry_button = pygame.Rect(right - 240, rect.y + 18, 152, 36)

        self._text(screen, "AI 棋局分析", (left, rect.y + 18),
                   self.title_font, theme.TEXT_CLR)
        status_label = STATUS_LABELS.get(self._status, self._status)
        self._status_chip(screen, (left, rect.y + 58), status_label)

        meta = []
        if self._model_name:
            meta.append(f"模型 {self._model_name}")
        if self._elapsed_ms is not None:
            meta.append(f"耗时 {float(self._elapsed_ms):.1f} ms")
        if self._position_id:
            meta.append(f"局面 {self._position_id[-8:]}")
        if meta:
            self._text(screen, " · ".join(meta), (left + 104, rect.y + 62),
                       self.small_font, theme.DIM_CLR)

        self._button(screen, self.close_button, "关闭")
        if self._status != "loading":
            self._button(screen, self.retry_button, "重新分析", accent=True)

        top = rect.y + 108
        bottom = rect.bottom - 42
        if self._status == "loading":
            self.retry_button = pygame.Rect(0, 0, 0, 0)
            dots = "." * (1 + (pygame.time.get_ticks() // 450) % 3)
            self._text(screen, f"正在请求大模型分析{dots}",
                       (left, top + 42), self.title_font, theme.GOLD_DEEP)
            self._draw_lines(
                screen,
                "分析在后台线程中执行，游戏计时、落子和界面保持响应。",
                (left, top + 88), content_width, self.body_font,
                theme.DIM_CLR, 3,
            )
            return
        if self._status in ("disabled", "error", "stale"):
            title = {
                "disabled": "大模型功能未启用",
                "stale": "分析结果已过期",
            }.get(self._status, "分析失败")
            self._text(screen, title, (left, top), self.title_font,
                       theme.GOLD_DEEP if self._status != "error" else theme.RED_DARK)
            detail = self._error or "分析暂不可用，正式对局不会受到影响。"
            self._draw_lines(screen, detail, (left, top + 44),
                             content_width, self.body_font, theme.TEXT_CLR, 8)
            self._text(screen, "Esc 或“关闭”返回棋盘", (left, bottom - 6),
                       self.small_font, theme.DIM_CLR)
            return

        data = analysis.as_dict() if hasattr(analysis, "as_dict") else analysis
        if not isinstance(data, dict):
            self._text(screen, "没有可显示的分析结果", (left, top),
                       self.title_font, theme.DIM_CLR)
            return

        summary = str(data.get("summary", ""))
        priority = PRIORITY_LABELS.get(data.get("priority"), data.get("priority", ""))
        self._text(screen, f"局势摘要 · {priority}", (left, top),
                   self.body_font, theme.GOLD_DEEP)
        y = top + 30
        y = self._draw_lines(
            screen, summary, (left, y), content_width,
            self.body_font, theme.TEXT_CLR, 4,
        )
        y += 14

        threats = list(data.get("threats", ()))
        if threats and y < bottom - 150:
            self._text(screen, "关键威胁", (left, y), self.body_font, theme.GOLD_DEEP)
            y += 6
            y += 26
            for item in threats[:4]:
                move = item.get("move")
                label = cell_name(*move) if move else "-"
                severity = SEVERITY_LABELS.get(item.get("severity"), item.get("severity", ""))
                reason = str(item.get("reason", "")).strip()
                text = f"{label} · {severity} · {reason}"
                y = self._draw_lines(
                    screen, text, (left, y), content_width,
                    self.small_font, theme.TEXT_CLR, 2,
                )
                if y >= bottom - 130:
                    break
            y += 10

        candidates = list(data.get("candidates", ()))
        if candidates and y < bottom - 105:
            self._text(screen, "候选分析", (left, y), self.body_font, theme.GOLD_DEEP)
            y += 6
            y += 26
            for item in candidates[:4]:
                move = item.get("move")
                label = cell_name(*move) if move else "-"
                pref = PREFERENCE_LABELS.get(
                    item.get("preference"), item.get("preference", "")
                )
                pros = str(item.get("pros", "")).strip()
                cons = str(item.get("cons", "")).strip()
                text = f"{label} · {pref} · 优点：{pros}  缺点：{cons}"
                y = self._draw_lines(
                    screen, text, (left, y), content_width,
                    self.small_font, theme.TEXT_CLR, 2,
                )
                if y >= bottom - 80:
                    break
            y += 8

        warnings = list(data.get("warnings", ()))
        if warnings and y < bottom - 45:
            self._text(screen, "提示", (left, y), self.body_font, theme.GOLD_DEEP)
            y += 24
            for warning in warnings[:2]:
                y = self._draw_lines(
                    screen, str(warning), (left, y), content_width,
                    self.small_font, theme.DIM_CLR, 2,
                )
                if y >= bottom - 28:
                    break

        confidence = data.get("subjective_confidence")
        footer = "主观置信度不是胜率，仅用于参考。"
        if confidence is not None:
            footer = f"主观置信度 {float(confidence):.2f}，不是胜率。"
        self._text(screen, footer, (left, rect.bottom - 36),
                   self.small_font, theme.DIM_CLR)
        return

    def _status_chip(self, screen, pos, label):
        font = self.small_font
        if font is None:
            return
        width = max(82, font.size(label)[0] + 28)
        rect = pygame.Rect(pos[0], pos[1] - 3, width, 28)
        theme.rrect(screen, rect, theme.CHIP, 9)
        pygame.draw.rect(screen, theme.CHIP_EDGE, rect, 1, border_radius=9)
        surface = font.render(label, True, theme.GOLD_DEEP)
        screen.blit(surface, surface.get_rect(center=rect.center))

    def _button(self, screen, rect, label, accent=False):
        fill = theme.GOLD if accent else theme.SIDE_BUTTON
        border = theme.GOLD_DEEP if accent else theme.SIDE_LINE
        theme.rrect(screen, rect, fill, 8)
        pygame.draw.rect(screen, border, rect, 1, border_radius=8)
        if self.body_font:
            surface = self.body_font.render(label, True, theme.TEXT_CLR)
            screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_lines(self, screen, text, start, width, font, color, max_lines):
        lines = self._wrap(text, font, width, max_lines)
        y = start[1]
        if font:
            line_height = font.get_linesize() + 3
            for line in lines:
                self._text(screen, line, (start[0], y), font, color)
                y += line_height
        return y

    @staticmethod
    def _wrap(text, font, width, max_lines):
        if font is None:
            return []
        chars = str(text).replace("\r", "").splitlines() or [""]
        lines = []
        for paragraph in chars:
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for char in paragraph:
                candidate = current + char
                if current and font.size(candidate)[0] > width:
                    lines.append(current)
                    current = char
                    if len(lines) >= max_lines:
                        break
                else:
                    current = candidate
            if len(lines) >= max_lines:
                break
            if current:
                lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        if lines and len("".join(lines)) < len(str(text).replace("\n", "")):
            line = lines[-1]
            lines[-1] = (line[:-1] if len(line) > 1 else "") + "…"
        return lines

    @staticmethod
    def _text(screen, text, pos, font, color):
        if font is None:
            return
        screen.blit(font.render(str(text), True, color), pos)