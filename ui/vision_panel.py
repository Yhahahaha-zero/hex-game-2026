# -*- coding: utf-8 -*-
"""Modal for previewing a read-only imported board screenshot."""
from __future__ import annotations

import pygame

from hex_board import cell_name
from ui import theme
from ui.board_renderer import BOARD_W, WINDOW_H
from ui.history_panel import PANEL_W


WINDOW_W = BOARD_W + PANEL_W


class VisionPanel:
    """Preview recognized board state; never writes into the live match."""

    def __init__(self):
        self.visible = False
        self.title_font = None
        self.body_font = None
        self.small_font = None
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.close_button = pygame.Rect(0, 0, 0, 0)
        self.reimport_button = pygame.Rect(0, 0, 0, 0)
        self.save_button = pygame.Rect(0, 0, 0, 0)
        self.session = None
        self.status = "idle"
        self.error = ""
        self._image_surface = None

    def set_fonts(self, title_font, body_font, small_font):
        self.title_font = title_font
        self.body_font = body_font
        self.small_font = small_font

    def show(self, *, status="idle", session=None, error=""):
        self.visible = True
        self.status = str(status)
        self.session = session
        self.error = str(error or "")
        self._image_surface = None
        if session is not None and getattr(session, "normalized_image", None) is not None:
            try:
                image = session.normalized_image.convert("RGB")
                self._image_surface = pygame.image.fromstring(
                    image.tobytes(), image.size, "RGB"
                )
            except Exception:
                self._image_surface = None

    def hide(self):
        self.visible = False

    def handle_click(self, pos):
        if not self.visible:
            return None
        if self.close_button.collidepoint(pos):
            self.hide()
            return "close"
        if self.reimport_button.collidepoint(pos):
            return "reimport"
        if self.save_button.collidepoint(pos):
            state = getattr(self.session, "state", None)
            if state is not None and state.valid and not state.uncertain_cells:
                return "save"
        return None

    def draw(self, screen):
        self.visible = True
        screen.blit(theme.veil_surface(*screen.get_size(), alpha=228), (0, 0))
        width = min(780, WINDOW_W - 72)
        height = min(820, WINDOW_H - 64)
        self.panel_rect = pygame.Rect(0, 0, width, height)
        self.panel_rect.center = (WINDOW_W // 2, WINDOW_H // 2)
        panel = theme.panel_surface(self.panel_rect, theme.GOLD_DEEP)
        screen.blit(panel, (self.panel_rect.x - 14, self.panel_rect.y - 14))
        rect = self.panel_rect
        left = rect.x + 28
        right = rect.right - 28
        self.close_button = pygame.Rect(right - 88, rect.y + 18, 88, 36)
        self.reimport_button = pygame.Rect(right - 276, rect.y + 18, 176, 36)
        self.save_button = pygame.Rect(right - 180, rect.bottom - 72, 180, 38)

        self._text(screen, "视觉棋盘导入", (left, rect.y + 18),
                   self.title_font, theme.TEXT_CLR)
        self._button(screen, self.close_button, "关闭", accent=False)
        self._button(screen, self.reimport_button, "重新选择图片", accent=False)

        if self.status == "processing":
            self._text(screen, "正在识别棋盘…", (left, rect.y + 100),
                       self.title_font, theme.GOLD_DEEP)
            self._draw_lines(
                screen,
                "图片在后台线程中处理，识别期间游戏仍可响应。",
                (left, rect.y + 150), right - left, self.body_font,
                theme.DIM_CLR, 3,
            )
            return
        if self.status == "error":
            self._text(screen, "视觉导入失败", (left, rect.y + 100),
                       self.title_font, theme.RED_DARK)
            self._draw_lines(screen, self.error, (left, rect.y + 145),
                             right - left, self.body_font, theme.TEXT_CLR, 10)
            return
        session = self.session
        state = getattr(session, "state", None)
        if state is None:
            self._text(screen, "没有可显示的结果", (left, rect.y + 100),
                       self.title_font, theme.DIM_CLR)
            return

        red_count = sum(cell.color == "RED" for cell in state.cells)
        blue_count = sum(cell.color == "BLUE" for cell in state.cells)
        self._text(
            screen,
            f"红方 {red_count} · 蓝方 {blue_count} · "
            f"未确定 {len(state.uncertain_cells)} · {state.elapsed_ms:.1f} ms",
            (left, rect.y + 62), self.small_font, theme.DIM_CLR,
        )

        preview_left = left
        preview_top = rect.y + 100
        preview_w = 330
        preview_h = rect.bottom - preview_top - 92
        pygame.draw.rect(screen, theme.CHIP,
                         (preview_left, preview_top, preview_w, preview_h),
                         border_radius=10)
        if self._image_surface is not None:
            image = self._image_surface
            inner_pad = 20
            scale = min((preview_w - inner_pad * 2) / image.get_width(),
                        (preview_h - inner_pad * 2) / image.get_height())
            target = pygame.transform.smoothscale(
                image,
                (max(1, int(image.get_width() * scale)),
                 max(1, int(image.get_height() * scale))),
            )
            screen.blit(target, target.get_rect(center=(
                preview_left + preview_w // 2,
                preview_top + preview_h // 2,
            )))

        info_left = preview_left + preview_w + 24
        info_width = right - info_left
        y = preview_top
        status_text = "校验通过" if state.valid else "需要人工检查"
        status_color = theme.TEXT_CLR if state.valid else theme.RED_DARK
        self._text(screen, status_text, (info_left, y),
                   self.body_font, status_color)
        y += 30
        self._text(screen, "识别棋盘", (info_left, y),
                   self.body_font, theme.GOLD_DEEP)
        y += 24
        self._text(screen, "R=红  B=蓝  .=空", (info_left, y),
                   self.small_font, theme.TEXT_CLR)
        y += 22
        for index, row in enumerate(state.board_rows()):
            if y > rect.bottom - 120:
                break
            self._text(screen, f"{index + 1:>2}  {row}",
                       (info_left, y), self.small_font, theme.TEXT_CLR)
            y += 19
        y += 8
        if state.uncertain_cells:
            uncertain_text = ", ".join(
                cell_name(q, r) for q, r in state.uncertain_cells[:8]
            )
            self._text(screen, f"未确定格子：{uncertain_text}",
                       (info_left, y), self.small_font, theme.RED_DARK)
            y += 24
        if state.warnings:
            self._text(screen, "警告", (info_left, y),
                       self.body_font, theme.GOLD_DEEP)
            y += 24
            for warning in state.warnings[:4]:
                y = self._draw_lines(
                    screen, warning, (info_left, y), info_width,
                    self.small_font, theme.DIM_CLR, 2,
                )
        note = (
            "确认只保存为只读局面快照，不会覆盖当前正式棋局。"
        )
        self._draw_lines(
            screen, note, (left, rect.bottom - 112),
            right - left - 210, self.small_font, theme.TEXT_CLR, 2,
        )
        save_enabled = state.valid and not state.uncertain_cells
        self._button(screen, self.save_button, "保存快照", accent=save_enabled,
                     enabled=save_enabled)

    def _button(self, screen, rect, label, *, accent=False, enabled=True):
        fill = theme.GOLD_DEEP if accent else theme.SIDE_BUTTON
        if not enabled:
            fill = (222, 216, 204)
        theme.rrect(screen, rect, fill, 8)
        pygame.draw.rect(screen, theme.SIDE_LINE, rect, 1, border_radius=8)
        if self.body_font:
            color = (theme.WHITE if accent else theme.TEXT_CLR) if enabled else theme.FAINT_CLR
            surface = self.body_font.render(label, True, color)
            screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_lines(self, screen, text, start, width, font, color, max_lines):
        y = start[1]
        for line in self._wrap(text, font, width, max_lines):
            self._text(screen, line, (start[0], y), font, color)
            y += (font.get_linesize() + 3) if font else 18
        return y

    @staticmethod
    def _wrap(text, font, width, max_lines):
        if font is None:
            return []
        lines = []
        for paragraph in str(text).splitlines() or [""]:
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
        return lines[:max_lines]

    @staticmethod
    def _text(screen, text, pos, font, color):
        if font is None:
            return
        screen.blit(font.render(str(text), True, color), pos)