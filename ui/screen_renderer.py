# -*- coding: utf-8 -*-
"""开始页、顶部栏和结束页等覆盖层绘制。"""
import math
import random

import pygame

from game_mode import GameMode
from hex_board import BLUE, RED, cell_name
from ui import theme
from ui.board_renderer import BOARD_W, WINDOW_H, hex_corners

CNAMES = {RED: "\u7ea2\u65b9", BLUE: "\u84dd\u65b9"}


def make_confetti(renderer):
    """生成一次获胜动画使用的彩纸参数。"""
    rng = random.Random(20261025)
    colors = [
        renderer.CC[RED], renderer.CL[RED], renderer.CC[BLUE],
        renderer.CL[BLUE], theme.GOLD, theme.SAGE,
    ]
    particles = []
    for _ in range(48):
        particles.append((
            rng.uniform(20, BOARD_W - 20),
            rng.uniform(-WINDOW_H * 0.7, -20),
            rng.uniform(-45, 45),
            rng.uniform(95, 210),
            rng.uniform(3, 7),
            rng.choice(colors),
            rng.uniform(0, math.tau),
            rng.uniform(2.0, 6.0),
        ))
    return particles


class OverlayRenderer:
    """绘制页面顶部栏、开始页、获胜页和本场结束页。"""

    def __init__(self, game):
        self.game = game

    def blit(self, text, pos, font, color, anchor="topleft"):
        surface = font.render(text, True, color)
        rect = surface.get_rect(**{anchor: pos})
        self.game.screen.blit(surface, rect)
        return rect

    @staticmethod
    def _pulse():
        return 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.004)

    def draw_hud(self):
        if self.game.state in (self.game.S_START, self.game.S_REPLAY):
            return
        self._draw_top_bar()
        self._draw_bottom_bar()

    def _draw_top_bar(self):
        screen = self.game.screen
        pygame.draw.rect(screen, theme.BAR, (0, 0, BOARD_W, 64))
        pygame.draw.line(screen, theme.BAR_LINE,
                         (0, 63), (BOARD_W, 63), 2)

        self.blit(f"第 {self.game.game_no} / 2 局", (16, 7),
                  self.game.font_s, theme.TEXT_CLR)
        p1, p2 = self.game.match_scores
        score = f"{self.game.player_label(1)}  {p1} : {p2}  {self.game.player_label(2)}"
        self.blit(score, (BOARD_W // 2, 8), self.game.font_m,
                  theme.GOLD_DEEP, anchor="midtop")

        mode_text = self.game.mode.label if self.game.mode else "未选择"
        red_name = self.game.player_label(self.game.red_player)
        self.blit(f"{mode_text} · {red_name} 执红先手", (16, 35),
                  self.game.font_xs, theme.DIM_CLR)

        if self.game.state != self.game.S_PLAY:
            return
        color = self.game.engine.current_color
        if color is None:
            return
        if self.game.mode == GameMode.AI_VS_AI and self.game.paused:
            text = "已暂停 · 空格继续"
            accent = theme.GOLD_DEEP
        else:
            owner = self.game.player_label(self.game.player_of(color))
            state_text = "行棋" if self.game.is_human_turn() else "AI 思考中"
            text = f"{CNAMES[color]} · {owner} {state_text}"
            accent = self.game.renderer.main_color(color)

        font = self.game.font_xs
        surface = font.render(text, True, theme.TEXT_CLR)
        width = surface.get_width() + 42
        rect = pygame.Rect(BOARD_W - 12 - width, 32, width, 28)
        theme.rrect(screen, rect, theme.CHIP, 14)
        pygame.draw.rect(screen, theme.CHIP_EDGE, rect, 1, border_radius=14)
        radius = 4 + int(round(self._pulse()))
        pygame.draw.circle(screen, accent, (rect.x + 15, rect.centery), radius)
        screen.blit(surface, surface.get_rect(midleft=(rect.x + 27, rect.centery)))
    def _draw_bottom_bar(self):
        screen = self.game.screen
        top = WINDOW_H - 38
        pygame.draw.rect(screen, theme.BAR, (0, top, BOARD_W, 38))
        pygame.draw.line(screen, theme.BAR_LINE, (0, top), (BOARD_W, top), 1)
        cy = top + 19

        if self.game.last_move:
            q, r, color, reason = self.game.last_move
            accent = self.game.renderer.main_color(color)
            pygame.draw.circle(screen, accent, (19, cy), 5)
            pygame.draw.circle(screen, self.game.renderer.CD[color], (19, cy), 5, 1)
            note = f"上一手 {CNAMES[color]} · {cell_name(q, r)}"
            if reason:
                note += f" · {str(reason)[:14]}"
            self.blit(note, (32, cy), self.game.font_s,
                      theme.TEXT_CLR, anchor="midleft")
        else:
            self.blit("红方先行，点击空位落子", (16, cy),
                      self.game.font_xs, theme.DIM_CLR, anchor="midleft")

        self.blit("C 配色 · Ctrl+S/E/P · F5 复盘", (BOARD_W - 14, cy),
                  self.game.font_xs, theme.DIM_CLR, anchor="midright")

    def draw_start_screen(self):
        screen = self.game.screen
        screen.blit(theme.veil_surface(*screen.get_size(), alpha=206), (0, 0))
        elapsed = pygame.time.get_ticks() - self.game.intro_at

        def ramp(start, duration=380):
            value = min(1.0, max(0.0, (elapsed - start) / duration))
            return 1 - (1 - value) ** 3

        decoration = pygame.Surface((BOARD_W, WINDOW_H), pygame.SRCALPHA)
        for cx, cy, size, color, alpha in (
            (58, 110, 42, self.game.renderer.CC[1], 70),
            (BOARD_W - 58, 155, 55, self.game.renderer.CC[2], 58),
            (72, WINDOW_H - 150, 52, theme.GOLD, 58),
            (BOARD_W - 88, WINDOW_H - 225, 40, theme.SAGE, 64),
            (135, 245, 10, theme.GOLD_DEEP, 70),
        ):
            pygame.draw.polygon(decoration, (*color, alpha),
                                hex_corners(cx, cy, size))
        decoration.set_alpha(int(255 * ramp(0, 500)))
        screen.blit(decoration, (0, 0))

        center_x = BOARD_W // 2
        amount = ramp(120, 520)
        if amount:
            title = self.game.font_xl.render("海克斯棋", True, theme.TEXT_CLR)
            title.set_alpha(int(255 * amount))
            screen.blit(title, title.get_rect(center=(center_x, 145 + (1 - amount) * 14)))
            pygame.draw.polygon(screen, theme.GOLD_DEEP,
                                hex_corners(center_x, 75, 26), 3)
            pygame.draw.polygon(screen, theme.GOLD,
                                hex_corners(center_x, 75, 15))
            pygame.draw.polygon(screen, theme.PANEL,
                                hex_corners(center_x, 75, 7))

        amount = ramp(650, 320)
        if amount:
            subtitle = self.game.font_xs.render("H E X   ·   连 路 即 胜", True, theme.DIM_CLR)
            subtitle.set_alpha(int(255 * amount))
            screen.blit(subtitle, subtitle.get_rect(center=(center_x, 205)))

        rules = (
            (self.game.renderer.CC[1], "红方连通上下两条红色边界获胜"),
            (self.game.renderer.CC[2], "蓝方连通左右两条蓝色边界获胜"),
            (theme.GOLD_DEEP, "每场两局 · 双方交换红蓝 · 红方每局先行"),
            (theme.SAGE, "坐标先纵后横：5C 表示第 5 行第 C 列"),
        )
        y = 258
        for index, (color, text) in enumerate(rules):
            amount = ramp(760 + index * 90, 300)
            if not amount:
                continue
            surface = self.game.font_s.render(text, True, theme.DIM_CLR)
            surface.set_alpha(int(255 * amount))
            rect = surface.get_rect(midleft=(center_x - 185, y + (1 - amount) * 7))
            pygame.draw.circle(screen, color, (rect.left - 19, rect.centery), 5)
            screen.blit(surface, rect)
            y += 31

        self.game.mode_buttons = []
        mouse = pygame.mouse.get_pos()
        card_x = center_x - 220
        y = 410
        for index, mode in enumerate(sorted(GameMode)):
            rect = pygame.Rect(card_x, y, 440, 66)
            self.game.mode_buttons.append((rect, mode))
            amount = ramp(1010 + index * 90, 360)
            if not amount:
                y += 76
                continue
            hover = rect.collidepoint(mouse)
            card = pygame.Surface((464, 88), pygame.SRCALPHA)
            body = pygame.Rect(12, 8, 440, 66)
            theme.rrect(card, body, theme.PANEL if not hover else theme.PANEL_HI, 12)
            pygame.draw.rect(card, theme.GOLD_DEEP if hover else theme.CHIP_EDGE,
                             body, 2 if hover else 1, border_radius=12)
            badge = pygame.Rect(28, 24, 34, 34)
            theme.rrect(card, badge, theme.GOLD if hover else theme.CHIP, 9)
            number = self.game.font_m.render(str(mode.value), True, theme.TEXT_CLR)
            card.blit(number, number.get_rect(center=badge.center))
            title = self.game.font_m.render(mode.label, True, theme.TEXT_CLR)
            card.blit(title, title.get_rect(midleft=(80, 30)))
            desc = self.game.font_xs.render(mode.description, True, theme.DIM_CLR)
            card.blit(desc, desc.get_rect(midleft=(80, 56)))
            card.set_alpha(int(255 * amount))
            offset = (1 - amount) * 18 - (3 if hover else 0)
            screen.blit(card, (card_x - 12, y - 8 + offset))
            y += 76

        amount = ramp(1460, 360)
        if amount:
            hint = self.game.font_xs.render(
                "点击卡片或按 1–4 选择模式      C 切换色盲友好配色      Esc 退出",
                True,
                theme.DIM_CLR,
            )
            hint.set_alpha(int(255 * amount))
            screen.blit(hint, hint.get_rect(center=(center_x, 768)))

    def draw_win_banner(self):
        game = self.game
        screen = game.screen
        screen.blit(theme.veil_surface(*screen.get_size(), alpha=196), (0, 0))
        elapsed = pygame.time.get_ticks() - (game.win_at or 0)
        self._draw_confetti(elapsed)

        winner = game.winner
        accent = game.renderer.main_color(winner)
        rect = pygame.Rect(0, 0, 500, 300)
        rect.center = (BOARD_W // 2, WINDOW_H // 2 - 8)
        panel = theme.panel_surface(rect, accent)
        screen.blit(panel, (rect.x - 14, rect.y - 14))

        cx = rect.centerx
        medal_y = rect.y + 53
        pygame.draw.polygon(screen, accent, [
            (cx - 16, medal_y - 29), (cx - 5, medal_y - 10), (cx - 12, medal_y - 5)
        ])
        pygame.draw.polygon(screen, accent, [
            (cx + 16, medal_y - 29), (cx + 5, medal_y - 10), (cx + 12, medal_y - 5)
        ])
        pygame.draw.circle(screen, theme.GOLD_DEEP, (cx, medal_y), 17)
        pygame.draw.circle(screen, theme.GOLD, (cx, medal_y), 14)
        pygame.draw.polygon(screen, accent, hex_corners(cx, medal_y, 7))

        self.blit(f"第 {game.game_no} 局 · GAME OVER", (cx, rect.y + 84),
                  game.font_xs, theme.GOLD_DEEP, anchor="center")
        self._glow_text(f"{CNAMES[winner]}获胜", game.font_l, accent,
                        (cx, rect.y + 126))
        how = "连通了上下红色边界" if winner == RED else "连通了左右蓝色边界"
        who = game.player_label(game.player_of(winner))
        self.blit(f"{who} · {how}", (cx, rect.y + 166),
                  game.font_s, theme.TEXT_CLR, anchor="center")
        pygame.draw.line(screen, theme.CHIP_EDGE,
                         (cx - 150, rect.y + 193), (cx + 150, rect.y + 193))
        self.blit("点击或按空格进入下一局 · 双方交换红蓝",
                  (cx, rect.y + 219), game.font_xs,
                  theme.DIM_CLR, anchor="center")
        p1, p2 = game.match_scores
        self.blit(f"当前比分  {p1} : {p2}", (cx, rect.y + 254),
                  game.font_xs, theme.FAINT_CLR, anchor="center")

    def draw_match_end(self):
        game = self.game
        screen = game.screen
        screen.blit(theme.veil_surface(*screen.get_size(), alpha=200), (0, 0))
        rect = pygame.Rect(0, 0, 520, 320)
        rect.center = (BOARD_W // 2, WINDOW_H // 2 - 8)
        panel = theme.panel_surface(rect, theme.GOLD_DEEP)
        screen.blit(panel, (rect.x - 14, rect.y - 14))

        cx = rect.centerx
        self._glow_text("本场结束", game.font_l, theme.TEXT_CLR,
                        (cx, rect.y + 58))
        p1, p2 = game.match_scores
        name1 = game.player_label(1)
        name2 = game.player_label(2)
        self.blit(f"{name1}   {p1}  :  {p2}   {name2}",
                  (cx, rect.y + 125), game.font_m,
                  theme.GOLD_DEEP, anchor="center")

        if p1 == p2:
            verdict = "两局 1 : 1 战平 · 单局无和棋，总分可平"
            verdict_color = theme.TEXT_CLR
        else:
            winner = 1 if p1 > p2 else 2
            verdict = f"{game.player_label(winner)} 本场获胜"
            verdict_color = theme.GOLD_DEEP
        self.blit(verdict, (cx, rect.y + 178), game.font_m,
                  verdict_color, anchor="center")
        pygame.draw.line(screen, theme.CHIP_EDGE,
                         (cx - 160, rect.y + 213), (cx + 160, rect.y + 213))
        self.blit("点击或按空格再来一场", (cx, rect.y + 242),
                  game.font_s, theme.TEXT_CLR, anchor="center")
        self.blit("Esc 退出", (cx, rect.y + 278),
                  game.font_xs, theme.DIM_CLR, anchor="center")

    def draw_replay_banner(self):
        game = self.game
        if game.replay is None:
            return
        cx = BOARD_W // 2
        text = f"复盘第 {game.replay.game_no} 局 · {game.replay.step}/{game.replay.total_moves} 手"
        surface = game.font_m.render(text, True, theme.GOLD_DEEP)
        rect = surface.get_rect(center=(cx, 104))
        panel = pygame.Surface((rect.width + 36, rect.height + 18), pygame.SRCALPHA)
        panel.fill((252, 248, 238, 232))
        pygame.draw.rect(panel, theme.CHIP_EDGE, panel.get_rect(),
                         1, border_radius=8)
        game.screen.blit(panel, panel.get_rect(center=rect.center))
        game.screen.blit(surface, rect)

    def _draw_confetti(self, elapsed_ms):
        game = self.game
        if not game.confetti:
            return
        elapsed = max(0.0, elapsed_ms / 1000.0)
        for x0, y0, vx, vy, size, color, phase, spin in game.confetti:
            x = x0 + vx * elapsed + math.sin(elapsed * 3 + phase) * 18
            y = y0 + vy * elapsed + 90 * elapsed * elapsed
            if y < -20 or y > WINDOW_H + 20:
                continue
            angle = phase + spin * elapsed
            dx = math.cos(angle) * size
            dy = math.sin(angle) * size
            pygame.draw.line(game.screen, color,
                             (x - dx, y - dy), (x + dx, y + dy), 2)
            pygame.draw.circle(game.screen, color, (int(x), int(y)), max(1, int(size / 3)))

    def _glow_text(self, text, font, color, center):
        screen = self.game.screen
        base = font.render(text, True, color)
        halo = font.render(text, True, theme.GOLD_DEEP)
        halo = pygame.transform.smoothscale(
            halo, (max(1, halo.get_width() // 2), max(1, halo.get_height() // 2))
        )
        halo = pygame.transform.smoothscale(
            halo, (base.get_width() + 16, base.get_height() + 16)
        )
        halo.set_alpha(42)
        screen.blit(halo, halo.get_rect(center=center))
        rect = base.get_rect(center=center)
        screen.blit(base, rect)
        return rect
