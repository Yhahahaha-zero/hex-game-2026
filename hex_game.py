# -*- coding: utf-8 -*-
"""海克斯棋 pygame 界面：只负责输入、协调和绘制。"""
import os

import pygame

from ai.worker import AIWorker
from hex_board import BLUE, COLS, EMPTY, IllegalMoveError, RED, ROWS, cell_name
from hex_engine import (HexEngine, STATE_GAME_OVER, STATE_MATCH_OVER,
                        STATE_PLAYING)
from ui.board_renderer import (BG, BLUE_CLR, BOARD_W, DIM_CLR, GOLD,
                               PIECE_COLORS, RED_CLR, TEXT_CLR, WHITE,
                               WINDOW_H, BoardRenderer, cell_center,
                               hex_corners, pixel_to_hex)
from ui.history_panel import HistoryPanel, PANEL_W


S_START, S_PLAY, S_WIN, S_MATCH_END = range(4)
WINDOW_W = BOARD_W + PANEL_W
MODES = {
    1: ("双人对战", "两名玩家轮流点击落子"),
    2: ("人机对战 · 玩家执红", "玩家先手"),
    3: ("人机对战 · 玩家执蓝", "AI 先手"),
    4: ("AI 对 AI 观战", "空格暂停 / 继续"),
}
AI_DELAY_MS = 350
WATCH_AUTO_MS = 1600
CNAMES = {RED: "红方", BLUE: "蓝方"}
CCOLORS = PIECE_COLORS
FONT_DIR = r"C:\Windows\Fonts"
FONT_NAMES = {
    False: ("msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
    True: ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
}


def get_font(size, bold=False):
    """查找支持中文的系统字体。"""
    if os.name == "nt":
        for name in FONT_NAMES[bold]:
            path = os.path.join(FONT_DIR, name)
            if os.path.isfile(path):
                try:
                    return pygame.font.Font(path, size)
                except pygame.error:
                    continue
    else:
        for name in ("microsoftyahei", "msyh", "simhei", "dengxian", "simsun", None):
            if name is None:
                break
            try:
                path = pygame.font.match_font(name, bold=bold)
            except Exception:
                continue
            if path:
                return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class HexGame:
    """界面控制器：规则和比赛状态全部委托给 HexEngine。"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("海克斯棋 Hex —— 红连上下，蓝连左右")
        self.clock = pygame.time.Clock()
        self.font_xs = get_font(15)
        self.font_s = get_font(18)
        self.font_m = get_font(22)
        self.font_l = get_font(40, bold=True)
        self.renderer = BoardRenderer(self.font_s)
        self.history_panel = HistoryPanel()
        self.history_panel.set_fonts(self.font_m, self.font_xs)
        self.mode = 0
        self.human = None
        self.ai = None
        self.ai_status_message = ""
        self.ai_request_id = 0
        self.ai_worker = None
        self.mode_buttons = []
        self.engine = HexEngine("玩家1", "玩家2")
        self.reset_match(show_intro=True)

    # ---------- 引擎同步 ----------
    def _sync_engine(self):
        self.board = self.engine.board
        self.game_no = self.engine.game_no
        self.red_player = 1 if self.engine.red_side == "玩家1" else 2
        scores = self.engine.match_score
        self.match_scores = [scores["玩家1"], scores["玩家2"]]
        self.turn = self.engine.current_color or self.engine.winner_color
        self.winner = self.engine.winner_color
        self.win_path = self.engine.win_path
        moves = self.engine.game_moves()
        if moves:
            move = moves[-1]
            self.last_move = (move["q"], move["r"], move["color"], move["reason"])
        else:
            self.last_move = None

    def _reset_transient_state(self):
        self.hover = None
        self.win_at = None
        self.paused = False
        self.pending_move = None
        self.pending_reason = None
        self.ai_act_at = 0
        self.ai_request_id += 1
        self.ai_worker = None

    def reset_match(self, show_intro=False):
        """开始一场新比赛，保留当前模式。"""
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.state = S_START if show_intro else S_PLAY

    def new_game(self):
        """兼容接口：重开第一局。"""
        self.engine.start_match()
        self._reset_transient_state()
        self._sync_engine()
        self.state = S_PLAY

    def start_mode(self, mode):
        self.mode = mode
        self.human = {2: 1, 3: 2}.get(mode)
        self.ai = None
        self.ai_status_message = ""
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.state = S_PLAY

    def player_of(self, color):
        return 1 if self.engine.side_of(color) == "玩家1" else 2

    def player_label(self, pid):
        if self.mode == 1 or self.human == pid:
            return f"玩家{pid}"
        if self.mode == 4:
            return f"AI{pid}"
        return "AI"

    def is_human_turn(self):
        if self.engine.state != STATE_PLAYING:
            return False
        if self.mode == 1:
            return True
        if self.mode == 4:
            return False
        return self.player_of(self.engine.current_color) == self.human

    def advance_match(self):
        if self.engine.state != STATE_GAME_OVER:
            return
        self.engine.next_game()
        self._reset_transient_state()
        self._sync_engine()
        self.state = S_MATCH_END if self.engine.state == STATE_MATCH_OVER else S_PLAY

    # ---------- 输入 ----------
    def pick(self, pos):
        q, r = pixel_to_hex(*pos)
        return (q, r) if self.board.in_range(q, r) else None

    def on_click(self, pos):
        if self.state == S_START:
            for rect, mode in self.mode_buttons:
                if rect.collidepoint(pos):
                    self.start_mode(mode)
                    return
        elif self.state == S_WIN:
            self.advance_match()
        elif self.state == S_MATCH_END:
            self.reset_match()
        elif self.state == S_PLAY:
            if not self.is_human_turn():
                return
            cell = self.pick(pos)
            if cell is None or not self.board.is_empty(*cell):
                return
            self.apply_move(*cell)

    def on_space(self):
        if self.state == S_START:
            self.start_mode(1)
        elif self.state == S_WIN:
            self.advance_match()
        elif self.state == S_MATCH_END:
            self.reset_match()
        elif self.state == S_PLAY and self.mode == 4:
            self.paused = not self.paused

    def apply_move(self, q, r, reason=None):
        """通过 HexEngine 完成落子、换手、判胜和计分。"""
        if self.engine.state != STATE_PLAYING:
            return False
        color = self.engine.current_color
        try:
            won = self.engine.play_move(q, r, reason=reason)
        except IllegalMoveError:
            return False
        self.last_move = (q, r, color, reason)
        self._sync_engine()
        if won or self.engine.state == STATE_GAME_OVER:
            self.state = S_WIN
            self.win_at = pygame.time.get_ticks()
            self.pending_move = None
            self.pending_reason = None
            self.ai_worker = None
        else:
            self.state = S_PLAY
        return won

    # ---------- AI ----------
    def _consume_ai_result(self, result):
        if result is None:
            return
        if (result.request_id != self.ai_request_id or self.state != S_PLAY
                or self.engine.current_color != result.color):
            return
        if result.ai is not None:
            self.ai = result.ai
        if result.init_error:
            print(f"[AI] {result.init_error}")
            self.ai_status_message = "模型不可用，已回退基础 AI"
        if result.inference_error:
            print(f"[AI] 推理失败: {result.inference_error}")
            self.ai_status_message = "模型推理失败，已回退基础 AI"
        if result.fallback_error:
            print(f"[AI] 基础 AI 回退失败: {result.fallback_error}")
            self.ai_status_message = "AI 执行失败"
            return
        if result.move is None:
            return
        self.pending_move = result.move
        self.pending_reason = result.reason
        self.ai_act_at = pygame.time.get_ticks() + AI_DELAY_MS

    def update_ai(self):
        """后台计算 AI 着法，主线程只消费结果并落子。"""
        if self.state != S_PLAY or self.is_human_turn():
            return
        now = pygame.time.get_ticks()
        if self.pending_move is not None:
            if now >= self.ai_act_at:
                q, r = self.pending_move
                reason, self.pending_move = self.pending_reason, None
                self.apply_move(q, r, reason)
            return
        if self.ai_worker is not None:
            if self.ai_worker.is_alive():
                return
            worker, self.ai_worker = self.ai_worker, None
            if worker.result is not None:
                self._consume_ai_result(worker.result)
                return
        color = self.engine.current_color
        self.ai_worker = AIWorker(
            self.board,
            color,
            self.ai_request_id,
            ai=self.ai,
        )
        self.ai_worker.start()

    def quit(self):
        pygame.quit()
        raise SystemExit

    # ---------- 主循环 ----------
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    if event.key == pygame.K_SPACE:
                        self.on_space()
                    elif self.state == S_START and pygame.K_1 <= event.key <= pygame.K_4:
                        self.start_mode(event.key - pygame.K_1 + 1)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
            self.hover = self.pick(pygame.mouse.get_pos())
            if self.state == S_PLAY and not self.paused:
                self.update_ai()
            elif (self.state == S_WIN and self.mode == 4 and not self.paused
                  and self.win_at and pygame.time.get_ticks() - self.win_at > WATCH_AUTO_MS):
                self.advance_match()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

    # ---------- 绘制 ----------
    def blit(self, text, pos, font, color, anchor="topleft"):
        surface = font.render(text, True, color)
        rect = surface.get_rect(**{anchor: pos})
        self.screen.blit(surface, rect)
        return rect

    def draw(self):
        self.screen.fill(BG)
        self.renderer.draw(
            self.screen,
            self.board,
            hover=self.hover,
            win_path=self.win_path,
            last_move=self.last_move,
            can_hover=self.state == S_PLAY and self.is_human_turn(),
        )
        if self.state != S_START:
            status = self.ai_status_message
            if self.state == S_PLAY and not self.is_human_turn() and not status:
                status = "AI 思考中…"
            self.history_panel.draw(
                self.screen,
                self.engine.game_moves(self.game_no),
                self.game_no,
                status,
            )
        self.draw_hud()
        if self.state == S_START:
            self.draw_start_screen()
        elif self.state == S_WIN:
            self.draw_win_banner()
        elif self.state == S_MATCH_END:
            self.draw_match_end()

    def draw_hud(self):
        if self.state == S_START:
            return
        p1, p2 = self.match_scores
        self.blit(f"第 {self.game_no}/2 局", (12, 8), self.font_m, TEXT_CLR)
        self.blit(f"比分  {self.player_label(1)} {p1} : {p2} {self.player_label(2)}",
                  (12, 38), self.font_s, DIM_CLR)
        mode_text = MODES[self.mode][0] if self.mode else "未选择"
        self.blit(f"模式：{mode_text}    执红（先手）：{self.player_label(self.red_player)}",
                  (12, 62), self.font_s, DIM_CLR)

        if self.state == S_PLAY:
            color = self.engine.current_color
            label = self.player_label(self.player_of(color))
            ai_note = "" if self.is_human_turn() else "  —— AI 思考中…"
            rect = self.blit(f"轮到 {CNAMES[color]}（{label}）{ai_note}",
                             (BOARD_W - 12, 12), self.font_m, TEXT_CLR, anchor="topright")
            pygame.draw.circle(self.screen, CCOLORS[color], (rect.left - 12, rect.centery), 7)
        if self.state == S_PLAY and self.mode == 4:
            self.blit("已暂停" if self.paused else "观战中…",
                      (BOARD_W - 12, 40), self.font_s,
                      GOLD if self.paused else DIM_CLR, anchor="topright")
        if self.last_move:
            q, r, color, reason = self.last_move
            extra = f"（{reason}）" if reason else ""
            self.blit(f"上一手：{CNAMES[color]} {cell_name(q, r)}{extra}",
                      (12, WINDOW_H - 52), self.font_s, TEXT_CLR)
        if self.hover:
            self.blit(f"悬停：{cell_name(*self.hover)}",
                      (12, WINDOW_H - 28), self.font_s, DIM_CLR)
        self.blit("空格=开始/继续   Esc=退出", (BOARD_W - 12, WINDOW_H - 24),
                  self.font_s, DIM_CLR, anchor="bottomright")

    def draw_start_screen(self):
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 185))
        self.screen.blit(veil, (0, 0))
        cx = BOARD_W // 2
        self.blit("海克斯棋 Hex", (cx, 120), self.font_l, TEXT_CLR, anchor="center")
        y = 200
        for text, dot in (("红方连通上下红色边界获胜", RED_CLR),
                          ("蓝方连通左右蓝色边界获胜", BLUE_CLR)):
            surface = self.font_m.render(text, True, TEXT_CLR)
            rect = surface.get_rect(center=(cx, y))
            self.screen.blit(surface, rect)
            pygame.draw.circle(self.screen, dot, (rect.left - 14, rect.centery), 7)
            y += 38
        y += 12
        for text in ("双方轮流落子，红方先行，先连通者胜（单局无和棋）",
                     "每场两局，双方交换红蓝，轮流先手",
                     "坐标先纵后横：如 5C 表示第 5 行、第 C 列"):
            self.blit(text, (cx, y), self.font_s, DIM_CLR, anchor="center")
            y += 28
        self.mode_buttons = []
        y += 20
        mouse = pygame.mouse.get_pos()
        for key, (text, desc) in sorted(MODES.items()):
            rect = pygame.Rect(cx - 195, y, 390, 46)
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, (70, 76, 92) if hover else (50, 54, 68),
                             rect, border_radius=8)
            pygame.draw.rect(self.screen, (110, 116, 132), rect, 1, border_radius=8)
            self.blit(f"[{key}] {text}", (rect.x + 16, rect.centery),
                      self.font_m, TEXT_CLR, anchor="midleft")
            self.blit(desc, (rect.right - 12, rect.centery),
                      self.font_s, DIM_CLR, anchor="midright")
            self.mode_buttons.append((rect, key))
            y += 58

    def draw_win_banner(self):
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 175))
        self.screen.blit(veil, (0, 0))
        cx = BOARD_W // 2
        how = "连通了上下红色边界" if self.winner == RED else "连通了左右蓝色边界"
        who = self.player_label(self.player_of(self.winner))
        self.blit(f"{CNAMES[self.winner]}（{who}）{how}，获胜！",
                  (cx, WINDOW_H // 2 - 40), self.font_m, GOLD, anchor="center")
        self.blit("点击或按空格进入下一局（双方交换红蓝）",
                  (cx, WINDOW_H // 2 + 10), self.font_s, TEXT_CLR, anchor="center")

    def draw_match_end(self):
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 175))
        self.screen.blit(veil, (0, 0))
        cx = BOARD_W // 2
        p1, p2 = self.match_scores
        self.blit("本场结束", (cx, WINDOW_H // 2 - 90), self.font_l, TEXT_CLR, anchor="center")
        self.blit(f"{self.player_label(1)}  {p1} : {p2}  {self.player_label(2)}",
                  (cx, WINDOW_H // 2 - 10), self.font_m, GOLD, anchor="center")
        if p1 == p2:
            verdict = "两局 1:1，本场战平（单局无和棋，总分可平）"
        else:
            winner = 1 if p1 > p2 else 2
            verdict = f"{self.player_label(winner)} 本场获胜"
        self.blit(verdict, (cx, WINDOW_H // 2 + 35), self.font_m, TEXT_CLR, anchor="center")
        self.blit("点击或按空格再来一场   Esc 退出", (cx, WINDOW_H // 2 + 80),
                  self.font_s, DIM_CLR, anchor="center")


def main():
    HexGame().run()


if __name__ == "__main__":
    main()
