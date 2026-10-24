# -*- coding: utf-8 -*-
"""
海克斯棋（Hex）—— 双人落子类游戏（pygame 图形版）

棋盘：11×11 个六边形单元格；上下边界为红色（横向坐标 A-K），
      左右边界为蓝色（纵向坐标 1-11）。
棋规：
  1. 双方交替落子，每次一颗，红方先行；
  2. 共边的相邻同色棋子相互连通；
  3. 红方连通上下两条红色边界获胜；蓝方连通左右两条蓝色边界获胜；
  4. 不存在和棋；
  5. 双方轮流先手，各赛一局（每场两局，一局结束后双方交换红蓝，红方每局先行）。
坐标描述：先纵后横，如 "5C" 表示第 5 行第 C 列。

模式（开始画面选择，也可按数字键 1-4）：
  1. 双人对战              2. 人机对战 · 玩家执红
  3. 人机对战 · 玩家执蓝   4. AI 对 AI 观战（空格暂停/继续）

操作：鼠标点击空单元格落子（悬停显示坐标），空格键开始/继续，Esc 退出。
运行：pip install pygame 之后执行 py hex_game.py
"""

import math
import os
import threading

import pygame

from hex_board import (ROWS, COLS, EMPTY, RED, BLUE, HexBoard, cell_name)

# ---------------------------- 常量 ----------------------------
SIZE = 28                           # 六边形边长（像素）
MARGIN = 68                         # 四周留白（画坐标标签）
SQ3 = math.sqrt(3)

ORIGIN_X = MARGIN + SIZE            # A 列中心的横坐标
ORIGIN_Y = MARGIN + SQ3 * SIZE      # 第 1 行中心的纵坐标
WINDOW_W = int(ORIGIN_X + 16 * SIZE + MARGIN)
WINDOW_H = int(ORIGIN_Y + SQ3 * SIZE * 15.5 + MARGIN)

# 颜色
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

# 界面状态：开始画面 / 对局中 / 本局结束 / 本场结束
S_START, S_PLAY, S_WIN, S_MATCH_END = range(4)

# 游戏模式
MODES = {
    1: ("双人对战", "两名玩家轮流点击落子"),
    2: ("人机对战 · 玩家执红", "玩家先手，AI 执蓝"),
    3: ("人机对战 · 玩家执蓝", "AI 先手，玩家执蓝"),
    4: ("AI 对 AI 观战", "空格暂停 / 继续"),
}

AI_DELAY_MS = 350                   # AI 落子前的演示间隔
WATCH_AUTO_MS = 1600                # 观战模式本局结束后自动进下一局的等待

CNAMES = {RED: "红方", BLUE: "蓝方"}
CCOLORS = {RED: RED_CLR, BLUE: BLUE_CLR}


# 中文系统字体候选（Windows 字体目录 + 字体文件名）
FONT_DIR = r"C:\Windows\Fonts"
FONT_NAMES = {
    False: ("msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
    True: ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
}


def get_font(size, bold=False):
    """找一个支持中文的系统字体（微软雅黑/黑体/等线/宋体），找不到就用默认字体。
    注意：直接按文件名加载，不调用 pygame.match_font —— 后者在部分中文版
    Windows 上枚举字体注册表时会崩溃（pygame 2.6.1 已知问题）。"""
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


# ---------------------------- 坐标换算 ----------------------------
def hex_corners(cx, cy):
    """平顶六边形的六个顶点（自右顶点 0° 起，顺时针 60° 一档）。"""
    return [(cx + SIZE * math.cos(math.radians(60 * i)),
             cy + SIZE * math.sin(math.radians(60 * i))) for i in range(6)]


def cell_center(q, r):
    """轴向坐标 (q, r) → 该单元格中心的像素坐标。"""
    return (ORIGIN_X + 1.5 * SIZE * q,
            ORIGIN_Y + SQ3 * SIZE * (r + q / 2))


def pixel_to_hex(x, y):
    """像素坐标 → 轴向坐标 (q, r)。用立方坐标取整，点中斜边也能选对格子。"""
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


# ---------------------------- 游戏主体 ----------------------------
class HexGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("海克斯棋 Hex —— 红连上下，蓝连左右")
        self.clock = pygame.time.Clock()
        self.font_s = get_font(18)
        self.font_m = get_font(22)
        self.font_l = get_font(40, bold=True)
        self.mode = 0                   # 0=未选择
        self.human = None               # 人机模式中人类对应的玩家编号（1/2）
        self.ai = None
        self.ai_status_message = ""
        self.ai_request_id = 0
        self.ai_thread = None
        self.ai_result = None
        self.mode_buttons = []
        self.reset_match(show_intro=True)

    # ---------- 状态管理 ----------
    def reset_match(self, show_intro=False):
        """开始一场比赛：共两局，双方轮流执红（先手）。"""
        self.match_scores = [0, 0]      # 玩家1 / 玩家2 的胜局数
        self.game_no = 1
        self.red_player = 1             # 本局执红（先手）的玩家编号
        self.new_game()
        if show_intro:
            self.state = S_START

    def new_game(self):
        self.board = HexBoard()
        self.turn = RED                 # 每局红方先行
        self.last_move = None
        self.hover = None
        self.win_path = None
        self.winner = None
        self.win_at = None              # 观战自动进下一局用
        self.paused = False             # 观战暂停
        self.pending_move = None        # AI 已算好、待演示的着法
        self.pending_reason = None
        self.ai_act_at = 0
        self.ai_request_id += 1
        self.ai_thread = None
        self.ai_result = None
        self.state = S_PLAY

    def start_mode(self, mode):
        """在开始画面选定模式并开始第一局。"""
        self.mode = mode
        self.human = {2: 1, 3: 2}.get(mode)
        self.ai = None
        self.ai_status_message = ""
        self.ai_request_id += 1
        self.ai_thread = None
        self.ai_result = None
        self.state = S_PLAY


    def player_of(self, color):
        """执该颜色的玩家编号（1 或 2）。"""
        return self.red_player if color == RED else 3 - self.red_player

    def player_label(self, pid):
        """玩家显示名：双人=玩家N；人机=玩家N/AI；观战=AI1/AI2。"""
        if self.mode == 1 or self.human == pid:
            return f"玩家{pid}"
        if self.mode == 4:
            return f"AI{pid}"
        return "AI"

    def is_human_turn(self):
        """当前回合是否由人类落子：双人模式恒为真，观战模式恒为假，
        人机模式看当前颜色归属。"""
        if self.mode == 1:
            return True
        if self.mode == 4:
            return False
        return self.player_of(self.turn) == self.human

    def advance_match(self):
        """一局结束：进入下一局（交换红蓝，即轮流先手）；两局打完整场结束。"""
        self.game_no += 1
        if self.game_no > 2:
            self.state = S_MATCH_END
            return
        self.red_player = 3 - self.red_player
        self.new_game()

    # ---------- 输入处理 ----------
    def pick(self, pos):
        """像素位置 → 棋盘单元格，点中棋盘外返回 None。"""
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
        else:  # S_PLAY
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
            self.paused = not self.paused     # 观战暂停/继续

    def apply_move(self, q, r, reason=None):
        """落子并判定胜负（人类与 AI 共用）。"""
        self.board.place(q, r, self.turn)
        self.last_move = (q, r, self.turn, reason)
        if self.board.has_won(self.turn):
            self.win_path = self.board.winning_path(self.turn)
            self.winner = self.turn
            self.match_scores[self.player_of(self.turn) - 1] += 1
            self.state = S_WIN
            self.win_at = pygame.time.get_ticks()
        else:
            self.turn = BLUE if self.turn == RED else RED

    # ---------- AI ----------
    def _ai_worker(self, board_snapshot, color, request_id):
        """后台加载并运行 AI，不直接修改正式棋盘。"""
        from hex_ai import BasicAI

        init_error = None
        inference_error = None
        fallback_error = None
        ai = self.ai

        if ai is None:
            from ai.neural_ai import create_ai_with_fallback
            ai, init_error = create_ai_with_fallback()
            self.ai = ai

        try:
            move, reason = ai.choose_move(board_snapshot, color)
            q, r = move
            if not board_snapshot.in_range(q, r) or not board_snapshot.is_empty(q, r):
                raise ValueError(f"AI 返回非法着法: {move!r}")
        except Exception as exc:
            inference_error = exc
            try:
                fallback = BasicAI()
                move, reason = fallback.choose_move(board_snapshot, color)
                self.ai = fallback
            except Exception as fallback_exc:
                fallback_error = fallback_exc
                move, reason = None, ""

        if request_id == self.ai_request_id:
            self.ai_result = (request_id, color, move, reason,
                              init_error, inference_error, fallback_error)

    def _consume_ai_result(self):
        result = self.ai_result
        self.ai_result = None
        self.ai_thread = None
        if result is None:
            return

        request_id, color, move, reason, init_error, inference_error, fallback_error = result
        if (request_id != self.ai_request_id or self.state != S_PLAY
                or self.turn != color):
            return
        if init_error:
            print(f"[AI] {init_error}")
            self.ai_status_message = "模型不可用，已回退基础 AI"
        if inference_error:
            print(f"[AI] 推理失败: {inference_error}")
            self.ai_status_message = "模型推理失败，已回退基础 AI"
        if fallback_error:
            print(f"[AI] 基础 AI 回退失败: {fallback_error}")
            self.ai_status_message = "AI 执行失败"
            return
        if move is None:
            return

        self.pending_move = move
        self.pending_reason = reason
        self.ai_act_at = pygame.time.get_ticks() + AI_DELAY_MS

    def update_ai(self):
        """AI 回合：后台计算着法，完成后延迟演示落子。"""
        if self.state != S_PLAY or self.is_human_turn():
            return

        now = pygame.time.get_ticks()
        if self.pending_move is not None:
            if now >= self.ai_act_at:
                q, r = self.pending_move
                reason, self.pending_move = self.pending_reason, None
                self.apply_move(q, r, reason)
            return

        if self.ai_result is not None:
            self._consume_ai_result()
            return

        if self.ai_thread is not None and self.ai_thread.is_alive():
            return

        self.ai_thread = None
        snapshot = HexBoard()
        snapshot.grid = [row[:] for row in self.board.grid]
        snapshot.history = list(self.board.history)
        request_id = self.ai_request_id
        color = self.turn
        self.ai_thread = threading.Thread(
            target=self._ai_worker,
            args=(snapshot, color, request_id),
            daemon=True,
        )
        self.ai_thread.start()


    def quit(self):
        pygame.quit()
        raise SystemExit

    # ---------- 主循环 ----------
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    elif event.key == pygame.K_SPACE:
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
                self.advance_match()    # 观战模式自动进入下一局
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

    # ---------- 绘制 ----------
    def blit(self, text, pos, font, color, anchor="topleft"):
        """按锚点画一行文本，返回其矩形（方便在旁边摆装饰）。"""
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def draw(self):
        self.screen.fill(BG)
        self.draw_cells()
        self.draw_borders()
        self.draw_labels()
        self.draw_pieces()
        self.draw_hover_highlight()
        self.draw_win_highlight()
        self.draw_last_marker()
        self.draw_hud()
        if self.state == S_START:
            self.draw_start_screen()
        elif self.state == S_WIN:
            self.draw_win_banner()
        elif self.state == S_MATCH_END:
            self.draw_match_end()

    def draw_cells(self):
        for q in range(COLS):
            for r in range(ROWS):
                cx, cy = cell_center(q, r)
                pts = hex_corners(cx, cy)
                pygame.draw.polygon(self.screen, CELL_FILL, pts)
                pygame.draw.polygon(self.screen, CELL_LINE, pts, 1)

    def draw_borders(self):
        """四条边界线：上下红、左右蓝（画在边界单元格的外侧边上）。"""
        for q in range(COLS):
            cx, cy = cell_center(q, 0)
            p = hex_corners(cx, cy)
            pygame.draw.lines(self.screen, RED_CLR, False, [p[4], p[5], p[0]], 4)
            cx, cy = cell_center(q, ROWS - 1)
            p = hex_corners(cx, cy)
            pygame.draw.lines(self.screen, RED_CLR, False, [p[1], p[2], p[3]], 4)
        for r in range(ROWS):
            cx, cy = cell_center(0, r)
            p = hex_corners(cx, cy)
            pygame.draw.lines(self.screen, BLUE_CLR, False, [p[2], p[3], p[4]], 4)
            cx, cy = cell_center(COLS - 1, r)
            p = hex_corners(cx, cy)
            pygame.draw.lines(self.screen, BLUE_CLR, False, [p[5], p[0], p[1]], 4)

    def draw_labels(self):
        """坐标标签：红色字母 A-K 在上下、蓝色数字 1-11 在左右。"""
        for q in range(COLS):
            cx, cy = cell_center(q, 0)
            self.blit(chr(65 + q), (cx + SIZE / 2, cy - SQ3 * SIZE / 2 - 16),
                      self.font_s, RED_CLR, anchor="center")
            cx, cy = cell_center(q, ROWS - 1)
            self.blit(chr(65 + q), (cx + SIZE / 2, cy + SQ3 * SIZE / 2 + 16),
                      self.font_s, RED_CLR, anchor="center")
        for r in range(ROWS):
            cx, cy = cell_center(0, r)
            self.blit(str(r + 1), (cx - SIZE - 18, cy), self.font_s, BLUE_CLR, anchor="center")
            cx, cy = cell_center(COLS - 1, r)
            self.blit(str(r + 1), (cx + SIZE + 18, cy), self.font_s, BLUE_CLR, anchor="center")

    def draw_pieces(self):
        radius = int(SIZE * 0.46)       # 棋子略小于六边形单元格
        for q in range(COLS):
            for r in range(ROWS):
                v = self.board.grid[r][q]
                if v == EMPTY:
                    continue
                cx, cy = cell_center(q, r)
                pygame.draw.circle(self.screen, RED_DARK if v == RED else BLUE_DARK,
                                   (cx, cy), radius + 2)
                pygame.draw.circle(self.screen, CCOLORS[v], (cx, cy), radius)

    def draw_hover_highlight(self):
        if self.state == S_PLAY and self.hover and self.is_human_turn():
            q, r = self.hover
            if self.board.is_empty(q, r):
                cx, cy = cell_center(q, r)
                pygame.draw.polygon(self.screen, WHITE, hex_corners(cx, cy), 2)

    def draw_win_highlight(self):
        """用金圈标出获胜通路。"""
        if self.win_path:
            for q, r in self.win_path:
                cx, cy = cell_center(q, r)
                pygame.draw.circle(self.screen, GOLD, (cx, cy), int(SIZE * 0.46) + 4, 3)

    def draw_last_marker(self):
        if self.last_move:
            q, r, _, _ = self.last_move
            cx, cy = cell_center(q, r)
            pygame.draw.circle(self.screen, WHITE, (cx, cy), 3)

    def draw_hud(self):
        p1, p2 = self.match_scores
        self.blit(f"第 {self.game_no}/2 局", (12, 8), self.font_m, TEXT_CLR)
        self.blit(f"比分  {self.player_label(1)} {p1} : {p2} {self.player_label(2)}",
                  (12, 38), self.font_s, DIM_CLR)
        mode_text = MODES[self.mode][0] if self.mode else "未选择"
        self.blit(f"模式：{mode_text}    执红（先手）：{self.player_label(self.red_player)}",
                  (12, 62), self.font_s, DIM_CLR)

        if self.state == S_PLAY:
            label = self.player_label(self.player_of(self.turn))
            ai_note = "" if self.is_human_turn() else f"  —— {self.ai_status_message or 'AI 思考中…'}"
            rect = self.blit(f"轮到 {CNAMES[self.turn]}（{label}）{ai_note}",
                             (WINDOW_W - 12, 12), self.font_m, TEXT_CLR, anchor="topright")
            pygame.draw.circle(self.screen, CCOLORS[self.turn],
                               (rect.left - 12, rect.centery), 7)
        if self.state == S_PLAY and self.mode == 4:
            self.blit("已暂停" if self.paused else "观战中…",
                      (WINDOW_W - 12, 40), self.font_s, GOLD if self.paused else DIM_CLR,
                      anchor="topright")

        if self.last_move:
            q, r, color, reason = self.last_move
            extra = f"（AI·{reason}）" if reason else ""
            self.blit(f"上一手：{CNAMES[color]} {cell_name(q, r)}{extra}",
                      (12, WINDOW_H - 52), self.font_s, TEXT_CLR)
        if self.hover:
            self.blit(f"悬停：{cell_name(*self.hover)}",
                      (12, WINDOW_H - 28), self.font_s, DIM_CLR)
        self.blit("空格=开始/继续   Esc=退出", (WINDOW_W - 12, WINDOW_H - 24),
                  self.font_s, DIM_CLR, anchor="bottomright")

    def draw_start_screen(self):
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 175))
        self.screen.blit(veil, (0, 0))
        cx = WINDOW_W // 2
        self.blit("海克斯棋 Hex", (cx, 120), self.font_l, TEXT_CLR, anchor="center")
        y = 200
        for text, dot in (("红方连通上下红色边界获胜", RED_CLR),
                          ("蓝方连通左右蓝色边界获胜", BLUE_CLR)):
            surf = self.font_m.render(text, True, TEXT_CLR)
            rect = surf.get_rect(center=(cx, y))
            self.screen.blit(surf, rect)
            pygame.draw.circle(self.screen, dot, (rect.left - 14, rect.centery), 7)
            y += 38
        y += 12
        for t in ("双方轮流落子，红方先行，先连通者胜（单局无和棋）",
                  "每场两局，双方交换红蓝，轮流先手",
                  "坐标先纵后横：如 5C 表示第 5 行、第 C 列"):
            self.blit(t, (cx, y), self.font_s, DIM_CLR, anchor="center")
            y += 28
        # 模式按钮
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
        cx = WINDOW_W // 2
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
        cx = WINDOW_W // 2
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
