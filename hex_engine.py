# -*- coding: utf-8 -*-
"""
对局引擎：统一负责局号、双方执色、当前回合、合法落子、判胜与获胜通路、
交换回合、两局交换红蓝、总比分与本场结束、完整历史记录。

GUI（hex_game.py）、自动比赛裁判（hex_match.py）、协议机器人（hex_bot.py）
都必须通过 HexEngine 落子与判胜，禁止各自复制换手/判胜逻辑。
"""
from hex_board import (HexBoard, RED, BLUE, COLOR_NAMES, cell_name,
                       IllegalMoveError)

MATCH_GAMES = 2      # 一场比赛固定两局，第二局交换红蓝（轮流先手）


class HexEngine:
    """一场两局制海克斯棋比赛引擎。

    - 每局红方先行；第一局结束后 next_game() 交换双方红蓝；
    - sides 是队伍（玩家/AI）名称，与颜色解耦，便于按队伍统计；
    - 状态机：playing → game_over（一局结束）→ match_over（两局打完）。
    """

    def __init__(self, red_side="红队", blue_side="蓝队"):
        self._base_sides = (str(red_side), str(blue_side))
        self.reset_match()

    # ---------- 只读属性 ----------
    @property
    def board(self):
        return self._board

    @property
    def turn(self):
        return self._turn

    @property
    def game_no(self):
        return self._game_no

    @property
    def state(self):
        return self._state           # playing / game_over / match_over

    @property
    def winner_color(self):
        return self._winner_color

    @property
    def winner_side(self):
        return self._winner_side

    @property
    def win_path(self):
        return self._win_path

    @property
    def end_reason(self):
        return self._end_reason

    def side_of(self, color):
        """执该颜色的队伍名。"""
        return self.red_side if color == RED else self.blue_side

    def color_of(self, side):
        """该队伍在当前局执的颜色。"""
        return RED if side == self.red_side else BLUE

    def legal_moves(self, player=None):
        return self._board.legal_moves(player)

    # ---------- 流程 ----------
    def reset_match(self):
        """开始一场新比赛：第 1 局，红队执红先行，比分与历史清零。"""
        self.red_side, self.blue_side = self._base_sides
        self.scores = {self.red_side: 0, self.blue_side: 0}   # 按队伍记胜局数
        self.history = []      # 全部手数（跨局）：
                               # {game, no, color, side, cell, q, r, ms, reason}
        self.results = []      # 每局结果：{game, winner_color, winner_side, reason}
        self._new_game(1)

    def _new_game(self, game_no):
        self._board = HexBoard()
        self._game_no = game_no
        self._turn = RED                      # 每局红方先行
        self._move_no = 1
        self._state = "playing"
        self._winner_color = None
        self._winner_side = None
        self._win_path = None
        self._end_reason = None

    def play(self, q, r, ms=0.0, reason=None):
        """当前方在 (q, r) 落子。非法落子抛 IllegalMoveError。
        返回该手是否取胜（True 时本局进入 game_over）。"""
        if self._state != "playing":
            raise IllegalMoveError("本局已结束，不能再落子")
        player = self._turn
        self._board.place(q, r, player)       # 越界/已占/非法玩家 → IllegalMoveError
        self.history.append({
            "game": self._game_no, "no": self._move_no,
            "color": player, "side": self.side_of(player),
            "cell": cell_name(q, r), "q": q, "r": r,
            "ms": round(float(ms), 2), "reason": reason or ""})
        if self._board.has_won(player):
            self._end_game(player, "connect")
            return True
        self._move_no += 1
        self._turn = BLUE if player == RED else RED
        return False

    def force_win(self, winner_color, reason):
        """裁判裁定本局结果（超时判负、非法落子判负等），无需在棋盘上落子。"""
        if self._state != "playing":
            raise IllegalMoveError("本局已结束，不能裁定")
        self._end_game(winner_color, reason)
        return True

    def _end_game(self, winner_color, reason):
        self._winner_color = winner_color
        self._winner_side = self.side_of(winner_color)
        self._end_reason = reason
        self._win_path = self._board.winning_path(winner_color)
        self.scores[self._winner_side] += 1
        self.results.append({"game": self._game_no,
                             "winner_color": winner_color,
                             "winner_side": self._winner_side,
                             "reason": reason})
        self._state = "game_over"

    def next_game(self):
        """进入下一局：双方交换红蓝（轮流先手）。第二局结束后进入 match_over。"""
        if self._state == "match_over":
            raise IllegalMoveError("比赛已结束")
        if self._state == "playing":
            raise IllegalMoveError("本局尚未结束")
        if self._game_no >= MATCH_GAMES:
            self._state = "match_over"
            return
        self.red_side, self.blue_side = self.blue_side, self.red_side
        self._new_game(self._game_no + 1)

    def match_winner(self):
        """本场胜者队伍名；1:1 返回 None（单局无和棋，总分可平）。"""
        if self._state != "match_over":
            return None
        a, b = self._base_sides
        if self.scores[a] > self.scores[b]:
            return a
        if self.scores[b] > self.scores[a]:
            return b
        return None

    def game_moves(self, game_no=None):
        """返回指定局（默认当前局）的手数记录列表。"""
        g = self._game_no if game_no is None else game_no
        return [h for h in self.history if h["game"] == g]
