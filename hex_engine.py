# -*- coding: utf-8 -*-
"""
对局引擎：统一负责局号、双方执色、当前回合、合法落子、判胜与获胜通路、
两局交换红蓝、总比分、比赛结束状态和完整历史记录。

页面层只能调用本引擎处理规则与对局状态，不得自行复制换手、判胜或计分逻辑。
"""
from hex_board import HexBoard, RED, BLUE, cell_name, IllegalMoveError

MATCH_GAMES = 2
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"
STATE_MATCH_OVER = "match_over"
VALID_COLORS = (RED, BLUE)


class HexEngine:
    """一场两局制海克斯棋比赛引擎。

    页面集成所需的正式接口：
      start_match()                 开始或重开一场比赛
      play_move(q, r)               当前方落子
      next_game()                   本局结束后进入下一局或结束比赛
      current_player                当前执棋队伍名；非 playing 状态为 None
      current_color                 当前执棋颜色；非 playing 状态为 None
      winner                        本局胜方队伍名；尚未结束时为 None
      match_score                   按队伍统计的比分快照
      move_history                  整场比赛的手数记录快照

    兼容接口：
      reset_match() 等价于 start_match()
      play() 等价于 play_move()
      turn 等价于当前颜色（保留供旧代码使用）
    """

    def __init__(self, red_side="红队", blue_side="蓝队"):
        red = str(red_side).strip()
        blue = str(blue_side).strip()
        if not red or not blue:
            raise ValueError("队伍名称不能为空")
        if red == blue:
            raise ValueError("红蓝双方必须使用不同队伍名称")
        self._base_sides = (red, blue)
        self.start_match()

    # ---------- 只读属性 ----------
    @property
    def board(self):
        return self._board

    @property
    def turn(self):
        """兼容旧接口：当前颜色。"""
        return self._turn

    @property
    def current_player(self):
        """当前执棋队伍名；本局或比赛结束后返回 None。"""
        if self._state != STATE_PLAYING:
            return None
        return self.side_of(self._turn)

    @property
    def current_color(self):
        """当前执棋颜色；本局或比赛结束后返回 None。"""
        if self._state != STATE_PLAYING:
            return None
        return self._turn

    @property
    def game_no(self):
        return self._game_no

    @property
    def state(self):
        return self._state

    @property
    def winner_color(self):
        return self._winner_color

    @property
    def winner_side(self):
        return self._winner_side

    @property
    def winner(self):
        """本局胜方队伍名；尚未结束或尚未产生胜者时为 None。"""
        return self._winner_side

    @property
    def match_score(self):
        """按队伍统计的比分快照，修改返回值不会改变引擎内部状态。"""
        return dict(self.scores)

    @property
    def move_history(self):
        """整场比赛的手数快照，修改返回值不会改变引擎内部记录。"""
        return tuple(dict(item) for item in self.history)

    @property
    def win_path(self):
        return self._win_path

    @property
    def end_reason(self):
        return self._end_reason

    def side_of(self, color):
        """返回执指定颜色的队伍名。"""
        if color not in VALID_COLORS:
            raise IllegalMoveError(f"非法颜色编号: {color!r}")
        return self.red_side if color == RED else self.blue_side

    def color_of(self, side):
        """返回指定队伍在当前局所执的颜色。"""
        if side == self.red_side:
            return RED
        if side == self.blue_side:
            return BLUE
        raise ValueError(f"未知队伍: {side!r}")

    def legal_moves(self, color=None):
        """返回空位；给定颜色时返回该方棋子的邻近候选点。"""
        return self._board.legal_moves(color)

    # ---------- 流程 ----------
    def start_match(self):
        """开始一场新比赛：第 1 局，红队执红先行，比分与历史清零。"""
        self.red_side, self.blue_side = self._base_sides
        self.scores = {self.red_side: 0, self.blue_side: 0}
        self.history = []
        self.results = []
        self._new_game(1)

    def reset_match(self):
        """兼容旧接口：重开比赛。"""
        self.start_match()

    def _new_game(self, game_no):
        self._board = HexBoard()
        self._game_no = game_no
        self._turn = RED
        self._move_no = 1
        self._state = STATE_PLAYING
        self._winner_color = None
        self._winner_side = None
        self._win_path = None
        self._end_reason = None

    def play_move(self, q, r, ms=0.0, reason=None):
        """当前方在 (q, r) 落子；获胜时返回 True，否则返回 False。"""
        if self._state != STATE_PLAYING:
            raise IllegalMoveError("本局已结束，不能再落子")
        color = self._turn
        self._board.place(q, r, color)
        self.history.append({
            "game": self._game_no,
            "no": self._move_no,
            "color": color,
            "side": self.side_of(color),
            "cell": cell_name(q, r),
            "q": q,
            "r": r,
            "ms": round(float(ms), 2),
            "reason": reason or "",
        })
        if self._board.has_won(color):
            self._end_game(color, "connect")
            return True
        self._move_no += 1
        self._turn = BLUE if color == RED else RED
        return False

    def play(self, q, r, ms=0.0, reason=None):
        """兼容旧接口：等价于 play_move()。"""
        return self.play_move(q, r, ms=ms, reason=reason)

    def force_win(self, winner_color, reason):
        """裁判裁定本局结果；无需在棋盘上落子。"""
        if self._state != STATE_PLAYING:
            raise IllegalMoveError("本局已结束，不能裁定")
        if winner_color not in VALID_COLORS:
            raise IllegalMoveError(f"非法颜色编号: {winner_color!r}")
        self._end_game(winner_color, reason)
        return True

    def _end_game(self, winner_color, reason):
        self._winner_color = winner_color
        self._winner_side = self.side_of(winner_color)
        self._end_reason = reason
        self._win_path = self._board.winning_path(winner_color)
        self.scores[self._winner_side] += 1
        self.results.append({
            "game": self._game_no,
            "winner_color": winner_color,
            "winner_side": self._winner_side,
            "reason": reason,
        })
        self._state = STATE_GAME_OVER

    def next_game(self):
        """本局结束后进入下一局；第二局结束后进入 match_over。"""
        if self._state == STATE_MATCH_OVER:
            raise IllegalMoveError("比赛已结束")
        if self._state == STATE_PLAYING:
            raise IllegalMoveError("本局尚未结束")
        if self._game_no >= MATCH_GAMES:
            self._state = STATE_MATCH_OVER
            return self._state
        self.red_side, self.blue_side = self.blue_side, self.red_side
        self._new_game(self._game_no + 1)
        return self._state

    def match_winner(self):
        """本场胜者队伍名；两局 1:1 时返回 None。"""
        if self._state != STATE_MATCH_OVER:
            return None
        first_side, second_side = self._base_sides
        if self.scores[first_side] > self.scores[second_side]:
            return first_side
        if self.scores[second_side] > self.scores[first_side]:
            return second_side
        return None

    def game_moves(self, game_no=None):
        """返回指定局（默认当前局）的手数记录快照。"""
        target = self._game_no if game_no is None else game_no
        return tuple(dict(item) for item in self.history if item["game"] == target)
