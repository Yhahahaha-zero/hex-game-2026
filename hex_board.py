# -*- coding: utf-8 -*-
"""
海克斯棋棋盘与规则核心（与界面无关，供图形界面、AI、自动对局三方共用）。

坐标约定：轴向坐标 (q, r)，q = 列（横向，A-K），r = 行（纵向，1-11）。
连通判定：红方连通上下边界（r=0 与 r=10），蓝方连通左右边界（q=0 与 q=10）。
坐标文本先纵后横，如 "5C" = 第 5 行第 C 列。

公开棋盘操作对非法坐标、非法颜色和非法落子统一抛出 IllegalMoveError。
parse_cell() 是输入解析函数，非法文本仍按约定返回 None。
"""
import re
from collections import deque

ROWS = COLS = 11
EMPTY, RED, BLUE = 0, 1, 2

COLOR_NAMES = {RED: "红", BLUE: "蓝"}

# 平顶六边形轴向坐标（q=列, r=行）的六个邻居方向
DIRS = ((1, 0), (-1, 0), (1, -1), (0, -1), (0, 1), (-1, 1))


class IllegalMoveError(ValueError):
    """非法落子、非法坐标、非法颜色或对局结束后继续操作。"""


def _valid_coord_value(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require_coord(q, r):
    """校验棋盘坐标，非法时统一抛出 IllegalMoveError。"""
    if not (_valid_coord_value(q) and _valid_coord_value(r)
            and 0 <= q < COLS and 0 <= r < ROWS):
        raise IllegalMoveError(f"坐标越界或类型非法: ({q!r}, {r!r})")


def _require_color(player):
    """校验棋子颜色，非法时统一抛出 IllegalMoveError。"""
    if not _valid_coord_value(player) or player not in (RED, BLUE):
        raise IllegalMoveError(f"非法玩家编号: {player!r}")


def cell_name(q, r):
    """返回先纵后横的坐标文本，如 (2, 4) -> 5C。"""
    _require_coord(q, r)
    return f"{r + 1}{chr(65 + q)}"


def parse_cell(s, strict=True):
    """解析坐标文本，返回 (q, r)；非法返回 None。

    strict=True（默认，正式协议用）：只接受题目规定的“先纵后横”格式，
    例如 5C。拒绝 C5、05C、5c、夹空格、0A、12A 等非标准写法。

    strict=False（显式开启的内部兼容模式）：允许先横后纵（C5），
    忽略首零与前后空格，但仍在 1-11 / A-K 范围内校验。
    """
    if not isinstance(s, str):
        return None
    if strict:
        if re.fullmatch(r"(?:[1-9]|1[01])[A-K]", s) is None:
            return None
        return ord(s[-1]) - 65, int(s[:-1]) - 1
    match = re.fullmatch(r"(\d{1,2})([A-Ka-k])|([A-Ka-k])(\d{1,2})",
                         s.strip().upper())
    if not match:
        return None
    number, letter = ((match.group(1), match.group(2)) if match.group(1)
                      else (match.group(4), match.group(3)))
    q, r = ord(letter) - 65, int(number) - 1
    return (q, r) if 0 <= q < COLS and 0 <= r < ROWS else None


class _DSU:
    """带边界掩码的并查集，用于 O(αN) 的连通性判定。"""

    def __init__(self, board, player):
        _require_color(player)
        n = ROWS * COLS
        self.parent = list(range(n))
        self.mask = [0] * n
        for r in range(ROWS):
            for q in range(COLS):
                if board.grid[r][q] != player:
                    continue
                idx = r * COLS + q
                if player == RED:
                    self.mask[idx] = ((1 if r == 0 else 0)
                                      | (2 if r == ROWS - 1 else 0))
                else:
                    self.mask[idx] = ((1 if q == 0 else 0)
                                      | (2 if q == COLS - 1 else 0))
        for r in range(ROWS):
            for q in range(COLS):
                idx = r * COLS + q
                if board.grid[r][q] != player:
                    continue
                for nq, nr in board.neighbors(q, r):
                    if board.grid[nr][nq] == player:
                        self.union(idx, nr * COLS + nq)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        self.parent[root_a] = root_b
        self.mask[root_b] |= self.mask[root_a]


class HexBoard:
    """11×11 六边形棋盘（grid[r][q]，r 为行、q 为列）。"""

    def __init__(self):
        self.grid = [[EMPTY] * COLS for _ in range(ROWS)]
        self.history = []
        self._dsu_cache = {RED: None, BLUE: None}

    def in_range(self, q, r):
        """坐标是否位于棋盘内。该谓词对非整数输入返回 False。"""
        return (_valid_coord_value(q) and _valid_coord_value(r)
                and 0 <= q < COLS and 0 <= r < ROWS)

    def neighbors(self, q, r):
        """返回与 (q, r) 共边的六边形邻居；非法坐标抛出 IllegalMoveError。"""
        _require_coord(q, r)
        for dq, dr in DIRS:
            nq, nr = q + dq, r + dr
            if self.in_range(nq, nr):
                yield nq, nr

    def is_empty(self, q, r):
        """指定坐标是否为空；非法坐标抛出 IllegalMoveError。"""
        _require_coord(q, r)
        return self.grid[r][q] == EMPTY

    def place(self, q, r, player):
        """落子；越界、占用位置和非法颜色均抛出 IllegalMoveError。"""
        _require_color(player)
        _require_coord(q, r)
        if self.grid[r][q] != EMPTY:
            raise IllegalMoveError(f"位置已占用: {cell_name(q, r)}")
        self.grid[r][q] = player
        self.history.append((q, r, player))
        self._dsu_cache[player] = None

    def undo(self):
        """悔一步棋，供搜索时试下回退。"""
        if not self.history:
            return False
        q, r, player = self.history.pop()
        self.grid[r][q] = EMPTY
        self._dsu_cache[player] = None
        return True

    def candidate_moves(self, player=None):
        """返回搜索候选点。

        player 为 None 时返回全部空位；给定颜色时返回与该方棋子共边的空位。
        这是搜索剪枝接口，并不表示 Hex 规则只允许在这些位置落子。
        """
        empties = [(q, r) for r in range(ROWS) for q in range(COLS)
                   if self.grid[r][q] == EMPTY]
        if player is None:
            return empties
        _require_color(player)
        candidates = set()
        has_piece = False
        for r in range(ROWS):
            for q in range(COLS):
                if self.grid[r][q] != player:
                    continue
                has_piece = True
                for nq, nr in self.neighbors(q, r):
                    if self.grid[nr][nq] == EMPTY:
                        candidates.add((nq, nr))
        return empties if not has_piece else list(candidates)

    def legal_moves(self, player=None):
        """兼容旧接口，等价于 candidate_moves()。"""
        return self.candidate_moves(player)

    def _dsu(self, player):
        _require_color(player)
        dsu = self._dsu_cache[player]
        if dsu is None:
            dsu = _DSU(self, player)
            self._dsu_cache[player] = dsu
        return dsu

    def has_won(self, player):
        """player 是否已连通己方两条边界；非法颜色抛出 IllegalMoveError。"""
        _require_color(player)
        dsu = self._dsu(player)
        for r in range(ROWS):
            for q in range(COLS):
                idx = r * COLS + q
                if (self.grid[r][q] == player
                        and dsu.mask[dsu.find(idx)] == 3):
                    return True
        return False

    def would_win(self, q, r, player):
        """虚拟落子判断是否立即获胜，不修改棋盘。

        非法坐标、已占位置或非法颜色均抛出 IllegalMoveError。
        """
        _require_color(player)
        _require_coord(q, r)
        if self.grid[r][q] != EMPTY:
            raise IllegalMoveError(f"位置已占用: {cell_name(q, r)}")
        dsu = self._dsu(player)
        if player == RED:
            mask = ((1 if r == 0 else 0)
                    | (2 if r == ROWS - 1 else 0))
        else:
            mask = ((1 if q == 0 else 0)
                    | (2 if q == COLS - 1 else 0))
        for nq, nr in self.neighbors(q, r):
            if self.grid[nr][nq] == player:
                mask |= dsu.mask[dsu.find(nr * COLS + nq)]
        return mask == 3

    def winning_path(self, player, algo="bfs"):
        """返回一条连接己方两侧边界的通路；未连通返回 None。

        valid algo: bfs / dfs。非法颜色抛出 IllegalMoveError，
        未知算法抛出 ValueError。
        """
        _require_color(player)
        if algo not in ("bfs", "dfs"):
            raise ValueError(f"未知连通搜索算法: {algo!r}")
        if player == RED:
            starts = [(q, 0) for q in range(COLS) if self.grid[0][q] == RED]
        else:
            starts = [(0, r) for r in range(ROWS) if self.grid[r][0] == BLUE]
        if not starts:
            return None
        seen = set(starts)
        parent = {}
        frontier = deque(starts) if algo == "bfs" else list(starts)
        pop = frontier.popleft if algo == "bfs" else frontier.pop
        goal = None
        while frontier:
            q, r = pop()
            if ((player == RED and r == ROWS - 1)
                    or (player == BLUE and q == COLS - 1)):
                goal = (q, r)
                break
            for nq, nr in self.neighbors(q, r):
                if (nq, nr) not in seen and self.grid[nr][nq] == player:
                    seen.add((nq, nr))
                    parent[(nq, nr)] = (q, r)
                    frontier.append((nq, nr))
        if goal is None:
            return None
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent.get(current)
        return path

    def render_text(self, path=None):
        """控制台文本棋盘；path 中非法坐标会抛出 IllegalMoveError。"""
        path_set = set(path or ())
        for q, r in path_set:
            _require_coord(q, r)
        lines = ["      " + "  ".join(chr(65 + q) for q in range(COLS))]
        for r in range(ROWS):
            indent = "  " if r % 2 == 1 else ""
            chars = []
            for q in range(COLS):
                value = self.grid[r][q]
                char = "R" if value == RED else ("B" if value == BLUE else ".")
                if (q, r) in path_set:
                    char = "*"
                chars.append(char)
            lines.append(f"{r + 1:>2}  " + indent + "  ".join(chars))
        return "\n".join(lines)
