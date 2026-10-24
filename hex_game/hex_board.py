# -*- coding: utf-8 -*-
"""
海克斯棋棋盘与规则核心（与界面无关，供图形界面、AI、自动对局三方共用）。

坐标约定：轴向坐标 (q, r)，q = 列（横向，A-K），r = 行（纵向，1-11）。
连通判定：红方连通上下边界（r=0 与 r=10），蓝方连通左右边界（q=0 与 q=10）。
坐标文本先纵后横，如 "5C" = 第 5 行第 C 列。
"""
import re
from collections import deque

ROWS = COLS = 11                     # 11×11 棋盘
EMPTY, RED, BLUE = 0, 1, 2           # 单元格三种状态

# 平顶六边形轴向坐标（q=列, r=行）的六个邻居方向
DIRS = ((1, 0), (-1, 0), (1, -1), (0, -1), (0, 1), (-1, 1))


def cell_name(q, r):
    """坐标文本：先纵（1-11）后横（A-K），如 5C = 第 5 行第 C 列。"""
    return f"{r + 1}{chr(65 + q)}"


def parse_cell(s):
    """解析坐标文本，兼容先纵后横（5C）或先横后纵（C5），返回 (q, r)；非法返回 None。"""
    m = re.fullmatch(r"(\d{1,2})([A-Ka-k])|([A-Ka-k])(\d{1,2})", s.strip().upper())
    if not m:
        return None
    num, letter = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    q, r = ord(letter) - 65, int(num) - 1
    return (q, r) if 0 <= q < COLS and 0 <= r < ROWS else None


class _DSU:
    """带边界掩码的并查集，用于 O(αN) 的连通性判定。

    掩码含义：红方 bit0=顶边界(r=0)、bit1=底边界(r=10)；
              蓝方 bit0=左边界(q=0)、bit1=右边界(q=10)。
    某连通分量掩码 == 3，即该分量已同时触及己方两条边界。
    """

    def __init__(self, board, player):
        n = ROWS * COLS
        self.parent = list(range(n))
        self.mask = [0] * n
        for r in range(ROWS):
            for q in range(COLS):
                if board.grid[r][q] != player:
                    continue
                idx = r * COLS + q
                if player == RED:
                    self.mask[idx] = (1 if r == 0 else 0) | (2 if r == ROWS - 1 else 0)
                else:
                    self.mask[idx] = (1 if q == 0 else 0) | (2 if q == COLS - 1 else 0)
        for r in range(ROWS):
            for q in range(COLS):
                idx = r * COLS + q
                if board.grid[r][q] != player:
                    continue
                for nq, nr in board.neighbors(q, r):
                    if board.grid[nr][nq] == player:
                        self.union(idx, nr * COLS + nq)

    def find(self, x):                 # 路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[ra] = rb
        self.mask[rb] |= self.mask[ra]


class HexBoard:
    """11×11 六边形棋盘（grid[r][q]，r 为行、q 为列）。"""

    def __init__(self):
        self.grid = [[EMPTY] * COLS for _ in range(ROWS)]
        self.history = []                              # (q, r, player) 落子序列
        self._dsu_cache = {RED: None, BLUE: None}      # 落子/悔棋后失效重建

    def in_range(self, q, r):
        return 0 <= q < COLS and 0 <= r < ROWS

    def neighbors(self, q, r):
        """与 (q, r) 共边的六个（边界处更少）邻居。"""
        for dq, dr in DIRS:
            nq, nr = q + dq, r + dr
            if self.in_range(nq, nr):
                yield nq, nr

    def is_empty(self, q, r):
        return self.grid[r][q] == EMPTY

    def place(self, q, r, player):
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

    def legal_moves(self, player=None):
        """全部空位；player 给定时只返回与该方棋子共边的空位（搜索剪枝，
        该方尚无棋子时返回全部空位）。"""
        empties = [(q, r) for r in range(ROWS) for q in range(COLS)
                   if self.grid[r][q] == EMPTY]
        if player is None:
            return empties
        cands = set()
        has_piece = False
        for r in range(ROWS):
            for q in range(COLS):
                if self.grid[r][q] != player:
                    continue
                has_piece = True
                for nq, nr in self.neighbors(q, r):
                    if self.grid[nr][nq] == EMPTY:
                        cands.add((nq, nr))
        return empties if not has_piece else list(cands)

    # ---------- 连通性判定 ----------
    def _dsu(self, player):
        dsu = self._dsu_cache[player]
        if dsu is None:
            dsu = _DSU(self, player)
            self._dsu_cache[player] = dsu
        return dsu

    def has_won(self, player):
        """并查集判定 player 是否已连通己方两条边界（每次落子后调用，很快）。"""
        dsu = self._dsu(player)
        for r in range(ROWS):
            for q in range(COLS):
                idx = r * COLS + q
                if self.grid[r][q] == player and dsu.mask[dsu.find(idx)] == 3:
                    return True
        return False

    def would_win(self, q, r, player):
        """虚拟落子：若 player 在空位 (q, r) 落子，是否立即获胜（O(邻居数)，不改棋盘）。"""
        dsu = self._dsu(player)
        if player == RED:
            m = (1 if r == 0 else 0) | (2 if r == ROWS - 1 else 0)
        else:
            m = (1 if q == 0 else 0) | (2 if q == COLS - 1 else 0)
        for nq, nr in self.neighbors(q, r):
            if self.grid[nr][nq] == player:
                m |= dsu.mask[dsu.find(nr * COLS + nq)]
        return m == 3

    def winning_path(self, player, algo="bfs"):
        """从一侧边界向对侧边界搜索同色连通通路，返回通路单元格列表；未连通返回 None。

        红方：从 r=0（上边界）搜到 r=10（下边界）；蓝方：从 q=0（左）搜到 q=10（右）。
        algo="bfs" 广度优先（队列）/"dfs" 深度优先（栈），连通性结论一致，通路形状不同。
        复杂度 O(N)，N=121。
        """
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
            if (player == RED and r == ROWS - 1) or (player == BLUE and q == COLS - 1):
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
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        return path

    # ---------- 文本显示 ----------
    def render_text(self, path=None):
        """控制台文本棋盘：奇数行右移两格模拟六边形错位。
        符号：. 空、R 红、B 蓝、* 为 path 中的通路。"""
        lines = ["      " + "  ".join(chr(65 + q) for q in range(COLS))]
        for r in range(ROWS):
            indent = "  " if r % 2 == 1 else ""
            chars = []
            for q in range(COLS):
                v = self.grid[r][q]
                ch = "R" if v == RED else ("B" if v == BLUE else ".")
                if path and (q, r) in path:
                    ch = "*"
                chars.append(ch)
            lines.append(f"{r + 1:>2}  " + indent + "  ".join(chars))
        return "\n".join(lines)
