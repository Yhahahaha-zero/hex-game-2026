# -*- coding: utf-8 -*-
"""基于单调时钟的双玩家计时器。"""
import time

from hex_board import BLUE, RED


def format_seconds(seconds):
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


class DualGameTimer:
    """分别累计红蓝双方的思考时间，不创建额外线程。"""

    def __init__(self, clock=None):
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self.totals = {RED: 0.0, BLUE: 0.0}
        self.active_color = None
        self.active_since = None
        self.paused_color = None

    def start(self, color):
        self.active_color = color
        self.active_since = self._clock()
        self.paused_color = None

    def _bank_active(self, now):
        if self.active_color is not None and self.active_since is not None:
            self.totals[self.active_color] += max(
                0.0, now - self.active_since
            )

    def switch_to(self, color):
        now = self._clock()
        self._bank_active(now)
        self.active_color = color
        self.active_since = now
        self.paused_color = None

    def pause(self):
        if self.active_color is None:
            return
        now = self._clock()
        self._bank_active(now)
        self.paused_color = self.active_color
        self.active_color = None
        self.active_since = None

    def resume(self):
        if self.paused_color is None:
            return
        self.active_color = self.paused_color
        self.active_since = self._clock()
        self.paused_color = None

    def stop(self):
        now = self._clock()
        self._bank_active(now)
        self.active_color = None
        self.active_since = None
        self.paused_color = None

    def elapsed(self, color):
        total = self.totals[color]
        if self.active_color == color and self.active_since is not None:
            total += max(0.0, self._clock() - self.active_since)
        return total

    def snapshot(self):
        return {
            RED: self.elapsed(RED),
            BLUE: self.elapsed(BLUE),
        }

    def formatted(self):
        values = self.snapshot()
        return {
            color: format_seconds(seconds)
            for color, seconds in values.items()
        }

    # =========================
    # 悔棋支持
    # =========================

    def savepoint(self):
        """保存当前计时状态，供悔棋恢复。"""
        return {
            "totals": dict(self.totals),
            "active_color": self.active_color,
            "active_since": self.active_since,
            "paused_color": self.paused_color,
        }

    def restore(self, snapshot):
        """恢复之前保存的计时状态。"""
        if not snapshot:
            return

        self.totals = dict(snapshot["totals"])
        self.active_color = snapshot["active_color"]
        self.paused_color = snapshot["paused_color"]

        # active_since 是基于 monotonic() 的时间点。
        # 恢复后重新建立当前玩家的计时起点，
        # 避免把悔棋期间的时间也算进去。
        if self.active_color is not None:
            self.active_since = self._clock()
        else:
            self.active_since = None