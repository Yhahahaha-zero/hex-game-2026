# -*- coding: utf-8 -*-
"""页面层使用的后台 AI 任务和不可变结果对象。"""
from dataclasses import dataclass
import threading
from typing import Any, Optional

from hex_board import BLUE, HexBoard, RED


@dataclass(frozen=True)
class AIResult:
    """一次 AI 请求的最终结果，由 UI 主线程消费。"""

    request_id: int
    color: int
    move: Optional[tuple[int, int]]
    reason: str
    ai: Optional[Any] = None
    init_error: Optional[str] = None
    inference_error: Optional[str] = None
    fallback_error: Optional[str] = None
    search_stats: tuple = ()

    @property
    def ok(self):
        return self.fallback_error is None and self.move is not None

    @property
    def used_fallback(self):
        return self.init_error is not None or self.inference_error is not None


def _snapshot_board(board):
    snapshot = HexBoard()
    snapshot.grid = [row[:] for row in board.grid]
    snapshot.history = list(board.history)
    return snapshot


def _default_ai_loader():
    from ai.neural_ai import create_ai_with_fallback
    return create_ai_with_fallback()


class AIWorker:
    """在后台线程加载 AI、获取着法并在失败时回退 BasicAI。"""

    def __init__(self, board, color, request_id, ai=None, ai_loader=None):
        if color not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {color!r}")
        self.request_id = request_id
        self.color = color
        self.board = _snapshot_board(board)
        self.ai = ai
        self._ai_loader = ai_loader or _default_ai_loader
        self._result = None
        self._thread = None
        self._progress_lock = threading.Lock()
        self._progress = {
            "phase": "idle",
            "completed": 0,
            "total": 0,
            "elapsed_ms": 0.0,
            "best_move": None,
        }

    @property
    def result(self):
        return self._result

    @property
    def progress(self):
        with self._progress_lock:
            return dict(self._progress)

    def _update_progress(self, **values):
        with self._progress_lock:
            self._progress.update(values)

    def _on_ai_progress(self, progress):
        self._update_progress(**progress)

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self._thread is not None:
            raise RuntimeError("AIWorker 只能启动一次")
        self._thread = threading.Thread(target=self.run, name="hex-ai", daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self):
        from hex_ai import BasicAI

        init_error = None
        inference_error = None
        fallback_error = None
        selected_ai = self.ai

        if selected_ai is None:
            self._update_progress(phase="loading")
            try:
                selected_ai, init_error = self._ai_loader()
            except Exception as exc:
                init_error = str(exc)
                selected_ai = BasicAI()

        self._update_progress(phase="thinking")
        set_callback = getattr(selected_ai, "set_progress_callback", None)
        if callable(set_callback):
            set_callback(self._on_ai_progress)

        try:
            move, reason = selected_ai.choose_move(self.board, self.color)
            if not self._legal_move(move):
                raise ValueError(f"AI 返回非法着法: {move!r}")
        except Exception as exc:
            inference_error = str(exc)
            try:
                selected_ai = BasicAI()
                move, reason = selected_ai.choose_move(self.board, self.color)
                if not self._legal_move(move):
                    raise ValueError(f"回退 AI 返回非法着法: {move!r}")
            except Exception as fallback_exc:
                fallback_error = str(fallback_exc)
                move, reason = None, ""

        self._update_progress(
            phase="done" if fallback_error is None else "error",
            best_move=move,
        )
        progress = self.progress
        self._result = AIResult(
            request_id=self.request_id,
            color=self.color,
            move=move,
            reason=reason,
            ai=selected_ai,
            init_error=init_error,
            inference_error=inference_error,
            fallback_error=fallback_error,
            search_stats=tuple(progress.get("candidates", ())),
        )

    def _legal_move(self, move):
        if not isinstance(move, tuple) or len(move) != 2:
            return False
        q, r = move
        return self.board.in_range(q, r) and self.board.is_empty(q, r)
