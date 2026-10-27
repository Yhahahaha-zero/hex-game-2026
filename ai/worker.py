# -*- coding: utf-8 -*-
"""页面层使用的可取消后台 AI 任务和不可变结果对象。"""
from dataclasses import dataclass
import threading
import time
from typing import Any, Optional

from ai.explanation import normalize_explanation
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
    explanation: Optional[dict] = None
    cancelled: bool = False
    ai_name: Optional[str] = None
    ai_version: Optional[str] = None
    model_sha256: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self):
        return (not self.cancelled and self.fallback_error is None
                and self.move is not None)

    @property
    def used_fallback(self):
        return self.init_error is not None or self.inference_error is not None


def _snapshot_board(board):
    snapshot = HexBoard()
    snapshot.grid = [row[:] for row in board.grid]
    snapshot.history = list(board.history)
    return snapshot


def _ai_metadata(ai):
    name = str(getattr(ai, "name", "") or type(ai).__name__)
    version = str(getattr(ai, "version", "") or "")
    model_sha256 = str(getattr(ai, "model_sha256", "") or "")
    selection = getattr(ai, "model_selection", None)
    if not model_sha256 and isinstance(selection, dict):
        model_sha256 = str(selection.get("sha256", "") or "")
    return name, version or None, model_sha256 or None


def _default_ai_loader():
    from ai.ai_factory import create_competition_ai
    return create_competition_ai()


class AIWorker:
    """在后台线程加载 AI、获取着法并在失败时回退 BasicAI。"""

    def __init__(self, board, color, request_id, ai=None, ai_loader=None,
                 memory_context=None):
        if color not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {color!r}")
        self.request_id = request_id
        self.color = color
        self.board = _snapshot_board(board)
        self.ai = ai
        self._ai_loader = ai_loader or _default_ai_loader
        self._memory_context = dict(memory_context or {})
        self._result = None
        self._thread = None
        self._progress_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._cancel_requested = False
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
    def cancelled(self):
        return self._cancel_requested

    @property
    def cancel_event(self):
        return self._cancel_event

    @property
    def progress(self):
        with self._progress_lock:
            return dict(self._progress)

    def _update_progress(self, **values):
        with self._progress_lock:
            self._progress.update(values)

    def _on_ai_progress(self, progress):
        if self._cancel_event.is_set():
            return
        self._update_progress(**progress)

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())

    def cancel(self):
        """Request cooperative cancellation; safe to call repeatedly."""
        self._cancel_requested = True
        self._cancel_event.set()
        self._update_progress(phase="cancelling", best_move=None)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("AIWorker 只能启动一次")
        self._thread = threading.Thread(target=self.run, name="hex-ai", daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        """Wait for the worker and return True when the thread has stopped."""
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_alive()

    def _cancel_result(self, selected_ai, init_error=None,
                       inference_error=None, fallback_error=None):
        self._update_progress(phase="cancelled", best_move=None)
        ai_name, ai_version, model_sha256 = _ai_metadata(selected_ai)
        progress = self.progress
        self._result = AIResult(
            request_id=self.request_id,
            color=self.color,
            move=None,
            reason="AI 任务已取消",
            ai=selected_ai,
            init_error=init_error,
            inference_error=inference_error,
            fallback_error=fallback_error,
            search_stats=tuple(progress.get("candidates", ())),
            cancelled=True,
            ai_name=ai_name,
            ai_version=ai_version,
            model_sha256=model_sha256,
        )

    def _attach_cancel_event(self, selected_ai):
        setter = getattr(selected_ai, "set_cancel_event", None)
        if callable(setter):
            setter(self._cancel_event)

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

        context_setter = getattr(selected_ai, "set_memory_context", None)
        if callable(context_setter):
            context_setter(**self._memory_context)
        self._attach_cancel_event(selected_ai)
        if self._cancel_event.is_set():
            self._cancel_result(selected_ai, init_error=init_error)
            return

        self._update_progress(phase="thinking")
        set_callback = getattr(selected_ai, "set_progress_callback", None)
        if callable(set_callback):
            set_callback(self._on_ai_progress)

        move = None
        reason = ""
        started = time.perf_counter()
        try:
            move, reason = selected_ai.choose_move(self.board, self.color)
            if self._cancel_event.is_set():
                self._cancel_result(selected_ai, init_error=init_error)
                return
            if not self._legal_move(move):
                raise ValueError(f"AI 返回非法着法: {move!r}")
        except Exception as exc:
            if self._cancel_event.is_set():
                self._cancel_result(
                    selected_ai,
                    init_error=init_error,
                    inference_error=str(exc),
                )
                return
            inference_error = str(exc)
            try:
                selected_ai = BasicAI()
                self._attach_cancel_event(selected_ai)
                move, reason = selected_ai.choose_move(self.board, self.color)
                if self._cancel_event.is_set():
                    self._cancel_result(
                        selected_ai,
                        init_error=init_error,
                        inference_error=inference_error,
                    )
                    return
                if not self._legal_move(move):
                    raise ValueError(f"回退 AI 返回非法着法: {move!r}")
            except Exception as fallback_exc:
                if self._cancel_event.is_set():
                    self._cancel_result(
                        selected_ai,
                        init_error=init_error,
                        inference_error=inference_error,
                        fallback_error=str(fallback_exc),
                    )
                    return
                fallback_error = str(fallback_exc)
                move, reason = None, ""

        self._update_progress(
            phase="done" if fallback_error is None else "error",
            best_move=move,
        )
        progress = self.progress
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        explanation = normalize_explanation(
            getattr(selected_ai, "last_explanation", None),
            move=move,
            reason=reason,
            elapsed_ms=elapsed_ms,
        )
        explanation_stats = tuple(explanation.get("candidates", ()))
        progress_stats = tuple(progress.get("candidates", ()))
        ai_name, ai_version, model_sha256 = _ai_metadata(selected_ai)
        self._result = AIResult(
            request_id=self.request_id,
            color=self.color,
            move=move,
            reason=reason,
            ai=selected_ai,
            init_error=init_error,
            inference_error=inference_error,
            fallback_error=fallback_error,
            search_stats=progress_stats or explanation_stats,
            explanation=explanation,
            ai_name=ai_name,
            ai_version=ai_version,
            model_sha256=model_sha256,
            elapsed_ms=elapsed_ms,
        )

    def _legal_move(self, move):
        if not isinstance(move, tuple) or len(move) != 2:
            return False
        q, r = move
        return self.board.in_range(q, r) and self.board.is_empty(q, r)
