# -*- coding: utf-8 -*-
"""Background worker for optional LLM analysis, separate from move AI."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import threading
import time
from typing import Optional

from ai.analysis_result import AnalysisResult


@dataclass(frozen=True)
class AnalysisWorkerResult:
    request_id: int
    position_id: str
    status: str
    analysis: Optional[AnalysisResult] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    cancelled: bool = False
    elapsed_ms: float = 0.0


def _default_advisor_loader():
    from ai.llm_advisor import LLMAdvisor
    from ai.llm_client import create_llm_client
    from ai.llm_config import load_llm_config
    return LLMAdvisor(create_llm_client(load_llm_config()))


class AnalysisWorker:
    """Run one analysis request in a daemon thread."""

    def __init__(self, observation, request_id, *, advisor=None,
                 advisor_loader=None):
        if not isinstance(observation, dict) or not observation.get("position_id"):
            raise ValueError("observation 缺少 position_id")
        self.request_id = int(request_id)
        self.observation = copy.deepcopy(observation)
        self.position_id = str(observation["position_id"])
        self.advisor = advisor
        self._advisor_loader = advisor_loader or _default_advisor_loader
        self._thread = None
        self._result = None
        self._cancel_event = threading.Event()
        self._cancel_requested = False
        self._phase_lock = threading.Lock()
        self._phase = "idle"

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
    def phase(self):
        with self._phase_lock:
            return self._phase

    def _set_phase(self, phase):
        with self._phase_lock:
            self._phase = str(phase)

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self._thread is not None:
            raise RuntimeError("AnalysisWorker 只能启动一次")
        self._thread = threading.Thread(
            target=self.run, name="hex-analysis", daemon=True
        )
        self._thread.start()

    def cancel(self):
        self._cancel_requested = True
        self._cancel_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_alive()

    def _make_result(self, status, *, analysis=None, error=None,
                     error_type=None, elapsed_ms=0.0):
        return AnalysisWorkerResult(
            request_id=self.request_id,
            position_id=self.position_id,
            status=status,
            analysis=analysis,
            error=str(error)[:300] if error else None,
            error_type=str(error_type)[:100] if error_type else None,
            cancelled=status == "cancelled",
            elapsed_ms=round(max(0.0, float(elapsed_ms)), 3),
        )

    def run(self):
        if self._cancel_event.is_set():
            self._result = self._make_result("cancelled")
            self._set_phase("cancelled")
            return
        selected_advisor = self.advisor
        started = time.perf_counter()
        try:
            if selected_advisor is None:
                self._set_phase("loading")
                selected_advisor = self._advisor_loader()
            if self._cancel_event.is_set():
                self._result = self._make_result("cancelled")
                self._set_phase("cancelled")
                return
            self._set_phase("analyzing")
            analysis = selected_advisor.analyze(self.observation)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self._cancel_event.is_set():
                self._result = self._make_result("cancelled")
                self._set_phase("cancelled")
                return
            self._result = self._make_result(
                "complete", analysis=analysis, elapsed_ms=elapsed_ms
            )
            self._set_phase("complete")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self._cancel_event.is_set():
                self._result = self._make_result(
                    "cancelled", elapsed_ms=elapsed_ms
                )
                self._set_phase("cancelled")
                return
            self._result = self._make_result(
                "error",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                elapsed_ms=elapsed_ms,
            )
            self._set_phase("error")