# -*- coding: utf-8 -*-
"""Background worker for non-blocking screenshot import."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Optional

from vision.types import VisionBoardState


@dataclass(frozen=True)
class VisionWorkerResult:
    request_id: int
    status: str
    state: Optional[VisionBoardState] = None
    session: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    cancelled: bool = False
    elapsed_ms: float = 0.0


class VisionImportWorker:
    """Run one standard-image import in a daemon thread."""

    def __init__(self, image_path, request_id, *, use_vlm=False, client=None,
                 model_name="qwen3-vl-plus", min_confidence=0.78,
                 max_image_mb=10, cache_enabled=True,
                 cache_dir=None):
        self.image_path = image_path
        self.request_id = int(request_id)
        self.use_vlm = bool(use_vlm)
        self.client = client
        self.model_name = str(model_name)
        self.min_confidence = float(min_confidence)
        self.max_image_mb = int(max_image_mb)
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = cache_dir
        self._thread = None
        self._result = None
        self._cancel_event = threading.Event()
        self._cancel_requested = False
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
        return self._phase

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self._thread is not None:
            raise RuntimeError("VisionImportWorker 只能启动一次")
        self._thread = threading.Thread(
            target=self.run, name="hex-vision-import", daemon=True
        )
        self._thread.start()

    def cancel(self):
        self._cancel_requested = True
        self._cancel_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_alive()

    def _cancelled_result(self, elapsed_ms=0.0):
        return VisionWorkerResult(
            request_id=self.request_id,
            status="cancelled",
            cancelled=True,
            elapsed_ms=elapsed_ms,
        )

    def run(self):
        from vision.import_session import import_standard_image

        if self._cancel_event.is_set():
            self._result = self._cancelled_result()
            self._phase = "cancelled"
            return
        started = time.perf_counter()
        selected_client = self.client
        try:
            if self.use_vlm and selected_client is None:
                from ai.llm_client import LLMDisabledError, create_llm_client
                from ai.llm_config import load_llm_config
                config = load_llm_config()
                if config.enabled:
                    selected_client = create_llm_client(config)
                else:
                    selected_client = None
            self._phase = "recognizing"
            session = import_standard_image(
                self.image_path,
                min_confidence=self.min_confidence,
                client=selected_client,
                model_name=self.model_name,
                use_vlm=self.use_vlm,
                cache_dir=self.cache_dir,
                max_image_mb=self.max_image_mb,
                cache_enabled=self.cache_enabled,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self._cancel_event.is_set():
                self._result = self._cancelled_result(elapsed_ms)
                self._phase = "cancelled"
                return
            self._result = VisionWorkerResult(
                request_id=self.request_id,
                status="complete",
                state=session.state,
                session=session,
                elapsed_ms=round(elapsed_ms, 3),
            )
            self._phase = "complete"
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._result = VisionWorkerResult(
                request_id=self.request_id,
                status="cancelled" if self._cancel_event.is_set() else "error",
                error=str(exc)[:300] or type(exc).__name__,
                error_type=type(exc).__name__,
                cancelled=self._cancel_event.is_set(),
                elapsed_ms=round(elapsed_ms, 3),
            )
            self._phase = self._result.status