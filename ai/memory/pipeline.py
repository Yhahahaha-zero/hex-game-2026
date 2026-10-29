# -*- coding: utf-8 -*-
"""Background-safe post-game validation, feature and outcome pipeline."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time

from ai.memory.collector import _env_setting, _read_env_file
from ai.memory.features import FEATURE_VERSION, features_json
from ai.memory.outcomes import attribute_match_outcomes
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore, MemoryJobRecord
from ai.memory.validator import validate_match
from hex_board import EMPTY, BLUE, RED, HexBoard


PIPELINE_MODES = ("off", "validate", "features", "full")


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _board_from_json(raw_value):
    document = json.loads(raw_value)
    board = HexBoard()
    board.grid = [list(row) for row in document["grid"]]
    board.history = []
    return board


class MemoryLearningPipeline:
    def __init__(self, store_factory, *, mode="full"):
        if mode not in PIPELINE_MODES or mode == "off":
            raise ValueError("pipeline mode must be validate, features or full")
        self._store_factory = store_factory
        self.mode = mode

    def _ensure_job(self, store, match_id):
        job_id = f"finalize:{match_id}"
        return store.save_memory_job(MemoryJobRecord(
            job_id=job_id,
            job_type="finalize_match",
            target_id=match_id,
            status="pending",
            created_at=_now(),
        ))

    def process_match(self, match_id):
        store = self._store_factory()
        try:
            job = self._ensure_job(store, match_id)
            if job["status"] == "completed":
                return {"match_id": match_id, "status": "already_completed"}
            attempts = int(job.get("attempts", 0)) + 1
            store.update_memory_job(
                job["job_id"],
                status="running",
                attempts=attempts,
                started_at=_now(),
            )
            try:
                report = validate_match(store, match_id, promote=True)
                if not report.valid:
                    raise ValueError("match validation failed")
                feature_count = 0
                outcome_counts = {}
                if self.mode in ("features", "full"):
                    positions = store.list_positions_missing_features(
                        limit=100000, match_id=match_id
                    )
                    for position in positions:
                        board = _board_from_json(position["board_json"])
                        store.update_position_features(
                            position["position_id"],
                            features_json(
                                board,
                                int(position["current_color"]),
                                int(position["ply"]),
                            ),
                            feature_version=FEATURE_VERSION,
                        )
                        feature_count += 1
                if self.mode == "full":
                    outcome_counts = attribute_match_outcomes(store, match_id)
                store.update_memory_job(
                    job["job_id"],
                    status="completed",
                    attempts=attempts,
                    finished_at=_now(),
                )
                return {
                    "match_id": match_id,
                    "status": "completed",
                    "validation": report.as_dict(),
                    "features_updated": feature_count,
                    "outcomes": outcome_counts,
                }
            except Exception as exc:
                store.update_memory_job(
                    job["job_id"],
                    status="failed",
                    attempts=attempts,
                    last_error=str(exc),
                    finished_at=_now(),
                )
                raise
        finally:
            store.close()

    def process_pending(self, limit=10):
        store = self._store_factory()
        try:
            completed = {
                job["target_id"] for job in store.list_memory_jobs(limit=100000)
                if job["job_type"] == "finalize_match"
                and job["status"] == "completed"
            }
            matches = [
                row["match_id"] for row in store.list_matches(limit=100000)
                if row["match_id"] not in completed
            ][:limit]
        finally:
            store.close()
        results = []
        for match_id in matches:
            try:
                results.append(self.process_match(match_id))
            except Exception as exc:
                results.append({
                    "match_id": match_id,
                    "status": "failed",
                    "error": str(exc),
                })
        return tuple(results)


class NullMemoryPipelineWorker:
    enabled = False

    def submit(self, match_id):
        return False

    def close(self, timeout=3.0):
        return None

    def stats(self):
        return {"enabled": False, "queued": 0, "processed": 0, "failed": 0}


class MemoryPipelineWorker:
    enabled = True

    def __init__(self, pipeline, *, queue_size=32):
        self.pipeline = pipeline
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = None
        self._stop_event = threading.Event()
        self._sentinel = object()
        self._processed = 0
        self._failed = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="hex-memory-pipeline", daemon=True
        )
        self._thread.start()

    def submit(self, match_id):
        if self._stop_event.is_set():
            return False
        try:
            self.queue.put_nowait(str(match_id))
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            match_id = self.queue.get()
            try:
                if match_id is self._sentinel:
                    return
                try:
                    self.pipeline.process_match(match_id)
                    self._processed += 1
                except Exception:
                    self._failed += 1
            finally:
                self.queue.task_done()

    def close(self, timeout=3.0):
        if self._thread is None:
            return
        self._stop_event.set()
        deadline = time.monotonic() + float(timeout)
        while self._thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(self._sentinel, timeout=0.05)
                break
            except queue.Full:
                continue
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if not self._thread.is_alive():
            self._thread = None

    def stats(self):
        return {
            "enabled": True,
            "queued": self.queue.qsize(),
            "processed": self._processed,
            "failed": self._failed,
        }


def create_memory_pipeline_from_env(*, store_factory=None, env_file=None, environ=None):
    root = Path(__file__).resolve().parents[2]
    env_path = Path(env_file) if env_file is not None else root / ".env.local"
    file_values = _read_env_file(env_path)
    environment = os.environ if environ is None else environ
    mode = _env_setting(
        file_values, environment, "HEX_MEMORY_PIPELINE", "off"
    ).lower()
    if mode not in PIPELINE_MODES or mode == "off":
        return NullMemoryPipelineWorker()
    if store_factory is None:
        configured_db = _env_setting(file_values, environment, "HEX_MEMORY_DB", "")
        db_path = configured_db or str(memory_db_path(environment))
        store_factory = lambda: ExperienceStore(db_path).open()
    queue_size = int(_env_setting(
        file_values, environment, "HEX_MEMORY_PIPELINE_QUEUE_SIZE", "32"
    ))
    worker = MemoryPipelineWorker(
        MemoryLearningPipeline(store_factory, mode=mode),
        queue_size=queue_size,
    )
    worker.start()
    return worker