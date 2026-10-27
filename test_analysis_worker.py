# -*- coding: utf-8 -*-
"""Tests for the independent LLM analysis worker."""
import threading
import time
import unittest

from ai.analysis_result import AnalysisResult
from ai.analysis_worker import AnalysisWorker


OBSERVATION = {
    "schema_version": 1,
    "position_id": "hex1-test-position",
    "board_rows": ["." * 11 for _ in range(11)],
    "candidates": [{"move": [5, 5], "cell": "6F"}],
    "stones": {"RED": [], "BLUE": []},
}


def make_analysis():
    return AnalysisResult(
        schema_version=1,
        prompt_version="test-v1",
        position_id=OBSERVATION["position_id"],
        summary="测试分析",
        priority="balanced",
        model_name="mock-model",
        elapsed_ms=1.0,
    )


class SlowAdvisor:
    def __init__(self, result=None, error=None, delay=0.03):
        self.result = result
        self.error = error
        self.delay = delay
        self.started = threading.Event()

    def analyze(self, observation):
        self.started.set()
        time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


class AnalysisWorkerTest(unittest.TestCase):
    def test_successful_analysis(self):
        worker = AnalysisWorker(
            OBSERVATION,
            request_id=7,
            advisor=SlowAdvisor(make_analysis()),
        )
        worker.start()
        self.assertTrue(worker.join(1.0))
        result = worker.result
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.request_id, 7)
        self.assertEqual(result.position_id, OBSERVATION["position_id"])
        self.assertEqual(result.analysis.summary, "测试分析")
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_advisor_loader_is_used(self):
        analysis = make_analysis()
        calls = []
        worker = AnalysisWorker(
            OBSERVATION,
            request_id=1,
            advisor_loader=lambda: (calls.append(1), SlowAdvisor(analysis))[1],
        )
        worker.start()
        self.assertTrue(worker.join(1.0))
        self.assertEqual(calls, [1])
        self.assertEqual(worker.result.status, "complete")

    def test_errors_are_returned_without_crashing_thread(self):
        worker = AnalysisWorker(
            OBSERVATION,
            request_id=2,
            advisor=SlowAdvisor(error=RuntimeError("analysis failed")),
        )
        worker.start()
        self.assertTrue(worker.join(1.0))
        result = worker.result
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("analysis failed", result.error)

    def test_cancel_before_start(self):
        worker = AnalysisWorker(
            OBSERVATION, request_id=3, advisor=SlowAdvisor(make_analysis())
        )
        worker.cancel()
        worker.start()
        self.assertTrue(worker.join(1.0))
        self.assertEqual(worker.result.status, "cancelled")
        self.assertTrue(worker.result.cancelled)

    def test_cancel_after_start_discards_completed_analysis(self):
        advisor = SlowAdvisor(make_analysis(), delay=0.08)
        worker = AnalysisWorker(OBSERVATION, request_id=4, advisor=advisor)
        worker.start()
        self.assertTrue(advisor.started.wait(1.0))
        worker.cancel()
        self.assertTrue(worker.join(1.0))
        self.assertEqual(worker.result.status, "cancelled")
        self.assertIsNone(worker.result.analysis)

    def test_start_can_only_be_called_once(self):
        worker = AnalysisWorker(
            OBSERVATION, request_id=5, advisor=SlowAdvisor(make_analysis())
        )
        worker.start()
        with self.assertRaises(RuntimeError):
            worker.start()
        self.assertTrue(worker.join(1.0))

    def test_observation_is_copied(self):
        observation = {
            **OBSERVATION,
            "stones": {"RED": [], "BLUE": []},
        }
        advisor = SlowAdvisor(make_analysis())
        worker = AnalysisWorker(observation, request_id=6, advisor=advisor)
        observation["stones"]["RED"].append([5, 5])
        worker.start()
        self.assertTrue(worker.join(1.0))
        self.assertEqual(worker.observation["stones"]["RED"], [])

    def test_invalid_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisWorker({}, request_id=1, advisor=SlowAdvisor(make_analysis()))


if __name__ == "__main__":
    unittest.main()