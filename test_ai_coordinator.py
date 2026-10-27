# -*- coding: utf-8 -*-
"""Offline tests for HybridAI + LLM candidate coordination."""
import copy
import json
import tempfile
import threading
import unittest

from ai.ai_coordinator import (
    AICoordinator,
    CandidateAnalysisCache,
    LLMAssistedAI,
)
from ai.analysis_result import AnalysisResult, CandidateAnalysis
from ai.explanation import AIExplanation
from ai.llm_client import LLMTimeoutError
from hex_board import RED, HexBoard


BASE = (5, 5)
ALT = (4, 5)
THIRD = (6, 5)


def make_base(candidates=None, move=BASE, reason="Hybrid基准"):
    candidates = candidates or [move, ALT, THIRD]

    class BaseAI:
        def __init__(self):
            self.calls = 0
            self.cancel_event = None
            self.seed = None
            self.last_explanation = AIExplanation(
                selected_move=move,
                tactic=reason,
                candidates=tuple({
                    "move": cell,
                    "score": 10.0 - index,
                    "reason": reason,
                } for index, cell in enumerate(candidates)),
                elapsed_ms=1.0,
                search_type="hybrid-monte-carlo",
                score_type="启发式综合分",
            )

        def choose_move(self, board, player):
            self.calls += 1
            return move, reason

        def analysis_candidates(self, limit=8):
            return tuple({
                "move": cell,
                "base_rank": index,
                "base_score": 10.0 - index,
                "base_score_type": "启发式综合分",
                "reason": reason,
                "visits": None,
            } for index, cell in enumerate(candidates[:limit], 1))

        def set_cancel_event(self, event):
            self.cancel_event = event

        def set_seed(self, seed):
            self.seed = seed

    return BaseAI()


class FakeAdvisor:
    def __init__(self, recommended=ALT, error=None, position_override=None):
        self.recommended = recommended
        self.error = error
        self.position_override = position_override
        self.calls = 0
        self.observations = []
        self.model = "mock-coordinator"

    def analyze(self, observation):
        self.calls += 1
        self.observations.append(copy.deepcopy(observation))
        if self.error is not None:
            raise self.error
        return AnalysisResult(
            schema_version=1,
            prompt_version="test-v1",
            position_id=self.position_override or observation["position_id"],
            summary="测试候选分析",
            priority="balanced",
            game_state="playing",
            is_terminal=False,
            candidates=(
                CandidateAnalysis(
                    move=ALT,
                    preference="prefer",
                    pros="改善连接",
                    cons="需要防守",
                ),
            ),
            recommended_move=self.recommended,
            subjective_confidence=0.7,
            model_name=self.model,
            elapsed_ms=20.0,
            usage={"total_tokens": 30},
        )


class AICoordinatorTest(unittest.TestCase):
    def test_analysis_only_does_not_change_base_move(self):
        base = make_base()
        advisor = FakeAdvisor()
        coordinator = AICoordinator(base, advisor, mode="analysis_only")
        move, reason = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertIn("LLM确认", reason)
        self.assertEqual(advisor.calls, 1)
        self.assertEqual(coordinator.last_result.decision_source, "hybrid")
        self.assertIsNotNone(coordinator.last_result.analysis)

    def test_rerank_with_alpha_can_select_llm_preferred_candidate(self):
        base = make_base()
        advisor = FakeAdvisor(recommended=ALT)
        coordinator = AICoordinator(
            base, advisor, mode="rerank", alpha=1.0
        )
        move, reason = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, ALT)
        self.assertEqual(reason, "Hybrid+LLM候选重排")
        self.assertEqual(coordinator.last_result.decision_source, "hybrid_llm")
        self.assertEqual(coordinator.last_result.final_move, ALT)

    def test_single_candidate_skips_llm(self):
        base = make_base(candidates=[BASE], move=BASE)
        advisor = FakeAdvisor()
        coordinator = AICoordinator(base, advisor, mode="rerank", alpha=1.0)
        move, reason = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertIn("候选不足", reason)
        self.assertEqual(advisor.calls, 0)
        self.assertIn("候选不足", coordinator.last_result.fallback_reason)
    def test_forced_win_skips_llm(self):
        board = HexBoard()
        for row in range(10):
            board.place(0, row, RED)
        base = make_base(candidates=[(0, 10)], move=(0, 10), reason="直接获胜")
        advisor = FakeAdvisor(recommended=ALT)
        coordinator = AICoordinator(base, advisor, mode="rerank", alpha=1.0)
        move, reason = coordinator.decide(board, RED)
        self.assertEqual(move, (0, 10))
        self.assertEqual(reason, "直接获胜")
        self.assertEqual(advisor.calls, 0)
        self.assertEqual(coordinator.last_result.decision_source, "forced")

    def test_llm_timeout_falls_back_to_base(self):
        base = make_base()
        advisor = FakeAdvisor(error=LLMTimeoutError("timeout"))
        coordinator = AICoordinator(base, advisor, mode="rerank", alpha=1.0)
        move, _ = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertEqual(coordinator.last_result.decision_source, "fallback")
        self.assertIn("LLMTimeoutError", coordinator.last_result.fallback_reason)

    def test_invalid_recommendation_falls_back_to_base(self):
        base = make_base()
        advisor = FakeAdvisor(recommended=(0, 0))
        coordinator = AICoordinator(base, advisor, mode="rerank", alpha=1.0)
        move, _ = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertEqual(coordinator.last_result.decision_source, "fallback")
        self.assertIn("合法候选", coordinator.last_result.fallback_reason)

    def test_position_mismatch_falls_back_to_base(self):
        base = make_base()
        advisor = FakeAdvisor(position_override="hex1-wrong")
        coordinator = AICoordinator(base, advisor, mode="rerank", alpha=1.0)
        move, _ = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertEqual(coordinator.last_result.decision_source, "fallback")
        self.assertIn("不一致", coordinator.last_result.fallback_reason)

    def test_candidate_observation_preserves_base_order(self):
        base = make_base()
        advisor = FakeAdvisor()
        coordinator = AICoordinator(base, advisor, mode="analysis_only")
        coordinator.decide(HexBoard(), RED)
        candidates = advisor.observations[0]["candidates"]
        self.assertEqual(candidates[0]["move"], list(BASE))
        self.assertEqual(candidates[1]["move"], list(ALT))
        self.assertEqual(candidates[0]["source"], "hybrid_base_rank")

    def test_board_is_read_only(self):
        board = HexBoard()
        board.place(5, 5, RED)
        before_grid = [row[:] for row in board.grid]
        before_history = list(board.history)
        coordinator = AICoordinator(
            make_base(candidates=[BASE, ALT], move=ALT),
            FakeAdvisor(recommended=BASE),
            mode="analysis_only",
        )
        coordinator.decide(board, RED)
        self.assertEqual(board.grid, before_grid)
        self.assertEqual(board.history, before_history)

    def test_cache_hit_avoids_second_model_call(self):
        board = HexBoard()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CandidateAnalysisCache(temp_dir)
            first_advisor = FakeAdvisor(recommended=ALT)
            first = AICoordinator(
                make_base(), first_advisor, mode="analysis_only",
                cache=cache, model_name="cache-model",
            )
            first.decide(board, RED)
            self.assertEqual(first_advisor.calls, 1)

            second_advisor = FakeAdvisor(error=AssertionError("should not call"))
            second = AICoordinator(
                make_base(), second_advisor, mode="analysis_only",
                cache=cache, model_name="cache-model",
            )
            move, _ = second.decide(board, RED)
        self.assertEqual(move, BASE)
        self.assertEqual(second_advisor.calls, 0)
        self.assertTrue(second.last_result.cache_hit)

    def test_cache_key_includes_candidate_order_and_model(self):
        first = CandidateAnalysisCache.make_key(
            position_id="p", model_name="m1", prompt_version="v1",
            candidates=[BASE, ALT],
        )
        second = CandidateAnalysisCache.make_key(
            position_id="p", model_name="m1", prompt_version="v1",
            candidates=[ALT, BASE],
        )
        third = CandidateAnalysisCache.make_key(
            position_id="p", model_name="m2", prompt_version="v1",
            candidates=[BASE, ALT],
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_cache_write_failure_does_not_discard_valid_analysis(self):
        class BrokenCache:
            def get(self, key):
                return None

            def put(self, key, analysis):
                raise OSError("read-only cache")

        coordinator = AICoordinator(
            make_base(), FakeAdvisor(), mode="analysis_only",
            cache=BrokenCache(), model_name="cache-model",
        )
        move, _ = coordinator.decide(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertIsNotNone(coordinator.last_result.analysis)
        self.assertEqual(coordinator.last_result.decision_source, "hybrid")
    def test_coordinator_result_is_json_serializable(self):
        coordinator = AICoordinator(make_base(), FakeAdvisor(), mode="analysis_only")
        coordinator.decide(HexBoard(), RED)
        encoded = json.dumps(
            coordinator.last_result.as_dict(), ensure_ascii=False, allow_nan=False
        )
        decoded = json.loads(encoded)
        self.assertEqual(decoded["base_move"], list(BASE))

    def test_llm_assisted_ai_adapter_exposes_explanation(self):
        base = make_base()
        ai = LLMAssistedAI(base, FakeAdvisor(), mode="analysis_only")
        move, reason = ai.choose_move(HexBoard(), RED)
        self.assertEqual(move, BASE)
        self.assertIn("LLM确认", reason)
        self.assertIsNotNone(ai.last_explanation)
        self.assertEqual(ai.last_explanation.selected_move, BASE)

    def test_cancel_event_is_forwarded_to_base_ai(self):
        base = make_base()
        event = threading.Event()
        ai = LLMAssistedAI(base, FakeAdvisor(), mode="analysis_only")
        ai.set_cancel_event(event)
        ai.set_seed(123)
        self.assertIs(base.cancel_event, event)
        self.assertEqual(base.seed, 123)


if __name__ == "__main__":
    unittest.main()