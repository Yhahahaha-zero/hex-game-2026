# -*- coding: utf-8 -*-
"""Offline tests for LLMAdvisor and structured analysis validation."""
import copy
import json
import unittest

from ai.analysis_result import (
    AnalysisValidationError,
    parse_analysis_text,
)
from ai.board_observation import build_engine_observation
from ai.llm_advisor import LLMAdvisor, SYSTEM_PROMPT, build_advisor_messages
from ai.llm_client import LLMResponse, LLMTimeoutError, MockLLMClient
from hex_engine import HexEngine


def make_observation():
    engine = HexEngine("红队", "蓝队")
    engine.play_move(5, 5, reason="中心开局")
    engine.play_move(4, 5, reason="防守")
    engine.play_move(5, 4, reason="连接")
    engine.play_move(6, 5, reason="阻断")
    return engine, build_engine_observation(engine, candidate_limit=6)


def make_payload(observation):
    candidate = observation["candidates"][0]["move"]
    return {
        "schema_version": 1,
        "position_id": observation["position_id"],
        "summary": "红方需要兼顾纵向连接并阻断蓝方横向通路。",
        "priority": "defense",
        "threats": [{
            "move": [0, 0],
            "severity": "high",
            "reason": "边缘关键点",
        }],
        "candidate_analysis": [{
            "move": candidate,
            "preference": "prefer",
            "pros": "可以改善连接",
            "cons": "需要关注对手反击",
        }],
        "recommended_move": candidate,
        "subjective_confidence": 0.7,
        "warnings": [],
    }


def mock_from_payload(payload, **response_kwargs):
    text = json.dumps(payload, ensure_ascii=False)
    response = LLMResponse(
        text=text,
        model=response_kwargs.get("model", "qwen3.8-flash"),
        elapsed_ms=response_kwargs.get("elapsed_ms", 12.5),
        status_code=200,
        request_id="request-test",
        usage=response_kwargs.get("usage", {"total_tokens": 20}),
    )
    return MockLLMClient(responses=[response])


class LLMAdvisorTest(unittest.TestCase):
    def setUp(self):
        self.engine, self.observation = make_observation()
        self.payload = make_payload(self.observation)
        self.candidate = tuple(self.payload["recommended_move"])

    def test_prompt_contains_observation_schema_and_rules(self):
        messages = build_advisor_messages(self.observation)
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[0]["role"], "system")
        prompt = messages[1]["content"]
        self.assertIn(self.observation["position_id"], prompt)
        self.assertIn('"candidate_analysis"', prompt)
        self.assertIn('"recommended_move"', prompt)
        self.assertIn("candidates", prompt)
        self.assertIn("红方连接上下边界", messages[0]["content"])
        self.assertIn("蓝方连接左右边界", messages[0]["content"])
        self.assertNotIn("隐藏思维链内容", prompt)
        self.assertEqual(messages, build_advisor_messages(self.observation))

    def test_valid_analysis_returns_immutable_result(self):
        advisor = LLMAdvisor(mock_from_payload(self.payload))
        result = advisor.analyze(self.observation)

        self.assertEqual(result.position_id, self.observation["position_id"])
        self.assertEqual(result.priority, "defense")
        self.assertEqual(result.recommended_move, self.candidate)
        self.assertEqual(result.model_name, "qwen3.8-flash")
        self.assertEqual(result.elapsed_ms, 12.5)
        self.assertEqual(result.usage["total_tokens"], 20)
        self.assertEqual(len(result.threats), 1)
        self.assertEqual(len(result.candidates), 1)

    def test_markdown_fenced_json_is_accepted(self):
        text = "```json\n" + json.dumps(self.payload, ensure_ascii=False) + "\n```"
        client = MockLLMClient(response_text=text)
        result = LLMAdvisor(client).analyze(self.observation)
        self.assertEqual(result.recommended_move, self.candidate)

    def test_small_surrounding_text_is_accepted(self):
        text = "分析如下：\n" + json.dumps(self.payload, ensure_ascii=False) + "\n分析结束。"
        result = parse_analysis_text(
            text, self.observation, model_name="test-model"
        )
        self.assertEqual(result.summary, self.payload["summary"])

    def test_multiple_json_objects_are_rejected(self):
        text = json.dumps(self.payload) + json.dumps(self.payload)
        with self.assertRaises(AnalysisValidationError):
            parse_analysis_text(text, self.observation)

    def test_position_id_mismatch_is_rejected(self):
        self.payload["position_id"] = "hex1-wrong"
        with self.assertRaisesRegex(AnalysisValidationError, "当前局面"):
            parse_analysis_text(json.dumps(self.payload), self.observation)

    def test_unsupported_schema_is_rejected(self):
        for value in (True, 2, "1"):
            with self.subTest(value=value):
                payload = dict(self.payload)
                payload["schema_version"] = value
                with self.assertRaises(AnalysisValidationError):
                    parse_analysis_text(json.dumps(payload), self.observation)

    def test_empty_summary_is_rejected(self):
        self.payload["summary"] = "   "
        with self.assertRaises(AnalysisValidationError):
            parse_analysis_text(json.dumps(self.payload), self.observation)

    def test_invalid_priority_is_rejected(self):
        for value in ("unknown", ["defense"]):
            with self.subTest(value=value):
                payload = dict(self.payload)
                payload["priority"] = value
                with self.assertRaises(AnalysisValidationError):
                    parse_analysis_text(json.dumps(payload), self.observation)

    def test_invalid_and_occupied_threats_are_removed(self):
        occupied = self.observation["stones"]["RED"][0]
        self.payload["threats"] = [
            {"move": [99, 0], "severity": "high", "reason": "越界"},
            {"move": occupied, "severity": "high", "reason": "已占用"},
            {"move": [0, 0], "severity": ["invalid"], "reason": "类型错误"},
            {"move": [0, 1], "severity": "medium", "reason": "有效威胁"},
        ]
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertEqual([item.move for item in result.threats], [(0, 1)])
        self.assertTrue(result.warnings)

    def test_duplicate_threat_is_removed(self):
        threat = {"move": [0, 0], "severity": "high", "reason": "威胁"}
        self.payload["threats"] = [threat, dict(threat)]
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertEqual(len(result.threats), 1)
        self.assertTrue(any("重复威胁" in item for item in result.warnings))

    def test_candidate_outside_candidate_set_is_removed(self):
        self.payload["candidate_analysis"] = [{
            "move": [0, 0],
            "preference": "prefer",
            "pros": "虚构候选",
            "cons": "",
        }]
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertEqual(result.candidates, ())
        self.assertTrue(any("候选集合" in item for item in result.warnings))

    def test_duplicate_candidate_analysis_is_removed(self):
        item = self.payload["candidate_analysis"][0]
        self.payload["candidate_analysis"] = [item, dict(item)]
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(any("重复候选" in item for item in result.warnings))

    def test_invalid_recommendation_is_cleared(self):
        self.payload["recommended_move"] = [0, 0]
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertIsNone(result.recommended_move)
        self.assertTrue(any("推荐着法" in item for item in result.warnings))

    def test_invalid_confidence_is_cleared(self):
        for value in (-0.1, 1.1, True, "high"):
            with self.subTest(value=value):
                payload = dict(self.payload)
                payload["subjective_confidence"] = value
                result = parse_analysis_text(json.dumps(payload), self.observation)
                self.assertIsNone(result.subjective_confidence)
                self.assertTrue(result.warnings)

    def test_long_summary_is_truncated(self):
        self.payload["summary"] = "长" * 600
        result = parse_analysis_text(json.dumps(self.payload), self.observation)
        self.assertEqual(len(result.summary), 500)
        self.assertTrue(any("截断" in item for item in result.warnings))

    def test_warnings_must_be_array(self):
        self.payload["warnings"] = "not-a-list"
        with self.assertRaises(AnalysisValidationError):
            parse_analysis_text(json.dumps(self.payload), self.observation)

    def test_invalid_json_raises_validation_error(self):
        with self.assertRaises(AnalysisValidationError):
            parse_analysis_text("not json", self.observation)

    def test_client_timeout_is_propagated(self):
        client = MockLLMClient(responses=[LLMTimeoutError("mock timeout")])
        with self.assertRaises(LLMTimeoutError):
            LLMAdvisor(client).analyze(self.observation)

    def test_analysis_does_not_mutate_observation_or_board(self):
        observation_before = copy.deepcopy(self.observation)
        grid_before = [row[:] for row in self.engine.board.grid]
        history_before = list(self.engine.board.history)

        LLMAdvisor(mock_from_payload(self.payload)).analyze(self.observation)

        self.assertEqual(self.observation, observation_before)
        self.assertEqual(self.engine.board.grid, grid_before)
        self.assertEqual(self.engine.board.history, history_before)

    def test_result_is_json_serializable(self):
        result = LLMAdvisor(mock_from_payload(self.payload)).analyze(self.observation)
        encoded = json.dumps(result.as_dict(), ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["recommended_move"], list(self.candidate))

    def test_advisor_passes_model_timeout_and_token_limit(self):
        client = mock_from_payload(self.payload)
        advisor = LLMAdvisor(
            client,
            model="custom-model",
            timeout=3.5,
            max_tokens=64,
        )
        advisor.analyze(self.observation)
        call = client.calls[0]
        self.assertEqual(call["model"], "custom-model")
        self.assertEqual(call["timeout"], 3.5)
        self.assertEqual(call["max_tokens"], 64)

    def test_oversized_prompt_is_rejected(self):
        observation = copy.deepcopy(self.observation)
        observation["unexpected_large_field"] = "x" * 31_000
        with self.assertRaisesRegex(AnalysisValidationError, "提示词"):
            build_advisor_messages(observation)


if __name__ == "__main__":
    unittest.main()