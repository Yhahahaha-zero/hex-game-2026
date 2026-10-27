# -*- coding: utf-8 -*-
"""Offline tests for the Qwen-compatible LLM client."""
import json
from pathlib import Path
import tempfile
import unittest

from ai.llm_client import (
    MAX_RESPONSE_BYTES,
    DisabledLLMClient,
    LLMAuthenticationError,
    LLMDisabledError,
    LLMRequestError,
    LLMResponse,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    MockLLMClient,
    QwenClient,
    create_llm_client,
)
from ai.llm_config import LLMConfig


TEST_KEY = "test-secret-key"


def make_config(**overrides):
    values = {
        "enabled": True,
        "provider": "qwen",
        "api_key": TEST_KEY,
        "base_url": "https://example.test/v1",
        "analysis_model": "qwen3.8-flash",
        "vision_model": "qwen3-vl-plus",
        "timeout_seconds": 8.0,
        "max_retries": 1,
        "max_output_tokens": 1000,
        "temperature": 0.1,
    }
    values.update(overrides)
    return LLMConfig(**values)


def response_body(text="OK", content=None, include_usage=True, include_id=True):
    payload = {
        "model": "qwen3.8-flash",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text if content is None else content,
            }
        }],
    }
    if include_usage:
        payload["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": 2,
            "total_tokens": 14,
        }
    if include_id:
        payload["id"] = "request-test-1"
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def ok_response(text="OK", content=None, **kwargs):
    return 200, {"Content-Type": "application/json"}, response_body(
        text=text, content=content, **kwargs
    )


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "payload": json.loads(body.decode("utf-8")),
            "timeout": timeout,
        })
        if not self.outcomes:
            raise AssertionError("FakeTransport 没有剩余响应")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class LLMClientTest(unittest.TestCase):
    def test_disabled_client_and_factory(self):
        config = make_config(enabled=False, api_key="", base_url="")
        client = create_llm_client(config)
        self.assertIsInstance(client, DisabledLLMClient)
        self.assertFalse(client.available)
        with self.assertRaises(LLMDisabledError):
            client.analyze_text("test")

    def test_factory_returns_qwen_client_when_enabled(self):
        client = create_llm_client(make_config(), transport=FakeTransport([]))
        self.assertIsInstance(client, QwenClient)

    def test_text_request_headers_body_and_response(self):
        transport = FakeTransport([ok_response("分析完成")])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        response = client.analyze_text(
            "分析局面", system_prompt="你是 Hex 助手", enable_thinking=False
        )

        self.assertEqual(response.text, "分析完成")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.request_id, "request-test-1")
        self.assertEqual(response.usage["total_tokens"], 14)
        call = transport.calls[0]
        self.assertEqual(
            call["url"],
            "https://example.test/v1/chat/completions",
        )
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {TEST_KEY}")
        self.assertEqual(call["payload"]["model"], "qwen3.8-flash")
        self.assertEqual(call["payload"]["messages"][0]["role"], "system")
        self.assertEqual(call["payload"]["messages"][1]["content"], "分析局面")
        self.assertFalse(call["payload"]["stream"])
        self.assertEqual(call["payload"]["temperature"], 0.1)
        self.assertEqual(call["payload"]["max_tokens"], 1000)
        self.assertFalse(call["payload"]["enable_thinking"])

    def test_model_and_token_limit_can_be_overridden(self):
        transport = FakeTransport([ok_response()])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        client.chat(
            [{"role": "user", "content": "test"}],
            model="custom-model",
            max_tokens=32,
        )
        self.assertEqual(transport.calls[0]["payload"]["model"], "custom-model")
        self.assertEqual(transport.calls[0]["payload"]["max_tokens"], 32)

    def test_multi_part_content_is_joined(self):
        transport = FakeTransport([ok_response(content=[
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"},
        ])])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        response = client.analyze_text("test")
        self.assertEqual(response.text, "第一段第二段")

    def test_usage_is_optional(self):
        transport = FakeTransport([ok_response(include_usage=False)])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        response = client.analyze_text("test")
        self.assertEqual(response.usage, {})

    def test_auth_error_is_not_retried(self):
        transport = FakeTransport([(401, {}, b"")])
        sleeps = []
        client = QwenClient(
            make_config(), transport=transport, sleep_func=sleeps.append
        )
        with self.assertRaises(LLMAuthenticationError) as caught:
            client.analyze_text("test")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(sleeps, [])
        self.assertNotIn(TEST_KEY, str(caught.exception))

    def test_rate_limit_is_retried_once(self):
        transport = FakeTransport([(429, {}, b""), ok_response()])
        sleeps = []
        client = QwenClient(
            make_config(max_retries=1),
            transport=transport,
            sleep_func=sleeps.append,
        )
        response = client.analyze_text("test")
        self.assertEqual(response.text, "OK")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_server_errors_exhaust_retry_budget(self):
        transport = FakeTransport([(500, {}, b""), (503, {}, b"")])
        sleeps = []
        client = QwenClient(
            make_config(max_retries=1),
            transport=transport,
            sleep_func=sleeps.append,
        )
        with self.assertRaises(LLMServerError):
            client.analyze_text("test")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_timeout_is_retried_then_succeeds(self):
        transport = FakeTransport([LLMTimeoutError("请求超时"), ok_response()])
        sleeps = []
        client = QwenClient(
            make_config(max_retries=1),
            transport=transport,
            sleep_func=sleeps.append,
        )
        response = client.analyze_text("test")
        self.assertEqual(response.text, "OK")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_timeout_is_raised_after_retry_budget(self):
        transport = FakeTransport([
            LLMTimeoutError("请求超时"),
            LLMTimeoutError("请求超时"),
        ])
        client = QwenClient(make_config(max_retries=1), transport=transport,
                            sleep_func=lambda _: None)
        with self.assertRaises(LLMTimeoutError):
            client.analyze_text("test")

    def test_invalid_json_is_rejected_without_retry(self):
        transport = FakeTransport([(200, {}, b"not-json")])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        with self.assertRaises(LLMResponseError):
            client.analyze_text("test")
        self.assertEqual(len(transport.calls), 1)

    def test_missing_choices_is_rejected(self):
        body = json.dumps({"model": "qwen3.8-flash"}).encode("utf-8")
        transport = FakeTransport([(200, {}, body)])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        with self.assertRaises(LLMResponseError):
            client.analyze_text("test")

    def test_empty_content_is_rejected(self):
        transport = FakeTransport([ok_response(content="")])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        with self.assertRaises(LLMResponseError):
            client.analyze_text("test")

    def test_oversized_response_is_rejected(self):
        transport = FakeTransport([(200, {}, b"x" * (MAX_RESPONSE_BYTES + 1))])
        client = QwenClient(make_config(), transport=transport, sleep_func=lambda _: None)
        with self.assertRaises(LLMResponseError):
            client.analyze_text("test")

    def test_invalid_request_values_are_rejected(self):
        client = QwenClient(make_config(), transport=FakeTransport([]),
                            sleep_func=lambda _: None)
        with self.assertRaises(LLMRequestError):
            client.chat([])
        with self.assertRaises(LLMRequestError):
            client.chat([{"role": "user", "content": "test"}], timeout=0)
        with self.assertRaises(LLMRequestError):
            client.chat([{"role": "user", "content": "test"}], max_tokens=8)

    def test_mock_client_is_deterministic_and_records_calls(self):
        client = MockLLMClient(responses=["第一个", "第二个"])
        first = client.analyze_text("问题")
        second = client.analyze_text("问题")
        self.assertEqual(first.text, "第一个")
        self.assertEqual(second.text, "第二个")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["messages"][0]["content"], "问题")

    def test_mock_client_can_return_error(self):
        client = MockLLMClient(responses=[LLMTimeoutError("mock timeout")])
        with self.assertRaises(LLMTimeoutError):
            client.analyze_text("test")

    def test_image_request_uses_data_url_and_vision_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "board.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")
            client = MockLLMClient(response_text="ok")
            response = client.analyze_image(
                "识别棋盘", path, model="qwen3-vl-plus"
            )

        self.assertEqual(response.text, "ok")
        call = client.calls[0]
        self.assertEqual(call["model"], "qwen3-vl-plus")
        content = call["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "识别棋盘"})
        self.assertTrue(content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        ))

    def test_invalid_or_missing_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "board.txt"
            text_path.write_text("not an image", encoding="utf-8")
            client = MockLLMClient()
            with self.assertRaises(LLMRequestError):
                client.analyze_image("test", text_path)
            with self.assertRaises(LLMRequestError):
                client.analyze_image("test", Path(temp_dir) / "missing.png")

    def test_response_object_contains_no_secret(self):
        response = LLMResponse(text="ok", model="qwen3.8-flash", elapsed_ms=1.0)
        self.assertNotIn(TEST_KEY, repr(response))
        self.assertNotIn(TEST_KEY, repr(response.safe_summary()))


if __name__ == "__main__":
    unittest.main()