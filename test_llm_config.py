# -*- coding: utf-8 -*-
"""Tests for safe LLM configuration and secret redaction."""
from pathlib import Path
import tempfile
import unittest

from ai.llm_config import LLMConfigError, load_llm_config


class LLMConfigTest(unittest.TestCase):
    def write_env(self, directory, text):
        path = Path(directory) / ".env.local"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    def test_disabled_config_needs_no_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(
                temp_dir,
                """
                LLM_ENABLED=0
                LLM_API_KEY=
                LLM_BASE_URL=
                """,
            )
            config = load_llm_config(path, environ={})

        self.assertFalse(config.enabled)
        self.assertFalse(config.key_configured)
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.analysis_timeout_seconds, 30.0)
        self.assertEqual(config.analysis_model, "qwen3.8-flash")
        self.assertEqual(config.vision_model, "qwen3-vl-plus")

    def test_enabled_config_accepts_valid_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(
                temp_dir,
                """
                LLM_ENABLED=1
                LLM_PROVIDER=qwen
                LLM_API_KEY="test-secret"
                LLM_BASE_URL="https://example.test/v1/"
                LLM_ANALYSIS_MODEL=qwen3.8-flash
                LLM_VISION_MODEL=qwen3-vl-plus
                LLM_TIMEOUT_SECONDS=8
                LLM_MAX_RETRIES=1
                LLM_MAX_OUTPUT_TOKENS=1000
                LLM_TEMPERATURE=0.1
                LLM_ENABLE_THINKING=1
                """,
            )
            config = load_llm_config(path, environ={})

        self.assertTrue(config.enabled)
        self.assertTrue(config.key_configured)
        self.assertTrue(config.enable_thinking)
        self.assertEqual(config.api_key, "test-secret")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(
            config.chat_completions_url,
            "https://example.test/v1/chat/completions",
        )

    def test_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(
                temp_dir,
                """
                LLM_ENABLED=0
                LLM_ANALYSIS_MODEL=file-model
                """,
            )
            config = load_llm_config(
                path,
                environ={
                    "LLM_ENABLED": "1",
                    "LLM_API_KEY": "environment-secret",
                    "LLM_BASE_URL": "http://localhost:11434/v1",
                    "LLM_ANALYSIS_MODEL": "environment-model",
                },
            )

        self.assertTrue(config.enabled)
        self.assertEqual(config.api_key, "environment-secret")
        self.assertEqual(config.analysis_model, "environment-model")

    def test_describe_never_contains_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(
                temp_dir,
                """
                LLM_ENABLED=1
                LLM_API_KEY=very-secret-value
                LLM_BASE_URL=https://example.test/v1
                """,
            )
            config = load_llm_config(path, environ={})

        description = config.describe()
        self.assertTrue(description["key_configured"])
        self.assertNotIn("very-secret-value", repr(description))
        self.assertNotIn("very-secret-value", repr(config))

    def test_enabled_config_requires_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(temp_dir, "LLM_ENABLED=1")
            with self.assertRaisesRegex(LLMConfigError, "LLM_API_KEY"):
                load_llm_config(path, environ={})

            path = self.write_env(
                temp_dir,
                "LLM_ENABLED=1\nLLM_API_KEY=test-secret",
            )
            with self.assertRaisesRegex(LLMConfigError, "LLM_BASE_URL"):
                load_llm_config(path, environ={})

    def test_invalid_values_fail_with_clear_errors(self):
        cases = (
            ("LLM_ENABLED=maybe", "LLM_ENABLED"),
            ("LLM_ENABLED=0\nLLM_ENABLE_THINKING=maybe", "LLM_ENABLE_THINKING"),
            ("LLM_ENABLED=0\nLLM_PROVIDER=unknown", "LLM_PROVIDER"),
            ("LLM_ENABLED=0\nLLM_BASE_URL=ftp://example.test", "LLM_BASE_URL"),
            ("LLM_ENABLED=0\nLLM_TIMEOUT_SECONDS=0", "LLM_TIMEOUT_SECONDS"),
            ("LLM_ENABLED=0\nLLM_MAX_RETRIES=9", "LLM_MAX_RETRIES"),
            ("LLM_ENABLED=0\nLLM_MAX_OUTPUT_TOKENS=0", "LLM_MAX_OUTPUT_TOKENS"),
            ("LLM_ENABLED=0\nLLM_TEMPERATURE=3", "LLM_TEMPERATURE"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for content, expected in cases:
                with self.subTest(expected=expected):
                    path = self.write_env(temp_dir, content)
                    with self.assertRaisesRegex(LLMConfigError, expected):
                        load_llm_config(path, environ={})

    def test_malformed_env_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_env(temp_dir, "NOT_A_SETTING")
            with self.assertRaisesRegex(LLMConfigError, "格式错误"):
                load_llm_config(path, environ={})

    def test_missing_env_file_uses_safe_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.env"
            config = load_llm_config(path, environ={})

        self.assertFalse(config.enabled)
        self.assertFalse(config.key_configured)
        self.assertEqual(config.chat_completions_url, "")


if __name__ == "__main__":
    unittest.main()