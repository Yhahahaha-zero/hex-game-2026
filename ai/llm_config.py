# -*- coding: utf-8 -*-
"""Safe loading and validation for optional LLM configuration.

The API key is deliberately excluded from repr(), logs and safe descriptions.
Environment variables take precedence over .env.local so temporary values can
be used without editing the local file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
SUPPORTED_PROVIDERS = {"qwen"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class LLMConfigError(ValueError):
    """Raised when an LLM setting is present but invalid."""


@dataclass(frozen=True)
class LLMConfig:
    """Immutable, validated LLM settings."""

    enabled: bool = False
    provider: str = "qwen"
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    analysis_model: str = "qwen3.8-flash"
    vision_model: str = "qwen3-vl-plus"
    timeout_seconds: float = 8.0
    max_retries: int = 1
    max_output_tokens: int = 1000
    temperature: float = 0.1
    enable_thinking: bool = False
    analysis_timeout_seconds: float = 30.0
    fusion_mode: str = "analysis_only"
    fusion_alpha: float = 0.0
    fusion_candidates: int = 8
    cache_enabled: bool = True

    @property
    def key_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def chat_completions_url(self) -> str:
        """Return the OpenAI-compatible chat endpoint URL."""
        if not self.base_url:
            return ""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def describe(self) -> dict:
        """Return diagnostics safe for logs and UI display."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "key_configured": self.key_configured,
            "base_url": self.base_url,
            "analysis_model": self.analysis_model,
            "vision_model": self.vision_model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "analysis_timeout_seconds": self.analysis_timeout_seconds,
            "fusion_mode": self.fusion_mode,
            "fusion_alpha": self.fusion_alpha,
            "fusion_candidates": self.fusion_candidates,
            "cache_enabled": self.cache_enabled,
        }


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a small KEY=VALUE env file without evaluating shell syntax."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LLMConfigError(f"无法读取 LLM 配置文件: {path.name}") from exc

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not name or not name.replace("_", "").isalnum():
            raise LLMConfigError(
                f"LLM 配置文件第 {line_no} 行格式错误，应为 KEY=VALUE"
            )
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def _setting(file_values, environ, name, default=""):
    if name in environ:
        return str(environ[name]).strip()
    return str(file_values.get(name, default)).strip()


def _parse_bool(raw_value: str, name: str) -> bool:
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise LLMConfigError(f"{name} 必须是 1/0、true/false、yes/no 或 on/off")


def _parse_float(raw_value: str, name: str) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"{name} 必须是数字") from exc


def _parse_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"{name} 必须是整数") from exc
    if str(value) != str(raw_value).strip():
        raise LLMConfigError(f"{name} 必须是整数")
    return value


def _normalize_base_url(raw_value: str) -> str:
    value = raw_value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigError("LLM_BASE_URL 必须是有效的 http/https 地址")
    if parsed.username or parsed.password:
        raise LLMConfigError("LLM_BASE_URL 不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise LLMConfigError("LLM_BASE_URL 不能包含查询参数或片段")
    return value


def load_llm_config(env_file=None, environ=None) -> LLMConfig:
    """Load config from an env file plus process environment overrides.

    ``environ`` is injectable for deterministic tests. The API key is required
    only when ``LLM_ENABLED`` is true.
    """
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    file_values = _read_env_file(path)
    env = os.environ if environ is None else environ

    enabled = _parse_bool(
        _setting(file_values, env, "LLM_ENABLED", "0"), "LLM_ENABLED"
    )
    provider = _setting(file_values, env, "LLM_PROVIDER", "qwen").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"不支持的 LLM_PROVIDER: {provider!r}，当前仅支持 qwen"
        )

    api_key = _setting(file_values, env, "LLM_API_KEY", "")
    base_url = _normalize_base_url(_setting(file_values, env, "LLM_BASE_URL", ""))
    analysis_model = _setting(
        file_values, env, "LLM_ANALYSIS_MODEL", "qwen3.8-flash"
    )
    vision_model = _setting(
        file_values, env, "LLM_VISION_MODEL", "qwen3-vl-plus"
    )
    if not analysis_model:
        raise LLMConfigError("LLM_ANALYSIS_MODEL 不能为空")
    if not vision_model:
        raise LLMConfigError("LLM_VISION_MODEL 不能为空")

    timeout_seconds = _parse_float(
        _setting(file_values, env, "LLM_TIMEOUT_SECONDS", "8"),
        "LLM_TIMEOUT_SECONDS",
    )
    if not 0.1 <= timeout_seconds <= 120:
        raise LLMConfigError("LLM_TIMEOUT_SECONDS 必须在 0.1 到 120 秒之间")

    max_retries = _parse_int(
        _setting(file_values, env, "LLM_MAX_RETRIES", "1"),
        "LLM_MAX_RETRIES",
    )
    if not 0 <= max_retries <= 5:
        raise LLMConfigError("LLM_MAX_RETRIES 必须在 0 到 5 之间")

    max_output_tokens = _parse_int(
        _setting(file_values, env, "LLM_MAX_OUTPUT_TOKENS", "1000"),
        "LLM_MAX_OUTPUT_TOKENS",
    )
    if not 16 <= max_output_tokens <= 32768:
        raise LLMConfigError("LLM_MAX_OUTPUT_TOKENS 必须在 16 到 32768 之间")

    temperature = _parse_float(
        _setting(file_values, env, "LLM_TEMPERATURE", "0.1"),
        "LLM_TEMPERATURE",
    )
    if not 0.0 <= temperature <= 2.0:
        raise LLMConfigError("LLM_TEMPERATURE 必须在 0 到 2 之间")

    analysis_timeout_seconds = _parse_float(
        _setting(file_values, env, "LLM_ANALYSIS_TIMEOUT_SECONDS", "30"),
        "LLM_ANALYSIS_TIMEOUT_SECONDS",
    )
    if not 5.0 <= analysis_timeout_seconds <= 120.0:
        raise LLMConfigError("LLM_ANALYSIS_TIMEOUT_SECONDS 必须在 5 到 120 秒之间")

    fusion_mode = _setting(
        file_values, env, "LLM_FUSION_MODE", "analysis_only"
    ).lower()
    if fusion_mode not in {"analysis_only", "rerank"}:
        raise LLMConfigError("LLM_FUSION_MODE 必须是 analysis_only 或 rerank")
    fusion_alpha = _parse_float(
        _setting(file_values, env, "LLM_FUSION_ALPHA", "0"),
        "LLM_FUSION_ALPHA",
    )
    if not 0.0 <= fusion_alpha <= 1.0:
        raise LLMConfigError("LLM_FUSION_ALPHA 必须在 0 到 1 之间")
    fusion_candidates = _parse_int(
        _setting(file_values, env, "LLM_FUSION_CANDIDATES", "8"),
        "LLM_FUSION_CANDIDATES",
    )
    if not 2 <= fusion_candidates <= 8:
        raise LLMConfigError("LLM_FUSION_CANDIDATES 必须在 2 到 8 之间")
    cache_enabled = _parse_bool(
        _setting(file_values, env, "LLM_CACHE_ENABLED", "1"),
        "LLM_CACHE_ENABLED",
    )

    enable_thinking = _parse_bool(
        _setting(file_values, env, "LLM_ENABLE_THINKING", "0"),
        "LLM_ENABLE_THINKING",
    )

    if enabled and not api_key:
        raise LLMConfigError("启用大模型时必须配置 LLM_API_KEY")
    if enabled and not base_url:
        raise LLMConfigError("启用大模型时必须配置 LLM_BASE_URL")

    return LLMConfig(
        enabled=enabled,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        analysis_model=analysis_model,
        vision_model=vision_model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        analysis_timeout_seconds=analysis_timeout_seconds,
        fusion_mode=fusion_mode,
        fusion_alpha=fusion_alpha,
        fusion_candidates=fusion_candidates,
        cache_enabled=cache_enabled,
    )