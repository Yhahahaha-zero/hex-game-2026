# -*- coding: utf-8 -*-
"""OpenAI-compatible Qwen client with safe errors and offline mocks.

Only this module constructs the Authorization header. API keys are never
included in exceptions, response objects, call logs or repr output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import http.client
import json
import mimetypes
from pathlib import Path
import socket
import time
from urllib import error, request

from ai.llm_config import LLMConfig, LLMConfigError, load_llm_config


MAX_RESPONSE_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class LLMClientError(RuntimeError):
    """Base class for safe, user-facing LLM client errors."""


class LLMDisabledError(LLMClientError):
    """Raised when an LLM request is attempted while disabled."""


class LLMTimeoutError(LLMClientError):
    """Raised when the request exceeds its configured timeout."""


class LLMTransportError(LLMClientError):
    """Raised for connection and transport failures."""


class LLMAuthenticationError(LLMClientError):
    """Raised for authentication or authorization failures."""


class LLMRateLimitError(LLMClientError):
    """Raised when the provider rejects a request due to rate limits."""


class LLMServerError(LLMClientError):
    """Raised when the provider has a temporary server-side failure."""


class LLMRequestError(LLMClientError):
    """Raised when the request itself is invalid or rejected."""


class LLMResponseError(LLMClientError):
    """Raised when a successful response cannot be parsed safely."""


@dataclass(frozen=True)
class LLMResponse:
    """Normalized provider response without raw request or credentials."""

    text: str
    model: str
    elapsed_ms: float
    status_code: int = 200
    request_id: str = ""
    usage: dict = field(default_factory=dict)

    def safe_summary(self):
        return {
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "text_length": len(self.text),
            "usage": dict(self.usage),
        }


def _validate_messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise LLMRequestError("messages 不能为空")
    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            raise LLMRequestError("每条 message 必须是字典")
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise LLMRequestError(f"非法消息角色: {role!r}")
        content = item.get("content")
        if isinstance(content, str):
            if not content.strip():
                raise LLMRequestError("消息内容不能为空")
        elif isinstance(content, (list, tuple)):
            if not content:
                raise LLMRequestError("多段消息内容不能为空")
        else:
            raise LLMRequestError("消息 content 必须是字符串或内容列表")
        normalized.append({"role": role, "content": content})
    return normalized


def _content_text(content):
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if item.get("type", "text") == "text" and isinstance(text_part, str):
                    parts.append(text_part)
        text = "".join(parts).strip()
    else:
        text = ""
    if not text:
        raise LLMResponseError("模型响应不包含可用文本")
    return text


def _usage_from_payload(payload):
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {}
    result = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    return result


def _image_data_url(image_path):
    path = Path(image_path).expanduser().resolve()
    extension = path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_TYPES.get(extension)
    if mime_type is None:
        raise LLMRequestError(f"不支持的图片格式: {extension or '无扩展名'}")
    if not path.is_file():
        raise LLMRequestError(f"图片文件不存在: {path}")
    try:
        size = path.stat().st_size
        if size <= 0:
            raise LLMRequestError("图片文件为空")
        if size > MAX_IMAGE_BYTES:
            raise LLMRequestError("图片文件超过 10MB 限制")
        payload = path.read_bytes()
    except LLMRequestError:
        raise
    except OSError as exc:
        raise LLMRequestError("无法读取图片文件") from exc
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _default_transport(url, headers, body, timeout):
    """Perform one HTTP POST and return status, headers and bounded bytes."""
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read(
                MAX_RESPONSE_BYTES + 1
            )
    except error.HTTPError as exc:
        # Read only a bounded body for status handling; never expose its text.
        try:
            exc.read(MAX_RESPONSE_BYTES + 1)
        except Exception:
            pass
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, b""
    except (TimeoutError, socket.timeout) as exc:
        raise LLMTimeoutError("请求超时") from exc
    except error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise LLMTimeoutError("请求超时") from exc
        raise LLMTransportError("网络连接失败") from exc
    except (http.client.HTTPException, OSError) as exc:
        raise LLMTransportError("网络传输失败") from exc


def _status_error(status_code):
    if status_code == 408:
        return LLMTimeoutError("请求超时（HTTP 408）")
    if status_code == 429:
        return LLMRateLimitError("千问 API 请求频率受限（HTTP 429）")
    if 500 <= status_code <= 599:
        return LLMServerError(f"千问 API 服务暂时不可用（HTTP {status_code}）")
    if status_code in (401, 403):
        return LLMAuthenticationError(f"千问 API 鉴权失败（HTTP {status_code}）")
    return LLMRequestError(f"千问 API 拒绝了请求（HTTP {status_code}）")


def _is_retryable(exc):
    return isinstance(
        exc,
        (LLMTimeoutError, LLMTransportError, LLMRateLimitError, LLMServerError),
    )


def _parse_response(raw_body, requested_model, status_code, elapsed_ms):
    if len(raw_body) > MAX_RESPONSE_BYTES:
        raise LLMResponseError("模型响应超过 1MB 限制")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMResponseError("模型返回了无效 JSON") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError("模型响应顶层必须是 JSON 对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("模型响应缺少 choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise LLMResponseError("模型响应缺少 message")
    text = _content_text(message.get("content"))
    model = payload.get("model")
    request_id = payload.get("id")
    return LLMResponse(
        text=text,
        model=str(model or requested_model)[:128],
        elapsed_ms=round(float(elapsed_ms), 3),
        status_code=int(status_code),
        request_id=str(request_id or "")[:256],
        usage=_usage_from_payload(payload),
    )


class LLMClient:
    """Minimal interface used by the analysis layer."""

    available = True

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None):
        raise NotImplementedError

    def analyze_text(self, prompt, system_prompt=None, *, model=None,
                     timeout=None, max_tokens=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt)})
        return self.chat(
            messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    def analyze_image(self, prompt, image_path, *, model=None,
                      timeout=None, max_tokens=None):
        image_url = _image_data_url(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        return self.chat(
            messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )


class DisabledLLMClient(LLMClient):
    available = False

    def __init__(self, reason="大模型功能未启用"):
        self.reason = str(reason)

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None):
        raise LLMDisabledError(self.reason)


class MockLLMClient(LLMClient):
    """Offline client for deterministic tests and advisor development."""

    def __init__(self, response_text="{}", responses=None, model="mock-llm"):
        if responses is None:
            responses = [response_text]
        elif isinstance(responses, (str, bytes, LLMResponse, Exception)):
            responses = [responses]
        self._responses = list(responses)
        self.model = str(model)
        self.calls = []

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None):
        normalized = _validate_messages(messages)
        self.calls.append({
            "messages": [dict(item) for item in normalized],
            "model": model or self.model,
            "timeout": timeout,
            "max_tokens": max_tokens,
        })
        if not self._responses:
            raise LLMResponseError("Mock 客户端没有可用响应")
        item = self._responses.pop(0)
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        text = str(item).strip()
        if not text:
            raise LLMResponseError("Mock 响应不能为空")
        return LLMResponse(
            text=text,
            model=str(model or self.model),
            elapsed_ms=0.0,
            request_id=f"mock-{len(self.calls)}",
        )


class QwenClient(LLMClient):
    """OpenAI-compatible HTTP client for the Qwen MaaS endpoint."""

    def __init__(self, config, *, transport=None, sleep_func=None):
        self.config = config
        if not config.enabled:
            raise LLMDisabledError("LLM_ENABLED=0")
        if config.provider != "qwen":
            raise LLMConfigError(f"不支持的 LLM_PROVIDER: {config.provider!r}")
        if not config.api_key:
            raise LLMConfigError("启用大模型时必须配置 LLM_API_KEY")
        if not config.chat_completions_url:
            raise LLMConfigError("启用大模型时必须配置 LLM_BASE_URL")
        self._transport = transport or _default_transport
        self._sleep = sleep_func or time.sleep

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None):
        normalized = _validate_messages(messages)
        selected_model = str(model or self.config.analysis_model).strip()
        if not selected_model:
            raise LLMRequestError("模型名称不能为空")
        try:
            timeout_value = float(
                self.config.timeout_seconds if timeout is None else timeout
            )
        except (TypeError, ValueError) as exc:
            raise LLMRequestError("timeout 必须是数字") from exc
        if not 0.1 <= timeout_value <= 120:
            raise LLMRequestError("timeout 必须在 0.1 到 120 秒之间")
        try:
            token_limit = int(
                self.config.max_output_tokens if max_tokens is None else max_tokens
            )
        except (TypeError, ValueError) as exc:
            raise LLMRequestError("max_tokens 必须是整数") from exc
        if not 16 <= token_limit <= 32768:
            raise LLMRequestError("max_tokens 必须在 16 到 32768 之间")

        payload = {
            "model": selected_model,
            "messages": normalized,
            "temperature": self.config.temperature,
            "max_tokens": token_limit,
            "stream": False,
        }
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LLMRequestError("请求内容无法序列化") from exc

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        url = self.config.chat_completions_url
        started = time.perf_counter()
        last_error = None
        attempts = self.config.max_retries + 1

        for attempt in range(attempts):
            try:
                status, _headers, raw_body = self._transport(
                    url, headers, body, timeout_value
                )
                if not isinstance(status, int):
                    raise LLMResponseError("HTTP 客户端未返回有效状态码")
                if not 200 <= status <= 299:
                    raise _status_error(status)
                elapsed_ms = (time.perf_counter() - started) * 1000
                return _parse_response(raw_body, selected_model, status, elapsed_ms)
            except LLMClientError as exc:
                last_error = exc
                if (attempt + 1 >= attempts or not _is_retryable(exc)):
                    raise
                self._sleep(min(0.5 * (2 ** attempt), 2.0))

        raise last_error or LLMTransportError("模型请求失败")


def create_llm_client(config=None, *, transport=None, sleep_func=None):
    """Create a disabled or Qwen client from validated configuration."""
    config = load_llm_config() if config is None else config
    if not config.enabled:
        return DisabledLLMClient("LLM_ENABLED=0")
    return QwenClient(config, transport=transport, sleep_func=sleep_func)