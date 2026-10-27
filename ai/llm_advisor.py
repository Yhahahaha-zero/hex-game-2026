# -*- coding: utf-8 -*-
"""Prompt construction and structured analysis via an injected LLM client."""
from __future__ import annotations

from ai.analysis_result import (
    PROMPT_VERSION,
    AnalysisValidationError,
    parse_analysis_text,
)
from ai.board_observation import observation_json


MAX_PROMPT_CHARS = 30_000
SYSTEM_PROMPT = """你是海克斯棋局面分析助手。
红方连接上下边界，蓝方连接左右边界。
你只能依据用户提供的结构化局面进行分析，不得虚构棋子、威胁、候选点或胜负结果。
不得修改棋盘，不得声称自己已经落子，不得把主观置信度描述为真实胜率。
候选分析和推荐着法只能使用输入 candidates 中出现的坐标。
如果信息不足，请在 warnings 中明确说明。
只给出简要结论和可核验依据，不输出隐藏思维链。
必须只返回一个合法 JSON 对象，不要输出 Markdown 代码块或额外说明。"""

OUTPUT_SCHEMA_PROMPT = """请严格按照以下 JSON 结构输出：
{
  "schema_version": 1,
  "position_id": "必须与输入完全一致",
  "summary": "不超过 500 字的局势摘要",
  "priority": "win|block_win|attack|defense|connect|develop|balanced",
  "threats": [
    {
      "move": [q, r],
      "severity": "immediate|high|medium",
      "reason": "简要依据"
    }
  ],
  "candidate_analysis": [
    {
      "move": [q, r],
      "preference": "prefer|neutral|avoid",
      "pros": "优点",
      "cons": "缺点"
    }
  ],
  "recommended_move": [q, r],
  "subjective_confidence": 0.0,
  "warnings": ["需要说明的限制"]
}
要求：
1. recommended_move 必须来自 candidates，若无法判断则输出 null。
2. threats 中的坐标必须是当前棋盘上的空位。
3. candidate_analysis 最多 8 条，warnings 最多 5 条。
4. subjective_confidence 只是主观值，范围为 0 到 1。
5. 不得输出 JSON 之外的任何文字。"""


def build_advisor_messages(observation):
    """Build deterministic system and user messages from an observation."""
    if not isinstance(observation, dict):
        raise AnalysisValidationError("observation 必须是字典")
    position_id = observation.get("position_id")
    if not isinstance(position_id, str) or not position_id:
        raise AnalysisValidationError("observation 缺少 position_id")
    try:
        encoded_observation = observation_json(observation)
    except (TypeError, ValueError) as exc:
        raise AnalysisValidationError("observation 无法序列化为 JSON") from exc
    prompt = (
        OUTPUT_SCHEMA_PROMPT
        + "\n\n当前棋局观察数据：\n"
        + encoded_observation
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        raise AnalysisValidationError("分析提示词超过长度限制")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


class LLMAdvisor:
    """Analyze a read-only board observation; never mutate game state."""

    def __init__(self, client, *, model=None, timeout=None,
                 max_tokens=1000, prompt_version=PROMPT_VERSION,
                 enable_thinking=False):
        if client is None:
            raise ValueError("client 不能为空")
        token_limit = int(max_tokens)
        if not 16 <= token_limit <= 32768:
            raise ValueError("max_tokens 必须在 16 到 32768 之间")
        self.client = client
        self.model = model
        self.timeout = timeout
        self.max_tokens = token_limit
        self.prompt_version = str(prompt_version)
        self.enable_thinking = bool(enable_thinking)

    def analyze(self, observation):
        messages = build_advisor_messages(observation)
        user_prompt = messages[-1]["content"]
        response = self.client.analyze_text(
            user_prompt,
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        return parse_analysis_text(
            response.text,
            observation,
            model_name=response.model,
            elapsed_ms=response.elapsed_ms,
            usage=response.usage,
            prompt_version=self.prompt_version,
        )