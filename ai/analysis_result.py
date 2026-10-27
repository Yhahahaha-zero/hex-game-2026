# -*- coding: utf-8 -*-
"""Structured LLM analysis results and strict validation.

The LLM output is untrusted text. This module converts it into immutable
objects only after checking coordinates, candidate membership and field types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re


ANALYSIS_SCHEMA_VERSION = 1
PROMPT_VERSION = "hex-llm-advisor-v1"
PRIORITIES = frozenset({
    "win", "block_win", "attack", "defense", "connect", "develop", "balanced",
})
PREFERENCES = frozenset({"prefer", "neutral", "avoid"})
SEVERITIES = frozenset({"immediate", "high", "medium"})
MAX_SUMMARY_CHARS = 500
MAX_REASON_CHARS = 200
MAX_THREATS = 8
MAX_CANDIDATES = 8
MAX_WARNINGS = 5
MAX_SURROUNDING_TEXT = 200
_FENCED_JSON = re.compile(
    r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$",
    re.IGNORECASE,
)


class AnalysisValidationError(ValueError):
    """Raised when model output cannot be converted into a safe analysis."""


@dataclass(frozen=True)
class ThreatAnalysis:
    move: tuple[int, int]
    severity: str
    reason: str

    def as_dict(self):
        return {
            "move": list(self.move),
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateAnalysis:
    move: tuple[int, int]
    preference: str
    pros: str
    cons: str

    def as_dict(self):
        return {
            "move": list(self.move),
            "preference": self.preference,
            "pros": self.pros,
            "cons": self.cons,
        }


@dataclass(frozen=True)
class AnalysisResult:
    schema_version: int
    prompt_version: str
    position_id: str
    summary: str
    priority: str
    threats: tuple[ThreatAnalysis, ...] = ()
    candidates: tuple[CandidateAnalysis, ...] = ()
    recommended_move: tuple[int, int] | None = None
    subjective_confidence: float | None = None
    warnings: tuple[str, ...] = ()
    model_name: str = ""
    elapsed_ms: float = 0.0
    usage: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "position_id": self.position_id,
            "summary": self.summary,
            "priority": self.priority,
            "threats": [item.as_dict() for item in self.threats],
            "candidates": [item.as_dict() for item in self.candidates],
            "recommended_move": (
                list(self.recommended_move)
                if self.recommended_move is not None else None
            ),
            "subjective_confidence": self.subjective_confidence,
            "warnings": list(self.warnings),
            "model_name": self.model_name,
            "elapsed_ms": self.elapsed_ms,
            "usage": dict(self.usage),
        }


def extract_json_object(text):
    """Extract one JSON object from strict JSON or a Markdown JSON block."""
    if not isinstance(text, str) or not text.strip():
        raise AnalysisValidationError("模型返回内容为空")
    candidate = text.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        raise AnalysisValidationError("模型返回内容不包含 JSON 对象")
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate, start)
    except json.JSONDecodeError as exc:
        raise AnalysisValidationError("模型返回了无效 JSON") from exc
    if not isinstance(payload, dict):
        raise AnalysisValidationError("模型分析结果顶层必须是对象")
    prefix = candidate[:start]
    suffix = candidate[end:]
    if "{ " in suffix or "{" in suffix:
        raise AnalysisValidationError("模型返回了多个 JSON 对象")
    if len(prefix.strip()) > MAX_SURROUNDING_TEXT:
        raise AnalysisValidationError("JSON 前置说明过长")
    if len(suffix.strip()) > MAX_SURROUNDING_TEXT:
        raise AnalysisValidationError("JSON 后置说明过长")
    return payload


def _append_warning(warnings, message):
    if len(warnings) < MAX_WARNINGS:
        warnings.append(str(message)[:MAX_REASON_CHARS])


def _clean_text(value, field_name, max_chars, warnings, required=False):
    if not isinstance(value, str):
        if required:
            raise AnalysisValidationError(f"{field_name} 必须是字符串")
        return ""
    cleaned = value.strip()
    if required and not cleaned:
        raise AnalysisValidationError(f"{field_name} 不能为空")
    if len(cleaned) > max_chars:
        _append_warning(warnings, f"{field_name} 过长，已截断")
        cleaned = cleaned[:max_chars]
    return cleaned


def _parse_move(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    q, r = value
    if (isinstance(q, bool) or isinstance(r, bool)
            or not isinstance(q, int) or not isinstance(r, int)):
        return None
    return q, r



def _empty_and_candidates(observation):
    rows = observation["board_rows"]
    candidates = set()
    for item in observation.get("candidates", ()):
        if not isinstance(item, dict):
            continue
        move = _parse_move(item.get("move"))
        if move is not None:
            candidates.add(move)
    return rows, candidates


def _move_is_empty(rows, move):
    if move is None:
        return False
    q, r = move
    return 0 <= r < len(rows) and 0 <= q < len(rows[0]) and rows[r][q] == "."


def _usage_copy(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in (
        "prompt_tokens", "completion_tokens", "total_tokens",
        "input_tokens", "output_tokens",
    ):
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
            result[key] = number
    return result


def parse_analysis_payload(payload, observation, *, model_name="", elapsed_ms=0.0,
                           usage=None, prompt_version=PROMPT_VERSION):
    """Validate a decoded JSON object and return an immutable result."""
    if not isinstance(payload, dict):
        raise AnalysisValidationError("模型分析结果顶层必须是对象")
    schema_version = payload.get("schema_version")
    if (isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != ANALYSIS_SCHEMA_VERSION):
        raise AnalysisValidationError("模型分析 schema_version 不受支持")
    expected_position = observation.get("position_id")
    if not isinstance(expected_position, str) or not expected_position:
        raise AnalysisValidationError("观察数据缺少 position_id")
    if payload.get("position_id") != expected_position:
        raise AnalysisValidationError("模型分析结果与当前局面不一致")

    warnings = []
    summary = _clean_text(
        payload.get("summary"), "summary", MAX_SUMMARY_CHARS,
        warnings, required=True,
    )
    priority = payload.get("priority")
    if not isinstance(priority, str) or priority not in PRIORITIES:
        raise AnalysisValidationError("模型返回的 priority 不合法")

    raw_threats = payload.get("threats", [])
    if not isinstance(raw_threats, list):
        raise AnalysisValidationError("threats 必须是数组")
    if len(raw_threats) > MAX_THREATS:
        _append_warning(warnings, "威胁数量过多，仅保留前 8 条")
    threats = []
    seen_threats = set()
    for item in raw_threats[:MAX_THREATS]:
        if not isinstance(item, dict):
            _append_warning(warnings, "忽略了格式错误的威胁")
            continue
        move = _parse_move(item.get("move"))
        if not _move_is_empty(observation["board_rows"], move):
            _append_warning(warnings, "忽略了越界或已占用的威胁坐标")
            continue
        severity = item.get("severity")
        if not isinstance(severity, str) or severity not in SEVERITIES:
            _append_warning(warnings, "忽略了严重程度不合法的威胁")
            continue
        if move in seen_threats:
            _append_warning(warnings, "忽略了重复威胁坐标")
            continue
        reason = _clean_text(
            item.get("reason"), "威胁理由", MAX_REASON_CHARS, warnings,
        )
        seen_threats.add(move)
        threats.append(ThreatAnalysis(move=move, severity=severity, reason=reason))

    raw_candidates = payload.get("candidate_analysis", [])
    if not isinstance(raw_candidates, list):
        raise AnalysisValidationError("candidate_analysis 必须是数组")
    if len(raw_candidates) > MAX_CANDIDATES:
        _append_warning(warnings, "候选分析过多，仅保留前 8 条")
    rows, candidate_moves = _empty_and_candidates(observation)
    candidates = []
    seen_candidates = set()
    for item in raw_candidates[:MAX_CANDIDATES]:
        if not isinstance(item, dict):
            _append_warning(warnings, "忽略了格式错误的候选分析")
            continue
        move = _parse_move(item.get("move"))
        if move not in candidate_moves or not _move_is_empty(rows, move):
            _append_warning(warnings, "忽略了不在候选集合中的分析")
            continue
        if move in seen_candidates:
            _append_warning(warnings, "忽略了重复候选分析")
            continue
        preference = item.get("preference")
        if not isinstance(preference, str) or preference not in PREFERENCES:
            _append_warning(warnings, "忽略了 preference 不合法的候选")
            continue
        pros = _clean_text(item.get("pros"), "候选优点", MAX_REASON_CHARS, warnings)
        cons = _clean_text(item.get("cons"), "候选缺点", MAX_REASON_CHARS, warnings)
        seen_candidates.add(move)
        candidates.append(CandidateAnalysis(
            move=move,
            preference=preference,
            pros=pros,
            cons=cons,
        ))

    recommended_move = _parse_move(payload.get("recommended_move"))
    if recommended_move is not None:
        if (recommended_move not in candidate_moves
                or not _move_is_empty(rows, recommended_move)):
            _append_warning(warnings, "推荐着法不在合法候选集合中，已忽略")
            recommended_move = None

    confidence = payload.get("subjective_confidence")
    if confidence is not None:
        if (isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0):
            _append_warning(warnings, "主观置信度不合法，已清除")
            confidence = None
        else:
            confidence = round(float(confidence), 4)

    raw_warnings = payload.get("warnings", [])
    if not isinstance(raw_warnings, list):
        raise AnalysisValidationError("warnings 必须是数组")
    for item in raw_warnings[:MAX_WARNINGS]:
        if isinstance(item, str) and item.strip():
            _append_warning(warnings, item)

    try:
        elapsed = max(0.0, float(elapsed_ms))
    except (TypeError, ValueError):
        elapsed = 0.0
    return AnalysisResult(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        prompt_version=str(prompt_version),
        position_id=expected_position,
        summary=summary,
        priority=priority,
        threats=tuple(threats),
        candidates=tuple(candidates),
        recommended_move=recommended_move,
        subjective_confidence=confidence,
        warnings=tuple(warnings),
        model_name=str(model_name or "")[:128],
        elapsed_ms=round(elapsed, 3),
        usage=_usage_copy(usage),
    )


def parse_analysis_text(text, observation, *, model_name="", elapsed_ms=0.0,
                        usage=None, prompt_version=PROMPT_VERSION):
    payload = extract_json_object(text)
    return parse_analysis_payload(
        payload,
        observation,
        model_name=model_name,
        elapsed_ms=elapsed_ms,
        usage=usage,
        prompt_version=prompt_version,
    )