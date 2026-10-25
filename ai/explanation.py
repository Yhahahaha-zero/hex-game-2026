# -*- coding: utf-8 -*-
"""统一的 AI 决策解释数据结构。"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class AIExplanation:
    """One AI decision with candidates and an explicit score type."""

    selected_move: Optional[tuple[int, int]]
    tactic: str
    candidates: tuple[dict[str, Any], ...] = ()
    elapsed_ms: Optional[float] = None
    search_type: str = ""
    score_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {
            "selected_move": self.selected_move,
            "tactic": self.tactic,
            "candidates": [dict(item) for item in self.candidates],
            "elapsed_ms": self.elapsed_ms,
            "search_type": self.search_type,
            "score_type": self.score_type,
            "metadata": dict(self.metadata),
        }


def normalize_explanation(value, move=None, reason="", elapsed_ms=None):
    """Convert an AI explanation or a legacy reason into a stable dictionary."""
    if value is None:
        return AIExplanation(
            selected_move=move,
            tactic=reason or "AI 落子",
            elapsed_ms=elapsed_ms,
        ).as_dict()
    if isinstance(value, AIExplanation):
        return value.as_dict()
    if isinstance(value, dict):
        data = dict(value)
        data.setdefault("selected_move", move)
        data.setdefault("tactic", reason or "AI 落子")
        data.setdefault("elapsed_ms", elapsed_ms)
        data.setdefault("candidates", [])
        data.setdefault("search_type", "")
        data.setdefault("score_type", "")
        data.setdefault("metadata", {})
        return data
    raise TypeError(f"不支持的 AI 解释类型: {type(value).__name__}")
