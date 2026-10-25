# -*- coding: utf-8 -*-
"""比赛 AI 工厂：默认使用实测更强的 HybridAI。"""
import os

from hex_ai import BasicAI


def create_competition_ai(engine=None):
    """按环境变量创建比赛 AI，默认 hybrid，可选 neural。"""
    engine = str(engine or os.environ.get("HEX_AI_ENGINE", "hybrid")).lower()
    if engine == "hybrid":
        try:
            from ai.hybrid_ai import HybridAI
            return HybridAI(time_limit=0.20), None
        except Exception as exc:
            return BasicAI(), f"HybridAI 不可用，已回退基础 AI：{exc}"
    if engine == "neural":
        from ai.neural_ai import create_ai_with_fallback
        return create_ai_with_fallback()
    if engine == "strategic":
        try:
            from ai.strategic_ai import StrategicAI
            return StrategicAI(), None
        except Exception as exc:
            return BasicAI(), f"StrategicAI ????????? AI?{exc}"
    if engine == "basic":
        return BasicAI(), None
    raise ValueError(f"未知 AI 引擎: {engine!r}")
