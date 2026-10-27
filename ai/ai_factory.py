# -*- coding: utf-8 -*-
"""比赛 AI 工厂：默认使用实测更强的 HybridAI。"""
import os

from hex_ai import BasicAI


def create_competition_ai(engine=None):
    """按环境变量创建比赛 AI，默认 hybrid，可选 neural。"""
    engine = str(engine or os.environ.get("HEX_AI_ENGINE", "hybrid")).lower()
    if engine in ("hybrid_llm", "hybrid+llm"):
        try:
            from ai.ai_coordinator import CandidateAnalysisCache, LLMAssistedAI
            from ai.hybrid_ai import HybridAI
            from ai.llm_advisor import LLMAdvisor
            from ai.llm_client import create_llm_client
            from ai.llm_config import load_llm_config
            config = load_llm_config()
            if not config.enabled:
                return HybridAI(time_limit=0.20), (
                    "LLM_ENABLED=0，已回退 HybridAI"
                )
            advisor = LLMAdvisor(
                create_llm_client(config),
                timeout=config.analysis_timeout_seconds,
                max_tokens=config.max_output_tokens,
                enable_thinking=config.enable_thinking,
            )
            cache = CandidateAnalysisCache() if config.cache_enabled else None
            return LLMAssistedAI(
                HybridAI(time_limit=0.20),
                advisor,
                mode=config.fusion_mode,
                alpha=config.fusion_alpha,
                candidate_limit=config.fusion_candidates,
                cache=cache,
                model_name=config.analysis_model,
            ), None
        except Exception as exc:
            from ai.hybrid_ai import HybridAI
            return HybridAI(time_limit=0.20), (
                f"HybridAI+LLM 不可用，已回退 HybridAI：{exc}"
            )

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