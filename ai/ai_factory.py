# -*- coding: utf-8 -*-
"""比赛 AI 工厂：默认使用实测更强的 HybridAI。"""
import os

from hex_ai import BasicAI


def _wrap_memory(ai, warning=None):
    try:
        from ai.memory.memory_ai import wrap_ai_with_memory
        return wrap_ai_with_memory(ai), warning
    except Exception as exc:
        message = f"memory retrieval unavailable: {exc}"
        if warning:
            message = f"{warning}; {message}"
        return ai, message


def create_competition_ai(engine=None):
    """Create the competition AI and optionally wrap it with memory."""
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
                return _wrap_memory(
                    HybridAI(time_limit=0.20),
                    "LLM_ENABLED=0\uff0c\u5df2\u56de\u9000 HybridAI",
                )
            advisor = LLMAdvisor(
                create_llm_client(config),
                timeout=config.analysis_timeout_seconds,
                max_tokens=config.max_output_tokens,
                enable_thinking=config.enable_thinking,
            )
            cache = CandidateAnalysisCache() if config.cache_enabled else None
            ai = LLMAssistedAI(
                HybridAI(time_limit=0.20),
                advisor,
                mode=config.fusion_mode,
                alpha=config.fusion_alpha,
                candidate_limit=config.fusion_candidates,
                cache=cache,
                model_name=config.analysis_model,
            )
            return _wrap_memory(ai)
        except Exception as exc:
            from ai.hybrid_ai import HybridAI
            return _wrap_memory(
                HybridAI(time_limit=0.20),
                f"HybridAI+LLM \u4e0d\u53ef\u7528\uff0c\u5df2\u56de\u9000 HybridAI\uff1a{exc}",
            )

    if engine == "hybrid":
        try:
            from ai.hybrid_ai import HybridAI
            return _wrap_memory(HybridAI(time_limit=0.20))
        except Exception as exc:
            return _wrap_memory(
                BasicAI(),
                f"HybridAI \u4e0d\u53ef\u7528\uff0c\u5df2\u56de\u9000\u57fa\u7840 AI\uff1a{exc}",
            )
    if engine == "neural":
        from ai.neural_ai import create_ai_with_fallback
        ai, warning = create_ai_with_fallback()
        return _wrap_memory(ai, warning)
    if engine == "strategic":
        try:
            from ai.strategic_ai import StrategicAI
            return _wrap_memory(StrategicAI())
        except Exception as exc:
            return _wrap_memory(
                BasicAI(),
                f"StrategicAI \u4e0d\u53ef\u7528\uff0c\u5df2\u56de\u9000\u57fa\u7840 AI\uff1a{exc}",
            )
    if engine == "basic":
        return _wrap_memory(BasicAI())
    raise ValueError(f"\u672a\u77e5 AI \u5f15\u64ce: {engine!r}")
