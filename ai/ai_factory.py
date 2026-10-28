# -*- coding: utf-8 -*-
"""Competition AI factory backed by one explicit profile."""
from dataclasses import replace
import os
from pathlib import Path

from ai.competition_config import (
    ALLOWED_AI_ENGINES,
    WORKSPACE_ROOT,
    load_competition_settings,
)
from hex_ai import BasicAI


def _normalize_engine(engine):
    normalized = str(engine).strip().lower()
    if normalized == "hybrid+llm":
        normalized = "hybrid_llm"
    if normalized not in ALLOWED_AI_ENGINES:
        raise ValueError(f"unknown AI engine: {engine!r}")
    return normalized


def _profile_memory_settings(settings):
    from ai.memory.memory_ai import MemoryRetrievalSettings
    env_settings = MemoryRetrievalSettings.from_env()
    db_path = settings.memory_db or env_settings.db_path
    if db_path:
        path = Path(db_path).expanduser()
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path
        db_path = str(path.resolve())
    return replace(
        env_settings,
        mode=settings.memory_mode,
        db_path=db_path,
    )


def _attach_profile(ai, settings):
    try:
        setattr(ai, "competition_profile", settings.metadata())
    except Exception:
        pass
    return ai


def _wrap_memory(ai, warning, settings):
    if settings.memory_mode == "off":
        return _attach_profile(ai, settings), warning
    try:
        from ai.memory.deployment import validate_deployment_manifest
        from ai.memory.memory_ai import wrap_ai_with_memory

        deployment = os.environ.get("HEX_MEMORY_DEPLOYMENT_MANIFEST", "").strip()
        if deployment:
            models_dir = WORKSPACE_ROOT / "models"
            validate_deployment_manifest(deployment, models_dir=models_dir)
        wrapped = wrap_ai_with_memory(ai, settings=_profile_memory_settings(settings))
        return _attach_profile(wrapped, settings), warning
    except Exception as exc:
        message = f"memory retrieval unavailable: {exc}"
        if warning:
            message = f"{warning}; {message}"
        return _attach_profile(ai, settings), message


def create_competition_ai(engine=None, *, settings=None):
    """Create the configured AI; explicit engine is reserved for tools and tests."""
    settings = settings or load_competition_settings()
    warning = None
    if engine is None:
        engine = settings.ai_engine
        environment_engine = str(os.environ.get("HEX_AI_ENGINE", "")).strip()
        if environment_engine and environment_engine.lower() != settings.ai_engine:
            warning = (
                f"HEX_AI_ENGINE={environment_engine!r} ignored; "
                f"competition profile selects {settings.ai_engine!r}"
            )
    else:
        engine = _normalize_engine(engine)
        settings = replace(settings, ai_engine=engine)

    engine = _normalize_engine(engine)
    time_limit_seconds = settings.search_time_limit_ms / 1000.0

    if engine == "hybrid_llm":
        try:
            from ai.ai_coordinator import CandidateAnalysisCache, LLMAssistedAI
            from ai.hybrid_ai import HybridAI
            from ai.llm_advisor import LLMAdvisor
            from ai.llm_client import create_llm_client
            from ai.llm_config import load_llm_config
            config = load_llm_config()
            if not config.enabled:
                return _wrap_memory(
                    HybridAI(time_limit=time_limit_seconds, seed=settings.random_seed),
                    "LLM_ENABLED=0, fallback to HybridAI",
                    settings,
                )
            advisor = LLMAdvisor(
                create_llm_client(config),
                timeout=config.analysis_timeout_seconds,
                max_tokens=config.max_output_tokens,
                enable_thinking=config.enable_thinking,
            )
            cache = CandidateAnalysisCache() if config.cache_enabled else None
            ai = LLMAssistedAI(
                HybridAI(time_limit=time_limit_seconds, seed=settings.random_seed),
                advisor,
                mode=config.fusion_mode,
                alpha=config.fusion_alpha,
                candidate_limit=config.fusion_candidates,
                cache=cache,
                model_name=config.analysis_model,
            )
            return _wrap_memory(ai, warning, settings)
        except Exception as exc:
            from ai.hybrid_ai import HybridAI
            message = (
                f"HybridAI+LLM unavailable, fallback to HybridAI: {exc}"
            )
            if warning:
                message = f"{warning}; {message}"
            return _wrap_memory(
                HybridAI(time_limit=time_limit_seconds, seed=settings.random_seed), message, settings
            )

    if engine == "hybrid":
        try:
            from ai.hybrid_ai import HybridAI
            return _wrap_memory(
                HybridAI(time_limit=time_limit_seconds, seed=settings.random_seed), warning, settings
            )
        except Exception as exc:
            message = f"HybridAI unavailable, fallback to BasicAI: {exc}"
            if warning:
                message = f"{warning}; {message}"
            return _wrap_memory(BasicAI(), message, settings)
    if engine == "neural":
        from ai.neural_ai import create_ai_with_fallback
        ai, fallback = create_ai_with_fallback(
            model_path=settings.neural_model_path,
            time_limit_ms=int(settings.search_time_limit_ms),
            seed=settings.random_seed,
        )
        if fallback:
            warning = f"{warning}; {fallback}" if warning else fallback
        return _wrap_memory(ai, warning, settings)
    if engine == "strategic":
        try:
            from ai.strategic_ai import StrategicAI
            return _wrap_memory(StrategicAI(), warning, settings)
        except Exception as exc:
            message = f"StrategicAI unavailable, fallback to BasicAI: {exc}"
            if warning:
                message = f"{warning}; {message}"
            return _wrap_memory(BasicAI(), message, settings)
    if engine == "basic":
        return _wrap_memory(BasicAI(), warning, settings)
    raise ValueError(f"unknown AI engine: {engine!r}")