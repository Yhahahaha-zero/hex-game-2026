# -*- coding: utf-8 -*-
"""Optional vision feature configuration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ai.llm_config import (
    DEFAULT_ENV_FILE,
    _parse_bool,
    _parse_float,
    _parse_int,
    _read_env_file,
    _setting,
)


class VisionConfigError(ValueError):
    """Raised for invalid vision configuration."""


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = True
    min_confidence: float = 0.78
    max_image_mb: int = 10
    cache_enabled: bool = True
    vlm_fallback: bool = True
    output_dir: str = ""

    def describe(self):
        return {
            "enabled": self.enabled,
            "min_confidence": self.min_confidence,
            "max_image_mb": self.max_image_mb,
            "cache_enabled": self.cache_enabled,
            "vlm_fallback": self.vlm_fallback,
            "output_dir": self.output_dir,
        }


def load_vision_config(env_file=None, environ=None):
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    values = _read_env_file(path)
    env = os.environ if environ is None else environ
    try:
        enabled = _parse_bool(
            _setting(values, env, "VISION_ENABLED", "1"), "VISION_ENABLED"
        )
        min_confidence = _parse_float(
            _setting(values, env, "VISION_MIN_CONFIDENCE", "0.78"),
            "VISION_MIN_CONFIDENCE",
        )
        max_image_mb = _parse_int(
            _setting(values, env, "VISION_MAX_IMAGE_MB", "10"),
            "VISION_MAX_IMAGE_MB",
        )
        cache_enabled = _parse_bool(
            _setting(values, env, "VISION_CACHE_ENABLED", "1"),
            "VISION_CACHE_ENABLED",
        )
        vlm_fallback = _parse_bool(
            _setting(values, env, "VISION_VLM_FALLBACK", "1"),
            "VISION_VLM_FALLBACK",
        )
    except ValueError as exc:
        raise VisionConfigError(str(exc)) from exc
    if not 0.0 <= min_confidence <= 1.0:
        raise VisionConfigError("VISION_MIN_CONFIDENCE 必须在 0 到 1 之间")
    if not 1 <= max_image_mb <= 50:
        raise VisionConfigError("VISION_MAX_IMAGE_MB 必须在 1 到 50 之间")
    return VisionConfig(
        enabled=enabled,
        min_confidence=min_confidence,
        max_image_mb=max_image_mb,
        cache_enabled=cache_enabled,
        vlm_fallback=vlm_fallback,
        output_dir=_setting(values, env, "VISION_OUTPUT_DIR", ""),
    )