# -*- coding: utf-8 -*-
"""???? AI ??"""


def create_ai_with_fallback(*args, **kwargs):
    from ai.neural_ai import create_ai_with_fallback as factory
    return factory(*args, **kwargs)


__all__ = ["create_ai_with_fallback"]
