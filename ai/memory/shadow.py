# -*- coding: utf-8 -*-
"""Shadow comparison records and aggregate reporting for memory retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowComparison:
    base_move: tuple[int, int]
    memory_move: tuple[int, int] | None
    final_move: tuple[int, int]
    applied: bool
    confidence: float

    @property
    def differs_from_base(self):
        return self.memory_move is not None and self.memory_move != self.base_move

    def as_dict(self):
        return {
            "base_move": list(self.base_move),
            "memory_move": list(self.memory_move) if self.memory_move else None,
            "final_move": list(self.final_move),
            "applied": self.applied,
            "confidence": self.confidence,
            "differs_from_base": self.differs_from_base,
        }


def summarize_shadow_events(events):
    total = len(events)
    with_advice = sum(1 for row in events if row.get("memory_q") is not None)
    differing = sum(
        1
        for row in events
        if row.get("memory_q") is not None
        and (row.get("memory_q"), row.get("memory_r"))
        != (row.get("base_q"), row.get("base_r"))
    )
    applied = sum(int(row.get("applied", 0)) for row in events)
    return {
        "events": total,
        "with_advice": with_advice,
        "coverage": round(with_advice / total, 6) if total else 0.0,
        "differs_from_base": differing,
        "difference_rate": round(differing / total, 6) if total else 0.0,
        "applied": applied,
        "apply_rate": round(applied / total, 6) if total else 0.0,
    }