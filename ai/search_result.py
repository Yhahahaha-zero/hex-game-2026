# -*- coding: utf-8 -*-
"""MCTS 搜索结果的不可变数据结构。"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ChildStat:
    """根节点下一个候选着法的统计。"""

    move: tuple[int, int]
    visits: int
    prior: float
    value: float

    def as_dict(self):
        return {
            "move": self.move,
            "visits": self.visits,
            "prior": round(self.prior, 6),
            "value": round(self.value, 6),
        }


@dataclass(frozen=True)
class SearchResult:
    """一次完整 MCTS 搜索的结果。"""

    move: tuple[int, int]
    visits: int
    simulations: int
    elapsed_ms: float
    children: tuple[ChildStat, ...] = ()
    root: Optional[Any] = field(default=None, repr=False, compare=False)

    @property
    def best_move(self):
        return self.move

    def summary(self):
        return {
            "move": self.move,
            "visits": self.visits,
            "simulations": self.simulations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "children": [child.as_dict() for child in self.children],
        }
