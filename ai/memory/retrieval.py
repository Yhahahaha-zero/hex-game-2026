# -*- coding: utf-8 -*-
"""Explainable exact, canonical and similar-position memory retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from ai.memory.canonical import (
    canonicalize_position,
    transform_move,
)
from ai.memory.features import (
    FEATURE_VERSION,
    extract_features,
    feature_similarity,
    parse_features,
)
from ai.memory.ids import build_memory_position_id
from ai.memory.schema import RETRIEVAL_VERSION
from hex_board import BLUE, EMPTY, RED, HexBoard


class MemoryRetrievalError(Exception):
    """Raised for invalid retrieval input."""


@dataclass(frozen=True)
class MemoryHit:
    position_id: str
    decision_id: str
    game_id: str
    match_id: str
    similarity: float
    similarity_type: str
    historical_move: tuple[int, int]
    mapped_move: tuple[int, int] | None
    result: float
    decision_quality: str

    def as_dict(self):
        return {
            "position_id": self.position_id,
            "decision_id": self.decision_id,
            "game_id": self.game_id,
            "match_id": self.match_id,
            "similarity": round(self.similarity, 6),
            "similarity_type": self.similarity_type,
            "historical_move": list(self.historical_move),
            "mapped_move": list(self.mapped_move) if self.mapped_move else None,
            "result": self.result,
            "decision_quality": self.decision_quality,
        }


@dataclass(frozen=True)
class MemoryAdvice:
    query_position_id: str
    canonical_key: str
    recommended_move: tuple[int, int] | None
    confidence: float
    support_count: int
    win_rate: float | None
    hits: tuple[MemoryHit, ...]
    evidence: dict
    retrieval_version: str = RETRIEVAL_VERSION

    @property
    def available(self):
        return self.recommended_move is not None

    def as_dict(self):
        return {
            "query_position_id": self.query_position_id,
            "canonical_key": self.canonical_key,
            "recommended_move": (
                list(self.recommended_move) if self.recommended_move else None
            ),
            "confidence": round(self.confidence, 6),
            "support_count": self.support_count,
            "win_rate": self.win_rate,
            "hits": [hit.as_dict() for hit in self.hits],
            "evidence": dict(self.evidence),
            "retrieval_version": self.retrieval_version,
        }


def _board_from_payload(raw_value):
    try:
        payload = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        grid = payload["grid"]
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise MemoryRetrievalError("invalid board_json in memory row") from exc
    if len(grid) != 11 or any(len(row) != 11 for row in grid):
        raise MemoryRetrievalError("invalid board dimensions in memory row")
    board = HexBoard()
    board.grid = [list(row) for row in grid]
    board.history = []
    return board


def _board_similarity(current, historical, current_color):
    current_canonical = canonicalize_position(current, current_color).board
    historical_canonical = canonicalize_position(
        historical, current_color
    ).board
    same_empty = 0
    union_empty = 0
    same_stones = 0
    union_stones = 0
    for r in range(11):
        for q in range(11):
            left = current_canonical.grid[r][q]
            right = historical_canonical.grid[r][q]
            if left == EMPTY and right == EMPTY:
                same_empty += 1
                union_empty += 1
            elif left != EMPTY or right != EMPTY:
                union_stones += 1
                if left == right and left != EMPTY:
                    same_stones += 1
    numerator = same_stones * 2.0 + same_empty * 0.2
    denominator = union_stones * 2.0 + union_empty * 0.2
    return numerator / denominator if denominator else 0.0


def _mapped_move(current_board, historical_board, current_color, move):
    current_canonical = canonicalize_position(current_board, current_color)
    historical_canonical = canonicalize_position(historical_board, current_color)
    if current_canonical.key != historical_canonical.key:
        return None
    canonical_move = transform_move(move, historical_canonical.transform)
    return transform_move(canonical_move, current_canonical.transform)


def _legal(board, move):
    return (
        isinstance(move, tuple)
        and len(move) == 2
        and board.in_range(*move)
        and board.is_empty(*move)
    )


class MemoryRetriever:
    """Retrieve validated AI decisions without changing the board."""

    def __init__(self, store, *, top_k=5, minimum_similarity=0.72,
                 minimum_support=3, ply_window=6, limit=2000,
                 allowed_types=None):
        if top_k < 1 or minimum_support < 1 or ply_window < 0 or limit < 1:
            raise ValueError("invalid retrieval limits")
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        self.store = store
        self.top_k = int(top_k)
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_support = int(minimum_support)
        self.ply_window = int(ply_window)
        self.limit = int(limit)
        allowed = tuple(allowed_types or ("exact", "canonical", "similar"))
        if any(kind not in ("exact", "canonical", "similar") for kind in allowed):
            raise ValueError("invalid retrieval type")
        self.allowed_types = frozenset(allowed)

    def query(self, board, color, *, ply=None, game_id=None, match_id=None,
              top_k=None, minimum_similarity=None, minimum_support=None):
        if color not in (RED, BLUE):
            raise MemoryRetrievalError("color must be RED or BLUE")
        ply = int(ply if ply is not None else len(board.history) + 1)
        if ply < 1:
            raise MemoryRetrievalError("ply must be positive")
        top_k = self.top_k if top_k is None else int(top_k)
        minimum_similarity = (
            self.minimum_similarity
            if minimum_similarity is None
            else float(minimum_similarity)
        )
        minimum_support = (
            self.minimum_support
            if minimum_support is None
            else int(minimum_support)
        )
        query_position_id = build_memory_position_id(board, color)
        current_canonical = canonicalize_position(board, color)
        current_features = extract_features(board, color, ply).as_dict()
        rows = self.store.query_validated_memory(
            current_color=color,
            ply=ply,
            feature_version=FEATURE_VERSION,
            ply_window=self.ply_window,
            exclude_match_id=match_id,
            exclude_game_id=game_id,
            limit=self.limit,
        )
        hits = []
        for row in rows:
            try:
                historical_board = _board_from_payload(row["board_json"])
                historical_features = parse_features(row["features_json"])
                if int(historical_features.get("current_color")) != color:
                    continue
                exact = row["position_id"] == query_position_id
                canonical = row["canonical_key"] == current_canonical.key
                historical_move = (int(row["chosen_q"]), int(row["chosen_r"]))
                if exact or canonical:
                    similarity = 1.0
                    similarity_type = "exact" if exact else "canonical"
                    mapped_move = (
                        historical_move if exact else
                        _mapped_move(board, historical_board, color, historical_move)
                    )
                else:
                    similarity_type = "similar"
                    similarity = (
                        0.55 * feature_similarity(
                            current_features, historical_features
                        )
                        + 0.45 * _board_similarity(board, historical_board, color)
                    )
                    mapped_move = historical_move
                if mapped_move is None or not _legal(board, mapped_move):
                    continue
                if similarity_type not in self.allowed_types:
                    continue
                if similarity < minimum_similarity:
                    continue
                hits.append(MemoryHit(
                    position_id=row["position_id"],
                    decision_id=row["decision_id"],
                    game_id=row["game_id"],
                    match_id=row["match_id"],
                    similarity=round(similarity, 6),
                    similarity_type=similarity_type,
                    historical_move=historical_move,
                    mapped_move=mapped_move,
                    result=float(row["result"]),
                    decision_quality=row["decision_quality"],
                ))
            except (MemoryRetrievalError, ValueError, TypeError, KeyError):
                continue
        hits.sort(key=lambda hit: (
            -hit.similarity,
            hit.position_id,
            hit.decision_id,
        ))
        hits = hits[:max(1, top_k)]
        evidence = self._aggregate(board, hits, minimum_support)
        return MemoryAdvice(
            query_position_id=query_position_id,
            canonical_key=current_canonical.key,
            recommended_move=evidence.pop("recommended_move"),
            confidence=evidence.pop("confidence"),
            support_count=evidence.pop("support_count"),
            win_rate=evidence.pop("win_rate"),
            hits=tuple(hits),
            evidence={
                **evidence,
                "query_features": current_features,
                "minimum_similarity": minimum_similarity,
                "minimum_support": minimum_support,
            },
        )

    def _aggregate(self, board, hits, minimum_support):
        groups = {}
        for hit in hits:
            if hit.mapped_move is None or not _legal(board, hit.mapped_move):
                continue
            group = groups.setdefault(hit.mapped_move, {
                "matches": set(),
                "hits": [],
                "votes": 0.0,
                "weighted_results": 0.0,
            })
            if hit.match_id not in group["matches"]:
                group["matches"].add(hit.match_id)
                weight = max(0.01, hit.similarity)
                group["hits"].append(hit)
                group["votes"] += weight
                group["weighted_results"] += weight * hit.result
        total_votes = sum(group["votes"] for group in groups.values())
        ranked = sorted(
            groups.items(),
            key=lambda item: (-item[1]["votes"], item[0]),
        )
        if not ranked:
            return {
                "recommended_move": None,
                "confidence": 0.0,
                "support_count": 0,
                "win_rate": None,
                "vote_share": 0.0,
                "conflict": False,
            }
        move, best = ranked[0]
        support = len(best["matches"])
        vote_share = best["votes"] / total_votes if total_votes else 0.0
        win_rate = (
            best["weighted_results"] / best["votes"] if best["votes"] else None
        )
        conflict = len(ranked) > 1 and ranked[1][1]["votes"] >= best["votes"] * 0.75
        recommended = (
            move
            if (
                support >= minimum_support
                and vote_share >= 0.60
                and not conflict
                and win_rate is not None
                and win_rate >= 0.50
            )
            else None
        )
        support_factor = min(1.0, support / max(1, minimum_support))
        average_similarity = (
            sum(hit.similarity for hit in best["hits"]) / len(best["hits"])
            if best["hits"] else 0.0
        )
        confidence = (
            0.45 * average_similarity
            + 0.25 * support_factor
            + 0.30 * vote_share
        )
        return {
            "recommended_move": recommended,
            "confidence": round(min(1.0, confidence), 6),
            "support_count": support,
            "win_rate": round(win_rate, 6) if win_rate is not None else None,
            "vote_share": round(vote_share, 6),
            "conflict": conflict,
            "candidate_votes": {
                f"{q},{r}": round(group["votes"], 6)
                for (q, r), group in ranked
            },
        }