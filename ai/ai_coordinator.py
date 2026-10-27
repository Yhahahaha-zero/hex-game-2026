# -*- coding: utf-8 -*-
"""Experimental HybridAI + LLM candidate coordination and cache."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from ai.analysis_result import (
    PROMPT_VERSION,
    AnalysisResult,
    CandidateAnalysis,
    ThreatAnalysis,
)
from ai.board_observation import build_board_observation
from ai.explanation import AIExplanation
from ai.hybrid_ai import HybridAI
from ai.tactics import opponent_threats
from hex_ai import winning_moves
from hex_board import BLUE, RED, cell_name


CACHE_ROOT = Path(__file__).resolve().parents[1] / "llm_cache" / "candidate_analysis"
PREFERENCE_BONUS = {"prefer": 1.0, "neutral": 0.5, "avoid": 0.0}
FORCED_MARKERS = ("直接获胜", "紧急防守", "多威胁最佳防守", "创造双威胁")


def _analysis_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("缓存分析数据格式错误")
    recommended = data.get("recommended_move")
    return AnalysisResult(
        schema_version=int(data["schema_version"]),
        prompt_version=str(data.get("prompt_version", PROMPT_VERSION)),
        position_id=str(data["position_id"]),
        summary=str(data["summary"]),
        priority=str(data["priority"]),
        game_state=str(data.get("game_state", "")),
        is_terminal=bool(data.get("is_terminal", False)),
        threats=tuple(ThreatAnalysis(
            move=tuple(item["move"]),
            severity=str(item["severity"]),
            reason=str(item.get("reason", "")),
        ) for item in data.get("threats", ())),
        candidates=tuple(CandidateAnalysis(
            move=tuple(item["move"]),
            preference=str(item["preference"]),
            pros=str(item.get("pros", "")),
            cons=str(item.get("cons", "")),
        ) for item in data.get("candidates", ())),
        recommended_move=tuple(recommended) if recommended is not None else None,
        subjective_confidence=data.get("subjective_confidence"),
        warnings=tuple(str(item) for item in data.get("warnings", ())),
        model_name=str(data.get("model_name", "")),
        elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        usage=dict(data.get("usage", {})),
    )


class CandidateAnalysisCache:
    """Small JSON cache keyed by position, model, prompt and candidate set."""

    def __init__(self, directory=None):
        self.directory = Path(directory or CACHE_ROOT)

    @staticmethod
    def make_key(*, position_id, model_name, prompt_version, candidates):
        payload = {
            "position_id": str(position_id),
            "model_name": str(model_name),
            "prompt_version": str(prompt_version),
            "candidates": [list(move) for move in candidates],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _path(self, key):
        return self.directory / f"{key}.json"

    def get(self, key):
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return _analysis_from_dict(data)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def put(self, key, analysis):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        document = analysis.as_dict()
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


@dataclass(frozen=True)
class CoordinatorCandidate:
    move: tuple[int, int]
    base_rank: int
    base_score: float | None
    base_score_type: str
    llm_preference: str | None
    final_score: float

    def as_dict(self):
        return {
            "move": list(self.move),
            "base_rank": self.base_rank,
            "base_score": self.base_score,
            "base_score_type": self.base_score_type,
            "llm_preference": self.llm_preference,
            "final_score": round(self.final_score, 6),
        }


@dataclass(frozen=True)
class CoordinatorResult:
    base_move: tuple[int, int]
    final_move: tuple[int, int]
    candidates: tuple[CoordinatorCandidate, ...]
    analysis: AnalysisResult | None
    decision_source: str
    fallback_reason: str | None
    position_id: str
    elapsed_ms: float
    cache_hit: bool = False

    def as_dict(self):
        return {
            "base_move": list(self.base_move),
            "final_move": list(self.final_move),
            "candidates": [item.as_dict() for item in self.candidates],
            "analysis": self.analysis.as_dict() if self.analysis else None,
            "decision_source": self.decision_source,
            "fallback_reason": self.fallback_reason,
            "position_id": self.position_id,
            "elapsed_ms": self.elapsed_ms,
            "cache_hit": self.cache_hit,
        }


class AICoordinator:
    """Use a validated LLM analysis only as a small candidate reranking signal."""

    def __init__(self, base_ai, advisor, *, mode="analysis_only", alpha=0.0,
                 candidate_limit=8, cache=None, model_name="",
                 prompt_version=PROMPT_VERSION):
        if mode not in {"analysis_only", "rerank"}:
            raise ValueError("mode 必须是 analysis_only 或 rerank")
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须在 0 到 1 之间")
        candidate_limit = int(candidate_limit)
        if not 2 <= candidate_limit <= 8:
            raise ValueError("candidate_limit 必须在 2 到 8 之间")
        self.base_ai = base_ai
        self.advisor = advisor
        self.mode = mode
        self.alpha = alpha
        self.candidate_limit = candidate_limit
        self.cache = cache
        self.model_name = str(model_name or "")
        self.prompt_version = str(prompt_version)
        self._cancel_event = None
        self.last_result = None
        self.last_explanation = None

    def set_cancel_event(self, event):
        self._cancel_event = event
        setter = getattr(self.base_ai, "set_cancel_event", None)
        if callable(setter):
            setter(event)

    def set_seed(self, seed):
        setter = getattr(self.base_ai, "set_seed", None)
        if callable(setter):
            setter(seed)

    def _candidate_rows(self, board, player, base_move):
        rows = ()
        exporter = getattr(self.base_ai, "analysis_candidates", None)
        if callable(exporter):
            rows = exporter(self.candidate_limit)
        if not rows:
            explanation = getattr(self.base_ai, "last_explanation", None)
            candidates = getattr(explanation, "candidates", ())
            rows = tuple({
                "move": tuple(item.get("move", ())),
                "base_rank": rank,
                "base_score": item.get("score"),
                "base_score_type": getattr(explanation, "score_type", ""),
                "reason": item.get("reason", ""),
                "visits": item.get("visits"),
            } for rank, item in enumerate(candidates[:self.candidate_limit], 1))
        normalized = []
        seen = set()
        for row in rows:
            try:
                move = tuple(row["move"])
            except (KeyError, TypeError):
                continue
            if len(move) != 2:
                continue
            if move == base_move:
                continue
            if move in seen or not board.in_range(*move) or not board.is_empty(*move):
                continue
            seen.add(move)
            normalized.append(dict(row))
        normalized.insert(0, {
            "move": tuple(base_move),
            "base_rank": 1,
            "base_score": None,
            "base_score_type": "",
            "reason": "HybridAI 基准选择",
            "visits": None,
        })
        for rank, row in enumerate(normalized, 1):
            row["base_rank"] = rank
        return normalized

    @staticmethod
    def _is_forced(board, player, reason):
        if winning_moves(board, player) or opponent_threats(board, player):
            return True
        return any(marker in str(reason) for marker in FORCED_MARKERS)

    def _cache_key(self, observation, candidates):
        model_name = self.model_name
        if not model_name:
            model_name = str(getattr(self.advisor, "model", "") or "")
        return CandidateAnalysisCache.make_key(
            position_id=observation["position_id"],
            model_name=model_name,
            prompt_version=self.prompt_version,
            candidates=candidates,
        )

    def decide(self, board, player):
        if player not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {player!r}")
        started = time.perf_counter()
        base_move, base_reason = self.base_ai.choose_move(board, player)
        base_move = tuple(base_move)
        if not board.in_range(*base_move) or not board.is_empty(*base_move):
            raise ValueError(f"HybridAI 返回非法着法: {base_move!r}")

        rows = self._candidate_rows(board, player, base_move)
        base_observation = build_board_observation(
            board,
            player,
            state="playing",
            current_color=player,
            history=board.history,
            candidates=[row["move"] for row in rows],
            candidate_limit=self.candidate_limit,
            history_limit=12,
        )
        candidate_moves = [tuple(row["move"]) for row in rows]
        base_observation["candidates"] = [
            {
                "rank": index,
                "move": list(row["move"]),
                "cell": cell_name(*row["move"]),
                "score": row.get("base_score"),
                "source": "hybrid_base_rank",
                "reason": row.get("reason", ""),
            }
            for index, row in enumerate(rows, 1)
        ]
        base_observation["candidate_score_type"] = "hybrid_base_rank"
        if not candidate_moves:
            candidate_moves = [base_move]

        forced = self._is_forced(board, player, base_reason)
        insufficient_candidates = len(candidate_moves) < 2
        if forced or insufficient_candidates or self.advisor is None:
            if forced:
                reason = base_reason
                source = "forced"
                fallback_reason = None
            elif insufficient_candidates:
                reason = f"{base_reason} · 候选不足"
                source = "hybrid"
                fallback_reason = "候选不足，无需调用模型"
            else:
                reason = f"{base_reason} · LLM不可用"
                source = "fallback"
                fallback_reason = "LLM不可用"
            result = self._finish(
                board, player, base_move, base_move, rows, None,
                source, reason,
                fallback_reason, base_observation["position_id"],
                started, False,
            )
            return result

        cache_key = self._cache_key(base_observation, candidate_moves)
        analysis = self.cache.get(cache_key) if self.cache is not None else None
        cache_hit = analysis is not None
        fallback_reason = None
        if analysis is None:
            if self._cancel_event is not None and self._cancel_event.is_set():
                fallback_reason = "cancelled"
            else:
                try:
                    analysis = self.advisor.analyze(base_observation)
                    if self.cache is not None:
                        try:
                            self.cache.put(cache_key, analysis)
                        except OSError:
                            pass
                except Exception as exc:
                    fallback_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
        if fallback_reason is not None:
            reason = f"{base_reason} · 大模型回退"
            return self._finish(
                board, player, base_move, base_move, rows, None,
                "fallback", reason, fallback_reason,
                base_observation["position_id"], started, False,
            )
        if self._cancel_event is not None and self._cancel_event.is_set():
            reason = f"{base_reason} · 分析已取消"
            return self._finish(
                board, player, base_move, base_move, rows, analysis,
                "fallback", reason, "cancelled",
                base_observation["position_id"], started, cache_hit,
            )

        recommended = analysis.recommended_move
        if analysis.position_id != base_observation["position_id"]:
            fallback_reason = "分析结果与当前候选局面不一致"
        elif analysis.is_terminal:
            fallback_reason = "终局分析不能用于落子"
        if fallback_reason is not None:
            reason = f"{base_reason} · 大模型回退"
            return self._finish(
                board, player, base_move, base_move, rows, analysis,
                "fallback", reason, fallback_reason,
                base_observation["position_id"], started, cache_hit,
            )
        invalid_recommendation = False
        if analysis.recommended_move is not None:
            if recommended not in candidate_moves:
                invalid_recommendation = True
            else:
                invalid_recommendation = not board.is_empty(*recommended)
        if invalid_recommendation:
            reason = f"{base_reason} · 大模型回退"
            return self._finish(
                board, player, base_move, base_move, rows, analysis,
                "fallback", reason, "LLM推荐着法不在合法候选集合",
                base_observation["position_id"], started, cache_hit,
            )

        preferences = {item.move: item.preference for item in analysis.candidates}
        scored = []
        count = len(candidate_moves)
        for rank, move in enumerate(candidate_moves, 1):
            preference = preferences.get(move)
            bonus = PREFERENCE_BONUS.get(preference, 0.0)
            if recommended == move:
                bonus = max(bonus, 1.0)
            base_component = 1.0 - (rank - 1) / max(1, count - 1)
            scored.append(CoordinatorCandidate(
                move=move,
                base_rank=rank,
                base_score=next(
                    (row.get("base_score") for row in rows if tuple(row["move"]) == move),
                    None,
                ),
                base_score_type=next(
                    (row.get("base_score_type", "") for row in rows
                     if tuple(row["move"]) == move),
                    "",
                ),
                llm_preference=preference,
                final_score=base_component + self.alpha * bonus,
            ))
        scored.sort(key=lambda item: (-item.final_score, item.base_rank, item.move))
        final_move = base_move if self.mode == "analysis_only" else scored[0].move
        if not board.in_range(*final_move) or not board.is_empty(*final_move):
            final_move = base_move
            fallback_reason = "融合结果非法"
        if final_move == base_move:
            reason = f"{base_reason} · LLM确认"
            source = "hybrid"
        else:
            reason = "Hybrid+LLM候选重排"
            source = "hybrid_llm"
        return self._finish(
            board, player, base_move, final_move, scored, analysis,
            source, reason, fallback_reason,
            base_observation["position_id"], started, cache_hit,
        )

    def _finish(self, board, player, base_move, final_move, rows, analysis,
                source, reason, fallback_reason, position_id, started,
                cache_hit):
        elapsed_ms = (time.perf_counter() - started) * 1000
        if rows and isinstance(rows[0], CoordinatorCandidate):
            candidates = tuple(rows)
        else:
            candidates = tuple(CoordinatorCandidate(
                move=tuple(row["move"]),
                base_rank=int(row["base_rank"]),
                base_score=row.get("base_score"),
                base_score_type=str(row.get("base_score_type", "")),
                llm_preference=None,
                final_score=1.0 - (rank - 1) / max(1, len(rows) - 1),
            ) for rank, row in enumerate(rows, 1))
        self.last_result = CoordinatorResult(
            base_move=base_move,
            final_move=final_move,
            candidates=candidates,
            analysis=analysis,
            decision_source=source,
            fallback_reason=fallback_reason,
            position_id=position_id,
            elapsed_ms=round(elapsed_ms, 3),
            cache_hit=cache_hit,
        )
        self.last_explanation = AIExplanation(
            selected_move=final_move,
            tactic=reason,
            candidates=tuple({
                "move": item.move,
                "score": round(item.final_score, 6),
                "reason": reason,
                "base_rank": item.base_rank,
                "llm_preference": item.llm_preference,
            } for item in candidates),
            elapsed_ms=round(elapsed_ms, 3),
            search_type="hybrid-llm",
            score_type="融合排序",
            metadata={
                "base_move": list(base_move),
                "llm_move": list(analysis.recommended_move)
                if analysis and analysis.recommended_move else None,
                "fusion_mode": self.mode,
                "fusion_alpha": self.alpha,
                "fallback_reason": fallback_reason,
                "position_id": position_id,
                "cache_hit": cache_hit,
            },
        )
        return final_move, reason


class LLMAssistedAI:
    """AI adapter for optional experimental HybridAI + LLM mode."""

    name = "HybridAI+LLM"
    version = "6.0.0"

    def __init__(self, base_ai=None, advisor=None, *, mode="analysis_only",
                 alpha=0.0, candidate_limit=8, cache=None, model_name=""):
        self.base_ai = base_ai or HybridAI()
        if advisor is None:
            from ai.llm_advisor import LLMAdvisor
            from ai.llm_client import create_llm_client
            from ai.llm_config import load_llm_config
            config = load_llm_config()
            if not config.enabled:
                raise RuntimeError("LLM_ENABLED=0")
            advisor = LLMAdvisor(
                create_llm_client(config),
                timeout=config.analysis_timeout_seconds,
                max_tokens=config.max_output_tokens,
                enable_thinking=config.enable_thinking,
            )
            model_name = model_name or config.analysis_model
        self.coordinator = AICoordinator(
            self.base_ai,
            advisor,
            mode=mode,
            alpha=alpha,
            candidate_limit=candidate_limit,
            cache=cache,
            model_name=model_name,
        )
        self.last_explanation = None
        self.last_result = None

    def set_cancel_event(self, event):
        self.coordinator.set_cancel_event(event)

    def set_seed(self, seed):
        self.coordinator.set_seed(seed)

    def choose_move(self, board, player):
        move, reason = self.coordinator.decide(board, player)
        self.last_explanation = self.coordinator.last_explanation
        self.last_result = self.coordinator.last_result
        return move, reason