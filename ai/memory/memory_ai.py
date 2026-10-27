# -*- coding: utf-8 -*-
"""Optional memory-retrieval wrapper around an existing Hex AI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import time
import uuid

from ai.explanation import AIExplanation
from ai.memory.collector import _env_setting, _read_env_file
from ai.memory.features import FEATURE_VERSION
from ai.memory.retrieval import MemoryRetriever
from ai.memory.schema import RETRIEVAL_VERSION, memory_db_path
from ai.memory.shadow import ShadowComparison
from ai.memory.store import ExperienceStore, RetrievalEventRecord
from ai.memory.trace import json_text
from ai.tactics import creates_double_threat
from hex_ai import winning_moves
from hex_board import BLUE, RED


RETRIEVAL_OFF = "off"
RETRIEVAL_SHADOW = "shadow"
RETRIEVAL_ASSIST = "assist"
RETRIEVAL_MODES = (RETRIEVAL_OFF, RETRIEVAL_SHADOW, RETRIEVAL_ASSIST)


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MemoryRetrievalSettings:
    mode: str = RETRIEVAL_OFF
    db_path: str | None = None
    top_k: int = 5
    minimum_similarity: float = 0.72
    minimum_support: int = 3
    minimum_confidence: float = 0.55
    alpha: float = 0.15
    ply_window: int = 6
    retrieval_types: tuple[str, ...] = ("exact", "canonical", "similar")

    @classmethod
    def from_env(cls, *, env_file=None, environ=None):
        root = Path(__file__).resolve().parents[2]
        env_path = Path(env_file) if env_file is not None else root / ".env.local"
        file_values = _read_env_file(env_path)
        env = os.environ if environ is None else environ
        mode = _env_setting(
            file_values, env, "HEX_MEMORY_RETRIEVAL", RETRIEVAL_OFF
        ).lower()
        if mode not in RETRIEVAL_MODES:
            mode = RETRIEVAL_OFF
        return cls(
            mode=mode,
            db_path=(
                _env_setting(file_values, env, "HEX_MEMORY_DB", "")
                or str(memory_db_path(env))
            ),
            top_k=max(1, _safe_int(
                _env_setting(file_values, env, "HEX_MEMORY_TOP_K", "5"), 5
            )),
            minimum_similarity=max(0.0, min(1.0, _safe_float(
                _env_setting(file_values, env, "HEX_MEMORY_MIN_SIMILARITY", "0.72"),
                0.72,
            ))),
            minimum_support=max(1, _safe_int(
                _env_setting(file_values, env, "HEX_MEMORY_MIN_SUPPORT", "3"), 3
            )),
            minimum_confidence=max(0.0, min(1.0, _safe_float(
                _env_setting(file_values, env, "HEX_MEMORY_MIN_CONFIDENCE", "0.55"),
                0.55,
            ))),
            alpha=max(0.0, min(1.0, _safe_float(
                _env_setting(file_values, env, "HEX_MEMORY_ALPHA", "0.15"), 0.15
            ))),
            ply_window=max(0, _safe_int(
                _env_setting(file_values, env, "HEX_MEMORY_PLY_WINDOW", "6"), 6
            )),
            retrieval_types=(
                tuple(
                    item.strip().lower()
                    for item in _env_setting(
                        file_values,
                        env,
                        "HEX_MEMORY_RETRIEVAL_TYPES",
                        "exact,canonical,similar",
                    ).split(",")
                    if item.strip() in ("exact", "canonical", "similar")
                )
                or ("exact", "canonical", "similar")
            ),
        )


class MemoryAssistedAI:
    """Wrap a base AI and optionally use validated memory advice."""

    def __init__(self, base_ai, *, settings=None, store=None,
                 store_factory=None, clock=None):
        self.base_ai = base_ai
        self.settings = settings or MemoryRetrievalSettings.from_env()
        if self.settings.mode not in RETRIEVAL_MODES:
            raise ValueError("invalid memory retrieval mode")
        self._store = store
        self._store_factory = store_factory
        self._clock = clock or _now
        self._context = {}
        self.last_memory_advice = None
        self.last_shadow_comparison = None
        self.last_memory_error = None
        self.last_explanation = None
        self.name = f"{getattr(base_ai, 'name', type(base_ai).__name__)}+Memory"
        self.version = str(getattr(base_ai, "version", "1.0.0") or "1.0.0")
        self.model_sha256 = getattr(base_ai, "model_sha256", None)

    @property
    def memory_enabled(self):
        return self.settings.mode != RETRIEVAL_OFF

    def set_memory_context(self, **context):
        self._context = dict(context)

    def set_cancel_event(self, event):
        setter = getattr(self.base_ai, "set_cancel_event", None)
        if callable(setter):
            setter(event)

    def set_seed(self, seed):
        setter = getattr(self.base_ai, "set_seed", None)
        if callable(setter):
            setter(seed)

    def set_progress_callback(self, callback):
        setter = getattr(self.base_ai, "set_progress_callback", None)
        if callable(setter):
            setter(callback)

    def analysis_candidates(self, limit=8):
        exporter = getattr(self.base_ai, "analysis_candidates", None)
        if callable(exporter):
            return exporter(limit)
        explanation = getattr(self.base_ai, "last_explanation", None)
        if isinstance(explanation, AIExplanation):
            return [
                {"rank": index, **dict(item)}
                for index, item in enumerate(explanation.candidates[:limit], 1)
            ]
        if isinstance(explanation, dict):
            return [
                {"rank": index, **dict(item)}
                for index, item in enumerate(
                    explanation.get("candidates", [])[:limit], 1
                )
            ]
        return []

    def _with_store(self):
        if self._store is not None:
            return self._store, False
        if self._store_factory is not None:
            return self._store_factory(), True
        return ExperienceStore(self.settings.db_path).open(), True
    def _candidate_rows(self, board, player, base_move):
        rows = list(self.analysis_candidates(limit=max(8, self.settings.top_k)))
        moves = []
        for row in rows:
            move = tuple(row.get("move", ())) if isinstance(row, dict) else ()
            if len(move) == 2 and board.in_range(*move) and board.is_empty(*move):
                moves.append(move)
        if base_move not in moves:
            moves.insert(0, base_move)
        return moves, rows

    @staticmethod
    def _forced_tactic(board, player, base_move):
        opponent = BLUE if player == RED else RED
        return bool(
            winning_moves(board, player)
            or winning_moves(board, opponent)
            or creates_double_threat(board, *base_move, player)
        )

    def _assist_move(self, board, player, base_move, advice):
        if advice is None or not advice.available:
            return base_move
        if advice.confidence < self.settings.minimum_confidence:
            return base_move
        candidates, _ = self._candidate_rows(board, player, base_move)
        memory_move = advice.recommended_move
        if memory_move not in candidates:
            if advice.confidence < 0.80:
                return base_move
            candidates.append(memory_move)
        scored = []
        for index, move in enumerate(candidates):
            base_score = (
                0.70 if move == base_move else max(0.20, 0.68 - 0.04 * index)
            )
            memory_bonus = (
                self.settings.alpha * advice.confidence
                if move == memory_move else 0.0
            )
            scored.append((base_score + memory_bonus, move))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1]

    def _log_event(self, store, *, query_position_id, advice, base_move,
                   shadow_move, final_move, elapsed_ms, evidence):
        selected_decision_id = None
        similarity = None
        if advice and advice.recommended_move:
            matching = [
                hit for hit in advice.hits
                if hit.mapped_move == advice.recommended_move
            ]
            if matching:
                selected_decision_id = matching[0].decision_id
                similarity = matching[0].similarity
        event = RetrievalEventRecord(
            retrieval_event_id=str(uuid.uuid4()),
            query_position_id=query_position_id,
            query_game_id=self._context.get("game_id"),
            query_match_id=self._context.get("match_id"),
            retrieval_mode=self.settings.mode,
            top_k=self.settings.top_k,
            selected_memory_decision_id=selected_decision_id,
            similarity=similarity,
            memory_move=shadow_move if shadow_move != base_move else None,
            base_move=base_move,
            shadow_move=shadow_move,
            final_move=final_move,
            confidence=advice.confidence if advice else None,
            applied=int(final_move != base_move),
            feature_version=FEATURE_VERSION,
            retrieval_version=RETRIEVAL_VERSION,
            elapsed_ms=elapsed_ms,
            evidence_json=json_text(evidence),
            created_at=self._clock(),
        )
        store.save_retrieval_event(event)

    def _skip_retrieval(self, reason, base_move, player, board):
        self.last_memory_advice = None
        self.last_shadow_comparison = ShadowComparison(
            base_move=base_move,
            memory_move=None,
            final_move=base_move,
            applied=False,
            confidence=0.0,
        )
        return {
            "retrieval_skipped": reason,
            "player": player,
            "ply": len(board.history) + 1,
        }

    def _build_explanation(self, board, player, base_move, base_reason,
                           advice, final_move, tactic, elapsed_ms):
        shadow = self.last_shadow_comparison.as_dict()
        memory_metadata = {
            "mode": self.settings.mode,
            "advice": advice.as_dict() if advice else None,
            "shadow": shadow,
            "elapsed_ms": elapsed_ms,
        }
        moves, _ = self._candidate_rows(board, player, base_move)
        rows = []
        for index, move in enumerate(moves):
            base_score = (
                0.70 if move == base_move else max(0.20, 0.68 - 0.04 * index)
            )
            memory_bonus = (
                self.settings.alpha * advice.confidence
                if advice and move == advice.recommended_move else 0.0
            )
            rows.append({
                "move": list(move),
                "score": round(base_score + memory_bonus, 6),
                "source": "memory_assist",
            })
        rows.sort(key=lambda row: (-row["score"], tuple(row["move"])))
        return AIExplanation(
            selected_move=final_move,
            tactic=tactic,
            candidates=tuple(rows),
            elapsed_ms=elapsed_ms,
            search_type=(
                "memory_assist" if self.settings.mode == RETRIEVAL_ASSIST
                else "memory_shadow"
            ),
            score_type="base_rank + memory_bonus",
            metadata=memory_metadata,
        ).as_dict()
    def choose_move(self, board, player):
        if player not in (RED, BLUE):
            raise ValueError(f"invalid player: {player!r}")
        base_move, base_reason = self.base_ai.choose_move(board, player)
        base_move = tuple(base_move)
        if not board.in_range(*base_move) or not board.is_empty(*base_move):
            raise ValueError(f"base AI returned illegal move: {base_move!r}")
        self.last_memory_advice = None
        self.last_shadow_comparison = None
        self.last_memory_error = None
        if not self.memory_enabled:
            self.last_explanation = AIExplanation(
                selected_move=base_move,
                tactic=base_reason,
                search_type="memory_off",
            ).as_dict()
            return base_move, base_reason
        started = time.perf_counter()
        advice = None
        evidence = {}
        store = None
        close_store = False
        try:
            if self._forced_tactic(board, player, base_move):
                evidence = self._skip_retrieval(
                    "forced_tactical_position", base_move, player, board
                )
            else:
                store, close_store = self._with_store()
                retriever = MemoryRetriever(
                    store,
                    top_k=self.settings.top_k,
                    minimum_similarity=self.settings.minimum_similarity,
                    minimum_support=self.settings.minimum_support,
                    ply_window=self.settings.ply_window,
                    allowed_types=self.settings.retrieval_types,
                )
                advice = retriever.query(
                    board,
                    player,
                    ply=len(board.history) + 1,
                    game_id=self._context.get("game_id"),
                    match_id=self._context.get("match_id"),
                )
                evidence = advice.as_dict()
            self.last_memory_advice = advice
            shadow_move = (
                advice.recommended_move
                if advice and advice.recommended_move else base_move
            )
            final_move = base_move
            if self.settings.mode == RETRIEVAL_ASSIST:
                final_move = self._assist_move(board, player, base_move, advice)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            self.last_shadow_comparison = ShadowComparison(
                base_move=base_move,
                memory_move=shadow_move if shadow_move != base_move else None,
                final_move=final_move,
                applied=final_move != base_move,
                confidence=advice.confidence if advice else 0.0,
            )
            if store is not None:
                self._log_event(
                    store,
                    query_position_id=advice.query_position_id if advice else "",
                    advice=advice,
                    base_move=base_move,
                    shadow_move=shadow_move,
                    final_move=final_move,
                    elapsed_ms=elapsed_ms,
                    evidence=evidence,
                )
            tactic = base_reason
            if advice and advice.recommended_move and self.settings.mode == RETRIEVAL_ASSIST:
                tactic += f" | memory {advice.confidence:.2f}"
            self.last_explanation = self._build_explanation(
                board,
                player,
                base_move,
                base_reason,
                advice,
                final_move,
                tactic,
                elapsed_ms,
            )
            reported_reason = (
                tactic if self.settings.mode == RETRIEVAL_ASSIST else base_reason
            )
            return final_move, reported_reason
        except Exception as exc:
            self.last_memory_error = str(exc)
            self.last_shadow_comparison = ShadowComparison(
                base_move=base_move,
                memory_move=None,
                final_move=base_move,
                applied=False,
                confidence=0.0,
            )
            self.last_explanation = AIExplanation(
                selected_move=base_move,
                tactic=base_reason,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                search_type="memory_fallback",
                metadata={"memory_error": str(exc)},
            ).as_dict()
            return base_move, base_reason
        finally:
            if close_store and store is not None:
                try:
                    store.close()
                except Exception:
                    pass

    def close(self):
        if self._store is not None:
            self._store.close()
        closer = getattr(self.base_ai, "close", None)
        if callable(closer):
            closer()


def wrap_ai_with_memory(ai, *, settings=None, store=None, store_factory=None):
    settings = settings or MemoryRetrievalSettings.from_env()
    if settings.mode == RETRIEVAL_OFF:
        return ai
    return MemoryAssistedAI(
        ai,
        settings=settings,
        store=store,
        store_factory=store_factory,
    )