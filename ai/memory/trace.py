# -*- coding: utf-8 -*-
"""Immutable runtime traces used by the P0-C shadow collector."""
from __future__ import annotations

from dataclasses import dataclass
import json

from hex_board import COLS, ROWS, HexBoard


@dataclass(frozen=True)
class ActorMetadata:
    actor_type: str
    actor_id: str | None = None
    ai_name: str | None = None
    ai_version: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True)
class DecisionTrace:
    move: tuple[int, int]
    color: int
    reason: str
    actor: ActorMetadata
    candidate_payload: object | None = None
    search_config: dict | None = None
    elapsed_ms: float = 0.0


def snapshot_board(board):
    """Return a deep copy that cannot alias the live HexBoard."""
    copied = HexBoard()
    copied.grid = [row[:] for row in board.grid]
    copied.history = list(board.history)
    return copied


def board_state_json(board):
    """Serialize only stable rule state, excluding move history and UI data."""
    return json.dumps(
        {
            "rows": ROWS,
            "cols": COLS,
            "grid": [list(row) for row in board.grid],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_text(value):
    if value is None:
        return None
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def human_actor(player_id):
    label = f"player{player_id}" if player_id is not None else "human"
    return ActorMetadata(actor_type="human", actor_id=label)


def ai_actor(ai):
    name = str(getattr(ai, "name", "") or type(ai).__name__)
    version = str(getattr(ai, "version", "") or "")
    model_sha256 = str(getattr(ai, "model_sha256", "") or "")
    selection = getattr(ai, "model_selection", None)
    if not model_sha256 and isinstance(selection, dict):
        model_sha256 = str(selection.get("sha256", "") or "")
    return ActorMetadata(
        actor_type="ai",
        actor_id=name,
        ai_name=name,
        ai_version=version or None,
        model_sha256=model_sha256 or None,
    )


def human_decision_trace(move, color, reason, player_id=None):
    return DecisionTrace(
        move=tuple(move),
        color=color,
        reason=str(reason or "human move"),
        actor=human_actor(player_id),
    )


def ai_decision_trace(result):
    fallback = ai_actor(result.ai)
    actor = ActorMetadata(
        actor_type="ai",
        actor_id=getattr(result, "ai_name", None) or fallback.actor_id,
        ai_name=getattr(result, "ai_name", None) or fallback.ai_name,
        ai_version=getattr(result, "ai_version", None) or fallback.ai_version,
        model_sha256=(
            getattr(result, "model_sha256", None) or fallback.model_sha256
        ),
    )
    search_config = {
        "search_stats": list(result.search_stats or ()),
        "init_error": result.init_error,
        "inference_error": result.inference_error,
        "fallback_error": result.fallback_error,
        "used_fallback": result.used_fallback,
    }
    return DecisionTrace(
        move=tuple(result.move),
        color=result.color,
        reason=str(result.reason or ""),
        actor=actor,
        candidate_payload=result.explanation,
        search_config=search_config,
        elapsed_ms=float(getattr(result, "elapsed_ms", 0.0) or 0.0),
    )