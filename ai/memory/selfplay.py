# -*- coding: utf-8 -*-
"""Explicit self-play generation for V3 experience data."""
from __future__ import annotations

from dataclasses import dataclass
import random

from ai.memory.collector import CAPTURE_SHADOW, MemoryCaptureConfig, MemoryCollector
from ai.memory.trace import ActorMetadata, DecisionTrace, ai_actor, snapshot_board
from hex_board import BLUE, RED
from hex_engine import HexEngine, STATE_GAME_OVER, STATE_PLAYING


@dataclass(frozen=True)
class SelfPlayMatchResult:
    match_id: str
    winner_a: int
    winner_b: int
    moves: int
    games: int

    def as_dict(self):
        return {
            "match_id": self.match_id,
            "winner_a": self.winner_a,
            "winner_b": self.winner_b,
            "moves": self.moves,
            "games": self.games,
        }


def _seed_ai(ai, seed):
    setter = getattr(ai, "set_seed", None)
    if callable(setter):
        setter(seed)
    else:
        random.seed(seed)


def _set_context(ai, context):
    setter = getattr(ai, "set_memory_context", None)
    if callable(setter):
        setter(**context)


def _actor(ai, side_name):
    actor = ai_actor(ai)
    return ActorMetadata(
        actor_type="ai",
        actor_id=actor.actor_id or side_name,
        ai_name=actor.ai_name or side_name,
        ai_version=actor.ai_version,
        model_sha256=actor.model_sha256,
    )


def play_memory_match(ai_a, ai_b, *, db_path, seed, source_type="self_play"):
    """Play one two-game match and atomically record both complete games."""
    collector = MemoryCollector(
        MemoryCaptureConfig(
            mode=CAPTURE_SHADOW,
            db_path=str(db_path),
            queue_size=2,
            flush_timeout_seconds=3.0,
            fail_open=False,
        ),
        synchronous=True,
    )
    actor_a = _actor(ai_a, "AI-A")
    actor_b = _actor(ai_b, "AI-B")
    total_moves = 0
    winner_a = 0
    winner_b = 0
    try:
        match_id = collector.start_match(
            mode_key=4,
            mode_name="AI vs AI self-play",
            red_team="AI-A",
            blue_team="AI-B",
            source_type=source_type,
            actor_types={RED: "ai", BLUE: "ai"},
            actor_metadata={RED: actor_a, BLUE: actor_b},
        )
        for game_no in (1, 2):
            if game_no == 2:
                collector.begin_game(
                    2,
                    sides={RED: "AI-B", BLUE: "AI-A"},
                    actor_types={RED: "ai", BLUE: "ai"},
                    actor_metadata={RED: actor_b, BLUE: actor_a},
                )
            engine = HexEngine("AI-A", "AI-B")
            side_map = engine.sides_for_game(game_no)
            color_ai = {RED: ai_a, BLUE: ai_b}
            color_actor = {RED: actor_a, BLUE: actor_b}
            if side_map[RED] == "AI-B":
                color_ai = {RED: ai_b, BLUE: ai_a}
                color_actor = {RED: actor_b, BLUE: actor_a}
            _seed_ai(ai_a, int(seed) + game_no * 2)
            _seed_ai(ai_b, int(seed) + game_no * 2 + 1)
            while engine.state == STATE_PLAYING:
                color = engine.current_color
                selected_ai = color_ai[color]
                _set_context(selected_ai, collector.current_context)
                board_before = snapshot_board(engine.board)
                move, reason = selected_ai.choose_move(engine.board, color)
                if not isinstance(move, tuple) or len(move) != 2:
                    raise ValueError(f"Self-play AI returned invalid move: {move!r}")
                if not engine.board.in_range(*move) or not engine.board.is_empty(*move):
                    raise ValueError(f"Self-play AI returned illegal move: {move!r}")
                trace = DecisionTrace(
                    move=tuple(move),
                    color=color,
                    reason=str(reason or "self-play"),
                    actor=color_actor[color],
                    candidate_payload=getattr(selected_ai, "last_explanation", None),
                    search_config={
                        "source": source_type,
                        "seed": int(seed),
                        "game_no": game_no,
                        "actor": side_map[color],
                    },
                    elapsed_ms=0.0,
                )
                engine.play_move(move[0], move[1], reason=trace.reason)
                collector.record_move(board_before, trace)
                total_moves += 1
            if engine.state != STATE_GAME_OVER:
                raise RuntimeError("Self-play game did not finish")
            winner_color = engine.winner_color
            collector.mark_game_over(
                winner_color,
                engine.results[-1].get("reason", "connect"),
            )
            if side_map[winner_color] == "AI-A":
                winner_a += 1
            else:
                winner_b += 1
        if not collector.accept_match():
            raise RuntimeError("Self-play match was not accepted")
    except Exception:
        collector.abort_match("selfplay_failed")
        raise
    finally:
        collector.close()
    return SelfPlayMatchResult(
        match_id=match_id,
        winner_a=winner_a,
        winner_b=winner_b,
        moves=total_moves,
        games=2,
    )
def run_selfplay_series(factory_a, factory_b, *, db_path, matches, seed):
    """Create explicit two-game self-play matches with independent factories."""
    count = int(matches)
    if count < 1:
        raise ValueError("matches must be positive")
    results = []
    for index in range(count):
        ai_a = factory_a()
        ai_b = factory_b()
        try:
            results.append(play_memory_match(
                ai_a,
                ai_b,
                db_path=db_path,
                seed=int(seed) + index * 100003,
            ))
        finally:
            for ai in (ai_a, ai_b):
                closer = getattr(ai, "close", None)
                if callable(closer):
                    closer()
    return tuple(results)