# -*- coding: utf-8 -*-
"""Shared V1 integration fixtures built from real HexEngine games."""
import json

from ai.memory.collector import CAPTURE_SHADOW, MemoryCaptureConfig, MemoryCollector
from ai.memory.features import FEATURE_VERSION, features_json
from ai.memory.store import ExperienceStore
from ai.memory.trace import ActorMetadata, DecisionTrace, snapshot_board
from ai.memory.validator import validate_match
from hex_board import BLUE, RED, HexBoard
from hex_engine import HexEngine


AI_ACTOR = ActorMetadata(
    actor_type="ai",
    actor_id="TestAI",
    ai_name="TestAI",
    ai_version="1.0.0",
    model_sha256="a" * 64,
)


def collect_scripted_match(db_path, *, red_column=0, blue_column=10):
    collector = MemoryCollector(
        MemoryCaptureConfig(
            mode=CAPTURE_SHADOW,
            db_path=str(db_path),
            queue_size=2,
            flush_timeout_seconds=2.0,
            fail_open=False,
        ),
        synchronous=True,
    )
    match_id = collector.start_match(
        mode_key=4,
        mode_name="AI vs AI",
        red_team="AI-R",
        blue_team="AI-B",
        source_type="ai_vs_ai",
        actor_types={RED: "ai", BLUE: "ai"},
    )
    for game_no in (1, 2):
        if game_no == 2:
            collector.begin_game(
                2,
                sides={RED: "AI-B", BLUE: "AI-R"},
                actor_types={RED: "ai", BLUE: "ai"},
            )
        engine = HexEngine("AI-R", "AI-B")
        winner = None
        for row in range(11):
            for color, column in ((RED, red_column), (BLUE, blue_column)):
                if engine.state != "playing":
                    break
                board_before = snapshot_board(engine.board)
                trace = DecisionTrace(
                    move=(column, row),
                    color=color,
                    reason="scripted",
                    actor=AI_ACTOR,
                    candidate_payload={"chosen": [column, row]},
                    search_config={"simulations": 10},
                    elapsed_ms=1.0,
                )
                won = engine.play_move(column, row, reason="scripted")
                collector.record_move(board_before, trace)
                if won:
                    winner = engine.winner_color
                    collector.mark_game_over(winner, "connect")
                    break
            if winner is not None:
                break
        if winner is None:
            raise RuntimeError("scripted game did not finish")
    if not collector.accept_match():
        raise RuntimeError("scripted match was not persisted")
    collector.close()
    return match_id


def populate_all_features(db_path):
    with ExperienceStore(db_path) as store:
        positions = store.list_positions_missing_features(limit=100000)
        for position in positions:
            document = json.loads(position["board_json"])
            board = HexBoard()
            board.grid = [list(row) for row in document["grid"]]
            board.history = []
            store.update_position_features(
                position["position_id"],
                features_json(
                    board,
                    int(position["current_color"]),
                    int(position["ply"]),
                ),
                feature_version=FEATURE_VERSION,
            )
        return len(positions)


def promote_match(db_path, match_id):
    with ExperienceStore(db_path) as store:
        return validate_match(store, match_id, promote=True)