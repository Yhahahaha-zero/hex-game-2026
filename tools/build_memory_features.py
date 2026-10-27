# -*- coding: utf-8 -*-
"""Backfill explainable V1 features for recorded positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.features import FEATURE_VERSION, features_json
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore
from hex_board import BLUE, EMPTY, RED, HexBoard


def _board(raw_value):
    document = json.loads(raw_value)
    board = HexBoard()
    board.grid = [list(row) for row in document["grid"]]
    board.history = []
    return board


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args(argv)
    updated = 0
    errors = []
    with ExperienceStore(args.db) as store:
        rows = (
            store.all_positions(limit=args.limit)
            if args.all else store.list_positions_missing_features(limit=args.limit)
        )
        for row in rows:
            try:
                board = _board(row["board_json"])
                value = features_json(
                    board, int(row["current_color"]), int(row["ply"])
                )
                store.update_position_features(
                    row["position_id"],
                    value,
                    feature_version=FEATURE_VERSION,
                )
                updated += 1
            except Exception as exc:
                errors.append({"position_id": row["position_id"], "error": str(exc)})
    print(json.dumps({
        "updated": updated,
        "errors": errors,
        "feature_version": FEATURE_VERSION,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())