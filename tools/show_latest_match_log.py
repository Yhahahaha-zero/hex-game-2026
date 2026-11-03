# -*- coding: utf-8 -*-
"""Print a concise result and move summary for the latest Hex match record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_records import latest_record, load_record
from hex_engine import STATE_MATCH_OVER


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="Optional explicit JSON record path")
    args = parser.parse_args(argv)
    path = Path(args.path).expanduser().resolve() if args.path else latest_record()
    if path is None or not Path(path).is_file():
        raise SystemExit("没有找到棋局记录，请先在游戏中完成对局并按 Ctrl+S 保存。")

    document = load_record(path)
    games = document.get("games", [])
    total_moves = sum(len(game.get("moves", [])) for game in games)
    print(f"[RECORD] {path}")
    print(f"[MODE] {document.get('mode', {}).get('name', '-')}")
    print(f"[SCORE] {document.get('match_score', {})}")
    for game in games:
        result = game.get("result") or {}
        print(
            f"[GAME {game.get('game', '?')}] "
            f"moves={len(game.get('moves', []))} "
            f"winner={result.get('winner_side', '-')} "
            f"winner_color={result.get('winner_color', '-')} "
            f"reason={result.get('reason', '-')}"
        )
    print(f"[TOTAL] moves={total_moves}")
    print(f"[STATE] {document.get('state', 'unknown')}")
    completed_games = sum(1 for game in games if game.get("result"))
    if document.get("state") == STATE_MATCH_OVER:
        winner = document.get("match_winner")
        print(f"[RESULT] match_winner={winner or '平局/待判定'}")
    elif len(games) == 2 and completed_games == 2:
        print("[NOTE] 两局棋均已结束，比分已完整；但记录状态仍为 game_over。")
        print("[NEXT] 请返回游戏结束界面，继续进入整场结算并再次按 Ctrl+S，才能得到 match_over。")
    else:
        print("[WARN] 当前比赛尚未完成两局，请完成比赛后重新按 Ctrl+S 保存。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())