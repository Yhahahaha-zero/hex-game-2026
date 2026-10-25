# -*- coding: utf-8 -*-
"""棋局报告、CSV 和界面截图导出。"""
import csv
from datetime import datetime
import json
from pathlib import Path

import pygame

from hex_board import COLOR_NAMES
from game_records import save_match_record


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
REPORTS_DIR = WORKSPACE_ROOT / "artifacts" / "reports"


def capture_screen(screen, output_dir=None, name=None):
    """保存当前 pygame 逻辑画布。"""
    output_dir = Path(output_dir or (REPORTS_DIR / "screenshots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = name or f"hex_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = output_dir / filename
    pygame.image.save(screen, str(path))
    return path


def export_report_bundle(engine, mode, player_labels, timer=None, screen=None,
                         search_stats=None, output_dir=None, note=None):
    """导出一次完整报告目录并返回所有产物路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(output_dir or REPORTS_DIR)
    report_dir = root / f"report_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)

    match_path = save_match_record(
        engine,
        mode,
        player_labels,
        timer,
        path=report_dir / "match.json",
    )
    moves = [dict(item) for item in engine.history]
    moves_path = report_dir / "moves.csv"
    _write_csv(moves_path, moves, [
        "game", "no", "color", "side", "cell", "q", "r", "ms", "reason",
    ])

    search_rows = []
    for index, item in enumerate(search_stats or (), 1):
        move = item.get("move")
        search_rows.append({
            "rank": index,
            "move": f"{move[0]},{move[1]}" if move else "",
            "visits": item.get("visits", 0),
            "prior": item.get("prior", 0),
            "value": item.get("value", 0),
        })
    stats_path = None
    if search_rows:
        stats_path = report_dir / "mcts_stats.csv"
        _write_csv(stats_path, search_rows, ["rank", "move", "visits", "prior", "value"])

    timer_values = timer.snapshot() if timer else {}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": {"key": int(mode), "name": mode.label},
        "players": dict(player_labels),
        "state": engine.state,
        "game_no": engine.game_no,
        "move_count": len(moves),
        "match_score": engine.match_score,
        "match_winner": engine.match_winner(),
        "timers_seconds": {COLOR_NAMES[key]: value for key, value in timer_values.items()},
        "note": note or "",
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    screenshot_path = None
    if screen is not None:
        screenshot_path = capture_screen(screen, report_dir, "screenshot.png")

    return {
        "directory": report_dir,
        "match_json": match_path,
        "moves_csv": moves_path,
        "mcts_csv": stats_path,
        "summary_json": summary_path,
        "screenshot": screenshot_path,
    }


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
