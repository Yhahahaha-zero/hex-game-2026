# -*- coding: utf-8 -*-
"""JSON 棋局记录、加载和逐步复盘。"""
from datetime import datetime
import json
from pathlib import Path

from hex_board import BLUE, COLS, RED, ROWS, HexBoard, parse_cell
from hex_engine import STATE_MATCH_OVER


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
RECORD_DIR = WORKSPACE_ROOT / "artifacts" / "records"


def build_match_record(engine, mode, player_labels, timer=None):
    """将引擎状态序列化为稳定的 JSON 文档。"""
    games = []
    for game_no in range(1, engine.game_no + 1):
        moves = [dict(move) for move in engine.history if move["game"] == game_no]
        result = next((dict(item) for item in engine.results if item["game"] == game_no), None)
        games.append({
            "game": game_no,
            "moves": moves,
            "result": result,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "mode": {"key": int(mode), "name": mode.label},
        "player_labels": dict(player_labels),
        "board": {"rows": ROWS, "cols": COLS},
        "state": engine.state,
        "match_score": engine.match_score,
        "match_winner": engine.match_winner() if engine.state == STATE_MATCH_OVER else None,
        "timers_seconds": timer.snapshot() if timer else {},
        "games": games,
    }


def save_match_record(engine, mode, player_labels, timer=None, path=None):
    """保存 JSON 记录并返回实际路径。"""
    document = build_match_record(engine, mode, player_labels, timer)
    if path is None:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RECORD_DIR / f"hex_match_{stamp}.json"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_record(path):
    """加载并做基础结构检查。"""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的记录版本: {document.get('schema_version')!r}")
    if not isinstance(document.get("games"), list):
        raise ValueError("记录缺少 games 列表")
    return document


def latest_record(directory=RECORD_DIR):
    """返回按修改时间排序的最新记录路径。"""
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


class ReplaySession:
    """从 JSON 记录构造可逐步前进/后退的只读局面。"""

    def __init__(self, document, game_no=None):
        self.document = document
        games = document.get("games", [])
        if game_no is None:
            game_no = games[-1]["game"] if games else 1
        game = next((item for item in games if item["game"] == game_no), None)
        if game is None:
            raise ValueError(f"记录中不存在第 {game_no} 局")
        self.game_no = game_no
        self.moves = list(game.get("moves", []))
        self.step = 0
        self.board = HexBoard()
        self.last_move = None
        self.goto(0)

    @property
    def total_moves(self):
        return len(self.moves)

    @property
    def finished(self):
        return self.step >= self.total_moves

    def goto(self, step):
        step = max(0, min(int(step), self.total_moves))
        board = HexBoard()
        for move in self.moves[:step]:
            color = move.get("color")
            if color not in (RED, BLUE):
                raise ValueError("复盘记录包含非法颜色")
            cell = move.get("cell")
            coords = (move.get("q"), move.get("r"))
            if coords[0] is None or coords[1] is None:
                coords = parse_cell(cell)
            if coords is None:
                raise ValueError(f"复盘记录包含非法坐标: {cell!r}")
            board.place(coords[0], coords[1], color)
        self.board = board
        self.step = step
        if step:
            move = self.moves[step - 1]
            self.last_move = (move["q"], move["r"], move["color"], move.get("reason", ""))
        else:
            self.last_move = None

    def next(self):
        if not self.finished:
            self.goto(self.step + 1)
        return self.board

    def previous(self):
        if self.step > 0:
            self.goto(self.step - 1)
        return self.board

    def current_moves(self):
        return self.moves[:self.step]
