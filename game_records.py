# -*- coding: utf-8 -*-
"""JSON 棋局记录、深度校验、原子保存和逐步复盘。"""
from collections import Counter
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import uuid

from hex_board import BLUE, COLS, RED, ROWS, HexBoard, IllegalMoveError, cell_name, parse_cell
from hex_engine import (STATE_GAME_OVER, STATE_MATCH_OVER, STATE_PLAYING)


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
VALID_STATES = (STATE_PLAYING, STATE_GAME_OVER, STATE_MATCH_OVER)
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
RECORD_DIR = WORKSPACE_ROOT / "artifacts" / "records"


class RecordValidationError(ValueError):
    """棋局记录结构、顺序、比分或完整性校验失败。"""


def _require(condition, message):
    if not condition:
        raise RecordValidationError(message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_record_sha256(document):
    """Calculate a stable SHA256 over the record excluding the hash field."""
    if not isinstance(document, dict):
        raise RecordValidationError("记录根节点必须是 JSON 对象")
    content = {key: value for key, value in document.items()
               if key != "record_sha256"}
    # Normalize through JSON so integer timer keys match their loaded string form.
    normalized = json.loads(json.dumps(content, ensure_ascii=False))
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def refresh_record_integrity(document):
    document["record_sha256"] = compute_record_sha256(document)
    return document


def build_match_record(engine, mode, player_labels, timer=None):
    """Serialize HexEngine state into the versioned record schema."""
    games = []
    for game_no in range(1, engine.game_no + 1):
        moves = [dict(move) for move in engine.history if move["game"] == game_no]
        result = next(
            (dict(item) for item in engine.results if item["game"] == game_no),
            None,
        )
        games.append({
            "game": game_no,
            "moves": moves,
            "result": result,
        })

    game_one_sides = engine.sides_for_game(1)
    teams = {
        "player1": game_one_sides[RED],
        "player2": game_one_sides[BLUE],
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(uuid.uuid4()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "mode": {"key": int(mode), "name": mode.label},
        "player_labels": dict(player_labels),
        "teams": teams,
        "board": {"rows": ROWS, "cols": COLS},
        "state": engine.state,
        "match_score": engine.match_score,
        "match_winner": (
            engine.match_winner() if engine.state == STATE_MATCH_OVER else None
        ),
        "timers_seconds": timer.snapshot() if timer else {},
        "games": games,
    }
    return refresh_record_integrity(document)


def _atomic_write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def save_match_record(engine, mode, player_labels, timer=None, path=None):
    """Validate, encode and atomically save a match record."""
    document = build_match_record(engine, mode, player_labels, timer)
    validate_match_record(document, verify_hash=True)
    if path is None:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = RECORD_DIR / f"hex_match_{stamp}.json"
    else:
        path = Path(path)
    return _atomic_write_json(path, document)


def _record_teams(document):
    teams = document.get("teams")
    if teams is not None:
        _require(isinstance(teams, dict), "teams 必须是对象")
        player1 = teams.get("player1")
        player2 = teams.get("player2")
        _require(isinstance(player1, str) and player1.strip(), "teams.player1 非法")
        _require(isinstance(player2, str) and player2.strip(), "teams.player2 非法")
        _require(player1 != player2, "两支队伍名称不能相同")
        return player1, player2

    scores = document.get("match_score")
    if isinstance(scores, dict) and len(scores) == 2:
        player1, player2 = list(scores.keys())
        return player1, player2
    return None, None


def _expected_side(teams, game_no, color):
    if teams is None:
        return None
    player1, player2 = teams
    if game_no == 1:
        return player1 if color == RED else player2
    return player2 if color == RED else player1


def _validate_game(game, expected_game_no, teams):
    _require(isinstance(game, dict), f"第 {expected_game_no} 局必须是对象")
    _require(game.get("game") == expected_game_no,
             f"局号不连续: 期望 {expected_game_no}")
    moves = game.get("moves")
    _require(isinstance(moves, list), f"第 {expected_game_no} 局缺少 moves 列表")

    board = HexBoard()
    expected_color = RED
    winner_color = None
    for index, move in enumerate(moves, 1):
        _require(isinstance(move, dict), f"第 {expected_game_no} 局第 {index} 手必须是对象")
        _require(winner_color is None, f"第 {expected_game_no} 局终局后仍存在额外落子")
        _require(move.get("game") == expected_game_no,
                 f"第 {expected_game_no} 局第 {index} 手局号错误")
        _require(move.get("no") == index,
                 f"第 {expected_game_no} 局手数应连续，期望 {index}")
        color = move.get("color")
        _require(color in (RED, BLUE),
                 f"第 {expected_game_no} 局第 {index} 手颜色非法")
        _require(color == expected_color,
                 f"第 {expected_game_no} 局第 {index} 手未交替行棋")

        q, r = move.get("q"), move.get("r")
        _require(_is_int(q) and _is_int(r),
                 f"第 {expected_game_no} 局第 {index} 手坐标类型非法")
        _require(board.in_range(q, r),
                 f"第 {expected_game_no} 局第 {index} 手坐标越界")
        _require(move.get("cell") == cell_name(q, r),
                 f"第 {expected_game_no} 局第 {index} 手 cell 与 q,r 不一致")
        expected_side = _expected_side(teams, expected_game_no, color)
        if expected_side is not None:
            _require(move.get("side") == expected_side,
                     f"第 {expected_game_no} 局第 {index} 手队伍颜色不一致")
        _require(_is_number(move.get("ms", 0)) and move.get("ms", 0) >= 0,
                 f"第 {expected_game_no} 局第 {index} 手耗时非法")
        _require(isinstance(move.get("reason", ""), str),
                 f"第 {expected_game_no} 局第 {index} 手 reason 非法")

        try:
            board.place(q, r, color)
        except IllegalMoveError as exc:
            raise RecordValidationError(
                f"第 {expected_game_no} 局第 {index} 手非法: {exc}"
            ) from exc
        if board.has_won(color):
            winner_color = color
        expected_color = BLUE if expected_color == RED else RED

    result = game.get("result")
    if winner_color is not None:
        _require(result is not None,
                 f"第 {expected_game_no} 局已经产生胜者，但缺少 result")
    if result is not None:
        _require(isinstance(result, dict),
                 f"第 {expected_game_no} 局 result 必须是对象")
        _require(result.get("game") == expected_game_no,
                 f"第 {expected_game_no} 局 result 局号错误")
        result_color = result.get("winner_color")
        _require(result_color in (RED, BLUE),
                 f"第 {expected_game_no} 局 result 颜色非法")
        _require(result_color == winner_color,
                 f"第 {expected_game_no} 局声明胜者与实际连通结果不一致")
        expected_side = _expected_side(teams, expected_game_no, result_color)
        if expected_side is not None:
            _require(result.get("winner_side") == expected_side,
                     f"第 {expected_game_no} 局胜方队伍错误")
    return winner_color, result


def validate_match_record(document, verify_hash=True):
    """Validate schema, full move sequence, winner, score and integrity."""
    _require(isinstance(document, dict), "记录根节点必须是 JSON 对象")
    version = document.get("schema_version")
    _require(version in SUPPORTED_SCHEMA_VERSIONS,
             f"不支持的记录版本: {version!r}")

    if version >= 2:
        record_id = document.get("record_id")
        _require(isinstance(record_id, str) and record_id, "记录缺少 record_id")
        try:
            uuid.UUID(record_id)
        except (ValueError, AttributeError) as exc:
            raise RecordValidationError("record_id 不是合法 UUID") from exc
        _require(isinstance(document.get("record_sha256"), str),
                 "记录缺少 record_sha256")
        if verify_hash:
            actual = compute_record_sha256(document)
            _require(hmac.compare_digest(document["record_sha256"].lower(), actual),
                     "记录 SHA256 校验失败，文件可能被修改或损坏")

    saved_at = document.get("saved_at")
    _require(isinstance(saved_at, str), "记录缺少保存时间")
    try:
        datetime.fromisoformat(saved_at)
    except ValueError as exc:
        raise RecordValidationError(f"保存时间格式非法: {saved_at!r}") from exc

    board_info = document.get("board")
    _require(isinstance(board_info, dict), "记录缺少 board 信息")
    _require(board_info.get("rows") == ROWS and board_info.get("cols") == COLS,
             f"棋盘尺寸不匹配，应为 {ROWS}×{COLS}")

    state = document.get("state")
    _require(state in VALID_STATES, f"非法对局状态: {state!r}")
    mode = document.get("mode")
    _require(isinstance(mode, dict) and _is_int(mode.get("key")),
             "记录 mode 信息非法")
    _require(isinstance(mode.get("name"), str) and mode["name"], "记录模式名称非法")
    labels = document.get("player_labels")
    _require(isinstance(labels, dict), "记录缺少 player_labels")
    _require(all(isinstance(key, str) and isinstance(value, str)
                 for key, value in labels.items()), "player_labels 非法")

    teams = _record_teams(document)
    games = document.get("games")
    _require(isinstance(games, list) and 1 <= len(games) <= 2,
             "games 必须包含 1 或 2 局")

    validated = []
    for expected_game_no, game in enumerate(games, 1):
        validated.append(_validate_game(game, expected_game_no, teams))

    for game_index, (winner_color, result) in enumerate(validated, 1):
        is_last = game_index == len(validated)
        if state == STATE_PLAYING and is_last:
            _require(result is None,
                     "进行中存档的当前局不能已经带有 result")
        else:
            _require(result is not None,
                     f"第 {game_index} 局已经结束，但缺少 result")

    if state == STATE_MATCH_OVER:
        _require(len(games) == 2, "match_over 记录必须包含两局")
        _require(all(result is not None for _, result in validated),
                 "match_over 记录缺少局结果")

    score = document.get("match_score")
    _require(isinstance(score, dict), "记录缺少 match_score")
    _require(all(isinstance(key, str) and _is_int(value) and value >= 0
                 for key, value in score.items()), "match_score 数值非法")
    if teams is not None:
        _require(set(score) == set(teams), "match_score 队伍与 teams 不一致")
    else:
        _require(len(score) == 2, "match_score 必须包含两支队伍")

    expected_score = Counter(
        result["winner_side"] for _, result in validated if result is not None
    )
    for side in score:
        _require(score[side] == expected_score[side],
                 f"比分与对局结果不一致: {side}")
    _require(sum(score.values()) == sum(expected_score.values()),
             "总分与已完成局数不一致")

    if len(score) == 2:
        first, second = list(score.keys())
        if score[first] > score[second]:
            derived_winner = first
        elif score[second] > score[first]:
            derived_winner = second
        else:
            derived_winner = None
    else:
        derived_winner = None
    if state == STATE_MATCH_OVER:
        _require(document.get("match_winner") == derived_winner,
                 "match_winner 与比分不一致")
    else:
        _require(document.get("match_winner") is None,
                 "未进入 match_over 时 match_winner 必须为空")

    timers = document.get("timers_seconds", {})
    _require(isinstance(timers, dict), "timers_seconds 必须是对象")
    _require(all(_is_number(value) and value >= 0 for value in timers.values()),
             "timers_seconds 包含非法耗时")
    return True


def load_record(path):
    """Load a record and perform full structural and integrity validation."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordValidationError(
            f"棋局记录 JSON 不完整或格式错误: {exc.msg}，位置 {exc.pos}"
        ) from exc
    except OSError as exc:
        raise RecordValidationError(f"无法读取棋局记录: {exc}") from exc
    validate_match_record(document, verify_hash=True)
    return document


def latest_record(directory=RECORD_DIR):
    """Return the most recently modified JSON record path."""
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


class ReplaySession:
    """Create a read-only stepwise replay from a validated record."""

    def __init__(self, document, game_no=None):
        validate_match_record(document, verify_hash=True)
        self.document = document
        games = document["games"]
        if game_no is None:
            game_no = games[-1]["game"]
        game = next((item for item in games if item["game"] == game_no), None)
        if game is None:
            raise RecordValidationError(f"记录中不存在第 {game_no} 局")
        self.game_no = game_no
        self.moves = list(game["moves"])
        self.step = 0
        self.board = HexBoard()
        self.last_move = None
        self.goto(0)

    @property
    def record_status(self):
        return "进行中存档" if self.document["state"] == STATE_PLAYING else "正常结束"

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
            color = move["color"]
            coords = (move.get("q"), move.get("r"))
            if coords[0] is None or coords[1] is None:
                coords = parse_cell(move.get("cell"))
            if coords is None:
                raise RecordValidationError(
                    f"复盘记录包含非法坐标: {move.get('cell')!r}"
                )
            board.place(coords[0], coords[1], color)
        self.board = board
        self.step = step
        if step:
            move = self.moves[step - 1]
            self.last_move = (
                move["q"], move["r"], move["color"], move.get("reason", "")
            )
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
