# -*- coding: utf-8 -*-
"""正式海克斯棋 TXT 棋谱的坐标、解析、校验和 GB2312 读写。"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Optional

from hex_board import BLUE, COLS, RED, ROWS, HexBoard, IllegalMoveError
from hex_engine import STATE_GAME_OVER, HexEngine


GAME_CODE = "HEX"
RESULT_FIRST = "先手胜"
RESULT_SECOND = "后手胜"
VALID_RESULTS = (RESULT_FIRST, RESULT_SECOND)
VALID_MARKS = (-2, -1, 0, 1, 2)
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
NOTATION_DIR = WORKSPACE_ROOT / "artifacts" / "notation"

COORD_RE = re.compile(r"^([A-K]),([1-9]|1[01])$")
HEADER_RE = re.compile(
    r"^\[HEX\]\[([^\[\]]+)\]\[([^\[\]]+)\]"
    r"\[(先手胜|后手胜)\]\[([^\[\]]+)\]\[([^\[\]]+)\]$"
)
MOVE_RE = re.compile(
    r"^([RB])\(([A-K]),([1-9]|1[01])\)"
    r"(?:MARK\[(-2|-1|0|1|2)\])?$"
)


class HexNotationError(ValueError):
    """正式棋谱格式或内容不合法。"""


class NotationEncodingError(HexNotationError):
    """棋谱无法按 GB2312 编码或解码。"""


@dataclass(frozen=True)
class NotationMetadata:
    red_team: str
    blue_team: str
    result: str
    time_location: str
    event_name: str


@dataclass(frozen=True)
class NotationMove:
    color: int
    q: int
    r: int
    mark: Optional[int] = None


@dataclass(frozen=True)
class HexNotation:
    metadata: NotationMetadata
    moves: tuple[NotationMove, ...]


def _clean_metadata_value(name, value):
    text = str(value).strip()
    if not text:
        raise HexNotationError(f"{name}不能为空")
    if any(char in text for char in "[];{}\r\n"):
        raise HexNotationError(f"{name}包含非法定界字符: {text!r}")
    return text


def _validated_metadata(metadata):
    red_team = _clean_metadata_value("先手队伍", metadata.red_team)
    blue_team = _clean_metadata_value("后手队伍", metadata.blue_team)
    if red_team == blue_team:
        raise HexNotationError("红蓝双方队伍名称不能相同")
    if metadata.result not in VALID_RESULTS:
        raise HexNotationError(f"非法比赛结果: {metadata.result!r}")
    time_location = _clean_metadata_value("比赛时间地点", metadata.time_location)
    event_name = _clean_metadata_value("赛事名称", metadata.event_name)
    return NotationMetadata(
        red_team=red_team,
        blue_team=blue_team,
        result=metadata.result,
        time_location=time_location,
        event_name=event_name,
    )


def format_notation_coord(q, r):
    """Convert internal (q, r) to formal notation such as E,7.

    The official coordinate origin is the bottom-left corner, so internal
    top-to-bottom row r=0..10 maps to notation rows 11..1.
    """
    if (not isinstance(q, int) or isinstance(q, bool)
            or not isinstance(r, int) or isinstance(r, bool)):
        raise HexNotationError(f"坐标类型非法: ({q!r}, {r!r})")
    if not (0 <= q < COLS and 0 <= r < ROWS):
        raise HexNotationError(f"坐标越界: ({q!r}, {r!r})")
    return f"{chr(65 + q)},{ROWS - r}"


def parse_notation_coord(value):
    """Parse formal coordinate E,7 into internal (q, r)."""
    if not isinstance(value, str):
        raise HexNotationError(f"坐标格式非法: {value!r}")
    match = COORD_RE.fullmatch(value.strip())
    if not match:
        raise HexNotationError(f"坐标格式非法: {value!r}")
    q = ord(match.group(1)) - 65
    notation_row = int(match.group(2))
    return q, ROWS - notation_row


def _format_move(move):
    if move.color not in (RED, BLUE):
        raise HexNotationError(f"非法棋子颜色: {move.color!r}")
    color = "R" if move.color == RED else "B"
    text = f"{color}({format_notation_coord(move.q, move.r)})"
    if move.mark is not None:
        if move.mark not in VALID_MARKS:
            raise HexNotationError(f"非法 MARK 值: {move.mark!r}")
        text += f"MARK[{move.mark}]"
    return text


def _validate_move_sequence(moves, verify_result=False, expected_result=None):
    if not moves:
        raise HexNotationError("棋谱至少需要一步落子")
    board = HexBoard()
    expected_color = RED
    winner = None
    for index, move in enumerate(moves, 1):
        if winner is not None:
            raise HexNotationError(f"第 {index} 手位于终局之后")
        if move.color != expected_color:
            expected = "R" if expected_color == RED else "B"
            actual = "R" if move.color == RED else "B"
            raise HexNotationError(
                f"第 {index} 手颜色错误: 期望 {expected}，实际 {actual}"
            )
        if move.mark is not None and move.mark not in VALID_MARKS:
            raise HexNotationError(f"第 {index} 手 MARK 非法: {move.mark!r}")
        try:
            board.place(move.q, move.r, move.color)
        except IllegalMoveError as exc:
            raise HexNotationError(f"第 {index} 手非法: {exc}") from exc
        if board.has_won(move.color):
            winner = move.color
        expected_color = BLUE if expected_color == RED else RED

    if verify_result:
        if expected_result not in VALID_RESULTS:
            raise HexNotationError(f"非法比赛结果: {expected_result!r}")
        expected_winner = RED if expected_result == RESULT_FIRST else BLUE
        if winner != expected_winner:
            expected = "先手/红方" if expected_winner == RED else "后手/蓝方"
            raise HexNotationError(f"棋谱结果与实际连通结果不一致，应为{expected}获胜")
    return board


def notation_to_text(notation, verify_result=True):
    """Serialize a HexNotation into the formal bracketed TXT format."""
    metadata = _validated_metadata(notation.metadata)
    moves = tuple(notation.moves)
    _validate_move_sequence(
        moves,
        verify_result=verify_result,
        expected_result=metadata.result,
    )
    header = (
        f"[{GAME_CODE}][{metadata.red_team}][{metadata.blue_team}]"
        f"[{metadata.result}][{metadata.time_location}][{metadata.event_name}]"
    )
    body = ";".join([header] + [_format_move(move) for move in moves])
    return "{" + body + "}"


def parse_notation(text, verify_result=False):
    """Parse formal notation text and optionally verify the declared winner."""
    if not isinstance(text, str):
        raise HexNotationError("棋谱内容必须是字符串")
    content = text.strip()
    if not content.startswith("{") or not content.endswith("}"):
        raise HexNotationError("棋谱必须由一对英文花括号 {} 包围")
    inner = content[1:-1]
    parts = inner.split(";")
    if len(parts) < 2:
        raise HexNotationError("棋谱缺少参赛信息或落子序列")

    header_match = HEADER_RE.fullmatch(parts[0])
    if not header_match:
        raise HexNotationError("棋谱头部格式非法")
    metadata = _validated_metadata(NotationMetadata(
        red_team=header_match.group(1),
        blue_team=header_match.group(2),
        result=header_match.group(3),
        time_location=header_match.group(4),
        event_name=header_match.group(5),
    ))

    moves = []
    for index, part in enumerate(parts[1:], 1):
        match = MOVE_RE.fullmatch(part)
        if not match:
            raise HexNotationError(f"第 {index} 手格式非法: {part!r}")
        q, r = parse_notation_coord(f"{match.group(2)},{match.group(3)}")
        color = RED if match.group(1) == "R" else BLUE
        mark = int(match.group(4)) if match.group(4) is not None else None
        moves.append(NotationMove(color=color, q=q, r=r, mark=mark))

    _validate_move_sequence(
        tuple(moves),
        verify_result=verify_result,
        expected_result=metadata.result,
    )
    return HexNotation(metadata=metadata, moves=tuple(moves))


def _encode_gb2312(text):
    try:
        return text.encode("gb2312", errors="strict")
    except UnicodeEncodeError as exc:
        bad = text[exc.start:exc.end]
        raise NotationEncodingError(
            f"棋谱包含无法用 GB2312 编码的字符: {bad!r}"
        ) from exc


def _decode_gb2312(data):
    try:
        return data.decode("gb2312", errors="strict")
    except UnicodeDecodeError as exc:
        raise NotationEncodingError(
            f"棋谱不是有效的 GB2312 文本，字节位置 {exc.start}"
        ) from exc


def save_notation_text(text, path):
    """Encode and atomically save formal notation as GB2312."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _encode_gb2312(text)
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


def load_notation_text(path):
    return _decode_gb2312(Path(path).read_bytes())


def load_notation(path, verify_result=False):
    return parse_notation(load_notation_text(path), verify_result=verify_result)


def sanitize_filename_component(value):
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value).strip())
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "未命名"


def default_notation_filename(metadata):
    metadata = _validated_metadata(metadata)
    return (
        f"HEX-{sanitize_filename_component(metadata.red_team)} vs "
        f"{sanitize_filename_component(metadata.blue_team)}-"
        f"{metadata.result}-{sanitize_filename_component(metadata.time_location)}-"
        f"{sanitize_filename_component(metadata.event_name)}.txt"
    )


def _marks_for_moves(moves, marks):
    result = []
    for move in moves:
        mark = move.get("mark")
        if marks is not None:
            mark = marks.get(move["no"], mark)
        if mark is not None:
            mark = int(mark)
            if mark not in VALID_MARKS:
                raise HexNotationError(f"第 {move['no']} 手 MARK 非法: {mark!r}")
        result.append(mark)
    return result


def notation_from_engine(engine, time_location, event_name, game_no=None,
                         red_team=None, blue_team=None, marks=None):
    """Build formal notation for one completed game in HexEngine."""
    target_game = engine.game_no if game_no is None else int(game_no)
    sides = engine.sides_for_game(target_game)
    result_record = next(
        (item for item in engine.results if item["game"] == target_game),
        None,
    )
    if result_record is None:
        raise HexNotationError(f"第 {target_game} 局尚未结束，无法导出正式棋谱")

    moves = engine.game_moves(target_game)
    if not moves:
        raise HexNotationError(f"第 {target_game} 局没有落子记录")
    move_marks = _marks_for_moves(moves, marks)
    notation_moves = tuple(
        NotationMove(
            color=move["color"],
            q=move["q"],
            r=move["r"],
            mark=mark,
        )
        for move, mark in zip(moves, move_marks)
    )
    winner_color = result_record["winner_color"]
    result = RESULT_FIRST if winner_color == RED else RESULT_SECOND
    metadata = _validated_metadata(NotationMetadata(
        red_team=red_team or sides[RED],
        blue_team=blue_team or sides[BLUE],
        result=result,
        time_location=time_location,
        event_name=event_name,
    ))
    notation = HexNotation(metadata=metadata, moves=notation_moves)
    _validate_move_sequence(
        notation.moves, verify_result=True, expected_result=metadata.result
    )
    return notation


def export_game_txt(engine, time_location, event_name, game_no=None,
                    output_dir=None, path=None, red_team=None, blue_team=None,
                    marks=None):
    """Export one completed HexEngine game to a formal GB2312 TXT file."""
    notation = notation_from_engine(
        engine,
        time_location=time_location,
        event_name=event_name,
        game_no=game_no,
        red_team=red_team,
        blue_team=blue_team,
        marks=marks,
    )
    if path is None:
        output_dir = Path(output_dir or NOTATION_DIR)
        path = output_dir / default_notation_filename(notation.metadata)
    text = notation_to_text(notation)
    return save_notation_text(text, path)


def export_match_txt(engine, time_location, event_name, output_dir=None,
                     marks_by_game=None):
    """Export every completed game as a separate formal TXT file."""
    paths = []
    for result in engine.results:
        game_no = result["game"]
        marks = None
        if marks_by_game:
            marks = marks_by_game.get(game_no)
        paths.append(export_game_txt(
            engine,
            time_location=time_location,
            event_name=event_name,
            game_no=game_no,
            output_dir=output_dir,
            marks=marks,
        ))
    if not paths:
        raise HexNotationError("没有可导出的已完成对局")
    return paths


def notation_to_engine(notation, verify_result=True):
    """Rebuild a HexEngine game from a parsed notation document."""
    _validate_move_sequence(
        notation.moves,
        verify_result=verify_result,
        expected_result=notation.metadata.result,
    )
    engine = HexEngine(notation.metadata.red_team, notation.metadata.blue_team)
    for move in notation.moves:
        reason = None if move.mark is None else f"MARK[{move.mark}]"
        engine.play_move(move.q, move.r, reason=reason)
    if verify_result and engine.state != STATE_GAME_OVER:
        raise HexNotationError("棋谱完成后引擎未进入 game_over 状态")
    return engine


def notation_to_match_record(notation, verify_result=True):
    """Convert notation into the existing JSON record schema for replay."""
    engine = notation_to_engine(notation, verify_result=verify_result)
    from game_mode import GameMode
    from game_records import build_match_record

    labels = {
        notation.metadata.red_team: notation.metadata.red_team,
        notation.metadata.blue_team: notation.metadata.blue_team,
    }
    document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, labels)
    document["source_format"] = "HEX-TXT"
    document["notation"] = {
        "time_location": notation.metadata.time_location,
        "event_name": notation.metadata.event_name,
        "result": notation.metadata.result,
    }
    return document
