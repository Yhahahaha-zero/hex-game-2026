# -*- coding: utf-8 -*-
"""正式海克斯棋 TXT 棋谱格式测试。"""
import tempfile
import unittest
from pathlib import Path

from game_records import ReplaySession
from hex_board import BLUE, RED
from hex_engine import STATE_GAME_OVER, HexEngine
from hex_notation import (HexNotationError, NotationEncodingError,
                          NotationMetadata, NotationMove,
                          default_notation_filename, export_game_txt,
                          format_notation_coord, load_notation,
                          load_notation_text, notation_from_engine,
                          notation_to_engine, notation_to_match_record,
                          notation_to_text, parse_notation,
                          parse_notation_coord, sanitize_filename_component,
                          save_notation_text)


OFFICIAL_SAMPLE = (
    "{[HEX][先手参赛队R][后手参赛队B][先手胜]"
    "[2017.07.29 14:00 重庆][2017 CCGC];"
    "R(E,7)MARK[1];B(E,6);R(F,7);B(G,7);"
    "R(D,6);B(F,6);R(C,6);B(G,6)}"
)


class HexNotationTest(unittest.TestCase):
    @staticmethod
    def _completed_engine():
        engine = HexEngine("红队", "蓝队")
        for row in range(11):
            engine.play_move(0, row)
            if engine.state == STATE_GAME_OVER:
                break
            engine.play_move(10, row)
        return engine

    def test_sides_for_game_handles_color_swap(self):
        engine = HexEngine("红队", "蓝队")
        self.assertEqual(engine.sides_for_game(1), {RED: "红队", BLUE: "蓝队"})
        engine.force_win(RED, "test")
        engine.next_game()
        self.assertEqual(engine.sides_for_game(2), {RED: "蓝队", BLUE: "红队"})

    def test_coordinate_round_trip_uses_bottom_left_origin(self):
        self.assertEqual(format_notation_coord(0, 10), "A,1")
        self.assertEqual(format_notation_coord(10, 0), "K,11")
        self.assertEqual(format_notation_coord(4, 4), "E,7")
        for q in range(11):
            for r in range(11):
                self.assertEqual(parse_notation_coord(format_notation_coord(q, r)),
                                 (q, r))

    def test_official_sample_parses_and_round_trips_syntax(self):
        notation = parse_notation(OFFICIAL_SAMPLE, verify_result=False)
        self.assertEqual(notation.metadata.red_team, "先手参赛队R")
        self.assertEqual(notation.metadata.blue_team, "后手参赛队B")
        self.assertEqual(notation.metadata.result, "先手胜")
        self.assertEqual(len(notation.moves), 8)
        self.assertEqual(notation.moves[0].mark, 1)
        self.assertEqual(notation_to_text(notation, verify_result=False),
                         OFFICIAL_SAMPLE)
        with self.assertRaises(HexNotationError):
            parse_notation(OFFICIAL_SAMPLE, verify_result=True)

    def test_complete_game_txt_engine_round_trip(self):
        engine = self._completed_engine()
        notation = notation_from_engine(
            engine,
            time_location="2026.10.25 14:00 成都",
            event_name="课程设计赛",
            game_no=1,
            marks={1: 1},
        )
        text = notation_to_text(notation)
        self.assertTrue(text.startswith(
            "{[HEX][红队][蓝队][先手胜][2026.10.25 14:00 成都][课程设计赛];"
        ))
        self.assertIn("R(A,11)MARK[1]", text)

        parsed = parse_notation(text, verify_result=True)
        rebuilt = notation_to_engine(parsed, verify_result=True)
        self.assertEqual(rebuilt.state, STATE_GAME_OVER)
        self.assertEqual(rebuilt.winner_color, RED)
        self.assertEqual(rebuilt.red_side, "红队")
        self.assertEqual(rebuilt.blue_side, "蓝队")

        original_moves = [(m["color"], m["q"], m["r"]) for m in engine.game_moves(1)]
        rebuilt_moves = [(m["color"], m["q"], m["r"]) for m in rebuilt.game_moves(1)]
        self.assertEqual(rebuilt_moves, original_moves)
        self.assertEqual(parsed.moves[0].mark, 1)
        self.assertEqual(rebuilt.game_moves(1)[0]["reason"], "MARK[1]")

    def test_notation_converts_to_replay_record(self):
        engine = self._completed_engine()
        notation = notation_from_engine(
            engine,
            time_location="2026.10.25 14:00 成都",
            event_name="课程设计赛",
        )
        document = notation_to_match_record(notation)
        self.assertEqual(document["source_format"], "HEX-TXT")
        self.assertEqual(document["notation"]["event_name"], "课程设计赛")
        replay = ReplaySession(document, game_no=1)
        self.assertEqual(replay.total_moves, len(notation.moves))
        replay.goto(replay.total_moves)
        self.assertTrue(replay.board.has_won(RED))

    def test_save_and_load_gb2312_file(self):
        engine = self._completed_engine()
        notation = notation_from_engine(
            engine,
            time_location="2026.10.25 14:00 成都",
            event_name="课程设计赛",
        )
        text = notation_to_text(notation)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / default_notation_filename(notation.metadata)
            saved = export_game_txt(
                engine,
                time_location="2026.10.25 14:00 成都",
                event_name="课程设计赛",
                path=path,
                marks={1: 1},
            )
            self.assertEqual(saved, path)
            raw = path.read_bytes()
            self.assertEqual(raw.decode("gb2312"), notation_to_text(
                notation_from_engine(
                    engine,
                    time_location="2026.10.25 14:00 成都",
                    event_name="课程设计赛",
                    marks={1: 1},
                )
            ))
            loaded = load_notation(path, verify_result=True)
            self.assertEqual(loaded.metadata.event_name, "课程设计赛")
            self.assertFalse(str(saved).startswith("\ufeff"))

    def test_filename_sanitization(self):
        self.assertEqual(sanitize_filename_component('A:B/C*D?'), "A_B_C_D_")
        metadata = NotationMetadata(
            red_team="红队",
            blue_team="蓝队",
            result="先手胜",
            time_location="2026.10.25 14:00 成都",
            event_name="课程设计赛",
        )
        filename = default_notation_filename(metadata)
        self.assertEqual(
            filename,
            "HEX-红队 vs 蓝队-先手胜-2026.10.25 14_00 成都-课程设计赛.txt",
        )

    def test_invalid_notation_cases(self):
        cases = (
            "R(E,7)",
            "{[HEX][红队][蓝队][和棋][时间][赛事];R(E,7)}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];R(L,7)}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];R(E,0)}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];R(E,7)MARK[3]}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];B(E,7)}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];R(E,7);R(F,7)}",
            "{[HEX][红队][蓝队][先手胜][时间][赛事];R(E,7);B(E,7)}",
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(HexNotationError):
                    parse_notation(text)

    def test_gb2312_encoding_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(NotationEncodingError):
                save_notation_text("{[HEX][红队😀][蓝队][先手胜][时间][赛事]}",
                                   Path(temp_dir) / "invalid.txt")
            bad = Path(temp_dir) / "bad.txt"
            bad.write_bytes(b"\xff\xfe")
            with self.assertRaises(NotationEncodingError):
                load_notation_text(bad)

    def test_inside_to_outside_coordinates(self):
        notation = parse_notation(OFFICIAL_SAMPLE, verify_result=False)
        self.assertEqual((notation.moves[0].q, notation.moves[0].r), (4, 4))
        self.assertEqual(notation.moves[1].color, BLUE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
