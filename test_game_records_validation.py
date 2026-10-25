# -*- coding: utf-8 -*-
"""JSON 棋局记录深度校验和原子保存测试。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from game_mode import GameMode
from game_records import (RecordValidationError, ReplaySession,
                          compute_record_sha256, load_record,
                          refresh_record_integrity, save_match_record,
                          validate_match_record)
from hex_board import BLUE, RED
from hex_engine import STATE_GAME_OVER, HexEngine


LABELS = {"红队": "红队", "蓝队": "蓝队"}


def completed_engine(second_game=False):
    engine = HexEngine("红队", "蓝队")
    for row in range(11):
        engine.play_move(0, row)
        if engine.state == STATE_GAME_OVER:
            break
        engine.play_move(10, row)
    if second_game:
        engine.next_game()
        for row in range(11):
            engine.play_move(0, row)
            if engine.state == STATE_GAME_OVER:
                break
            engine.play_move(10, row)
        engine.next_game()
    return engine


class RecordValidationTest(unittest.TestCase):
    def test_valid_record_contains_uuid_and_sha256(self):
        engine = completed_engine()
        from game_records import build_match_record
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        self.assertTrue(validate_match_record(document))
        self.assertTrue(document["record_id"])
        self.assertEqual(document["record_sha256"], compute_record_sha256(document))
        other = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        self.assertNotEqual(other["record_id"], document["record_id"])

    def test_atomic_save_and_load(self):
        engine = completed_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.json"
            save_match_record(
                engine,
                GameMode.HUMAN_VS_HUMAN,
                LABELS,
                path=path,
            )
            loaded = load_record(path)
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_tampering_is_detected_by_hash(self):
        from game_records import build_match_record
        engine = completed_engine()
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        document["games"][0]["moves"][0]["reason"] = "tampered"
        with self.assertRaisesRegex(RecordValidationError, "SHA256"):
            validate_match_record(document)

    def test_duplicate_and_non_alternating_moves(self):
        from game_records import build_match_record
        engine = completed_engine()
        base = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)

        duplicate = deepcopy(base)
        first = duplicate["games"][0]["moves"][0]
        second = duplicate["games"][0]["moves"][1]
        second["q"], second["r"], second["cell"] = first["q"], first["r"], first["cell"]
        refresh_record_integrity(duplicate)
        with self.assertRaisesRegex(RecordValidationError, "非法"):
            validate_match_record(duplicate)

        alternating = deepcopy(base)
        alternating["games"][0]["moves"][1]["color"] = RED
        refresh_record_integrity(alternating)
        with self.assertRaisesRegex(RecordValidationError, "交替"):
            validate_match_record(alternating)

    def test_extra_move_after_win_is_rejected(self):
        from game_records import build_match_record
        engine = completed_engine()
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        moves = document["games"][0]["moves"]
        last = moves[-1]
        moves.append({
            "game": 1,
            "no": last["no"] + 1,
            "color": BLUE,
            "side": "蓝队",
            "cell": "B6",
            "q": 1,
            "r": 5,
            "ms": 0,
            "reason": "after win",
        })
        refresh_record_integrity(document)
        with self.assertRaisesRegex(RecordValidationError, "终局后"):
            validate_match_record(document)

    def test_winner_score_and_state_consistency(self):
        from game_records import build_match_record
        engine = completed_engine()
        base = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)

        wrong_winner = deepcopy(base)
        wrong_winner["games"][0]["result"]["winner_color"] = BLUE
        wrong_winner["games"][0]["result"]["winner_side"] = "蓝队"
        refresh_record_integrity(wrong_winner)
        with self.assertRaisesRegex(RecordValidationError, "实际连通结果"):
            validate_match_record(wrong_winner)

        wrong_score = deepcopy(base)
        wrong_score["match_score"]["红队"] = 0
        refresh_record_integrity(wrong_score)
        with self.assertRaisesRegex(RecordValidationError, "比分"):
            validate_match_record(wrong_score)

        wrong_state = deepcopy(base)
        wrong_state["state"] = "match_over"
        refresh_record_integrity(wrong_state)
        with self.assertRaisesRegex(RecordValidationError, "两局"):
            validate_match_record(wrong_state)

    def test_board_dimensions_are_validated(self):
        from game_records import build_match_record
        engine = completed_engine()
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        document["board"]["rows"] = 10
        refresh_record_integrity(document)
        with self.assertRaisesRegex(RecordValidationError, "棋盘尺寸"):
            validate_match_record(document)

    def test_partial_and_finished_replay_status(self):
        from game_records import build_match_record
        partial = HexEngine("红队", "蓝队")
        partial.play_move(5, 5)
        partial_document = build_match_record(
            partial, GameMode.HUMAN_VS_HUMAN, LABELS
        )
        self.assertEqual(ReplaySession(partial_document).record_status, "进行中存档")

        finished = build_match_record(
            completed_engine(), GameMode.HUMAN_VS_HUMAN, LABELS
        )
        self.assertEqual(ReplaySession(finished).record_status, "正常结束")

    def test_second_game_match_over_record(self):
        from game_records import build_match_record
        engine = completed_engine(second_game=True)
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        self.assertEqual(document["state"], "match_over")
        self.assertEqual(len(document["games"]), 2)
        self.assertTrue(validate_match_record(document))
        self.assertEqual(ReplaySession(document, game_no=2).record_status, "正常结束")

    def test_legacy_v1_record_is_validated_without_hash(self):
        from game_records import build_match_record
        engine = completed_engine()
        document = build_match_record(engine, GameMode.HUMAN_VS_HUMAN, LABELS)
        document["schema_version"] = 1
        document.pop("record_id", None)
        document.pop("record_sha256", None)
        document.pop("teams", None)
        self.assertTrue(validate_match_record(document))

    def test_truncated_json_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text('{"schema_version": 2, "games": [', encoding="utf-8")
            with self.assertRaisesRegex(RecordValidationError, "JSON"):
                load_record(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
