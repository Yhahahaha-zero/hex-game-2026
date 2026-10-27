# -*- coding: utf-8 -*-
"""Tests for the read-only board observation contract."""
import json
import unittest

from ai.board_observation import (
    build_board_observation,
    build_engine_observation,
    build_position_id,
    observation_json,
)
from hex_board import BLUE, RED, HexBoard
from hex_engine import HexEngine


class BoardObservationTest(unittest.TestCase):
    def test_initial_observation_contract(self):
        engine = HexEngine("红队", "蓝队")
        observation = build_engine_observation(engine)

        self.assertEqual(observation["schema_version"], 1)
        self.assertTrue(observation["position_id"].startswith("hex1-"))
        self.assertEqual(observation["game_no"], 1)
        self.assertEqual(observation["move_count"], 0)
        self.assertEqual(observation["next_move_no"], 1)
        self.assertEqual(observation["state"], "playing")
        self.assertEqual(observation["analysis_color"], "RED")
        self.assertEqual(observation["current_color"], "RED")
        self.assertEqual(observation["current_player"], "红队")
        self.assertEqual(observation["sides"], {"RED": "红队", "BLUE": "蓝队"})
        self.assertEqual(len(observation["board_rows"]), 11)
        self.assertTrue(all(len(row) == 11 for row in observation["board_rows"]))
        self.assertEqual(observation["stones"], {"RED": [], "BLUE": []})
        self.assertEqual(observation["legal_move_count"], 121)
        self.assertEqual(len(observation["candidates"]), 8)
        self.assertIsNone(observation["last_move"])
        json.loads(observation_json(observation))

    def test_position_id_is_stable_and_changes_with_move_and_undo(self):
        engine = HexEngine("红队", "蓝队")
        first = build_engine_observation(engine)
        second = build_engine_observation(engine)
        self.assertEqual(first["position_id"], second["position_id"])

        engine.play_move(5, 5, reason="中心开局")
        moved = build_engine_observation(engine)
        self.assertNotEqual(first["position_id"], moved["position_id"])

        engine.undo_move(1)
        restored = build_engine_observation(engine)
        self.assertEqual(first["position_id"], restored["position_id"])

    def test_position_id_changes_with_game_state(self):
        board = HexBoard()
        playing = build_position_id(
            board, RED, game_no=1, state="playing", current_color=RED
        )
        finished = build_position_id(
            board, RED, game_no=1, state="game_over", current_color=None
        )
        self.assertNotEqual(playing, finished)

    def test_stones_history_and_last_move(self):
        engine = HexEngine("红队", "蓝队")
        engine.play_move(5, 5, reason="中心开局")
        engine.play_move(4, 5, reason="防守")
        observation = build_engine_observation(engine)

        self.assertEqual(observation["move_count"], 2)
        self.assertEqual(observation["next_move_no"], 3)
        self.assertEqual(observation["stones"]["RED"], [[5, 5]])
        self.assertEqual(observation["stones"]["BLUE"], [[4, 5]])
        self.assertEqual(observation["last_move"]["move"], [4, 5])
        self.assertEqual(observation["last_move"]["cell"], "6E")
        self.assertEqual(observation["last_move"]["reason"], "防守")
        self.assertEqual(len(observation["recent_history"]), 2)
        self.assertEqual(observation["recent_history"][0]["cell"], "6F")

    def test_history_and_candidate_limits(self):
        engine = HexEngine("红队", "蓝队")
        engine.play_move(5, 5)
        engine.play_move(4, 5)
        engine.play_move(5, 4)
        observation = build_engine_observation(
            engine, history_limit=2, candidate_limit=3
        )

        self.assertEqual(len(observation["recent_history"]), 2)
        self.assertEqual(observation["recent_history"][-1]["move"], [5, 4])
        self.assertEqual(len(observation["candidates"]), 3)
        self.assertEqual([row["rank"] for row in observation["candidates"]], [1, 2, 3])
        for row in observation["candidates"]:
            q, r = row["move"]
            self.assertTrue(engine.board.is_empty(q, r))
            self.assertEqual(row["source"], "distance_heuristic")

    def test_winning_move_and_opponent_threat_are_exposed(self):
        winning_board = HexBoard()
        for r in range(10):
            winning_board.place(5, r, RED)
        winning = build_board_observation(
            winning_board, RED, candidate_limit=4
        )
        self.assertIn(
            {"move": [5, 10], "cell": "11F"},
            winning["tactics"]["winning_moves"],
        )

        threat_board = HexBoard()
        for q in range(10):
            threat_board.place(q, 5, BLUE)
        threatened = build_board_observation(
            threat_board, RED, candidate_limit=4
        )
        self.assertIn(
            {"move": [10, 5], "cell": "6K"},
            threatened["tactics"]["opponent_winning_moves"],
        )

    def test_observation_is_read_only(self):
        engine = HexEngine("红队", "蓝队")
        engine.play_move(5, 5)
        engine.play_move(4, 5)
        grid_before = [row[:] for row in engine.board.grid]
        history_before = list(engine.board.history)
        engine_history_before = engine.move_history

        build_engine_observation(engine)

        self.assertEqual(engine.board.grid, grid_before)
        self.assertEqual(engine.board.history, history_before)
        self.assertEqual(engine.move_history, engine_history_before)

    def test_terminal_observation_uses_winner_perspective(self):
        engine = HexEngine("红队", "蓝队")
        engine.force_win(BLUE, "test")
        observation = build_engine_observation(engine)

        self.assertEqual(observation["state"], "game_over")
        self.assertTrue(observation["is_terminal"])
        self.assertEqual(observation["candidates"], [])
        self.assertEqual(observation["analysis_color"], "BLUE")
        self.assertIsNone(observation["current_color"])
        self.assertEqual(observation["winner_color"], "BLUE")
        self.assertEqual(observation["winner_side"], "蓝队")

    def test_second_game_swaps_side_labels(self):
        engine = HexEngine("红队", "蓝队")
        engine.force_win(RED, "test")
        engine.next_game()
        observation = build_engine_observation(engine)

        self.assertEqual(observation["game_no"], 2)
        self.assertEqual(observation["current_color"], "RED")
        self.assertEqual(observation["sides"]["RED"], "蓝队")
        self.assertEqual(observation["sides"]["BLUE"], "红队")

    def test_unreachable_cost_is_serialized_as_null(self):
        board = HexBoard()
        for q in range(11):
            board.place(q, 5, BLUE)
        observation = build_board_observation(
            board, RED, candidate_limit=2, history_limit=0
        )

        self.assertIsNone(observation["tactics"]["connection_cost"]["RED"])
        self.assertEqual(observation["recent_history"], [])
        parsed = json.loads(observation_json(observation))
        self.assertIsNone(parsed["tactics"]["connection_cost"]["RED"])

    def test_invalid_arguments_are_rejected(self):
        board = HexBoard()
        cases = (
            {"game_no": 0},
            {"history_limit": -1},
            {"history_limit": 51},
            {"candidate_limit": -1},
            {"candidate_limit": 21},
            {"state": ""},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_board_observation(board, RED, **kwargs)

        with self.assertRaises(ValueError):
            build_board_observation(board, 99)
        with self.assertRaises(ValueError):
            build_board_observation(board, RED, candidates=[(True, 0)])

        board.place(5, 5, BLUE)
        with self.assertRaises(ValueError):
            build_board_observation(board, RED, candidates=[(5, 5)])


if __name__ == "__main__":
    unittest.main()