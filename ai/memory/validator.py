# -*- coding: utf-8 -*-
"""Independent replay validation and quality promotion for memory data."""
from __future__ import annotations

from dataclasses import dataclass, field
import json

from hex_board import BLUE, EMPTY, RED, HexBoard


VALIDATOR_VERSION = "1.0.0"


@dataclass(frozen=True)
class ValidationIssue:
    game_no: int | None
    ply: int | None
    decision_id: str | None
    reason: str


@dataclass(frozen=True)
class ValidationReport:
    match_id: str
    valid: bool
    games_checked: int = 0
    decisions_checked: int = 0
    ai_decisions_eligible: int = 0
    rejected_decisions: int = 0
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def as_dict(self):
        return {
            "match_id": self.match_id,
            "valid": self.valid,
            "games_checked": self.games_checked,
            "decisions_checked": self.decisions_checked,
            "ai_decisions_eligible": self.ai_decisions_eligible,
            "rejected_decisions": self.rejected_decisions,
            "issues": [
                {
                    "game_no": issue.game_no,
                    "ply": issue.ply,
                    "decision_id": issue.decision_id,
                    "reason": issue.reason,
                }
                for issue in self.issues
            ],
        }


def _board_from_json(raw_value):
    try:
        document = json.loads(raw_value)
        grid = document["grid"]
        rows = int(document.get("rows", len(grid)))
        cols = int(document.get("cols", len(grid[0]) if grid else 0))
    except (TypeError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("board_json is not a valid board state") from exc
    if rows != 11 or cols != 11 or len(grid) != rows:
        raise ValueError("board_json has an invalid board size")
    if any(not isinstance(row, list) or len(row) != cols for row in grid):
        raise ValueError("board_json has invalid grid rows")
    if any(cell not in (EMPTY, RED, BLUE) for row in grid for cell in row):
        raise ValueError("board_json contains invalid cell values")
    board = HexBoard()
    board.grid = [list(row) for row in grid]
    board.history = []
    return board


def _same_grid(left, right):
    return left.grid == right.grid


def _issue(game_no, ply, decision_id, reason):
    return ValidationIssue(
        game_no=game_no,
        ply=ply,
        decision_id=decision_id,
        reason=reason,
    )


def _decision_updates(document, *, quality, training_eligible,
                      integrity, rule, result_status, reason):
    updates = []
    for row in document["decisions"]:
        eligible = int(
            quality == "validated"
            and row.get("actor_type") == "ai"
            and training_eligible
        )
        updates.append({
            "decision_id": row["decision_id"],
            "decision_quality": quality,
            "training_eligible": eligible,
            "integrity_status": integrity,
            "rule_status": rule,
            "result_status": result_status,
            "validation_reason": reason,
            "validator_version": VALIDATOR_VERSION,
        })
    return updates


def validate_match(store, match_id, *, promote=False):
    """Replay every game and optionally promote validated decisions."""
    document = store.load_match_for_validation(match_id)
    if document is None:
        raise ValueError(f"Unknown match_id: {match_id}")
    issues = []
    match = document["match"]
    games = document["games"]
    decisions_by_game = {1: [], 2: []}
    for row in document["decisions"]:
        decisions_by_game.setdefault(int(row["game_no"]), []).append(row)
    if match.get("status") != "completed":
        issues.append(_issue(None, None, None, "match is not completed"))
    if len(games) != 2 or tuple(int(game["game_no"]) for game in games) != (1, 2):
        issues.append(_issue(None, None, None, "match must contain exactly two games"))
    checked = 0
    eligible = 0
    for game in games:
        game_no = int(game["game_no"])
        winner = game.get("winner_color")
        if game.get("status") != "completed":
            issues.append(_issue(game_no, None, None, "game is not completed"))
        if winner not in (RED, BLUE):
            issues.append(_issue(game_no, None, None, "game winner is invalid"))
        board = HexBoard()
        expected_color = RED
        decisions = sorted(
            decisions_by_game.get(game_no, ()),
            key=lambda row: (int(row["ply"]), str(row["decision_id"])),
        )
        if not decisions:
            issues.append(_issue(game_no, None, None, "game has no decisions"))
            continue
        expected_ply = 1
        game_won_before_last = False
        for index, decision in enumerate(decisions):
            checked += 1
            ply = int(decision.get("position_ply") or decision.get("ply"))
            decision_id = decision.get("decision_id")
            if ply != expected_ply:
                issues.append(_issue(game_no, ply, decision_id, "non-contiguous ply"))
            if decision.get("actor_color") != expected_color:
                issues.append(_issue(game_no, ply, decision_id, "non-alternating color"))
            if decision.get("current_color") != expected_color:
                issues.append(_issue(game_no, ply, decision_id, "position color mismatch"))
            try:
                stored_board = _board_from_json(decision["board_json"])
            except ValueError as exc:
                issues.append(_issue(game_no, ply, decision_id, str(exc)))
                break
            if not _same_grid(board, stored_board):
                issues.append(_issue(game_no, ply, decision_id, "board snapshot mismatch"))
            q, r = decision.get("chosen_q"), decision.get("chosen_r")
            if not isinstance(q, int) or not isinstance(r, int):
                issues.append(_issue(game_no, ply, decision_id, "chosen move is missing"))
                break
            if not board.in_range(q, r) or not board.is_empty(q, r):
                issues.append(_issue(game_no, ply, decision_id, "chosen move is illegal"))
                break
            if board.has_won(decision.get("actor_color")):
                game_won_before_last = True
            try:
                board.place(q, r, decision.get("actor_color"))
            except Exception as exc:
                issues.append(_issue(game_no, ply, decision_id, str(exc)))
                break
            if board.has_won(decision.get("actor_color")) and index != len(decisions) - 1:
                game_won_before_last = True
            expected_result = (
                1.0 if decision.get("actor_color") == winner else 0.0
            )
            if decision.get("result") != expected_result:
                issues.append(_issue(game_no, ply, decision_id, "result mismatch"))
            expected_color = BLUE if expected_color == RED else RED
            expected_ply += 1
        if winner in (RED, BLUE) and not board.has_won(winner):
            issues.append(_issue(game_no, None, None, "winner never connected"))
        if game_won_before_last:
            issues.append(_issue(game_no, None, None, "moves exist after game over"))
    valid = not issues
    updates = _decision_updates(
        document,
        quality="validated" if valid else "rejected",
        training_eligible=valid,
        integrity="valid" if valid else "invalid",
        rule="valid" if valid else "invalid",
        result_status="known" if valid else "invalid",
        reason="independent_replay_validated" if valid else "independent_replay_failed",
    )
    if valid:
        eligible = sum(update["training_eligible"] for update in updates)
    if promote:
        store.batch_update_decision_validation(updates)
    return ValidationReport(
        match_id=match_id,
        valid=valid,
        games_checked=len(games),
        decisions_checked=checked,
        ai_decisions_eligible=eligible,
        rejected_decisions=0 if valid else len(updates),
        issues=tuple(issues),
    )


def validate_all(store, *, promote=False):
    reports = []
    for match in store.list_matches(limit=100000):
        reports.append(validate_match(store, match["match_id"], promote=promote))
    return tuple(reports)