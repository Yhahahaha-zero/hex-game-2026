# -*- coding: utf-8 -*-
"""Persistent SQLite storage for the Hex experience-memory system.

This module is deliberately independent from the UI, HexEngine and AI classes.
Callers create typed records and explicitly send them to ExperienceStore. P0-B
does not install any automatic game-flow collector.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
import tempfile

from hex_board import BLUE, RED

from ai.memory.schema import (
    ACTOR_TYPES,
    CAPTURE_MODES,
    COLLECTOR_VERSION,
    DECISION_QUALITIES,
    DEFAULT_BUSY_TIMEOUT_MS,
    FEATURE_VERSION,
    GAME_STATUSES,
    INDEX_DDL,
    INTEGRITY_STATUSES,
    MATCH_STATUSES,
    MEMORY_INDEXES,
    MEMORY_SCHEMA_VERSION,
    MEMORY_TABLES,
    MIGRATION_DDL,
    RESULT_STATUSES,
    RETRIEVAL_MODES,
    RETRIEVAL_VERSION,
    RULES_VERSION,
    RULE_STATUSES,
    SOURCE_TYPES,
    TABLE_COLUMNS,
    TABLE_DDL,
    memory_db_path,
)


_POSITION_ID_PATTERN = re.compile(r"^pos1-[0-9a-f]{32}$")
_CANONICAL_ID_PATTERN = re.compile(r"^can1-[0-9a-f]{32}$")
_DECISION_ID_PATTERN = re.compile(r"^dec1-[0-9a-f]{32}$")


class MemoryStoreError(Exception):
    """Base error for database-backed experience storage."""


class MemoryValidationError(MemoryStoreError):
    """Raised when a record does not satisfy the storage contract."""


class MemoryConflictError(MemoryStoreError):
    """Raised when a stable identifier is reused with different content."""


class MemoryClosedError(MemoryStoreError):
    """Raised when an operation requires an open store."""


class MemoryDatabaseError(MemoryStoreError):
    """Raised for recoverable or non-structural SQLite failures."""


class MemorySchemaError(MemoryDatabaseError):
    """Raised when the database schema version or layout is unsupported."""


class MemoryIntegrityError(MemoryDatabaseError):
    """Raised when SQLite reports malformed or inconsistent data."""


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    mode_name: str
    red_team: str
    blue_team: str
    source_type: str
    started_at: str
    mode_key: int | None = None
    schema_version: int = MEMORY_SCHEMA_VERSION
    rules_version: int = RULES_VERSION
    ended_at: str | None = None
    status: str = "active"
    record_id: str | None = None
    capture_mode: str = "shadow"
    collector_version: str = COLLECTOR_VERSION


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    match_id: str
    game_no: int
    red_actor_type: str
    blue_actor_type: str
    started_at: str
    red_actor_id: str | None = None
    blue_actor_id: str | None = None
    red_ai_name: str | None = None
    blue_ai_name: str | None = None
    red_ai_version: str | None = None
    blue_ai_version: str | None = None
    winner_color: int | None = None
    termination_reason: str | None = None
    status: str = "active"
    ended_at: str | None = None
    record_sha256: str | None = None


@dataclass(frozen=True)
class PositionRecord:
    position_id: str
    experience_key: str
    canonical_key: str
    board_json: str
    current_color: int
    ply: int
    created_at: str
    rules_version: int = RULES_VERSION
    feature_version: int = FEATURE_VERSION
    features_json: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    game_id: str
    position_id: str
    ply: int
    actor_type: str
    actor_color: int
    created_at: str
    actor_id: str | None = None
    ai_name: str | None = None
    ai_version: str | None = None
    candidate_json: str | None = None
    search_config_json: str | None = None
    chosen_q: int | None = None
    chosen_r: int | None = None
    result: float | None = None
    decision_quality: str = "unverified"
    training_eligible: int = 0
    integrity_status: str = "unknown"
    rule_status: str = "unknown"
    result_status: str = "unknown"
    validation_reason: str | None = None
    validator_version: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True)
class RetrievalEventRecord:
    retrieval_event_id: str
    query_position_id: str
    retrieval_mode: str
    top_k: int
    feature_version: int
    retrieval_version: str
    elapsed_ms: float
    created_at: str
    query_game_id: str | None = None
    query_match_id: str | None = None
    selected_memory_decision_id: str | None = None
    similarity: float | None = None
    memory_move: tuple[int, int] | None = None
    base_move: tuple[int, int] | None = None
    shadow_move: tuple[int, int] | None = None
    final_move: tuple[int, int] | None = None
    confidence: float | None = None
    applied: int = 0
    evidence_json: str | None = None
    query_ply: int | None = None
    selection_probability: float | None = None
    exploration_epsilon: float | None = None
    policy_version: str | None = None


@dataclass(frozen=True)
class RetrievalOutcomeRecord:
    retrieval_event_id: str
    game_id: str
    decision_id: str
    applied: int
    decision_result: float
    game_result: float
    retrieval_reward: float
    retrieval_mode: str
    created_at: str
    similarity: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class MemoryJobRecord:
    job_id: str
    job_type: str
    target_id: str
    status: str
    created_at: str
    attempts: int = 0
    last_error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class DatasetVersionRecord:
    dataset_id: str
    dataset_dir: str
    manifest_sha256: str
    samples_sha256: str
    feature_version: int
    encoding_version: str
    sample_count: int
    split_counts_json: str
    source_match_count: int
    created_at: str


@dataclass(frozen=True)
class TrainingRunRecord:
    training_run_id: str
    dataset_id: str
    status: str
    started_at: str
    parent_model_sha256: str | None = None
    candidate_model_sha256: str | None = None
    metrics_json: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class TrustModelRecord:
    trust_model_id: str
    model_path: str
    sha256: str
    feature_version: int
    status: str
    created_at: str
    metrics_json: str | None = None


@dataclass(frozen=True)
class ActiveLearningCandidateRecord:
    candidate_id: str
    position_id: str
    decision_id: str
    priority: float
    created_at: str
    reason_json: str | None = None
    selected: int = 0
    dataset_id: str | None = None


@dataclass(frozen=True)
class OpponentProfileRecord:
    opponent_id: str
    ai_name: str
    created_at: str
    ai_version: str | None = None
    model_sha256: str | None = None
    elo: float | None = None
    metrics_json: str | None = None


@dataclass(frozen=True)
class LeagueRunRecord:
    league_run_id: str
    status: str
    started_at: str
    candidate_sha256: str | None = None
    metrics_json: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class DriftSnapshotRecord:
    snapshot_id: str
    window_start: str
    window_end: str
    metrics_json: str
    alerts_json: str
    created_at: str


@dataclass(frozen=True)
class GameExperience:
    game: GameRecord
    decisions: tuple[tuple[PositionRecord, DecisionRecord], ...]


@dataclass(frozen=True)
class MatchExperience:
    match: MatchRecord
    games: tuple[GameExperience, ...]


@dataclass(frozen=True)
class MatchWriteResult:
    match_changed: bool
    games_changed: int
    positions_changed: int
    decisions_changed: int

    @property
    def any_changed(self):
        return any((
            self.match_changed,
            self.games_changed,
            self.positions_changed,
            self.decisions_changed,
        ))


@dataclass(frozen=True)
class StoreWriteResult:
    match_changed: bool
    game_changed: bool
    position_changed: bool
    decision_changed: bool

    @property
    def any_changed(self):
        return any((
            self.match_changed,
            self.game_changed,
            self.position_changed,
            self.decision_changed,
        ))


@dataclass(frozen=True)
class MemoryHealth:
    database_path: str
    schema_version: int
    journal_mode: str
    synchronous: int
    foreign_keys: bool
    busy_timeout_ms: int
    integrity: str
    foreign_key_violations: int
    counts: dict[str, int]
    size_bytes: int
    sqlite_version: str

def _require_text(value, field, *, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field} must be a non-empty string")


def _require_int(value, field, *, minimum=None, optional=False):
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise MemoryValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise MemoryValidationError(f"{field} must be >= {minimum}")


def _require_enum(value, field, choices):
    if value not in choices:
        raise MemoryValidationError(f"{field} must be one of {choices!r}")


def _require_color(value, field, *, optional=False):
    if value is None and optional:
        return
    if value not in (RED, BLUE):
        raise MemoryValidationError(f"{field} must be RED or BLUE")


def _require_timestamp(value, field, *, optional=False):
    if value is None and optional:
        return
    _require_text(value, field)
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MemoryValidationError(f"{field} must be an ISO-8601 timestamp") from exc


def _require_json_text(value, field, *, optional=False):
    if value is None and optional:
        return
    _require_text(value, field)
    try:
        json.loads(value)
    except (TypeError, ValueError) as exc:
        raise MemoryValidationError(f"{field} must contain valid JSON") from exc


def _validate_match(record):
    if not isinstance(record, MatchRecord):
        raise MemoryValidationError("record must be MatchRecord")
    _require_text(record.match_id, "match_id")
    _require_int(record.schema_version, "schema_version", minimum=1)
    if record.schema_version != MEMORY_SCHEMA_VERSION:
        raise MemoryValidationError("schema_version is not supported")
    _require_int(record.rules_version, "rules_version", minimum=1)
    _require_int(record.mode_key, "mode_key", minimum=0, optional=True)
    _require_text(record.mode_name, "mode_name")
    _require_text(record.red_team, "red_team")
    _require_text(record.blue_team, "blue_team")
    _require_enum(record.source_type, "source_type", SOURCE_TYPES)
    _require_timestamp(record.started_at, "started_at")
    _require_timestamp(record.ended_at, "ended_at", optional=True)
    _require_enum(record.status, "status", MATCH_STATUSES)
    _require_text(record.record_id, "record_id", optional=True)
    _require_enum(record.capture_mode, "capture_mode", CAPTURE_MODES)
    _require_text(record.collector_version, "collector_version")


def _validate_game(record):
    if not isinstance(record, GameRecord):
        raise MemoryValidationError("record must be GameRecord")
    _require_text(record.game_id, "game_id")
    _require_text(record.match_id, "match_id")
    _require_int(record.game_no, "game_no", minimum=1)
    if record.game_no not in (1, 2):
        raise MemoryValidationError("game_no must be 1 or 2")
    _require_enum(record.red_actor_type, "red_actor_type", ACTOR_TYPES)
    _require_enum(record.blue_actor_type, "blue_actor_type", ACTOR_TYPES)
    _require_timestamp(record.started_at, "started_at")
    for field in (
        "red_actor_id", "blue_actor_id", "red_ai_name", "blue_ai_name",
        "red_ai_version", "blue_ai_version", "termination_reason",
    ):
        _require_text(getattr(record, field), field, optional=True)
    _require_color(record.winner_color, "winner_color", optional=True)
    _require_enum(record.status, "status", GAME_STATUSES)
    _require_timestamp(record.ended_at, "ended_at", optional=True)
    _require_text(record.record_sha256, "record_sha256", optional=True)


def _validate_position(record):
    if not isinstance(record, PositionRecord):
        raise MemoryValidationError("record must be PositionRecord")
    _require_text(record.position_id, "position_id")
    _require_text(record.experience_key, "experience_key")
    _require_text(record.canonical_key, "canonical_key")
    if not _POSITION_ID_PATTERN.fullmatch(record.position_id):
        raise MemoryValidationError("position_id has an invalid format")
    if record.experience_key != record.position_id:
        raise MemoryValidationError("experience_key must equal position_id")
    if not _CANONICAL_ID_PATTERN.fullmatch(record.canonical_key):
        raise MemoryValidationError("canonical_key has an invalid format")
    _require_int(record.rules_version, "rules_version", minimum=1)
    _require_int(record.feature_version, "feature_version", minimum=1)
    _require_json_text(record.board_json, "board_json")
    _require_color(record.current_color, "current_color")
    _require_int(record.ply, "ply", minimum=1)
    _require_json_text(record.features_json, "features_json", optional=True)
    _require_timestamp(record.created_at, "created_at")


def _validate_decision(record):
    if not isinstance(record, DecisionRecord):
        raise MemoryValidationError("record must be DecisionRecord")
    _require_text(record.decision_id, "decision_id")
    _require_text(record.game_id, "game_id")
    _require_text(record.position_id, "position_id")
    if not _DECISION_ID_PATTERN.fullmatch(record.decision_id):
        raise MemoryValidationError("decision_id has an invalid format")
    if not _POSITION_ID_PATTERN.fullmatch(record.position_id):
        raise MemoryValidationError("position_id has an invalid format")
    _require_int(record.ply, "ply", minimum=1)
    _require_enum(record.actor_type, "actor_type", ACTOR_TYPES)
    _require_color(record.actor_color, "actor_color")
    _require_timestamp(record.created_at, "created_at")
    for field in ("actor_id", "ai_name", "ai_version", "validation_reason",
                  "validator_version"):
        _require_text(getattr(record, field), field, optional=True)
    _require_json_text(record.candidate_json, "candidate_json", optional=True)
    _require_json_text(record.search_config_json, "search_config_json", optional=True)
    _require_int(record.chosen_q, "chosen_q", minimum=0, optional=True)
    _require_int(record.chosen_r, "chosen_r", minimum=0, optional=True)
    if (record.chosen_q is None) != (record.chosen_r is None):
        raise MemoryValidationError("chosen_q and chosen_r must both be set or both be null")
    if record.chosen_q is not None and record.chosen_q > 10:
        raise MemoryValidationError("chosen_q must be <= 10")
    if record.chosen_r is not None and record.chosen_r > 10:
        raise MemoryValidationError("chosen_r must be <= 10")
    if record.result is not None:
        if isinstance(record.result, bool) or record.result not in (0.0, 0.5, 1.0):
            raise MemoryValidationError("result must be 0.0, 0.5, 1.0 or null")
    _require_enum(record.decision_quality, "decision_quality", DECISION_QUALITIES)
    if record.training_eligible not in (0, 1):
        raise MemoryValidationError("training_eligible must be 0 or 1")
    _require_enum(record.integrity_status, "integrity_status", INTEGRITY_STATUSES)
    _require_enum(record.rule_status, "rule_status", RULE_STATUSES)
    _require_enum(record.result_status, "result_status", RESULT_STATUSES)
    _require_text(record.model_sha256, "model_sha256", optional=True)


def _validate_bundle(match, game, position, decision):
    _validate_match(match)
    _validate_game(game)
    _validate_position(position)
    _validate_decision(decision)
    if game.match_id != match.match_id:
        raise MemoryValidationError("game.match_id must match the match record")
    if decision.game_id != game.game_id:
        raise MemoryValidationError("decision.game_id must match the game record")
    if decision.position_id != position.position_id:
        raise MemoryValidationError("decision.position_id must match the position record")
    if decision.ply != position.ply:
        raise MemoryValidationError("decision.ply must match position.ply")
    if decision.actor_color != position.current_color:
        raise MemoryValidationError("decision.actor_color must match current_color")
    expected_actor = (
        game.red_actor_type if decision.actor_color == RED else game.blue_actor_type
    )
    if decision.actor_type != expected_actor:
        raise MemoryValidationError("decision.actor_type does not match the game side")

def _validate_retrieval_event(record):
    if not isinstance(record, RetrievalEventRecord):
        raise MemoryValidationError("record must be RetrievalEventRecord")
    for field in ("retrieval_event_id", "query_position_id", "retrieval_version"):
        _require_text(getattr(record, field), field)
    for field in ("query_game_id", "query_match_id", "selected_memory_decision_id"):
        _require_text(getattr(record, field), field, optional=True)
    _require_enum(record.retrieval_mode, "retrieval_mode", RETRIEVAL_MODES)
    _require_int(record.top_k, "top_k", minimum=1)
    _require_int(record.feature_version, "feature_version", minimum=1)
    _require_int(record.query_ply, "query_ply", minimum=1, optional=True)
    for field in ("selection_probability", "exploration_epsilon"):
        value = getattr(record, field)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise MemoryValidationError(f"{field} must be between 0 and 1")
    _require_text(record.policy_version, "policy_version", optional=True)
    if isinstance(record.elapsed_ms, bool) or not isinstance(record.elapsed_ms, (int, float)):
        raise MemoryValidationError("elapsed_ms must be numeric")
    if record.elapsed_ms < 0:
        raise MemoryValidationError("elapsed_ms must be non-negative")
    _require_timestamp(record.created_at, "created_at")
    if record.applied not in (0, 1):
        raise MemoryValidationError("applied must be 0 or 1")
    for field, value in (
        ("memory_move", record.memory_move),
        ("base_move", record.base_move),
        ("shadow_move", record.shadow_move),
        ("final_move", record.final_move),
    ):
        if value is None:
            continue
        if not isinstance(value, tuple) or len(value) != 2:
            raise MemoryValidationError(f"{field} must be a two-item tuple")
        for coordinate in value:
            _require_int(coordinate, field, minimum=0)
            if coordinate > 10:
                raise MemoryValidationError(f"{field} coordinate must be <= 10")
    for field in ("similarity", "confidence"):
        value = getattr(record, field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MemoryValidationError(f"{field} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise MemoryValidationError(f"{field} must be between 0 and 1")
    _require_json_text(record.evidence_json, "evidence_json", optional=True)


def _validate_match_experience(experience):
    if not isinstance(experience, MatchExperience):
        raise MemoryValidationError("experience must be MatchExperience")
    _validate_match(experience.match)
    if experience.match.status != "completed":
        raise MemoryValidationError("complete match must have completed status")
    if not isinstance(experience.games, tuple) or len(experience.games) != 2:
        raise MemoryValidationError("complete match must contain exactly two games")
    game_numbers = []
    for item in experience.games:
        if not isinstance(item, GameExperience):
            raise MemoryValidationError("games must contain GameExperience values")
        game = item.game
        _validate_game(game)
        game_numbers.append(game.game_no)
        if game.match_id != experience.match.match_id:
            raise MemoryValidationError("every game must belong to the match")
        if game.status != "completed" or game.winner_color not in (RED, BLUE):
            raise MemoryValidationError("every game must have a completed winner")
        if game.ended_at is None:
            raise MemoryValidationError("completed games require ended_at")
        if not isinstance(item.decisions, tuple) or not item.decisions:
            raise MemoryValidationError("every completed game requires decisions")
        for position, decision in item.decisions:
            _validate_bundle(experience.match, game, position, decision)
            expected = 1.0 if decision.actor_color == game.winner_color else 0.0
            if decision.result != expected:
                raise MemoryValidationError("decision result must reflect the game winner")
            if decision.result_status != "known":
                raise MemoryValidationError("completed decisions require known result_status")
    if tuple(sorted(game_numbers)) != (1, 2):
        raise MemoryValidationError("complete match must contain game 1 and game 2")


def _retrieval_event_dict(record):
    value = {
        "retrieval_event_id": record.retrieval_event_id,
        "query_position_id": record.query_position_id,
        "query_game_id": record.query_game_id,
        "query_match_id": record.query_match_id,
        "retrieval_mode": record.retrieval_mode,
        "top_k": record.top_k,
        "selected_memory_decision_id": record.selected_memory_decision_id,
        "similarity": record.similarity,
        "memory_q": record.memory_move[0] if record.memory_move else None,
        "memory_r": record.memory_move[1] if record.memory_move else None,
        "base_q": record.base_move[0] if record.base_move else None,
        "base_r": record.base_move[1] if record.base_move else None,
        "shadow_q": record.shadow_move[0] if record.shadow_move else None,
        "shadow_r": record.shadow_move[1] if record.shadow_move else None,
        "final_q": record.final_move[0] if record.final_move else None,
        "final_r": record.final_move[1] if record.final_move else None,
        "confidence": record.confidence,
        "applied": int(record.applied),
        "feature_version": record.feature_version,
        "retrieval_version": record.retrieval_version,
        "elapsed_ms": float(record.elapsed_ms),
        "evidence_json": record.evidence_json,
        "created_at": record.created_at,
        "query_ply": record.query_ply,
        "selection_probability": record.selection_probability,
        "exploration_epsilon": record.exploration_epsilon,
        "policy_version": record.policy_version,
    }
    return value


def _simple_record_values(record, fields):
    return {field: getattr(record, field) for field in fields}


def _record_dict(record):
    return {
        "match_id": record.match_id,
        "schema_version": record.schema_version,
        "rules_version": record.rules_version,
        "mode_key": record.mode_key,
        "mode_name": record.mode_name,
        "red_team": record.red_team,
        "blue_team": record.blue_team,
        "source_type": record.source_type,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "status": record.status,
        "record_id": record.record_id,
        "capture_mode": record.capture_mode,
        "collector_version": record.collector_version,
    }


def _game_dict(record):
    return {
        "game_id": record.game_id,
        "match_id": record.match_id,
        "game_no": record.game_no,
        "red_actor_type": record.red_actor_type,
        "blue_actor_type": record.blue_actor_type,
        "red_actor_id": record.red_actor_id,
        "blue_actor_id": record.blue_actor_id,
        "red_ai_name": record.red_ai_name,
        "blue_ai_name": record.blue_ai_name,
        "red_ai_version": record.red_ai_version,
        "blue_ai_version": record.blue_ai_version,
        "winner_color": record.winner_color,
        "termination_reason": record.termination_reason,
        "status": record.status,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "record_sha256": record.record_sha256,
    }


def _position_dict(record):
    return {
        "position_id": record.position_id,
        "experience_key": record.experience_key,
        "canonical_key": record.canonical_key,
        "rules_version": record.rules_version,
        "feature_version": record.feature_version,
        "board_json": record.board_json,
        "current_color": record.current_color,
        "ply": record.ply,
        "features_json": record.features_json,
        "created_at": record.created_at,
    }


def _decision_dict(record):
    return {
        "decision_id": record.decision_id,
        "game_id": record.game_id,
        "position_id": record.position_id,
        "ply": record.ply,
        "actor_type": record.actor_type,
        "actor_color": record.actor_color,
        "actor_id": record.actor_id,
        "ai_name": record.ai_name,
        "ai_version": record.ai_version,
        "candidate_json": record.candidate_json,
        "search_config_json": record.search_config_json,
        "chosen_q": record.chosen_q,
        "chosen_r": record.chosen_r,
        "result": record.result,
        "decision_quality": record.decision_quality,
        "training_eligible": int(record.training_eligible),
        "integrity_status": record.integrity_status,
        "rule_status": record.rule_status,
        "result_status": record.result_status,
        "validation_reason": record.validation_reason,
        "validator_version": record.validator_version,
        "created_at": record.created_at,
        "model_sha256": record.model_sha256,
    }


def _coerce_database_path(path):
    if str(path) == ":memory:":
        return ":memory:"
    return Path(path).expanduser().resolve()


def _path_text(path):
    return ":memory:" if path == ":memory:" else str(path)


def _quick_check(connection):
    rows = connection.execute("PRAGMA quick_check").fetchall()
    values = [str(row[0]) for row in rows]
    if values != ["ok"]:
        raise MemoryIntegrityError("SQLite quick_check failed: " + "; ".join(values))
    return "ok"


def _foreign_key_violations(connection):
    return len(connection.execute("PRAGMA foreign_key_check").fetchall())


def _is_integrity_exception(exc):
    message = str(exc).lower()
    return any(token in message for token in (
        "file is not a database",
        "database disk image is malformed",
        "database malformed",
        "quick_check",
    ))


def _translate_database_error(exc):
    if _is_integrity_exception(exc):
        return MemoryIntegrityError(str(exc))
    return MemoryDatabaseError(str(exc))


def _quarantine_database(path):
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    quarantine = path.parent / f"{path.name}.corrupt-{stamp}"
    counter = 1
    while quarantine.exists():
        quarantine = path.parent / f"{path.name}.corrupt-{stamp}-{counter}"
        counter += 1
    quarantine.mkdir(parents=False, exist_ok=False)
    moved = 0
    for source in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if source.exists():
            os.replace(source, quarantine / source.name)
            moved += 1
    if not moved:
        quarantine.rmdir()
        raise MemoryIntegrityError("No database files were available to quarantine")
    return quarantine


def recover_database(path, *, busy_timeout_ms=DEFAULT_BUSY_TIMEOUT_MS):
    """Quarantine a structurally corrupt database and create a clean one.

    Healthy databases and databases with an unsupported future schema are never
    quarantined. The damaged file and any WAL/SHM sidecars are preserved under
    a timestamped directory next to the original database.
    """
    normalized = _coerce_database_path(path)
    if normalized == ":memory:":
        raise MemoryValidationError("In-memory databases cannot be recovered")
    probe = ExperienceStore(normalized, busy_timeout_ms=busy_timeout_ms)
    try:
        probe.open()
    except MemoryIntegrityError:
        probe.close()
        quarantine = _quarantine_database(normalized)
        rebuilt = ExperienceStore(normalized, busy_timeout_ms=busy_timeout_ms)
        try:
            rebuilt.open()
        finally:
            rebuilt.close()
        return quarantine
    else:
        probe.close()
        raise MemoryStoreError("Database passes integrity checks; recovery was not started")

class ExperienceStore:
    """Thread-safe SQLite store with explicit P0-B lifecycle management."""

    def __init__(self, path=None, *, busy_timeout_ms=DEFAULT_BUSY_TIMEOUT_MS,
                 recover_corrupt=False):
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int):
            raise MemoryValidationError("busy_timeout_ms must be an integer")
        if busy_timeout_ms < 1:
            raise MemoryValidationError("busy_timeout_ms must be positive")
        self._database_path = _coerce_database_path(path if path is not None else memory_db_path())
        self._busy_timeout_ms = busy_timeout_ms
        self._recover_corrupt = bool(recover_corrupt)
        self._connection = None
        self._lock = RLock()
        self._last_recovery_path = None

    @property
    def database_path(self):
        return _path_text(self._database_path)

    @property
    def is_open(self):
        return self._connection is not None

    @property
    def last_recovery_path(self):
        return self._last_recovery_path

    def open(self, *, recover_corrupt=None):
        with self._lock:
            if self._connection is not None:
                return self
            should_recover = (
                self._recover_corrupt if recover_corrupt is None else bool(recover_corrupt)
            )
            try:
                self._open_once()
            except MemoryIntegrityError:
                if not should_recover or self._database_path == ":memory:":
                    raise
                self._close_connection_unlocked()
                quarantine = _quarantine_database(self._database_path)
                self._open_once()
                self._last_recovery_path = quarantine
            return self

    def _open_once(self):
        if self._database_path != ":memory:":
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        target = _path_text(self._database_path)
        try:
            connection = sqlite3.connect(
                target,
                timeout=self._busy_timeout_ms / 1000.0,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise _translate_database_error(exc) from exc
        connection.row_factory = sqlite3.Row
        self._connection = connection
        try:
            self._configure_connection()
            self._migrate_or_verify()
            self._verify_schema()
            _quick_check(connection)
            if _foreign_key_violations(connection):
                raise MemoryIntegrityError("Foreign-key violations detected")
        except MemoryStoreError:
            self._close_connection_unlocked()
            raise
        except sqlite3.DatabaseError as exc:
            self._close_connection_unlocked()
            raise _translate_database_error(exc) from exc

    def _configure_connection(self):
        connection = self._require_connection_unlocked()
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA temp_store = MEMORY")
        if self._database_path != ":memory:":
            journal_mode = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).lower()
            if journal_mode != "wal":
                raise MemoryDatabaseError(
                    f"Could not enable WAL journal mode: {journal_mode}"
                )
        connection.execute("PRAGMA synchronous = FULL")
        if not bool(connection.execute("PRAGMA foreign_keys").fetchone()[0]):
            raise MemoryDatabaseError("Could not enable SQLite foreign keys")

    def _migrate_or_verify(self):
        connection = self._require_connection_unlocked()
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > MEMORY_SCHEMA_VERSION:
            raise MemorySchemaError(
                f"Database schema {version} is newer than supported {MEMORY_SCHEMA_VERSION}"
            )
        if version == 0:
            self._apply_initial_schema()
            return
        while version < MEMORY_SCHEMA_VERSION:
            statements = MIGRATION_DDL.get(version)
            if statements is None:
                raise MemorySchemaError(f"No migration path from schema version {version}")
            self._apply_migration(version, statements)
            version += 1

    def _apply_initial_schema(self):
        connection = self._require_connection_unlocked()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for statement in (*TABLE_DDL, *INDEX_DDL):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {MEMORY_SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise _translate_database_error(exc) from exc

    def _apply_migration(self, source_version, statements):
        connection = self._require_connection_unlocked()
        target_version = source_version + 1
        try:
            connection.execute("BEGIN IMMEDIATE")
            for statement in statements:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {target_version}")
            connection.commit()
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise _translate_database_error(exc) from exc

    def _verify_schema(self):
        connection = self._require_connection_unlocked()
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != MEMORY_SCHEMA_VERSION:
            raise MemorySchemaError(
                f"Expected schema version {MEMORY_SCHEMA_VERSION}, found {version}"
            )
        tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = set(MEMORY_TABLES) - tables
        if missing_tables:
            raise MemorySchemaError(
                "Missing tables: " + ", ".join(sorted(missing_tables))
            )
        for table in MEMORY_TABLES:
            actual = tuple(
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if actual != TABLE_COLUMNS[table]:
                raise MemorySchemaError(f"Unexpected columns for table {table}")
        indexes = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        missing_indexes = set(MEMORY_INDEXES) - indexes
        if missing_indexes:
            raise MemorySchemaError(
                "Missing indexes: " + ", ".join(sorted(missing_indexes))
            )

    def _close_connection_unlocked(self):
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass

    def close(self):
        with self._lock:
            self._close_connection_unlocked()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _require_connection_unlocked(self):
        if self._connection is None:
            raise MemoryClosedError("ExperienceStore is not open")
        return self._connection

    @contextmanager
    def _write_transaction(self):
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
            try:
                yield connection
            except MemoryStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise MemoryConflictError(str(exc)) from exc
            except sqlite3.DatabaseError as exc:
                connection.rollback()
                raise _translate_database_error(exc) from exc
            except Exception:
                connection.rollback()
                raise
            try:
                connection.commit()
            except sqlite3.DatabaseError as exc:
                connection.rollback()
                raise _translate_database_error(exc) from exc

    def _select_one(self, connection, table, key_column, key_value):
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {key_column} = ?",
            (key_value,),
        ).fetchone()
        return None if row is None else dict(row)

    def _insert_row(self, connection, table, values):
        columns = TABLE_COLUMNS[table]
        placeholders = ",".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
            tuple(values[column] for column in columns),
        )

    def _update_row(self, connection, table, key_column, key_value, updates):
        if not updates:
            return False
        assignments = ", ".join(f"{column} = ?" for column in updates)
        connection.execute(
            f"UPDATE {table} SET {assignments} WHERE {key_column} = ?",
            (*updates.values(), key_value),
        )
        return True

    def _save_match_connection(self, connection, record):
        new = _record_dict(record)
        current = self._select_one(connection, "matches", "match_id", record.match_id)
        if current is None:
            self._insert_row(connection, "matches", new)
            return True
        immutable = (
            "schema_version", "rules_version", "mode_key", "mode_name",
            "red_team", "blue_team", "source_type", "started_at",
        )
        for field in immutable:
            if current[field] != new[field]:
                raise MemoryConflictError(f"match_id already exists with a different {field}")
        if current["record_id"] and new["record_id"] and current["record_id"] != new["record_id"]:
            raise MemoryConflictError("match_id already has a different record_id")
        mutable = ("ended_at", "status", "record_id")
        new["record_id"] = new["record_id"] or current["record_id"]
        updates = {field: new[field] for field in mutable if current[field] != new[field]}
        return self._update_row(connection, "matches", "match_id", record.match_id, updates)

    def _save_game_connection(self, connection, record):
        new = _game_dict(record)
        current = self._select_one(connection, "games", "game_id", record.game_id)
        if current is None:
            if self._select_one(connection, "matches", "match_id", record.match_id) is None:
                raise MemoryConflictError("game references a match that does not exist")
            self._insert_row(connection, "games", new)
            return True
        immutable = (
            "match_id", "game_no", "red_actor_type", "blue_actor_type",
            "red_actor_id", "blue_actor_id", "red_ai_name", "blue_ai_name",
            "red_ai_version", "blue_ai_version", "started_at",
        )
        for field in immutable:
            if current[field] != new[field]:
                raise MemoryConflictError(f"game_id already exists with a different {field}")
        mutable = ("winner_color", "termination_reason", "status", "ended_at")
        updates = {field: new[field] for field in mutable if current[field] != new[field]}
        return self._update_row(connection, "games", "game_id", record.game_id, updates)

    def _save_position_connection(self, connection, record):
        new = _position_dict(record)
        current = self._select_one(
            connection, "positions", "position_id", record.position_id
        )
        by_experience = self._select_one(
            connection, "positions", "experience_key", record.experience_key
        )
        if current is None and by_experience is not None:
            raise MemoryConflictError("experience_key already belongs to another position_id")
        if current is None:
            self._insert_row(connection, "positions", new)
            return True
        core = (
            "experience_key", "canonical_key", "rules_version", "feature_version",
            "board_json", "current_color", "ply",
        )
        for field in core:
            if current[field] != new[field]:
                raise MemoryConflictError(
                    f"position_id already exists with a different {field}"
                )
        updates = {}
        if current["features_json"] is None and new["features_json"] is not None:
            updates["features_json"] = new["features_json"]
        elif (current["features_json"] is not None
              and new["features_json"] is not None
              and current["features_json"] != new["features_json"]):
            raise MemoryConflictError("position_id already has different features_json")
        return self._update_row(
            connection, "positions", "position_id", record.position_id, updates
        )

    def _save_decision_connection(self, connection, record):
        new = _decision_dict(record)
        current = self._select_one(
            connection, "decisions", "decision_id", record.decision_id
        )
        if current is None:
            if self._select_one(connection, "games", "game_id", record.game_id) is None:
                raise MemoryConflictError("decision references a game that does not exist")
            if self._select_one(
                connection, "positions", "position_id", record.position_id
            ) is None:
                raise MemoryConflictError("decision references a position that does not exist")
            self._insert_row(connection, "decisions", new)
            return True
        immutable = (
            "game_id", "position_id", "ply", "actor_type", "actor_color",
            "actor_id", "ai_name", "ai_version", "candidate_json",
            "search_config_json", "chosen_q", "chosen_r",
        )
        for field in immutable:
            if current[field] != new[field]:
                raise MemoryConflictError(
                    f"decision_id already exists with a different {field}"
                )
        mutable = (
            "result", "decision_quality", "training_eligible",
            "integrity_status", "rule_status", "result_status",
            "validation_reason", "validator_version",
        )
        updates = {field: new[field] for field in mutable if current[field] != new[field]}
        return self._update_row(
            connection, "decisions", "decision_id", record.decision_id, updates
        )

    def save_match(self, record):
        _validate_match(record)
        with self._write_transaction() as connection:
            return self._save_match_connection(connection, record)

    def save_game(self, record):
        _validate_game(record)
        with self._write_transaction() as connection:
            return self._save_game_connection(connection, record)

    def save_position(self, record):
        _validate_position(record)
        with self._write_transaction() as connection:
            return self._save_position_connection(connection, record)

    def save_decision(self, record):
        _validate_decision(record)
        with self._write_transaction() as connection:
            return self._save_decision_connection(connection, record)

    def save_decision_bundle(self, match, game, position, decision):
        """Atomically persist one match/game/position/decision snapshot."""
        _validate_bundle(match, game, position, decision)
        with self._write_transaction() as connection:
            match_changed = self._save_match_connection(connection, match)
            game_changed = self._save_game_connection(connection, game)
            position_changed = self._save_position_connection(connection, position)
            decision_changed = self._save_decision_connection(connection, decision)
        return StoreWriteResult(
            match_changed=match_changed,
            game_changed=game_changed,
            position_changed=position_changed,
            decision_changed=decision_changed,
        )

    def save_complete_match(self, experience):
        """Atomically persist a complete two-game match and all decisions."""
        _validate_match_experience(experience)
        with self._write_transaction() as connection:
            match_changed = self._save_match_connection(connection, experience.match)
            games_changed = 0
            positions_changed = 0
            decisions_changed = 0
            for item in experience.games:
                games_changed += int(
                    self._save_game_connection(connection, item.game)
                )
                for position, decision in item.decisions:
                    positions_changed += int(
                        self._save_position_connection(connection, position)
                    )
                    decisions_changed += int(
                        self._save_decision_connection(connection, decision)
                    )
        return MatchWriteResult(
            match_changed=match_changed,
            games_changed=games_changed,
            positions_changed=positions_changed,
            decisions_changed=decisions_changed,
        )

    def update_position_features(self, position_id, features_json, *,
                                 feature_version=1):
        _require_text(position_id, "position_id")
        _require_json_text(features_json, "features_json")
        _require_int(feature_version, "feature_version", minimum=1)
        with self._write_transaction() as connection:
            if self._select_one(
                connection, "positions", "position_id", position_id
            ) is None:
                raise MemoryConflictError("position_id does not exist")
            connection.execute(
                "UPDATE positions SET features_json = ?, feature_version = ? "
                "WHERE position_id = ?",
                (features_json, feature_version, position_id),
            )
        return True

    def save_retrieval_event(self, record):
        _validate_retrieval_event(record)
        values = _retrieval_event_dict(record)
        with self._write_transaction() as connection:
            current = self._select_one(
                connection,
                "retrieval_events",
                "retrieval_event_id",
                record.retrieval_event_id,
            )
            if current is None:
                self._insert_row(connection, "retrieval_events", values)
                return True
            for key, value in values.items():
                if current[key] != value:
                    raise MemoryConflictError(
                        f"retrieval_event_id already exists with a different {key}"
                    )
            return False

    def list_retrieval_events(self, *, query_game_id=None, limit=500):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        parameters = []
        where = ""
        if query_game_id is not None:
            _require_text(query_game_id, "query_game_id")
            where = " WHERE query_game_id = ?"
            parameters.append(query_game_id)
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM retrieval_events"
                    + where
                    + " ORDER BY created_at, retrieval_event_id LIMIT ?",
                    tuple(parameters),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def list_positions_missing_features(self, *, limit=1000, match_id=None):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        parameters = []
        where = "features_json IS NULL"
        if match_id is not None:
            _require_text(match_id, "match_id")
            where += (
                " AND EXISTS ("
                "SELECT 1 FROM decisions d JOIN games g ON g.game_id = d.game_id "
                "WHERE d.position_id = positions.position_id AND g.match_id = ?)"
            )
            parameters.append(match_id)
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM positions WHERE "
                    + where
                    + " ORDER BY created_at, position_id LIMIT ?",
                    tuple(parameters),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def all_positions(self, *, limit=100000):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM positions ORDER BY created_at, position_id LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def query_training_samples(self, *, limit=1000000):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        query = (
            "SELECT d.*, p.board_json, p.current_color, "
            "p.ply AS position_ply, g.match_id, g.game_no "
            "FROM decisions d "
            "JOIN positions p ON p.position_id = d.position_id "
            "JOIN games g ON g.game_id = d.game_id "
            "WHERE d.training_eligible = 1 "
            "AND d.decision_quality = 'validated' "
            "AND d.actor_type = 'ai' "
            "AND d.chosen_q IS NOT NULL AND d.chosen_r IS NOT NULL "
            "AND d.result IS NOT NULL "
            "ORDER BY g.match_id, g.game_no, d.ply, d.decision_id "
            "LIMIT ?"
        )
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(query, (limit,)).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def query_validated_memory(self, *, current_color, ply, feature_version,
                               ply_window=6, exclude_match_id=None,
                               exclude_game_id=None, limit=2000):
        _require_color(current_color, "current_color")
        _require_int(ply, "ply", minimum=1)
        _require_int(feature_version, "feature_version", minimum=1)
        _require_int(ply_window, "ply_window", minimum=0)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        clauses = [
            "d.training_eligible = 1",
            "d.decision_quality = 'validated'",
            "d.actor_type = 'ai'",
            "p.current_color = ?",
            "p.rules_version = ?",
            "p.feature_version = ?",
            "p.features_json IS NOT NULL",
            "p.ply BETWEEN ? AND ?",
        ]
        parameters = [
            current_color,
            RULES_VERSION,
            feature_version,
            max(1, ply - ply_window),
            ply + ply_window,
        ]
        if exclude_match_id:
            clauses.append("g.match_id != ?")
            parameters.append(str(exclude_match_id))
        if exclude_game_id:
            clauses.append("g.game_id != ?")
            parameters.append(str(exclude_game_id))
        parameters.append(limit)
        query = (
            "SELECT d.*, p.canonical_key, p.board_json, p.features_json, "
            "p.feature_version, p.ply AS position_ply, g.match_id, "
            "g.winner_color "
            "FROM decisions d "
            "JOIN positions p ON p.position_id = d.position_id "
            "JOIN games g ON g.game_id = d.game_id "
            "WHERE "
            + " AND ".join(clauses)
            + " ORDER BY d.decision_id LIMIT ?"
        )
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(query, tuple(parameters)).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def load_match_for_validation(self, match_id):
        _require_text(match_id, "match_id")
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                match_row = connection.execute(
                    "SELECT * FROM matches WHERE match_id = ?", (match_id,)
                ).fetchone()
                if match_row is None:
                    return None
                game_rows = connection.execute(
                    "SELECT * FROM games WHERE match_id = ? ORDER BY game_no",
                    (match_id,),
                ).fetchall()
                rows = connection.execute(
                    "SELECT d.*, p.board_json, p.current_color, "
                    "p.ply AS position_ply, p.experience_key, p.canonical_key, "
                    "p.feature_version, p.features_json, g.match_id, "
                    "g.game_no, g.winner_color, g.status AS game_status "
                    "FROM decisions d "
                    "JOIN positions p ON p.position_id = d.position_id "
                    "JOIN games g ON g.game_id = d.game_id "
                    "WHERE g.match_id = ? "
                    "ORDER BY g.game_no, d.ply, d.decision_id",
                    (match_id,),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return {
            "match": dict(match_row),
            "games": [dict(row) for row in game_rows],
            "decisions": [dict(row) for row in rows],
        }

    def batch_update_decision_validation(self, updates):
        if not updates:
            return 0
        prepared = []
        for update in updates:
            decision_id = update.get("decision_id")
            _require_text(decision_id, "decision_id")
            quality = update.get("decision_quality", "unverified")
            _require_enum(quality, "decision_quality", DECISION_QUALITIES)
            training_eligible = int(update.get("training_eligible", 0))
            if training_eligible not in (0, 1):
                raise MemoryValidationError("training_eligible must be 0 or 1")
            integrity = update.get("integrity_status", "unknown")
            rule = update.get("rule_status", "unknown")
            result = update.get("result_status", "unknown")
            _require_enum(integrity, "integrity_status", INTEGRITY_STATUSES)
            _require_enum(rule, "rule_status", RULE_STATUSES)
            _require_enum(result, "result_status", RESULT_STATUSES)
            prepared.append((
                quality,
                training_eligible,
                integrity,
                rule,
                result,
                update.get("validation_reason"),
                update.get("validator_version"),
                decision_id,
            ))
        with self._write_transaction() as connection:
            connection.executemany(
                "UPDATE decisions SET decision_quality = ?, training_eligible = ?, "
                "integrity_status = ?, rule_status = ?, result_status = ?, "
                "validation_reason = ?, validator_version = ? "
                "WHERE decision_id = ?",
                prepared,
            )
        return len(prepared)

    def save_retrieval_outcome(self, record):
        fields = (
            "retrieval_event_id", "game_id", "decision_id", "applied",
            "decision_result", "game_result", "retrieval_reward",
            "similarity", "confidence", "retrieval_mode", "created_at",
        )
        values = _simple_record_values(record, fields)
        if record.applied not in (0, 1):
            raise MemoryValidationError("applied must be 0 or 1")
        if record.decision_result not in (0.0, 0.5, 1.0):
            raise MemoryValidationError("decision_result is invalid")
        if record.game_result not in (0.0, 0.5, 1.0):
            raise MemoryValidationError("game_result is invalid")
        if record.retrieval_reward not in (-1.0, 0.0, 1.0):
            raise MemoryValidationError("retrieval_reward is invalid")
        _require_enum(record.retrieval_mode, "retrieval_mode", RETRIEVAL_MODES)
        if record.similarity is not None and not 0.0 <= float(record.similarity) <= 1.0:
            raise MemoryValidationError("similarity must be between 0 and 1")
        if record.confidence is not None and not 0.0 <= float(record.confidence) <= 1.0:
            raise MemoryValidationError("confidence must be between 0 and 1")
        with self._write_transaction() as connection:
            current = self._select_one(
                connection, "retrieval_outcomes", "retrieval_event_id",
                record.retrieval_event_id,
            )
            if current is None:
                self._insert_row(connection, "retrieval_outcomes", values)
                return True
            for key, value in values.items():
                if current[key] != value:
                    raise MemoryConflictError(
                        f"retrieval outcome already exists with a different {key}"
                    )
            return False

    def list_retrieval_outcomes(self, *, game_id=None, limit=10000):
        parameters = []
        where = ""
        if game_id is not None:
            _require_text(game_id, "game_id")
            where = " WHERE game_id = ?"
            parameters.append(game_id)
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM retrieval_outcomes"
                + where
                + " ORDER BY created_at, retrieval_event_id LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_memory_job(self, record):
        fields = (
            "job_id", "job_type", "target_id", "status", "attempts",
            "last_error", "created_at", "started_at", "finished_at",
        )
        _require_enum(record.status, "status", (
            "pending", "running", "completed", "failed", "cancelled"
        ))
        values = _simple_record_values(record, fields)
        with self._write_transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM memory_jobs WHERE job_type = ? AND target_id = ?",
                (record.job_type, record.target_id),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            self._insert_row(connection, "memory_jobs", values)
        return values

    def update_memory_job(self, job_id, *, status, attempts=None,
                          last_error=None, started_at=None, finished_at=None):
        _require_enum(status, "status", (
            "pending", "running", "completed", "failed", "cancelled"
        ))
        updates = {"status": status}
        if attempts is not None:
            updates["attempts"] = int(attempts)
        if last_error is not None:
            updates["last_error"] = last_error
        if started_at is not None:
            updates["started_at"] = started_at
        if finished_at is not None:
            updates["finished_at"] = finished_at
        with self._write_transaction() as connection:
            self._update_row(connection, "memory_jobs", "job_id", job_id, updates)
        return True

    def list_memory_jobs(self, *, status=None, limit=1000):
        parameters = []
        where = ""
        if status is not None:
            where = " WHERE status = ?"
            parameters.append(status)
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM memory_jobs"
                + where
                + " ORDER BY created_at, job_id LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_dataset_version(self, record):
        fields = (
            "dataset_id", "dataset_dir", "manifest_sha256", "samples_sha256",
            "feature_version", "encoding_version", "sample_count",
            "split_counts_json", "source_match_count", "created_at",
        )
        values = _simple_record_values(record, fields)
        with self._write_transaction() as connection:
            if self._select_one(
                connection, "dataset_versions", "dataset_id", record.dataset_id
            ) is not None:
                return False
            self._insert_row(connection, "dataset_versions", values)
        return True

    def list_dataset_versions(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM dataset_versions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_training_run(self, record):
        fields = (
            "training_run_id", "dataset_id", "parent_model_sha256",
            "candidate_model_sha256", "status", "metrics_json",
            "started_at", "finished_at",
        )
        values = _simple_record_values(record, fields)
        with self._write_transaction() as connection:
            if self._select_one(
                connection, "training_runs", "training_run_id",
                record.training_run_id,
            ) is not None:
                return False
            self._insert_row(connection, "training_runs", values)
        return True

    def update_training_run(self, training_run_id, *, status,
                            candidate_model_sha256=None, metrics_json=None,
                            finished_at=None):
        updates = {"status": status}
        if candidate_model_sha256 is not None:
            updates["candidate_model_sha256"] = candidate_model_sha256
        if metrics_json is not None:
            updates["metrics_json"] = metrics_json
        if finished_at is not None:
            updates["finished_at"] = finished_at
        with self._write_transaction() as connection:
            self._update_row(
                connection, "training_runs", "training_run_id",
                training_run_id, updates,
            )
        return True

    def list_training_runs(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _save_immutable_row(self, table, key_column, key_value, values):
        with self._write_transaction() as connection:
            current = self._select_one(connection, table, key_column, key_value)
            if current is None:
                self._insert_row(connection, table, values)
                return True
            for key, value in values.items():
                if current[key] != value:
                    raise MemoryConflictError(
                        f"{key_column} already exists with a different {key}"
                    )
            return False

    def save_trust_model(self, record):
        fields = (
            "trust_model_id", "model_path", "sha256", "feature_version",
            "metrics_json", "status", "created_at",
        )
        _require_enum(record.status, "status", (
            "candidate", "selected", "rejected", "rolled_back"
        ))
        _require_int(record.feature_version, "feature_version", minimum=1)
        return self._save_immutable_row(
            "trust_models", "trust_model_id", record.trust_model_id,
            _simple_record_values(record, fields),
        )

    def list_trust_models(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM trust_models ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_active_learning_candidate(self, record):
        fields = (
            "candidate_id", "position_id", "decision_id", "priority",
            "reason_json", "selected", "dataset_id", "created_at",
        )
        if record.selected not in (0, 1):
            raise MemoryValidationError("selected must be 0 or 1")
        return self._save_immutable_row(
            "active_learning_queue", "candidate_id", record.candidate_id,
            _simple_record_values(record, fields),
        )

    def list_active_learning_candidates(self, *, selected=None, limit=1000):
        where = ""
        parameters = []
        if selected is not None:
            where = " WHERE selected = ?"
            parameters.append(int(selected))
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM active_learning_queue"
                + where
                + " ORDER BY priority DESC, created_at, candidate_id LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_opponent_profile(self, record):
        fields = (
            "opponent_id", "ai_name", "ai_version", "model_sha256", "elo",
            "metrics_json", "created_at",
        )
        return self._save_immutable_row(
            "opponent_profiles", "opponent_id", record.opponent_id,
            _simple_record_values(record, fields),
        )

    def list_opponent_profiles(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM opponent_profiles ORDER BY elo DESC, opponent_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_league_run(self, record):
        fields = (
            "league_run_id", "candidate_sha256", "status", "metrics_json",
            "started_at", "finished_at",
        )
        return self._save_immutable_row(
            "league_runs", "league_run_id", record.league_run_id,
            _simple_record_values(record, fields),
        )

    def update_league_run(self, league_run_id, *, status, metrics_json=None,
                          finished_at=None):
        updates = {"status": status}
        if metrics_json is not None:
            updates["metrics_json"] = metrics_json
        if finished_at is not None:
            updates["finished_at"] = finished_at
        with self._write_transaction() as connection:
            self._update_row(
                connection, "league_runs", "league_run_id",
                league_run_id, updates,
            )
        return True

    def list_league_runs(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM league_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_drift_snapshot(self, record):
        fields = (
            "snapshot_id", "window_start", "window_end", "metrics_json",
            "alerts_json", "created_at",
        )
        _require_json_text(record.metrics_json, "metrics_json")
        _require_json_text(record.alerts_json, "alerts_json")
        return self._save_immutable_row(
            "drift_snapshots", "snapshot_id", record.snapshot_id,
            _simple_record_values(record, fields),
        )

    def list_drift_snapshots(self, *, limit=100):
        with self._lock:
            connection = self._require_connection_unlocked()
            rows = connection.execute(
                "SELECT * FROM drift_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _get_row(self, table, key_column, key_value):
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                return self._select_one(connection, table, key_column, key_value)
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc

    def get_match(self, match_id):
        return self._get_row("matches", "match_id", match_id)

    def list_matches(self, *, limit=100):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM matches ORDER BY started_at, match_id LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def get_game(self, game_id):
        return self._get_row("games", "game_id", game_id)

    def get_position(self, position_id):
        return self._get_row("positions", "position_id", position_id)

    def get_decision(self, decision_id):
        return self._get_row("decisions", "decision_id", decision_id)

    def list_decisions(self, *, game_id=None, training_eligible=None, limit=500):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        if training_eligible is not None and training_eligible not in (0, 1):
            raise MemoryValidationError("training_eligible must be 0 or 1")
        clauses = []
        parameters = []
        if game_id is not None:
            _require_text(game_id, "game_id")
            clauses.append("game_id = ?")
            parameters.append(game_id)
        if training_eligible is not None:
            clauses.append("training_eligible = ?")
            parameters.append(int(training_eligible))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM decisions"
                    + where
                    + " ORDER BY ply, created_at, decision_id LIMIT ?",
                    tuple(parameters),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def find_positions_by_canonical_key(self, canonical_key, *, limit=50):
        _require_text(canonical_key, "canonical_key")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryValidationError("limit must be a positive integer")
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                rows = connection.execute(
                    "SELECT * FROM positions WHERE canonical_key = ? "
                    "ORDER BY created_at, position_id LIMIT ?",
                    (canonical_key, limit),
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
        return [dict(row) for row in rows]

    def counts(self):
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                return {
                    table: int(connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0])
                    for table in MEMORY_TABLES
                }
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc

    def health_check(self):
        with self._lock:
            connection = self._require_connection_unlocked()
            try:
                self._verify_schema()
                integrity = _quick_check(connection)
                violations = _foreign_key_violations(connection)
                if violations:
                    raise MemoryIntegrityError(
                        f"Foreign-key violations detected: {violations}"
                    )
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                journal_mode = str(
                    connection.execute("PRAGMA journal_mode").fetchone()[0]
                ).lower()
                synchronous = int(
                    connection.execute("PRAGMA synchronous").fetchone()[0]
                )
                foreign_keys = bool(
                    connection.execute("PRAGMA foreign_keys").fetchone()[0]
                )
                busy_timeout_ms = int(
                    connection.execute("PRAGMA busy_timeout").fetchone()[0]
                )
            except sqlite3.DatabaseError as exc:
                raise _translate_database_error(exc) from exc
            counts = {
                table: int(connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0])
                for table in MEMORY_TABLES
            }
        size_bytes = (
            0 if self._database_path == ":memory:"
            else self._database_path.stat().st_size
        )
        return MemoryHealth(
            database_path=_path_text(self._database_path),
            schema_version=schema_version,
            journal_mode=journal_mode,
            synchronous=synchronous,
            foreign_keys=foreign_keys,
            busy_timeout_ms=busy_timeout_ms,
            integrity=integrity,
            foreign_key_violations=violations,
            counts=counts,
            size_bytes=size_bytes,
            sqlite_version=sqlite3.sqlite_version,
        )

    def backup_to(self, destination):
        """Create a consistent standalone backup using SQLite's backup API."""
        target_path = _coerce_database_path(destination)
        if target_path == ":memory:":
            raise MemoryValidationError("Backup destination must be a file path")
        if (self._database_path != ":memory:"
                and target_path == self._database_path):
            raise MemoryValidationError("Backup destination must differ from the source")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            with self._lock:
                source = self._require_connection_unlocked()
                try:
                    target = sqlite3.connect(temp_path)
                    try:
                        source.backup(target)
                        target.execute("PRAGMA journal_mode = DELETE")
                        check = target.execute("PRAGMA quick_check").fetchall()
                        if [str(row[0]) for row in check] != ["ok"]:
                            raise MemoryIntegrityError("Backup quick_check failed")
                    finally:
                        target.close()
                except sqlite3.DatabaseError as exc:
                    raise _translate_database_error(exc) from exc
            with open(temp_path, "rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return target_path

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass