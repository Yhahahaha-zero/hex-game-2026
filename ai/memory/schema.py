# -*- coding: utf-8 -*-
"""Versioned SQLite schema contract for experience memory.

Importing this module is side-effect free. ExperienceStore owns migrations,
connections, transactions and all database I/O.
"""
from __future__ import annotations

import os
from pathlib import Path


MEMORY_SCHEMA_VERSION = 1
RULES_VERSION = 1
FEATURE_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_DB = PROJECT_ROOT.parent / "artifacts" / "memory" / "hex_experience.sqlite3"

MATCH_STATUSES = ("active", "completed", "aborted", "invalid")
GAME_STATUSES = ("active", "completed", "aborted", "invalid")
ACTOR_TYPES = ("ai", "human")
DECISION_QUALITIES = ("validated", "unverified", "rejected")
INTEGRITY_STATUSES = ("valid", "invalid", "unknown")
RULE_STATUSES = ("valid", "invalid", "unknown")
RESULT_STATUSES = ("known", "unknown", "invalid")
SOURCE_TYPES = ("human_vs_ai", "ai_vs_ai", "human_vs_human", "imported", "self_play")
MEMORY_TABLES = ("matches", "games", "positions", "decisions")
MEMORY_INDEXES = (
    "idx_positions_canonical",
    "idx_positions_rules_ply",
    "idx_decisions_game",
    "idx_decisions_position",
    "idx_decisions_quality",
)
TABLE_COLUMNS = {
    "matches": (
        "match_id", "schema_version", "rules_version", "mode_key", "mode_name",
        "red_team", "blue_team", "source_type", "started_at", "ended_at",
        "status", "record_id",
    ),
    "games": (
        "game_id", "match_id", "game_no", "red_actor_type", "blue_actor_type",
        "red_actor_id", "blue_actor_id", "red_ai_name", "blue_ai_name",
        "red_ai_version", "blue_ai_version", "winner_color",
        "termination_reason", "status", "started_at", "ended_at",
    ),
    "positions": (
        "position_id", "experience_key", "canonical_key", "rules_version",
        "feature_version", "board_json", "current_color", "ply",
        "features_json", "created_at",
    ),
    "decisions": (
        "decision_id", "game_id", "position_id", "ply", "actor_type",
        "actor_color", "actor_id", "ai_name", "ai_version", "candidate_json",
        "search_config_json", "chosen_q", "chosen_r", "result",
        "decision_quality", "training_eligible", "integrity_status",
        "rule_status", "result_status", "validation_reason",
        "validator_version", "created_at",
    ),
}
SUPPORTED_SCHEMA_VERSIONS = tuple(range(MEMORY_SCHEMA_VERSION + 1))
DEFAULT_BUSY_TIMEOUT_MS = 5000

PRAGMAS = (
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode = WAL;",
)

TABLE_DDL = (
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL,
        rules_version INTEGER NOT NULL,
        mode_key INTEGER,
        mode_name TEXT NOT NULL,
        red_team TEXT NOT NULL,
        blue_team TEXT NOT NULL,
        source_type TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL,
        record_id TEXT,
        CHECK(status IN ('active','completed','aborted','invalid'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        match_id TEXT NOT NULL,
        game_no INTEGER NOT NULL CHECK(game_no IN (1,2)),
        red_actor_type TEXT NOT NULL,
        blue_actor_type TEXT NOT NULL,
        red_actor_id TEXT,
        blue_actor_id TEXT,
        red_ai_name TEXT,
        blue_ai_name TEXT,
        red_ai_version TEXT,
        blue_ai_version TEXT,
        winner_color INTEGER CHECK(winner_color IN (1,2) OR winner_color IS NULL),
        termination_reason TEXT,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id),
        UNIQUE(match_id, game_no),
        CHECK(red_actor_type IN ('ai','human')),
        CHECK(blue_actor_type IN ('ai','human')),
        CHECK(status IN ('active','completed','aborted','invalid'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        position_id TEXT PRIMARY KEY,
        experience_key TEXT NOT NULL UNIQUE,
        canonical_key TEXT NOT NULL,
        rules_version INTEGER NOT NULL,
        feature_version INTEGER NOT NULL,
        board_json TEXT NOT NULL,
        current_color INTEGER NOT NULL CHECK(current_color IN (1,2)),
        ply INTEGER NOT NULL CHECK(ply >= 1),
        features_json TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY,
        game_id TEXT NOT NULL,
        position_id TEXT NOT NULL,
        ply INTEGER NOT NULL CHECK(ply >= 1),
        actor_type TEXT NOT NULL,
        actor_color INTEGER NOT NULL CHECK(actor_color IN (1,2)),
        actor_id TEXT,
        ai_name TEXT,
        ai_version TEXT,
        candidate_json TEXT,
        search_config_json TEXT,
        chosen_q INTEGER CHECK(chosen_q BETWEEN 0 AND 10 OR chosen_q IS NULL),
        chosen_r INTEGER CHECK(chosen_r BETWEEN 0 AND 10 OR chosen_r IS NULL),
        result REAL CHECK(result IN (0.0, 0.5, 1.0) OR result IS NULL),
        decision_quality TEXT NOT NULL DEFAULT 'unverified',
        training_eligible INTEGER NOT NULL DEFAULT 0 CHECK(training_eligible IN (0,1)),
        integrity_status TEXT NOT NULL DEFAULT 'unknown',
        rule_status TEXT NOT NULL DEFAULT 'unknown',
        result_status TEXT NOT NULL DEFAULT 'unknown',
        validation_reason TEXT,
        validator_version TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(game_id) REFERENCES games(game_id),
        FOREIGN KEY(position_id) REFERENCES positions(position_id),
        UNIQUE(game_id, ply, actor_type, actor_color),
        CHECK(actor_type IN ('ai','human')),
        CHECK(decision_quality IN ('validated','unverified','rejected')),
        CHECK(integrity_status IN ('valid','invalid','unknown')),
        CHECK(rule_status IN ('valid','invalid','unknown')),
        CHECK(result_status IN ('known','unknown','invalid'))
    );
    """,
)

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_positions_canonical ON positions(canonical_key);",
    "CREATE INDEX IF NOT EXISTS idx_positions_rules_ply ON positions(rules_version, ply);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_game ON decisions(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_position ON decisions(position_id);",
    (
        "CREATE INDEX IF NOT EXISTS idx_decisions_quality "
        "ON decisions(decision_quality, training_eligible);"
    ),
)

SCHEMA_DDL = "\n".join(TABLE_DDL + INDEX_DDL)


def memory_db_path(environ=None):
    """Resolve the database path from env override or the project root."""
    env = os.environ if environ is None else environ
    override = str(env.get("HEX_MEMORY_DB", "")).strip()
    return Path(override).expanduser().resolve() if override else DEFAULT_MEMORY_DB.resolve()


# P0-A records these rules for the later collector; no write path exists yet.
DO_NOT_WRITE_REASONS = (
    "proposal_not_committed",
    "move_undone",
    "game_restarted_before_finish",
    "mode_switched_before_finish",
    "process_interrupted",
    "replay_session",
    "duplicate_decision_id",
    "invalid_board",
    "invalid_record_hash",
    "vision_only_import",
    "move_after_game_over",
    "forced_admin_result_without_normal_move",
)