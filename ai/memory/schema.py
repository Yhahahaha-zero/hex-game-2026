# -*- coding: utf-8 -*-
"""Versioned SQLite schema contract for experience memory.

Importing this module is side-effect free. ExperienceStore owns migrations,
connections, transactions and all database I/O.
"""
from __future__ import annotations

import os
from pathlib import Path


MEMORY_SCHEMA_VERSION = 5
RULES_VERSION = 1
FEATURE_VERSION = 1
COLLECTOR_VERSION = "1.0.0"
RETRIEVAL_VERSION = "1.0.0"
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
CAPTURE_MODES = ("shadow",)
RETRIEVAL_MODES = ("shadow", "assist", "exact", "canonical", "similar")
MEMORY_TABLES = (
    "matches", "games", "positions", "decisions", "retrieval_events",
    "retrieval_outcomes", "memory_jobs", "dataset_versions", "training_runs",
    "trust_models", "active_learning_queue", "opponent_profiles", "league_runs",
    "drift_snapshots",
)
MEMORY_INDEXES = (
    "idx_positions_canonical",
    "idx_positions_rules_ply",
    "idx_decisions_game",
    "idx_decisions_position",
    "idx_decisions_quality",
    "idx_decisions_training",
    "idx_retrieval_query_game",
    "idx_retrieval_applied",
    "idx_retrieval_created",
    "idx_outcomes_game",
    "idx_jobs_status",
    "idx_training_dataset",
    "idx_retrieval_query_ply",
    "idx_active_learning_priority",
    "idx_trust_models_status",
    "idx_league_runs_status",
    "idx_drift_created",
)
TABLE_COLUMNS = {
    "matches": (
        "match_id", "schema_version", "rules_version", "mode_key", "mode_name",
        "red_team", "blue_team", "source_type", "started_at", "ended_at",
        "status", "record_id", "capture_mode", "collector_version",
    ),
    "games": (
        "game_id", "match_id", "game_no", "red_actor_type", "blue_actor_type",
        "red_actor_id", "blue_actor_id", "red_ai_name", "blue_ai_name",
        "red_ai_version", "blue_ai_version", "winner_color",
        "termination_reason", "status", "started_at", "ended_at",
        "record_sha256",
    ),
    "positions": (
        "position_id", "experience_key", "canonical_key", "rules_version",
        "feature_version", "board_json", "current_color", "ply",
        "features_json", "created_at",
    ),
    "retrieval_events": (
        "retrieval_event_id", "query_position_id", "query_game_id",
        "query_match_id", "retrieval_mode", "top_k",
        "selected_memory_decision_id", "similarity", "memory_q", "memory_r",
        "base_q", "base_r", "shadow_q", "shadow_r", "final_q", "final_r",
        "confidence", "applied", "feature_version", "retrieval_version",
        "elapsed_ms", "evidence_json", "created_at", "query_ply",
        "selection_probability", "exploration_epsilon", "policy_version",
    ),
    "retrieval_outcomes": (
        "retrieval_event_id", "game_id", "decision_id", "applied",
        "decision_result", "game_result", "retrieval_reward", "similarity",
        "confidence", "retrieval_mode", "created_at",
    ),
    "memory_jobs": (
        "job_id", "job_type", "target_id", "status", "attempts",
        "last_error", "created_at", "started_at", "finished_at",
    ),
    "dataset_versions": (
        "dataset_id", "dataset_dir", "manifest_sha256", "samples_sha256",
        "feature_version", "encoding_version", "sample_count",
        "split_counts_json", "source_match_count", "created_at",
    ),
    "training_runs": (
        "training_run_id", "dataset_id", "parent_model_sha256",
        "candidate_model_sha256", "status", "metrics_json",
        "started_at", "finished_at",
    ),
    "trust_models": (
        "trust_model_id", "model_path", "sha256", "feature_version",
        "metrics_json", "status", "created_at",
    ),
    "active_learning_queue": (
        "candidate_id", "position_id", "decision_id", "priority",
        "reason_json", "selected", "dataset_id", "created_at",
    ),
    "opponent_profiles": (
        "opponent_id", "ai_name", "ai_version", "model_sha256", "elo",
        "metrics_json", "created_at",
    ),
    "league_runs": (
        "league_run_id", "candidate_sha256", "status", "metrics_json",
        "started_at", "finished_at",
    ),
    "drift_snapshots": (
        "snapshot_id", "window_start", "window_end", "metrics_json",
        "alerts_json", "created_at",
    ),
    "decisions": (
        "decision_id", "game_id", "position_id", "ply", "actor_type",
        "actor_color", "actor_id", "ai_name", "ai_version", "candidate_json",
        "search_config_json", "chosen_q", "chosen_r", "result",
        "decision_quality", "training_eligible", "integrity_status",
        "rule_status", "result_status", "validation_reason",
        "validator_version", "created_at", "model_sha256",
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
        capture_mode TEXT NOT NULL DEFAULT 'shadow'
            CHECK(capture_mode IN ('shadow')),
        collector_version TEXT NOT NULL DEFAULT '1.0.0',
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
        record_sha256 TEXT,
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
        model_sha256 TEXT,
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
    """
    CREATE TABLE IF NOT EXISTS retrieval_events (
        retrieval_event_id TEXT PRIMARY KEY,
        query_position_id TEXT NOT NULL,
        query_game_id TEXT,
        query_match_id TEXT,
        retrieval_mode TEXT NOT NULL,
        top_k INTEGER NOT NULL CHECK(top_k >= 1),
        selected_memory_decision_id TEXT,
        similarity REAL,
        memory_q INTEGER CHECK(memory_q BETWEEN 0 AND 10 OR memory_q IS NULL),
        memory_r INTEGER CHECK(memory_r BETWEEN 0 AND 10 OR memory_r IS NULL),
        base_q INTEGER CHECK(base_q BETWEEN 0 AND 10 OR base_q IS NULL),
        base_r INTEGER CHECK(base_r BETWEEN 0 AND 10 OR base_r IS NULL),
        shadow_q INTEGER CHECK(shadow_q BETWEEN 0 AND 10 OR shadow_q IS NULL),
        shadow_r INTEGER CHECK(shadow_r BETWEEN 0 AND 10 OR shadow_r IS NULL),
        final_q INTEGER CHECK(final_q BETWEEN 0 AND 10 OR final_q IS NULL),
        final_r INTEGER CHECK(final_r BETWEEN 0 AND 10 OR final_r IS NULL),
        confidence REAL,
        applied INTEGER NOT NULL DEFAULT 0 CHECK(applied IN (0,1)),
        feature_version INTEGER NOT NULL,
        retrieval_version TEXT NOT NULL,
        elapsed_ms REAL NOT NULL,
        evidence_json TEXT,
        created_at TEXT NOT NULL,
        query_ply INTEGER,
        selection_probability REAL,
        exploration_epsilon REAL,
        policy_version TEXT,
        FOREIGN KEY(selected_memory_decision_id)
            REFERENCES decisions(decision_id),
        CHECK(retrieval_mode IN (
            'shadow','assist','exact','canonical','similar'
        ))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_outcomes (
        retrieval_event_id TEXT PRIMARY KEY,
        game_id TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        applied INTEGER NOT NULL CHECK(applied IN (0,1)),
        decision_result REAL CHECK(decision_result IN (0.0,0.5,1.0)),
        game_result REAL CHECK(game_result IN (0.0,0.5,1.0)),
        retrieval_reward REAL CHECK(retrieval_reward IN (-1.0,0.0,1.0)),
        similarity REAL,
        confidence REAL,
        retrieval_mode TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(retrieval_event_id) REFERENCES retrieval_events(retrieval_event_id),
        FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        UNIQUE(job_type, target_id),
        CHECK(status IN ('pending','running','completed','failed','cancelled'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_versions (
        dataset_id TEXT PRIMARY KEY,
        dataset_dir TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        samples_sha256 TEXT NOT NULL,
        feature_version INTEGER NOT NULL,
        encoding_version TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        split_counts_json TEXT NOT NULL,
        source_match_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        training_run_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        parent_model_sha256 TEXT,
        candidate_model_sha256 TEXT,
        status TEXT NOT NULL,
        metrics_json TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        FOREIGN KEY(dataset_id) REFERENCES dataset_versions(dataset_id),
        CHECK(status IN ('running','completed','failed','cancelled'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trust_models (
        trust_model_id TEXT PRIMARY KEY,
        model_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        feature_version INTEGER NOT NULL,
        metrics_json TEXT,
        status TEXT NOT NULL CHECK(status IN ('candidate','selected','rejected','rolled_back')),
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS active_learning_queue (
        candidate_id TEXT PRIMARY KEY,
        position_id TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        priority REAL NOT NULL,
        reason_json TEXT,
        selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
        dataset_id TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS opponent_profiles (
        opponent_id TEXT PRIMARY KEY,
        ai_name TEXT NOT NULL,
        ai_version TEXT,
        model_sha256 TEXT,
        elo REAL,
        metrics_json TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS league_runs (
        league_run_id TEXT PRIMARY KEY,
        candidate_sha256 TEXT,
        status TEXT NOT NULL,
        metrics_json TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS drift_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        window_start TEXT NOT NULL,
        window_end TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        alerts_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
)

MIGRATION_DDL = {
    1: (
        (
            "ALTER TABLE matches ADD COLUMN capture_mode TEXT NOT NULL "
            "DEFAULT 'shadow';"
        ),
        (
            "ALTER TABLE matches ADD COLUMN collector_version TEXT NOT NULL "
            "DEFAULT '1.0.0';"
        ),
        "ALTER TABLE games ADD COLUMN record_sha256 TEXT;",
        "ALTER TABLE decisions ADD COLUMN model_sha256 TEXT;",
    ),
    2: (
        """
        CREATE TABLE IF NOT EXISTS retrieval_events (
            retrieval_event_id TEXT PRIMARY KEY,
            query_position_id TEXT NOT NULL,
            query_game_id TEXT,
            query_match_id TEXT,
            retrieval_mode TEXT NOT NULL,
            top_k INTEGER NOT NULL CHECK(top_k >= 1),
            selected_memory_decision_id TEXT,
            similarity REAL,
            memory_q INTEGER CHECK(memory_q BETWEEN 0 AND 10 OR memory_q IS NULL),
            memory_r INTEGER CHECK(memory_r BETWEEN 0 AND 10 OR memory_r IS NULL),
            base_q INTEGER CHECK(base_q BETWEEN 0 AND 10 OR base_q IS NULL),
            base_r INTEGER CHECK(base_r BETWEEN 0 AND 10 OR base_r IS NULL),
            shadow_q INTEGER CHECK(shadow_q BETWEEN 0 AND 10 OR shadow_q IS NULL),
            shadow_r INTEGER CHECK(shadow_r BETWEEN 0 AND 10 OR shadow_r IS NULL),
            final_q INTEGER CHECK(final_q BETWEEN 0 AND 10 OR final_q IS NULL),
            final_r INTEGER CHECK(final_r BETWEEN 0 AND 10 OR final_r IS NULL),
            confidence REAL,
            applied INTEGER NOT NULL DEFAULT 0 CHECK(applied IN (0,1)),
            feature_version INTEGER NOT NULL,
            retrieval_version TEXT NOT NULL,
            elapsed_ms REAL NOT NULL,
            evidence_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(selected_memory_decision_id)
                REFERENCES decisions(decision_id),
            CHECK(retrieval_mode IN (
                'shadow','assist','exact','canonical','similar'
            ))
        );
        """,
        (
            "CREATE INDEX IF NOT EXISTS idx_decisions_training "
            "ON decisions(training_eligible, decision_quality);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_retrieval_query_game "
            "ON retrieval_events(query_game_id, created_at);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_retrieval_applied "
            "ON retrieval_events(applied, retrieval_mode);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_retrieval_created "
            "ON retrieval_events(created_at);"
        ),
    ),
    3: (
        "ALTER TABLE retrieval_events ADD COLUMN query_ply INTEGER;",
        """
        CREATE TABLE IF NOT EXISTS retrieval_outcomes (
            retrieval_event_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            applied INTEGER NOT NULL CHECK(applied IN (0,1)),
            decision_result REAL CHECK(decision_result IN (0.0,0.5,1.0)),
            game_result REAL CHECK(game_result IN (0.0,0.5,1.0)),
            retrieval_reward REAL CHECK(retrieval_reward IN (-1.0,0.0,1.0)),
            similarity REAL,
            confidence REAL,
            retrieval_mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(retrieval_event_id) REFERENCES retrieval_events(retrieval_event_id),
            FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            UNIQUE(job_type, target_id),
            CHECK(status IN ('pending','running','completed','failed','cancelled'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            dataset_id TEXT PRIMARY KEY,
            dataset_dir TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            samples_sha256 TEXT NOT NULL,
            feature_version INTEGER NOT NULL,
            encoding_version TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            split_counts_json TEXT NOT NULL,
            source_match_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            training_run_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            parent_model_sha256 TEXT,
            candidate_model_sha256 TEXT,
            status TEXT NOT NULL,
            metrics_json TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY(dataset_id) REFERENCES dataset_versions(dataset_id),
            CHECK(status IN ('running','completed','failed','cancelled'))
        );
        """,
        (
            "CREATE INDEX IF NOT EXISTS idx_outcomes_game "
            "ON retrieval_outcomes(game_id);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_jobs_status "
            "ON memory_jobs(status, job_type);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_training_dataset "
            "ON training_runs(dataset_id, status);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_retrieval_query_ply "
            "ON retrieval_events(query_game_id, query_ply);"
        ),
    ),
    4: (
        "ALTER TABLE retrieval_events ADD COLUMN selection_probability REAL;",
        "ALTER TABLE retrieval_events ADD COLUMN exploration_epsilon REAL;",
        "ALTER TABLE retrieval_events ADD COLUMN policy_version TEXT;",
        """
        CREATE TABLE IF NOT EXISTS trust_models (
            trust_model_id TEXT PRIMARY KEY,
            model_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            feature_version INTEGER NOT NULL,
            metrics_json TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS active_learning_queue (
            candidate_id TEXT PRIMARY KEY,
            position_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            priority REAL NOT NULL,
            reason_json TEXT,
            selected INTEGER NOT NULL DEFAULT 0,
            dataset_id TEXT,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS opponent_profiles (
            opponent_id TEXT PRIMARY KEY,
            ai_name TEXT NOT NULL,
            ai_version TEXT,
            model_sha256 TEXT,
            elo REAL,
            metrics_json TEXT,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS league_runs (
            league_run_id TEXT PRIMARY KEY,
            candidate_sha256 TEXT,
            status TEXT NOT NULL,
            metrics_json TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS drift_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            alerts_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
        (
            "CREATE INDEX IF NOT EXISTS idx_active_learning_priority "
            "ON active_learning_queue(selected, priority DESC);"
        ),
        "CREATE INDEX IF NOT EXISTS idx_trust_models_status ON trust_models(status, created_at);",
        "CREATE INDEX IF NOT EXISTS idx_league_runs_status ON league_runs(status, started_at);",
        "CREATE INDEX IF NOT EXISTS idx_drift_created ON drift_snapshots(created_at);",
    ),
}

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_positions_canonical ON positions(canonical_key);",
    "CREATE INDEX IF NOT EXISTS idx_positions_rules_ply ON positions(rules_version, ply);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_game ON decisions(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_position ON decisions(position_id);",
    (
        "CREATE INDEX IF NOT EXISTS idx_decisions_quality "
        "ON decisions(decision_quality, training_eligible);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_decisions_training "
        "ON decisions(training_eligible, decision_quality);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_retrieval_query_game "
        "ON retrieval_events(query_game_id, created_at);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_retrieval_applied "
        "ON retrieval_events(applied, retrieval_mode);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_retrieval_created ON retrieval_events(created_at);",
    (
        "CREATE INDEX IF NOT EXISTS idx_active_learning_priority "
        "ON active_learning_queue(selected, priority DESC);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_trust_models_status ON trust_models(status, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_league_runs_status ON league_runs(status, started_at);",
    "CREATE INDEX IF NOT EXISTS idx_drift_created ON drift_snapshots(created_at);",
    (
        "CREATE INDEX IF NOT EXISTS idx_outcomes_game "
        "ON retrieval_outcomes(game_id);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_jobs_status "
        "ON memory_jobs(status, job_type);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_training_dataset "
        "ON training_runs(dataset_id, status);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_retrieval_query_ply "
        "ON retrieval_events(query_game_id, query_ply);"
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