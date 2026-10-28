# -*- coding: utf-8 -*-
"""Build a training-only experience database for memory ablation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import file_sha256, load_competition_settings
from ai.memory.dataset import split_match_ids
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "artifacts" / "memory" / "ablation"
SPLIT_SCHEMA_VERSION = 1


def _atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
    return path


def _copy_database(source, destination):
    source_uri = f"file:{Path(source).resolve().as_posix()}?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _clean_training_database(path, training_match_ids):
    training_match_ids = sorted(str(value) for value in training_match_ids)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TEMP TABLE keep_matches(match_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TEMP TABLE keep_games(game_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TEMP TABLE keep_decisions(decision_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TEMP TABLE keep_positions(position_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO keep_matches(match_id) VALUES (?)",
            [(value,) for value in training_match_ids],
        )
        connection.execute(
            "INSERT INTO keep_games(game_id) "
            "SELECT game_id FROM games WHERE match_id IN (SELECT match_id FROM keep_matches)"
        )
        connection.execute(
            "INSERT INTO keep_decisions(decision_id) "
            "SELECT decision_id FROM decisions "
            "WHERE game_id IN (SELECT game_id FROM keep_games)"
        )
        connection.execute(
            "INSERT INTO keep_positions(position_id) "
            "SELECT DISTINCT position_id FROM decisions "
            "WHERE decision_id IN (SELECT decision_id FROM keep_decisions)"
        )
        connection.execute(
            "DELETE FROM retrieval_outcomes "
            "WHERE game_id NOT IN (SELECT game_id FROM keep_games) "
            "OR decision_id NOT IN (SELECT decision_id FROM keep_decisions) "
            "OR retrieval_event_id NOT IN ("
            "  SELECT retrieval_event_id FROM retrieval_events "
            "  WHERE query_game_id IN (SELECT game_id FROM keep_games) "
            "  AND query_match_id IN (SELECT match_id FROM keep_matches) "
            "  AND (selected_memory_decision_id IS NULL "
            "       OR selected_memory_decision_id IN "
            "          (SELECT decision_id FROM keep_decisions))"
            ")"
        )
        connection.execute(
            "DELETE FROM retrieval_events "
            "WHERE query_game_id NOT IN (SELECT game_id FROM keep_games) "
            "OR query_match_id NOT IN (SELECT match_id FROM keep_matches) "
            "OR (selected_memory_decision_id IS NOT NULL "
            "    AND selected_memory_decision_id NOT IN "
            "        (SELECT decision_id FROM keep_decisions))"
        )
        connection.execute("DELETE FROM active_learning_queue")
        connection.execute("DELETE FROM memory_jobs")
        connection.execute("DELETE FROM training_runs")
        connection.execute("DELETE FROM dataset_versions")
        connection.execute("DELETE FROM trust_models")
        connection.execute("DELETE FROM opponent_profiles")
        connection.execute("DELETE FROM league_runs")
        connection.execute("DELETE FROM drift_snapshots")
        connection.execute(
            "DELETE FROM decisions "
            "WHERE decision_id NOT IN (SELECT decision_id FROM keep_decisions)"
        )
        connection.execute(
            "DELETE FROM positions "
            "WHERE position_id NOT IN (SELECT position_id FROM keep_positions)"
        )
        connection.execute(
            "DELETE FROM games WHERE game_id NOT IN (SELECT game_id FROM keep_games)"
        )
        connection.execute(
            "DELETE FROM matches "
            "WHERE match_id NOT IN (SELECT match_id FROM keep_matches)"
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok":
            raise ValueError(f"training database integrity_check failed: {integrity}")
        if foreign_keys:
            raise ValueError("training database has foreign-key violations")
        connection.execute("VACUUM")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _training_feature_count(path, training_match_ids):
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TEMP TABLE selected_matches(match_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO selected_matches(match_id) VALUES (?)",
            [(str(value),) for value in training_match_ids],
        )
        return int(connection.execute(
            "SELECT COUNT(*) FROM decisions d "
            "JOIN games g ON g.game_id = d.game_id "
            "JOIN positions p ON p.position_id = d.position_id "
            "WHERE g.match_id IN (SELECT match_id FROM selected_matches) "
            "AND d.training_eligible = 1 "
            "AND d.decision_quality = 'validated' "
            "AND p.features_json IS NOT NULL"
        ).fetchone()[0])
    finally:
        connection.close()

def _training_matches(store):
    rows = store.query_training_samples(limit=1000000)
    return sorted({str(row["match_id"]) for row in rows})


def prepare_training_database(*, db_path, output_dir=DEFAULT_OUTPUT_DIR,
                              seed=20261025,
                              ratios=(0.70, 0.15, 0.15),
                              dataset_manifest=None, force=False):
    source = Path(db_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"memory database not found: {source}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "training_only.sqlite3"
    provenance_path = output_dir / "training_provenance.json"
    split_path = output_dir / "split_manifest.json"
    if not force:
        existing = [
            path for path in (database_path, provenance_path, split_path)
            if path.exists()
        ]
        if existing:
            raise ValueError(
                "output files already exist; use --force to replace: "
                + ", ".join(str(path) for path in existing)
            )

    source_sha256 = file_sha256(source)
    with ExperienceStore(source) as store:
        source_match_ids = _training_matches(store)
        source_counts = store.counts()
    if len(source_match_ids) < 3:
        raise ValueError(
            "at least three matches with validated training samples are required"
        )
    splits = split_match_ids(source_match_ids, seed=seed, ratios=ratios)
    training_ids = list(splits["train"])
    validation_ids = list(splits["validation"])
    test_ids = list(splits["test"])
    if not training_ids or not validation_ids or not test_ids:
        raise ValueError("all three match groups must be non-empty")
    retrievable_training_decisions = _training_feature_count(source, training_ids)
    if retrievable_training_decisions <= 0:
        raise ValueError(
            "training split has no validated decisions with position features; "
            "run tools/build_memory_features.py first"
        )
    heldout_ids = set(validation_ids) | set(test_ids)

    descriptor, temp_name = tempfile.mkstemp(
        dir=output_dir, prefix=".training_only.", suffix=".sqlite3"
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        _copy_database(source, temp_path)
        _clean_training_database(temp_path, training_ids)
        with ExperienceStore(temp_path) as store:
            output_counts = store.counts()
            output_match_ids = {
                str(row["match_id"]) for row in store.list_matches(limit=1000000)
            }
        leaked = output_match_ids & heldout_ids
        if leaked:
            raise ValueError(f"held-out matches remain in output: {sorted(leaked)!r}")
        if output_match_ids != set(training_ids):
            raise ValueError("output matches do not exactly match the training split")
        os.replace(temp_path, database_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    output_sha256 = file_sha256(database_path)
    manifest_reference = None
    if dataset_manifest:
        manifest_path = Path(dataset_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(f"dataset manifest not found: {manifest_path}")
        manifest_reference = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        }
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    provenance = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": created_at,
        "source_db": str(source),
        "source_db_sha256": source_sha256,
        "training_db": str(database_path),
        "db_sha256": output_sha256,
        "split_seed": int(seed),
        "split_ratios": [float(value) for value in ratios],
        "training_match_ids": training_ids,
        "validation_match_ids": validation_ids,
        "test_match_ids": test_ids,
        "dataset_manifest": manifest_reference,
        "source_counts": source_counts,
        "training_counts": output_counts,
        "retrievable_training_decisions": retrievable_training_decisions,
    }
    split_manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": created_at,
        "seed": int(seed),
        "ratios": [float(value) for value in ratios],
        "source_match_count": len(source_match_ids),
        "splits": {
            "train": training_ids,
            "validation": validation_ids,
            "test": test_ids,
        },
        "training_only_database": database_path.name,
        "training_only_sha256": output_sha256,
    }
    _atomic_json(provenance_path, provenance)
    _atomic_json(split_path, split_manifest)
    if file_sha256(source) != source_sha256:
        raise RuntimeError("source memory database changed during preparation")
    return {
        "database": database_path,
        "provenance": provenance_path,
        "split_manifest": split_path,
        "document": provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare a training-only memory database and provenance"
    )
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_competition_settings(args.profile)
        seed = settings.random_seed if args.seed is None else args.seed
        result = prepare_training_database(
            db_path=args.db,
            output_dir=args.output_dir,
            seed=seed,
            ratios=(
                args.train_ratio,
                args.validation_ratio,
                args.test_ratio,
            ),
            dataset_manifest=args.dataset_manifest,
            force=args.force,
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    document = result["document"]
    print(json.dumps({
        "database": str(result["database"]),
        "provenance": str(result["provenance"]),
        "split_manifest": str(result["split_manifest"]),
        "db_sha256": document["db_sha256"],
        "train_matches": len(document["training_match_ids"]),
        "validation_matches": len(document["validation_match_ids"]),
        "test_matches": len(document["test_match_ids"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())