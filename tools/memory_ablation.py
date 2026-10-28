# -*- coding: utf-8 -*-
"""Run isolated memory off/shadow/assist ablation benchmarks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import (
    load_competition_settings,
    file_sha256,
    reproducibility_metadata,
    require_clean_worktree,
)
from ai.memory.store import ExperienceStore
from tools import benchmark_ai


DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "evaluation" / "memory"
MODES = ("off", "shadow", "assist")


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


@contextmanager
def _memory_environment(db_path, mode):
    names = ("HEX_MEMORY_DB", "HEX_MEMORY_RETRIEVAL")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["HEX_MEMORY_DB"] = str(Path(db_path).resolve())
    os.environ["HEX_MEMORY_RETRIEVAL"] = mode
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def validate_training_provenance(db_path, provenance_path, *, allow_unverified=False):
    db_path = Path(db_path)
    if not db_path.is_file():
        raise ValueError(f"memory database not found: {db_path}")
    actual_sha256 = file_sha256(db_path)
    if provenance_path is None:
        if not allow_unverified:
            raise ValueError(
                "training provenance is required; use --allow-unverified-db only for smoke tests"
            )
        return {
            "verified": False,
            "db_sha256": actual_sha256,
            "warning": "unverified database; not valid evidence for a formal claim",
        }
    provenance_path = Path(provenance_path)
    if not provenance_path.is_file():
        raise ValueError(f"training provenance not found: {provenance_path}")
    try:
        document = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError("training provenance cannot be parsed") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported training provenance schema")
    expected_sha = str(document.get("db_sha256", "")).lower()
    if expected_sha != actual_sha256.lower():
        raise ValueError("memory database SHA256 does not match provenance")
    training_ids = {str(value) for value in document.get("training_match_ids", [])}
    validation_ids = {
        str(value) for value in document.get("validation_match_ids", [])
    }
    test_ids = {str(value) for value in document.get("test_match_ids", [])}
    if not training_ids:
        raise ValueError("provenance requires training_match_ids")
    if not test_ids:
        raise ValueError("provenance requires test_match_ids")
    heldout_ids = validation_ids | test_ids
    overlap = training_ids & heldout_ids
    if overlap:
        raise ValueError(f"training/held-out match leakage: {sorted(overlap)!r}")
    if validation_ids & test_ids:
        raise ValueError("validation/test match leakage")
    with ExperienceStore(db_path) as store:
        database_ids = {str(row["match_id"]) for row in store.list_matches(limit=1000000)}
    leaked = database_ids & heldout_ids
    if leaked:
        raise ValueError(f"memory database contains held-out matches: {sorted(leaked)!r}")
    missing = training_ids - database_ids
    if missing:
        raise ValueError(f"memory database is missing training matches: {sorted(missing)!r}")
    return {
        "verified": True,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": file_sha256(provenance_path),
        "db_sha256": actual_sha256,
        "training_match_count": len(training_ids),
        "validation_match_count": len(validation_ids),
        "test_match_count": len(test_ids),
        "training_match_ids": sorted(training_ids),
        "validation_match_ids": sorted(validation_ids),
        "test_match_ids": sorted(test_ids),
        "dataset_manifest": document.get("dataset_manifest"),
    }


def _memory_evidence(store, *, event_ids_before, ai_moves):
    events = store.list_retrieval_events(limit=1000000)
    new_events = [
        event for event in events
        if event["retrieval_event_id"] not in event_ids_before
    ]
    delta = len(new_events)
    applied = sum(1 for event in new_events if int(event.get("applied", 0)))
    return {
        "events": delta,
        "applied": applied,
        "coverage": round(delta / ai_moves, 6) if ai_moves else 0.0,
        "applied_rate": round(applied / delta, 6) if delta else 0.0,
    }


def _game_key(row):
    return (
        int(row.get("repeat", 1)),
        int(row.get("pair", 0)),
        str(row.get("orientation", "")),
        int(row.get("seed", 0)),
    )


def _game_score(row):
    winner = row.get("winner_ai")
    if winner == "A":
        return 1.0
    if winner == "B":
        return 0.0
    return 0.5


def _percentile(values, probability):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_comparison(off_games, mode_games, *, resamples=2000,
                                minimum_pairs=20, seed=20261025):
    def completed_by_key(rows):
        values = {}
        for row in rows:
            if row.get("status") != "completed":
                continue
            key = _game_key(row)
            if key in values:
                raise ValueError(f"duplicate game pairing key: {key!r}")
            values[key] = _game_score(row)
        return values

    off = completed_by_key(off_games)
    mode = completed_by_key(mode_games)
    common = sorted(set(off) & set(mode))
    differences = [mode[key] - off[key] for key in common]
    count = len(differences)
    difference_pp = (
        round(sum(differences) / count * 100.0, 4) if count else 0.0
    )
    if count:
        rng = random.Random(int(seed))
        bootstrap = []
        for _ in range(max(1, int(resamples))):
            total = sum(rng.choice(differences) for _ in range(count))
            bootstrap.append(total / count * 100.0)
        ci_low = round(_percentile(bootstrap, 0.025), 4)
        ci_high = round(_percentile(bootstrap, 0.975), 4)
    else:
        ci_low = 0.0
        ci_high = 0.0
    if count < int(minimum_pairs):
        status = "insufficient_pairs"
    elif ci_low > 0.0:
        status = "significant_improvement"
    elif ci_high < 0.0:
        status = "significant_regression"
    else:
        status = "no_significant_difference"
    return {
        "method": "paired_bootstrap",
        "resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "minimum_pairs": int(minimum_pairs),
        "games_compared": count,
        "unmatched_off_games": len(set(off) - set(mode)),
        "unmatched_mode_games": len(set(mode) - set(off)),
        "score_mean_off": round(sum(off.values()) / len(off), 6) if off else 0.0,
        "score_mean_mode": round(sum(mode.values()) / len(mode), 6) if mode else 0.0,
        "difference_pp": difference_pp,
        "ci95_low_pp": ci_low,
        "ci95_high_pp": ci_high,
        "mode_only_wins": sum(1 for value in differences if value > 0),
        "off_only_wins": sum(1 for value in differences if value < 0),
        "ties": sum(1 for value in differences if value == 0),
        "status": status,
    }

def run_ablation(*, db_path, provenance_path=None, base_ai="competition",
                 opponent="basic", games=None, seed=None, output_dir=DEFAULT_OUTPUT_DIR,
                 name=None, quiet=False, move_time_limit_ms=None,
                 competition_settings=None, allow_unverified_db=False,
                 require_clean=False, bootstrap_resamples=2000,
                 minimum_pairs=200, bootstrap_seed=None,
                 apply_profile=False, profile_path=None):
    if apply_profile and not require_clean:
        raise ValueError("--apply-profile requires --require-clean")
    if require_clean:
        require_clean_worktree()
    settings = competition_settings or load_competition_settings()
    provenance = validate_training_provenance(
        db_path, provenance_path, allow_unverified=allow_unverified_db
    )
    games = settings.tournament_games if games is None else int(games)
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even integer")
    seed = settings.random_seed if seed is None else int(seed)
    bootstrap_seed = seed if bootstrap_seed is None else int(bootstrap_seed)
    if move_time_limit_ms is None:
        move_time_limit_ms = settings.hard_move_limit_ms
    move_time_limit_ms = float(move_time_limit_ms)

    output_dir = Path(output_dir)
    run_name = name or datetime.now().strftime("memory_ablation_%Y%m%d_%H%M%S")
    run_root = output_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    off_settings = replace(settings, memory_mode="off")
    profile_snapshot_path = run_root / "competition_profile.snapshot.json"
    profile_source_path = Path(profile_path or settings.profile_path).resolve()
    profile_document = json.loads(
        profile_source_path.read_text(encoding="utf-8-sig")
    )
    _atomic_json(profile_snapshot_path, profile_document)
    original_sha256 = file_sha256(db_path)
    results = {}
    documents = {}
    evidence = {}
    memory_configs = {}
    run_paths = {}

    for mode in MODES:
        mode_dir = run_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        isolated_db = mode_dir / "memory.sqlite3"
        _copy_database(db_path, isolated_db)
        with ExperienceStore(isolated_db) as store:
            event_ids_before = {
                row["retrieval_event_id"]
                for row in store.list_retrieval_events(limit=1000000)
            }
        ai_a_spec = base_ai if mode == "off" else f"memory-{mode}:{base_ai}"
        with _memory_environment(isolated_db, mode):
            from ai.memory.memory_ai import MemoryRetrievalSettings
            memory_configs[mode] = asdict(MemoryRetrievalSettings.from_env())
            document, paths = benchmark_ai.run_benchmark(
                ai_a_spec,
                opponent,
                games=games,
                repeats=1,
                seed=seed,
                output_dir=mode_dir,
                name=mode,
                quiet=quiet,
                move_time_limit_ms=move_time_limit_ms,
                competition_settings=off_settings,
            )
        ai_moves = sum(int(game.get("a_move_count", 0)) for game in document["games"])
        with ExperienceStore(isolated_db) as store:
            evidence[mode] = _memory_evidence(
                store, event_ids_before=event_ids_before, ai_moves=ai_moves
            )
        results[mode] = document["summary"]
        documents[mode] = document
        run_paths[mode] = {key: str(value) for key, value in paths.items()}

    if file_sha256(db_path) != original_sha256:
        raise RuntimeError("source memory database changed during ablation")

    baseline = results["off"]
    paired_comparisons = {
        mode: paired_bootstrap_comparison(
            documents["off"]["games"],
            documents[mode]["games"],
            resamples=bootstrap_resamples,
            minimum_pairs=minimum_pairs,
            seed=bootstrap_seed,
        )
        for mode in ("shadow", "assist")
    }
    deltas = {}
    for mode in ("shadow", "assist"):
        deltas[mode] = {
            key: round(float(results[mode].get(key, 0.0)) - float(baseline.get(key, 0.0)), 4)
            for key in (
                "ai_a_win_rate",
                "ai_a_red_win_rate",
                "ai_a_blue_win_rate",
                "ai_a_average_move_ms",
                "ai_a_p95_move_ms",
                "ai_a_p99_move_ms",
                "move_timeout_rate",
                "errors",
            )
        }
        deltas[mode]["memory_events"] = evidence[mode]["events"]
        deltas[mode]["memory_applied"] = evidence[mode]["applied"]
    assist_status = paired_comparisons["assist"]["status"]
    if results["assist"]["errors"] != 0:
        recommendation_status = "invalid_result"
    elif assist_status == "significant_improvement":
        recommendation_status = "assist_supported"
    elif assist_status == "significant_regression":
        recommendation_status = "off_recommended"
    elif assist_status == "insufficient_pairs":
        recommendation_status = "insufficient_evidence"
    else:
        recommendation_status = "no_stable_evidence"
    recommendation = {
        "status": recommendation_status,
        "assist_paired_status": assist_status,
        "paired_difference_pp": paired_comparisons["assist"]["difference_pp"],
        "paired_ci95_pp": [
            paired_comparisons["assist"]["ci95_low_pp"],
            paired_comparisons["assist"]["ci95_high_pp"],
        ],
        "note": "Only paired CI evidence plus leakage review supports a formal claim.",
    }
    profile_update = {
        "applied": False,
        "profile_file": str(profile_source_path),
        "profile_snapshot": str(profile_snapshot_path),
    }
    payload = {
        "ablation_schema_version": 2,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "base_ai": base_ai,
            "opponent": opponent,
            "games": games,
            "seed": seed,
            "move_time_limit_ms": move_time_limit_ms,
            "modes": list(MODES),
            "bootstrap_resamples": int(bootstrap_resamples),
            "minimum_pairs": int(minimum_pairs),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "provenance": provenance,
        "reproducibility": reproducibility_metadata(settings),
        "results": results,
        "memory_configs": memory_configs,
        "memory_evidence": evidence,
        "paired_comparisons": paired_comparisons,
        "deltas_vs_off": deltas,
        "recommendation": recommendation,
        "run_paths": run_paths,
    }
    json_path = run_root / "ablation.json"
    csv_path = run_root / "ablation_summary.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "mode", "games_completed", "errors", "ai_a_win_rate",
        "ai_a_red_win_rate", "ai_a_blue_win_rate", "ai_a_average_move_ms",
        "ai_a_p95_move_ms", "ai_a_p99_move_ms", "move_timeout_rate",
        "memory_events", "memory_applied", "memory_coverage",
        "paired_games", "paired_delta_pp", "paired_ci95_low_pp",
        "paired_ci95_high_pp", "paired_status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            summary = results[mode]
            writer.writerow({
                "mode": mode,
                **{key: summary.get(key, 0) for key in fields if key in summary},
                "memory_events": evidence[mode]["events"],
                "memory_applied": evidence[mode]["applied"],
                "memory_coverage": evidence[mode]["coverage"],
                "paired_games": (
                    paired_comparisons[mode]["games_compared"]
                    if mode in paired_comparisons else ""
                ),
                "paired_delta_pp": (
                    paired_comparisons[mode]["difference_pp"]
                    if mode in paired_comparisons else ""
                ),
                "paired_ci95_low_pp": (
                    paired_comparisons[mode]["ci95_low_pp"]
                    if mode in paired_comparisons else ""
                ),
                "paired_ci95_high_pp": (
                    paired_comparisons[mode]["ci95_high_pp"]
                    if mode in paired_comparisons else ""
                ),
                "paired_status": (
                    paired_comparisons[mode]["status"]
                    if mode in paired_comparisons else "baseline"
                ),
            })
    if apply_profile:
        if recommendation_status in ("insufficient_evidence", "invalid_result"):
            raise ValueError(
                "memory result is not sufficient to update competition_profile"
            )
        if file_sha256(profile_source_path) != settings.profile_sha256:
            raise RuntimeError("competition profile changed during ablation")
        selected_mode = "assist" if recommendation_status == "assist_supported" else "off"
        profile_document["memory_mode"] = selected_mode
        profile_document["memory_selection"] = {
            "status": recommendation_status,
            "paired_status": assist_status,
            "difference_pp": paired_comparisons["assist"]["difference_pp"],
            "ci95_low_pp": paired_comparisons["assist"]["ci95_low_pp"],
            "ci95_high_pp": paired_comparisons["assist"]["ci95_high_pp"],
            "ablation_file": os.path.relpath(
                json_path, PROJECT_ROOT.parent
            ).replace("\\", "/"),
            "ablation_sha256": file_sha256(json_path),
            "profile_snapshot": os.path.relpath(
                profile_snapshot_path, PROJECT_ROOT.parent
            ).replace("\\", "/"),
            "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _atomic_json(profile_source_path, profile_document)
        load_competition_settings(profile_source_path)
        profile_update.update({
            "applied": True,
            "memory_mode": selected_mode,
            "selected_at": profile_document["memory_selection"]["selected_at"],
        })
        payload["profile_update"] = profile_update
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload, {"json": json_path, "summary_csv": csv_path, "root": run_root}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated memory ablation benchmark")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--db", required=True, help="training-only memory database")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--allow-unverified-db", action="store_true")
    parser.add_argument("--base-ai", default="competition")
    parser.add_argument("--opponent", default="basic")
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--move-time-limit-ms", type=float, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--minimum-pairs", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--require-clean", action="store_true",
                        help="fail when the Git worktree is dirty")
    parser.add_argument("--apply-profile", action="store_true",
                        help="write the conservative memory mode into competition_profile")
    args = parser.parse_args(argv)
    try:
        settings = load_competition_settings(args.profile)
        payload, paths = run_ablation(
            db_path=args.db,
            provenance_path=args.provenance,
            base_ai=args.base_ai,
            opponent=args.opponent,
            games=args.games,
            seed=args.seed,
            output_dir=args.output_dir,
            name=args.name,
            quiet=args.quiet,
            move_time_limit_ms=args.move_time_limit_ms,
            competition_settings=settings,
            allow_unverified_db=args.allow_unverified_db,
            require_clean=args.require_clean,
            apply_profile=args.apply_profile,
            profile_path=args.profile,
            bootstrap_resamples=args.bootstrap_resamples,
            minimum_pairs=args.minimum_pairs,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    if not args.quiet:
        for mode in MODES:
            summary = payload["results"][mode]
            print(
                mode,
                "win_rate", summary["ai_a_win_rate"],
                "errors", summary["errors"],
                "P95", summary["ai_a_p95_move_ms"],
            )
        for mode in ("shadow", "assist"):
            paired = payload["paired_comparisons"][mode]
            print(
                f"{mode} paired delta: {paired['difference_pp']:.2f} pp "
                f"(95% CI {paired['ci95_low_pp']:.2f}, "
                f"{paired['ci95_high_pp']:.2f}); {paired['status']}"
            )
        print("recommendation:", payload["recommendation"]["status"])
        print("profile update:", payload["profile_update"]["applied"])
        print(f"JSON: {paths['json']}")
        print(f"CSV: {paths['summary_csv']}")
    failed = any(payload["results"][mode]["errors"] for mode in MODES)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())