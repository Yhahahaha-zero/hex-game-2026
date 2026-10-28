# -*- coding: utf-8 -*-
"""Validate benchmark, tournament and memory-ablation result documents."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import load_competition_settings


LATENCY_KEYS = ("ai_a_p50_move_ms", "ai_a_p95_move_ms", "ai_a_p99_move_ms")


def _add_check(checks, errors, name, condition, message):
    checks[name] = bool(condition)
    if not condition:
        errors.append(message)


def _common_checks(document, checks, errors, *, allow_dirty=False, expected_profile_sha256=None):
    reproducibility = document.get("reproducibility")
    _add_check(
        checks, errors, "reproducibility_present",
        isinstance(reproducibility, dict), "reproducibility metadata is missing",
    )
    if not isinstance(reproducibility, dict):
        return
    profile = reproducibility.get("competition_profile", {})
    _add_check(
        checks, errors, "code_commit_present",
        bool(reproducibility.get("code_commit")), "Git commit is missing",
    )
    _add_check(
        checks, errors, "profile_hash_present",
        isinstance(profile.get("profile_sha256"), str)
        and len(profile.get("profile_sha256", "")) == 64,
        "competition profile SHA256 is missing or invalid",
    )
    if expected_profile_sha256:
        _add_check(
            checks, errors, "profile_hash_matches",
            profile.get("profile_sha256") == expected_profile_sha256,
            "result profile SHA256 does not match the current competition profile",
        )
    if not allow_dirty:
        _add_check(
            checks, errors, "clean_worktree",
            reproducibility.get("working_tree_dirty") is False,
            "result was produced from a dirty worktree",
        )


def _latency_checks(summary, checks, errors, *, max_p99_ms):
    for key in LATENCY_KEYS:
        _add_check(
            checks, errors, key,
            summary.get(key) is not None and float(summary.get(key, 0.0)) >= 0,
            f"missing latency metric: {key}",
        )
    p99 = float(summary.get("ai_a_p99_move_ms", 0.0) or 0.0)
    _add_check(
        checks, errors, "p99_within_limit",
        p99 <= max_p99_ms,
        f"AI-A P99 {p99:.3f} ms exceeds {max_p99_ms:.3f} ms",
    )


def verify_benchmark(document, *, max_p99_ms=500.0, allow_dirty=False,
                     allow_timeouts=False, expected_profile_sha256=None):
    checks = {}
    errors = []
    _add_check(
        checks, errors, "benchmark_schema",
        int(document.get("benchmark_schema_version", 0)) >= 2,
        "unsupported benchmark schema",
    )
    _common_checks(
        document, checks, errors, allow_dirty=allow_dirty,
        expected_profile_sha256=expected_profile_sha256,
    )
    summary = document.get("summary", {})
    requested = int(summary.get("games_requested", 0))
    completed = int(summary.get("games_completed", 0))
    _add_check(
        checks, errors, "games_complete",
        requested > 0 and completed == requested,
        f"completed games {completed} do not match requested {requested}",
    )
    _add_check(
        checks, errors, "zero_errors", int(summary.get("errors", 0)) == 0,
        "benchmark contains errors",
    )
    _add_check(
        checks, errors, "zero_illegal_moves",
        int(summary.get("illegal_move_games", 0)) == 0,
        "benchmark contains illegal moves",
    )
    if not allow_timeouts:
        _add_check(
            checks, errors, "zero_timeouts",
            int(summary.get("move_timeout_count", 0)) == 0,
            "benchmark contains move timeouts",
        )
    _add_check(
        checks, errors, "both_colors_played",
        int(summary.get("ai_a_red_games", 0)) > 0
        and int(summary.get("ai_a_blue_games", 0)) > 0,
        "AI-A did not play both colors",
    )
    _latency_checks(summary, checks, errors, max_p99_ms=max_p99_ms)
    return {"kind": "benchmark", "passed": not errors,
            "checks": checks, "errors": errors}


def verify_tournament(document, *, max_p99_ms=500.0, allow_dirty=False,
                      allow_timeouts=False, require_recommendation=True, expected_profile_sha256=None):
    checks = {}
    errors = []
    _add_check(
        checks, errors, "tournament_schema",
        int(document.get("tournament_schema_version", 0)) >= 2,
        "unsupported tournament schema",
    )
    _common_checks(
        document, checks, errors, allow_dirty=allow_dirty,
        expected_profile_sha256=expected_profile_sha256,
    )
    totals = document.get("totals", {})
    requested = int(totals.get("games_requested", 0))
    completed = int(totals.get("games_completed", 0))
    _add_check(
        checks, errors, "games_complete",
        requested > 0 and completed == requested,
        f"completed games {completed} do not match requested {requested}",
    )
    _add_check(
        checks, errors, "zero_errors", int(totals.get("errors", 0)) == 0,
        "tournament contains errors",
    )
    _add_check(
        checks, errors, "zero_illegal_moves",
        float(totals.get("illegal_move_rate", 0.0)) == 0.0,
        "tournament contains illegal moves",
    )
    if not allow_timeouts:
        _add_check(
            checks, errors, "zero_timeouts",
            float(totals.get("move_timeout_rate", 0.0)) == 0.0,
            "tournament contains move timeouts",
        )
    pairings = document.get("pairings", [])
    _add_check(
        checks, errors, "pairings_present", bool(pairings),
        "tournament has no pairings",
    )
    pair_valid = True
    for pairing in pairings:
        if int(pairing.get("games_completed", 0)) != int(
            pairing.get("games_requested", 0)
        ):
            pair_valid = False
        if int(pairing.get("errors", 0)) != 0:
            pair_valid = False
        if float(pairing.get("illegal_move_rate", 0.0)) != 0.0:
            pair_valid = False
        if (not allow_timeouts
                and float(pairing.get("move_timeout_rate", 0.0)) != 0.0):
            pair_valid = False
        for key in (
            "ai_a_p95_move_ms", "ai_a_p99_move_ms",
            "ai_b_p95_move_ms", "ai_b_p99_move_ms",
        ):
            if pairing.get(key) is None:
                pair_valid = False
    _add_check(
        checks, errors, "pairings_complete", pair_valid,
        "one or more tournament pairings are incomplete or missing metrics",
    )
    specs = document.get("config", {}).get("opponents", [])
    matrix = document.get("matrix_win_rate", {})
    matrix_complete = bool(specs) and all(
        matrix.get(left, {}).get(right) is not None
        for left in specs for right in specs if left != right
    )
    _add_check(
        checks, errors, "matrix_complete", matrix_complete,
        "tournament win-rate matrix is incomplete",
    )
    standings = document.get("standings", [])
    _add_check(
        checks, errors, "standings_present", bool(standings),
        "tournament standings are missing",
    )
    if standings:
        max_p99 = max(float(row.get("max_p99_move_ms", 0.0)) for row in standings)
        _add_check(
            checks, errors, "standings_p99_within_limit",
            max_p99 <= max_p99_ms,
            f"maximum AI P99 {max_p99:.3f} ms exceeds {max_p99_ms:.3f} ms",
        )
    recommendation = document.get("recommendation", {})
    if require_recommendation:
        _add_check(
            checks, errors, "recommendation_eligible",
            recommendation.get("eligible") is True,
            "tournament recommendation is not eligible",
        )
        _add_check(
            checks, errors, "recommendation_unique",
            bool(recommendation.get("recommended_ai")),
            "tournament has no unique recommended AI",
        )
    return {"kind": "tournament", "passed": not errors,
            "checks": checks, "errors": errors}


def verify_ablation(document, *, max_p99_ms=500.0, allow_dirty=False,
                    allow_timeouts=False, allow_unverified_db=False,
                    expected_profile_sha256=None, require_minimum_pairs=True):
    checks = {}
    errors = []
    _add_check(
        checks, errors, "ablation_schema",
        int(document.get("ablation_schema_version", 0)) >= 2,
        "unsupported ablation schema",
    )
    _common_checks(
        document, checks, errors, allow_dirty=allow_dirty,
        expected_profile_sha256=expected_profile_sha256,
    )
    provenance = document.get("provenance", {})
    if not allow_unverified_db:
        _add_check(
            checks, errors, "verified_provenance",
            provenance.get("verified") is True,
            "memory ablation uses an unverified database",
        )
    results = document.get("results", {})
    _add_check(
        checks, errors, "three_modes",
        set(results) == {"off", "shadow", "assist"},
        "ablation must contain off, shadow and assist results",
    )
    games = int(document.get("config", {}).get("games", 0))
    all_complete = True
    all_clean = True
    latency_ok = True
    for mode in ("off", "shadow", "assist"):
        summary = results.get(mode, {})
        if int(summary.get("games_completed", 0)) != games or games <= 0:
            all_complete = False
        if int(summary.get("errors", 0)) != 0:
            all_clean = False
        if (int(summary.get("illegal_move_games", 0)) != 0
                or float(summary.get("illegal_move_rate", 0.0)) != 0.0):
            all_clean = False
        if not allow_timeouts and int(summary.get("move_timeout_count", 0)) != 0:
            all_clean = False
        if any(summary.get(key) is None for key in LATENCY_KEYS):
            latency_ok = False
        if float(summary.get("ai_a_p99_move_ms", 0.0) or 0.0) > max_p99_ms:
            latency_ok = False
    _add_check(checks, errors, "modes_complete", all_complete,
               "one or more ablation modes are incomplete")
    _add_check(checks, errors, "modes_clean", all_clean,
               "one or more ablation modes contain errors or timeouts")
    _add_check(checks, errors, "latency_available", latency_ok,
               "ablation latency metrics are missing or exceed the limit")
    minimum_pairs = int(document.get("config", {}).get("minimum_pairs", 0))
    if require_minimum_pairs:
        _add_check(
            checks, errors, "minimum_pairs_configured",
            minimum_pairs >= 2,
            "memory ablation minimum_pairs is missing or invalid",
        )
    paired = document.get("paired_comparisons", {})
    paired_complete = set(paired) == {"shadow", "assist"}
    paired_valid = paired_complete
    if paired_complete:
        for mode in ("shadow", "assist"):
            comparison = paired.get(mode, {})
            if comparison.get("method") != "paired_bootstrap":
                paired_valid = False
            if int(comparison.get("games_compared", -1)) != games:
                paired_valid = False
            if (require_minimum_pairs
                    and int(comparison.get("games_compared", 0)) < minimum_pairs):
                paired_valid = False
            for key in ("ci95_low_pp", "ci95_high_pp", "difference_pp"):
                if comparison.get(key) is None:
                    paired_valid = False
            if comparison.get("status") not in (
                "insufficient_pairs", "significant_improvement",
                "significant_regression", "no_significant_difference",
            ):
                paired_valid = False
    _add_check(
        checks, errors, "paired_comparisons_present", paired_complete,
        "paired shadow/assist comparisons are missing",
    )
    _add_check(
        checks, errors, "paired_comparisons_valid", paired_valid,
        "paired bootstrap metrics are invalid or incomplete",
    )
    _add_check(
        checks, errors, "memory_configs_present",
        set(document.get("memory_configs", {})) == {"off", "shadow", "assist"},
        "memory configuration metadata is incomplete",
    )
    return {"kind": "memory_ablation", "passed": not errors,
            "checks": checks, "errors": errors}


def detect_kind(document):
    if "benchmark_schema_version" in document:
        return "benchmark"
    if "tournament_schema_version" in document:
        return "tournament"
    if "ablation_schema_version" in document:
        return "memory_ablation"
    return None


def verify_document(document, *, max_p99_ms=500.0, allow_dirty=False,
                    allow_timeouts=False, allow_unverified_db=False,
                    require_recommendation=True, expected_profile_sha256=None,
                    require_minimum_pairs=True):
    kind = detect_kind(document)
    if kind == "benchmark":
        result = verify_benchmark(
            document, max_p99_ms=max_p99_ms, allow_dirty=allow_dirty,
            allow_timeouts=allow_timeouts,
            expected_profile_sha256=expected_profile_sha256,
        )
    elif kind == "tournament":
        result = verify_tournament(
            document, max_p99_ms=max_p99_ms, allow_dirty=allow_dirty,
            allow_timeouts=allow_timeouts,
            require_recommendation=require_recommendation,
            expected_profile_sha256=expected_profile_sha256,
        )
    elif kind == "memory_ablation":
        result = verify_ablation(
            document, max_p99_ms=max_p99_ms, allow_dirty=allow_dirty,
            allow_timeouts=allow_timeouts,
            allow_unverified_db=allow_unverified_db,
            expected_profile_sha256=expected_profile_sha256,
            require_minimum_pairs=require_minimum_pairs,
        )
    else:
        raise ValueError("unrecognized competition result document")
    return result


def _expand_inputs(values):
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        else:
            paths.append(path)
    return [path for path in paths if path.name != "validation.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate competition benchmark/tournament/ablation JSON"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-p99-ms", type=float, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-timeouts", action="store_true")
    parser.add_argument("--allow-unverified-db", action="store_true")
    parser.add_argument("--allow-no-recommendation", action="store_true")
    parser.add_argument("--allow-profile-mismatch", action="store_true")
    parser.add_argument("--allow-minimum-pairs", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_competition_settings(args.profile)
        max_p99 = (
            settings.hard_move_limit_ms
            if args.max_p99_ms is None else args.max_p99_ms
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2
    reports = []
    for path in _expand_inputs(args.input):
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            result = verify_document(
                document,
                max_p99_ms=max_p99,
                allow_dirty=args.allow_dirty,
                allow_timeouts=args.allow_timeouts,
                allow_unverified_db=args.allow_unverified_db,
                require_recommendation=not args.allow_no_recommendation,
                expected_profile_sha256=(
                    None if args.allow_profile_mismatch
                    else settings.profile_sha256
                ),
                require_minimum_pairs=not args.allow_minimum_pairs,
            )
            result["path"] = str(path.resolve())
            reports.append(result)
        except (OSError, ValueError, TypeError) as exc:
            reports.append({
                "path": str(path.resolve()),
                "kind": "unknown",
                "passed": False,
                "checks": {},
                "errors": [str(exc)],
            })
    passed = bool(reports) and all(report["passed"] for report in reports)
    payload = {
        "validation_schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "document_count": len(reports),
        "documents": reports,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    if not reports:
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())