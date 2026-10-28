# -*- coding: utf-8 -*-
"""Fixed-seed cross-AI tournament with color-balanced metrics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import (
    load_competition_settings,
    reproducibility_metadata,
    require_clean_worktree,
)
from tools import benchmark_ai

DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "evaluation" / "tournament_results"
DEFAULT_OPPONENTS = (
    "basic",
    "strategic",
    "hybrid",
    "neural",
    "neural:../models/hex_model_v5_reference.pt",
)
DEFAULT_GAMES = 200
DEFAULT_SEED = 20261025

DISPLAY_NAMES = {
    "competition": "CompetitionProfile",
    "profile": "CompetitionProfile",
    "basic": "BasicAI",
    "strategic": "StrategicAI",
    "hybrid": "HybridAI",
    "hybrid_llm": "HybridAI+LLM",
    "neural": "NeuralMCTS-official",
    "neural:../models/hex_model_v5_reference.pt": "NeuralMCTS-reference",
}


def opponent_label(spec):
    return DISPLAY_NAMES.get(spec, spec)


def _safe_name(value):
    return benchmark_ai._safe_filename(value)


def _matrix_csv_path(output_dir, stem):
    return Path(output_dir) / f"{stem}_matrix.csv"


def _write_matrix_csv(path, opponents, matrix):
    labels = [opponent_label(spec) for spec in opponents]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["AI \\ 对手"] + labels)
        for row_spec, row_label in zip(opponents, labels):
            writer.writerow(
                [row_label]
                + [
                    "" if row_spec == col_spec else matrix[row_spec][col_spec]
                    for col_spec in opponents
                ]
            )


def _pair_summary(ai_a, ai_b, document):
    summary = document["summary"]
    return {
        "ai_a": ai_a,
        "ai_b": ai_b,
        "ai_a_name": opponent_label(ai_a),
        "ai_b_name": opponent_label(ai_b),
        "games_requested": summary["games_requested"],
        "games_completed": summary["games_completed"],
        "errors": summary["errors"],
        "ai_a_wins": summary["ai_a_wins"],
        "ai_b_wins": summary["ai_b_wins"],
        "ai_a_win_rate": summary["ai_a_win_rate"],
        "ai_b_win_rate": summary["ai_b_win_rate"],
        "ai_a_red_win_rate": summary["ai_a_red_win_rate"],
        "ai_a_blue_win_rate": summary["ai_a_blue_win_rate"],
        "ci95_low": summary["ai_a_win_rate_ci95_low"],
        "ci95_high": summary["ai_a_win_rate_ci95_high"],
        "average_moves": summary["average_moves"],
        "ai_a_p50_move_ms": summary.get("ai_a_p50_move_ms", 0.0),
        "ai_a_p95_move_ms": summary.get("ai_a_p95_move_ms", 0.0),
        "ai_a_p99_move_ms": summary.get("ai_a_p99_move_ms", 0.0),
        "ai_b_p50_move_ms": summary.get("ai_b_p50_move_ms", 0.0),
        "ai_b_p95_move_ms": summary.get("ai_b_p95_move_ms", 0.0),
        "ai_b_p99_move_ms": summary.get("ai_b_p99_move_ms", 0.0),
        "illegal_move_rate": summary.get("illegal_move_rate", 0.0),
        "move_timeout_rate": summary.get("move_timeout_rate", 0.0),
        "draws": summary.get("draws", 0),
        "error_types": dict(summary.get("error_types", {})),
    }


def _build_standings(opponents, pairings):
    stats = {
        opponent: {
            "opponent_id": opponent,
            "name": opponent_label(opponent),
            "games": 0,
            "wins": 0,
            "losses": 0,
            "matchups": 0,
            "max_p95_move_ms": 0.0,
            "max_p99_move_ms": 0.0,
            "max_timeout_rate": 0.0,
        }
        for opponent in opponents
    }
    for pairing in pairings:
        left = stats[pairing["ai_a"]]
        right = stats[pairing["ai_b"]]
        completed = int(pairing["games_completed"])
        left["games"] += completed
        right["games"] += completed
        left["wins"] += int(pairing["ai_a_wins"])
        left["losses"] += int(pairing["ai_b_wins"])
        right["wins"] += int(pairing["ai_b_wins"])
        right["losses"] += int(pairing["ai_a_wins"])
        left["matchups"] += 1
        right["matchups"] += 1
        left["max_p95_move_ms"] = max(
            left["max_p95_move_ms"], float(pairing["ai_a_p95_move_ms"])
        )
        right["max_p95_move_ms"] = max(
            right["max_p95_move_ms"], float(pairing["ai_b_p95_move_ms"])
        )
        left["max_p99_move_ms"] = max(
            left["max_p99_move_ms"], float(pairing["ai_a_p99_move_ms"])
        )
        right["max_p99_move_ms"] = max(
            right["max_p99_move_ms"], float(pairing["ai_b_p99_move_ms"])
        )
        left["max_timeout_rate"] = max(
            left["max_timeout_rate"], float(pairing["move_timeout_rate"])
        )
        right["max_timeout_rate"] = max(
            right["max_timeout_rate"], float(pairing["move_timeout_rate"])
        )
    rows = []
    for row in stats.values():
        games = row["games"]
        row["win_rate"] = round(row["wins"] / games * 100, 2) if games else 0.0
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["win_rate"],
            row["max_timeout_rate"],
            row["max_p95_move_ms"],
            row["opponent_id"],
        )
    )
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def _recommendation(standings, totals, games_per_pair):
    expected_games = len(standings) * (len(standings) - 1) // 2 * games_per_pair
    complete = (
        totals["errors"] == 0
        and totals["games_completed"] == expected_games
        and totals["games_requested"] == expected_games
    )
    top_rate = standings[0]["win_rate"] if standings else None
    tied = [
        row["opponent_id"] for row in standings
        if top_rate is not None and row["win_rate"] == top_rate
    ]
    recommended = tied[0] if complete and len(tied) == 1 else None
    return {
        "eligible": complete,
        "recommended_ai": recommended,
        "tied_ai": tied,
        "reason": (
            "completed tournament without errors"
            if complete else "errors, incomplete games or tied ranking"
        ),
    }


def run_tournament(opponents, games=None, seed=None, output_dir=DEFAULT_OUTPUT_DIR,
                   name=None, quiet=False, move_time_limit_ms=None,
                   competition_settings=None, require_clean=False):
    """Run every unordered matchup and write a complete cross-evaluation."""
    if require_clean:
        require_clean_worktree()
    settings = competition_settings or load_competition_settings()
    opponents = list(dict.fromkeys(str(value).strip() for value in opponents))
    if len(opponents) < 2:
        raise ValueError("opponent pool requires at least two different AIs")
    games = settings.tournament_games if games is None else int(games)
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even integer")
    seed = settings.random_seed if seed is None else int(seed)
    if move_time_limit_ms is None:
        move_time_limit_ms = settings.hard_move_limit_ms
    move_time_limit_ms = float(move_time_limit_ms)

    output_dir = Path(output_dir)
    stem = _safe_name(name or datetime.now().strftime("tournament_%Y%m%d_%H%M%S"))
    individual_dir = output_dir / f"{stem}_matches"
    individual_dir.mkdir(parents=True, exist_ok=True)

    matrix = {ai: {other: None for other in opponents} for ai in opponents}
    matrix_ci_low = {ai: {other: None for other in opponents} for ai in opponents}
    matrix_ci_high = {ai: {other: None for other in opponents} for ai in opponents}
    matrix_p95 = {ai: {other: None for other in opponents} for ai in opponents}
    matrix_p99 = {ai: {other: None for other in opponents} for ai in opponents}
    matrix_timeout = {ai: {other: None for other in opponents} for ai in opponents}
    pairings = []
    for left_index, ai_a in enumerate(opponents):
        for ai_b in opponents[left_index + 1:]:
            matchup = (
                f"{_safe_name(opponent_label(ai_a))}_vs_"
                f"{_safe_name(opponent_label(ai_b))}"
            )
            if not quiet:
                print(f"[tournament] {matchup}")
            document, _paths = benchmark_ai.run_benchmark(
                ai_a,
                ai_b,
                games=games,
                repeats=1,
                seed=seed,
                output_dir=individual_dir,
                name=matchup,
                quiet=quiet,
                move_time_limit_ms=move_time_limit_ms,
                competition_settings=settings,
            )
            item = _pair_summary(ai_a, ai_b, document)
            pairings.append(item)
            matrix[ai_a][ai_b] = item["ai_a_win_rate"]
            matrix[ai_b][ai_a] = item["ai_b_win_rate"]
            matrix_ci_low[ai_a][ai_b] = item["ci95_low"]
            matrix_ci_high[ai_a][ai_b] = item["ci95_high"]
            matrix_p95[ai_a][ai_b] = item["ai_a_p95_move_ms"]
            matrix_p95[ai_b][ai_a] = item["ai_b_p95_move_ms"]
            matrix_p99[ai_a][ai_b] = item["ai_a_p99_move_ms"]
            matrix_p99[ai_b][ai_a] = item["ai_b_p99_move_ms"]
            matrix_timeout[ai_a][ai_b] = item["move_timeout_rate"]
            matrix_timeout[ai_b][ai_a] = item["move_timeout_rate"]

    standings = _build_standings(opponents, pairings)
    totals = {
        "matchups": len(pairings),
        "games_requested": sum(item["games_requested"] for item in pairings),
        "games_completed": sum(item["games_completed"] for item in pairings),
        "errors": sum(item["errors"] for item in pairings),
        "draws": sum(item["draws"] for item in pairings),
        "illegal_move_rate": round(
            sum(item["illegal_move_rate"] for item in pairings) / max(1, len(pairings)),
            2,
        ),
        "move_timeout_rate": round(
            sum(item["move_timeout_rate"] for item in pairings) / max(1, len(pairings)),
            2,
        ),
    }
    payload = {
        "tournament_schema_version": 2,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "opponents": opponents,
            "opponent_names": [opponent_label(spec) for spec in opponents],
            "games_per_pair": games,
            "seed": seed,
            "pairing": "AB/BA with colors swapped",
            "move_time_limit_ms": move_time_limit_ms,
        },
        "reproducibility": reproducibility_metadata(settings),
        "matrix_win_rate": matrix,
        "matrix_ci95_low": matrix_ci_low,
        "matrix_ci95_high": matrix_ci_high,
        "matrix_p95_move_ms": matrix_p95,
        "matrix_p99_move_ms": matrix_p99,
        "matrix_timeout_rate": matrix_timeout,
        "pairings": pairings,
        "standings": standings,
        "totals": totals,
        "recommendation": _recommendation(standings, totals, games),
    }
    json_path = output_dir / f"{stem}.json"
    matrix_path = _matrix_csv_path(output_dir, stem)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_matrix_csv(matrix_path, opponents, matrix)
    return payload, {
        "json": json_path,
        "matrix_csv": matrix_path,
        "matches": individual_dir,
    }


def _parse_opponents(values):
    opponents = []
    for value in values:
        opponents.extend(part.strip() for part in value.split(",") if part.strip())
    return opponents


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-seed cross-AI tournament")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS),
                        help="AI list; comma-separated values are accepted")
    parser.add_argument("--stage", choices=("screening", "final"),
                        default="screening",
                        help="use screening or final games from the profile")
    parser.add_argument("--games", type=int, default=None,
                        help="games per pair; defaults to competition profile")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--move-time-limit-ms", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--require-clean", action="store_true",
                        help="fail when the Git worktree is dirty")
    args = parser.parse_args(argv)
    try:
        settings = load_competition_settings(args.profile)
        games = args.games
        if games is None:
            games = (
                settings.tournament_final_games
                if args.stage == "final"
                else settings.tournament_games
            )
        payload, paths = run_tournament(
            _parse_opponents(args.opponents),
            games,
            seed=args.seed,
            output_dir=args.output_dir,
            name=args.name,
            quiet=args.quiet,
            move_time_limit_ms=args.move_time_limit_ms,
            competition_settings=settings,
            require_clean=args.require_clean,
        )
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    if not args.quiet:
        print("Standings:")
        for row in payload["standings"]:
            print(
                row["rank"], row["name"], row["win_rate"],
                "P95", row["max_p95_move_ms"],
                "P99", row["max_p99_move_ms"],
            )
        print("recommended:", payload["recommendation"]["recommended_ai"])
        print(f"JSON: {paths['json']}")
        print(f"matrix CSV: {paths['matrix_csv']}")
    return 0 if payload["totals"]["games_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())