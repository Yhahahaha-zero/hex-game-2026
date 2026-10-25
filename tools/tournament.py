# -*- coding: utf-8 -*-
"""固定 AI 对手池的交叉评测与胜率矩阵。"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import benchmark_ai

DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "evaluation" / "tournament_results"
DEFAULT_OPPONENTS = (
    "basic",
    "strategic",
    "hybrid",
    "neural",
    "neural:../models/hex_model_v5_reference.pt",
)
DEFAULT_GAMES = 20
DEFAULT_SEED = 20261025

DISPLAY_NAMES = {
    "basic": "BasicAI",
    "strategic": "StrategicAI",
    "hybrid": "HybridAI",
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
                    "" if row_spec == col_spec
                    else matrix[row_spec][col_spec]
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
        "ai_a_p95_move_ms": summary.get("ai_a_p95_move_ms", 0.0),
        "ai_b_p95_move_ms": summary.get("ai_b_p95_move_ms", 0.0),
        "illegal_move_rate": summary.get("illegal_move_rate", 0.0),
        "move_timeout_rate": summary.get("move_timeout_rate", 0.0),
        "draws": summary.get("draws", 0),
        "error_types": dict(summary.get("error_types", {})),
    }


def run_tournament(opponents, games, seed=DEFAULT_SEED,
                   output_dir=DEFAULT_OUTPUT_DIR, name=None, quiet=False,
                   move_time_limit_ms=500):
    """Run every unordered matchup and write a complete cross-evaluation."""
    opponents = list(dict.fromkeys(opponents))
    if len(opponents) < 2:
        raise ValueError("对手池至少需要两个不同 AI")
    if games <= 0 or games % 2:
        raise ValueError("games 必须是大于 0 的偶数")

    output_dir = Path(output_dir)
    stem = _safe_name(name or datetime.now().strftime("tournament_%Y%m%d_%H%M%S"))
    individual_dir = output_dir / f"{stem}_matches"
    individual_dir.mkdir(parents=True, exist_ok=True)

    matrix = {ai: {other: None for other in opponents} for ai in opponents}
    pairings = []
    for left_index, ai_a in enumerate(opponents):
        for ai_b in opponents[left_index + 1:]:
            matchup = f"{_safe_name(opponent_label(ai_a))}_vs_{_safe_name(opponent_label(ai_b))}"
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
            )
            item = _pair_summary(ai_a, ai_b, document)
            pairings.append(item)
            matrix[ai_a][ai_b] = item["ai_a_win_rate"]
            matrix[ai_b][ai_a] = item["ai_b_win_rate"]

    payload = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "opponents": opponents,
            "opponent_names": [opponent_label(spec) for spec in opponents],
            "games_per_pair": games,
            "seed": seed,
            "pairing": "AB/BA with colors swapped",
            "move_time_limit_ms": float(move_time_limit_ms),
        },
        "matrix_win_rate": matrix,
        "pairings": pairings,
        "totals": {
            "matchups": len(pairings),
            "games_requested": sum(item["games_requested"] for item in pairings),
            "games_completed": sum(item["games_completed"] for item in pairings),
            "errors": sum(item["errors"] for item in pairings),
            "draws": sum(item["draws"] for item in pairings),
            "illegal_move_rate": round(
                sum(item["illegal_move_rate"] for item in pairings) / max(1, len(pairings)), 2
            ),
            "move_timeout_rate": round(
                sum(item["move_timeout_rate"] for item in pairings) / max(1, len(pairings)), 2
            ),
        },
    }
    json_path = output_dir / f"{stem}.json"
    matrix_path = _matrix_csv_path(output_dir, stem)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_matrix_csv(matrix_path, opponents, matrix)
    return payload, {"json": json_path, "matrix_csv": matrix_path,
                     "matches": individual_dir}


def _parse_opponents(values):
    opponents = []
    for value in values:
        opponents.extend(part.strip() for part in value.split(",") if part.strip())
    return opponents


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="固定种子、AB/BA 配对的 AI 对手池交叉评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS),
                        help="AI 列表，可使用逗号分隔")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="每一对 AI 的总局数，必须为偶数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--move-time-limit-ms", type=float, default=500)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload, paths = run_tournament(
            _parse_opponents(args.opponents),
            args.games,
            seed=args.seed,
            output_dir=args.output_dir,
            name=args.name,
            quiet=args.quiet,
            move_time_limit_ms=args.move_time_limit_ms,
        )
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    if not args.quiet:
        print("胜率矩阵（行 AI 对列 AI）：")
        names = payload["config"]["opponent_names"]
        for index, row_name in enumerate(names):
            row = ["" if index == col else
                   payload["matrix_win_rate"][payload["config"]["opponents"][index]][
                       payload["config"]["opponents"][col]
                   ] for col in range(len(names))]
            print(row_name, row)
        print(f"JSON: {paths['json']}")
        print(f"矩阵 CSV: {paths['matrix_csv']}")
    return 0 if payload["totals"]["games_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
