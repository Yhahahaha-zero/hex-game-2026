# -*- coding: utf-8 -*-
"""Generate Markdown and CSV reports from validated competition results."""
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

from ai.competition_config import load_competition_settings
from tools import verify_competition_results


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _cell(value):
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = [
        "| " + " | ".join(_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _matrix_table(tournament):
    config = tournament["config"]
    specs = config["opponents"]
    labels = config["opponent_names"]
    matrix = tournament["matrix_win_rate"]
    rows = []
    for row_spec, row_label in zip(specs, labels):
        rows.append([
            row_label,
            *[
                "" if row_spec == col_spec else matrix[row_spec][col_spec]
                for col_spec in specs
            ],
        ])
    return _table(["AI", *labels], rows)


def _standings_table(tournament):
    rows = []
    for row in tournament.get("standings", []):
        rows.append([
            row.get("rank"),
            row.get("name"),
            row.get("games"),
            row.get("wins"),
            row.get("losses"),
            row.get("win_rate"),
            row.get("max_p95_move_ms"),
            row.get("max_p99_move_ms"),
            row.get("max_timeout_rate"),
        ])
    return _table(
        ["Rank", "AI", "Games", "Wins", "Losses", "Win %",
         "Max P95 ms", "Max P99 ms", "Max timeout %"],
        rows,
    )


def _pair_table(tournament):
    rows = []
    for row in tournament.get("pairings", []):
        rows.append([
            f"{row['ai_a_name']} vs {row['ai_b_name']}",
            row.get("games_completed"),
            row.get("errors"),
            row.get("ai_a_win_rate"),
            row.get("ai_a_red_win_rate"),
            row.get("ai_a_blue_win_rate"),
            row.get("ai_a_p95_move_ms"),
            row.get("ai_a_p99_move_ms"),
            row.get("ai_b_p95_move_ms"),
            row.get("ai_b_p99_move_ms"),
            row.get("move_timeout_rate"),
            row.get("illegal_move_rate"),
        ])
    return _table(
        ["Matchup", "Games", "Errors", "A win %", "A red %", "A blue %",
         "A P95", "A P99", "B P95", "B P99", "Timeout %", "Illegal %"],
        rows,
    )


def _ablation_table(ablation):
    rows = []
    for mode in ("off", "shadow", "assist"):
        summary = ablation["results"][mode]
        evidence = ablation["memory_evidence"][mode]
        rows.append([
            mode,
            summary.get("games_completed"),
            summary.get("errors"),
            summary.get("ai_a_win_rate"),
            summary.get("ai_a_red_win_rate"),
            summary.get("ai_a_blue_win_rate"),
            summary.get("ai_a_p95_move_ms"),
            summary.get("ai_a_p99_move_ms"),
            summary.get("move_timeout_rate"),
            evidence.get("events"),
            evidence.get("applied"),
            evidence.get("coverage"),
        ])
    return _table(
        ["Mode", "Games", "Errors", "Win %", "Red %", "Blue %",
         "P95", "P99", "Timeout %", "Events", "Applied", "Coverage"],
        rows,
    )


def _ablation_delta_table(ablation):
    rows = []
    paired = ablation.get("paired_comparisons", {})
    for mode in ("shadow", "assist"):
        delta = ablation["deltas_vs_off"][mode]
        comparison = paired.get(mode, {})
        rows.append([
            mode,
            delta.get("ai_a_win_rate"),
            comparison.get("ci95_low_pp"),
            comparison.get("ci95_high_pp"),
            comparison.get("status"),
            delta.get("ai_a_red_win_rate"),
            delta.get("ai_a_blue_win_rate"),
            delta.get("ai_a_p95_move_ms"),
            delta.get("ai_a_p99_move_ms"),
            delta.get("move_timeout_rate"),
            delta.get("memory_events"),
            delta.get("memory_applied"),
        ])
    return _table(
        ["Mode", "Raw win delta pp", "Paired CI low", "Paired CI high",
         "Paired status", "Red delta pp", "Blue delta pp",
         "P95 delta ms", "P99 delta ms", "Timeout delta pp",
         "Events", "Applied"],
        rows,
    )


def _write_csv(path, tournament, ablation=None):
    rows = []
    for standing in tournament.get("standings", []):
        rows.append({
            "section": "tournament_standing",
            "name": standing.get("name"),
            "rank": standing.get("rank"),
            "games": standing.get("games"),
            "wins": standing.get("wins"),
            "losses": standing.get("losses"),
            "win_rate": standing.get("win_rate"),
            "p95_move_ms": standing.get("max_p95_move_ms"),
            "p99_move_ms": standing.get("max_p99_move_ms"),
            "timeout_rate": standing.get("max_timeout_rate"),
        })
    for pairing in tournament.get("pairings", []):
        rows.append({
            "section": "tournament_pairing",
            "name": f"{pairing.get('ai_a_name')} vs {pairing.get('ai_b_name')}",
            "games": pairing.get("games_completed"),
            "errors": pairing.get("errors"),
            "win_rate": pairing.get("ai_a_win_rate"),
            "red_win_rate": pairing.get("ai_a_red_win_rate"),
            "blue_win_rate": pairing.get("ai_a_blue_win_rate"),
            "p95_move_ms": pairing.get("ai_a_p95_move_ms"),
            "p99_move_ms": pairing.get("ai_a_p99_move_ms"),
            "timeout_rate": pairing.get("move_timeout_rate"),
        })
    if ablation:
        for mode in ("off", "shadow", "assist"):
            summary = ablation["results"][mode]
            evidence = ablation["memory_evidence"][mode]
            comparison = ablation.get("paired_comparisons", {}).get(mode, {})
            rows.append({
                "section": "memory_ablation",
                "name": mode,
                "games": summary.get("games_completed"),
                "errors": summary.get("errors"),
                "win_rate": summary.get("ai_a_win_rate"),
                "red_win_rate": summary.get("ai_a_red_win_rate"),
                "blue_win_rate": summary.get("ai_a_blue_win_rate"),
                "p95_move_ms": summary.get("ai_a_p95_move_ms"),
                "p99_move_ms": summary.get("ai_a_p99_move_ms"),
                "timeout_rate": summary.get("move_timeout_rate"),
                "memory_events": evidence.get("events"),
                "memory_applied": evidence.get("applied"),
                "paired_games": comparison.get("games_compared"),
                "paired_delta_pp": comparison.get("difference_pp"),
                "paired_ci95_low_pp": comparison.get("ci95_low_pp"),
                "paired_ci95_high_pp": comparison.get("ci95_high_pp"),
                "paired_status": comparison.get("status", "baseline"),
            })
    fields = [
        "section", "name", "rank", "games", "wins", "losses", "errors",
        "win_rate", "red_win_rate", "blue_win_rate", "p95_move_ms",
        "p99_move_ms", "timeout_rate", "memory_events", "memory_applied",
        "paired_games", "paired_delta_pp", "paired_ci95_low_pp",
        "paired_ci95_high_pp", "paired_status",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(*, tournament_path, ablation_path=None, output_dir=None,
                    allow_dirty=False, allow_timeouts=False,
                    allow_unverified_db=False, profile_path=None):
    tournament_path = Path(tournament_path).resolve()
    tournament = _load(tournament_path)
    settings = load_competition_settings(profile_path)
    tournament_validation = verify_competition_results.verify_document(
        tournament,
        max_p99_ms=settings.hard_move_limit_ms,
        allow_dirty=allow_dirty,
        allow_timeouts=allow_timeouts,
        require_recommendation=True,
        expected_profile_sha256=settings.profile_sha256,
    )
    if not tournament_validation["passed"]:
        raise ValueError(
            "tournament failed validation: "
            + "; ".join(tournament_validation["errors"])
        )
    ablation = None
    ablation_validation = None
    if ablation_path:
        ablation_path = Path(ablation_path).resolve()
        ablation = _load(ablation_path)
        ablation_validation = verify_competition_results.verify_document(
            ablation,
            max_p99_ms=settings.hard_move_limit_ms,
            allow_dirty=allow_dirty,
            allow_timeouts=allow_timeouts,
            allow_unverified_db=allow_unverified_db,
            require_recommendation=False,
            expected_profile_sha256=settings.profile_sha256,
        )
        if not ablation_validation["passed"]:
            raise ValueError(
                "memory ablation failed validation: "
                + "; ".join(ablation_validation["errors"])
            )
    output_dir = Path(output_dir or tournament_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "competition_report.md"
    csv_path = output_dir / "competition_report.csv"
    reproducibility = tournament.get("reproducibility", {})
    profile = reproducibility.get("competition_profile", {})
    totals = tournament.get("totals", {})
    recommendation = tournament.get("recommendation", {})

    sections = [
        "# Hex Competition Evaluation Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Baseline",
        "",
        _table(["Field", "Value"], [
            ["Git commit", reproducibility.get("code_commit")],
            ["Working tree dirty", reproducibility.get("working_tree_dirty")],
            ["Profile", profile.get("profile_name")],
            ["Profile SHA256", profile.get("profile_sha256")],
            ["AI engine", profile.get("ai_engine")],
            ["Search budget ms", profile.get("search_time_limit_ms")],
            ["Hard move limit ms", profile.get("hard_move_limit_ms")],
            ["Memory mode", profile.get("memory_mode")],
            ["Random seed", profile.get("random_seed")],
        ]),
        "",
        "## Tournament Totals",
        "",
        _table(["Field", "Value"], [
            ["Matchups", totals.get("matchups")],
            ["Games requested", totals.get("games_requested")],
            ["Games completed", totals.get("games_completed")],
            ["Errors", totals.get("errors")],
            ["Draws", totals.get("draws")],
            ["Illegal move rate", totals.get("illegal_move_rate")],
            ["Timeout rate", totals.get("move_timeout_rate")],
        ]),
        "",
        "## Standings",
        "",
        _standings_table(tournament),
        "",
        "## Win-Rate Matrix",
        "",
        _matrix_table(tournament),
        "",
        "## Pair Details",
        "",
        _pair_table(tournament),
        "",
        "## Recommendation",
        "",
        _table(["Field", "Value"], [
            ["Eligible", recommendation.get("eligible")],
            ["Recommended AI", recommendation.get("recommended_ai")],
            ["Tied AIs", ", ".join(recommendation.get("tied_ai", []))],
            ["Reason", recommendation.get("reason")],
        ]),
    ]
    if ablation:
        sections.extend([
            "",
            "## Memory Ablation",
            "",
            _table(["Field", "Value"], [
                ["Base AI", ablation.get("config", {}).get("base_ai")],
                ["Opponent", ablation.get("config", {}).get("opponent")],
                ["Games per mode", ablation.get("config", {}).get("games")],
                ["Provenance verified", ablation.get("provenance", {}).get("verified")],
                ["Database SHA256", ablation.get("provenance", {}).get("db_sha256")],
                ["Recommendation", ablation.get("recommendation", {}).get("status")],
            ]),
            "",
            _ablation_table(ablation),
            "",
            "### Delta Versus Memory Off",
            "",
            _ablation_delta_table(ablation),
        ])
    sections.extend([
        "",
        "## Notes",
        "",
        "- The automated recommendation is evidence for selection, not a substitute for manual review.",
        "- A 2-game or other small smoke run must not be presented as final competitive evidence.",
        "- Formal results must be regenerated from a clean Git worktree.",
        "",
    ])
    markdown_path.write_text("\n".join(sections), encoding="utf-8")
    _write_csv(csv_path, tournament, ablation)
    validation_path = output_dir / "report_validation.json"
    validation_path.write_text(
        json.dumps({
            "tournament": tournament_validation,
            "memory_ablation": ablation_validation,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "markdown": markdown_path,
        "csv": csv_path,
        "validation": validation_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Markdown/CSV competition evaluation reports"
    )
    parser.add_argument("--tournament", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-timeouts", action="store_true")
    parser.add_argument("--allow-unverified-db", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = generate_report(
            tournament_path=args.tournament,
            ablation_path=args.ablation,
            output_dir=args.output_dir,
            allow_dirty=args.allow_dirty,
            allow_timeouts=args.allow_timeouts,
            allow_unverified_db=args.allow_unverified_db,
            profile_path=args.profile,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(f"Markdown: {paths['markdown']}")
    print(f"CSV: {paths['csv']}")
    print(f"Validation: {paths['validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())