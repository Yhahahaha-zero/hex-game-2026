# -*- coding: utf-8 -*-
"""Run screening, finalists, validation and profile selection in one flow."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import (
    ALLOWED_AI_ENGINES,
    load_competition_settings,
    require_clean_worktree,
)
from tools import (
    competition_report,
    tournament,
    verify_competition_results,
)


DEFAULT_SCREENING_DIR = PROJECT_ROOT.parent / "evaluation" / "tournament_results"
DEFAULT_FINAL_DIR = PROJECT_ROOT.parent / "evaluation" / "tournament_results"
DEFAULT_REPORT_DIR = PROJECT_ROOT.parent / "evaluation" / "competition_results"


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


def select_finalists(document):
    standings = document.get("standings", [])
    if len(standings) < 2:
        raise ValueError("screening did not produce at least two ranked AIs")
    first = standings[0]
    second = standings[1]
    if len(standings) > 2:
        third = standings[2]
        if float(third.get("win_rate", 0.0)) == float(second.get("win_rate", 0.0)):
            raise ValueError("the second/third place cutoff is tied; run more games")
    return [first["opponent_id"], second["opponent_id"]]


def _validate(document, settings, *, require_recommendation=True):
    result = verify_competition_results.verify_document(
        document,
        max_p99_ms=settings.hard_move_limit_ms,
        allow_dirty=False,
        allow_timeouts=False,
        allow_unverified_db=False,
        require_recommendation=require_recommendation,
        expected_profile_sha256=settings.profile_sha256,
    )
    if not result["passed"]:
        raise ValueError("result validation failed: " + "; ".join(result["errors"]))
    return result


def _resume_document(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def update_profile_selection(profile_path, *, winner, benchmark_file, freeze_commit,
                             selected_at=None):
    path = Path(profile_path).resolve()
    load_competition_settings(path)
    if winner not in ALLOWED_AI_ENGINES:
        raise ValueError(f"unsupported winner: {winner!r}")
    if not str(freeze_commit or "").strip():
        raise ValueError("freeze_commit is required")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported competition profile schema")
    document["ai_engine"] = str(winner)
    document["freeze_commit"] = str(freeze_commit)
    document["benchmark_file"] = str(benchmark_file)
    document["selected_at"] = selected_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    _atomic_json(path, document)
    return load_competition_settings(path)


def run_selection(*, profile_path=None, screening_games=None, final_games=None,
                  screening_dir=DEFAULT_SCREENING_DIR,
                  final_dir=DEFAULT_FINAL_DIR,
                  report_dir=DEFAULT_REPORT_DIR,
                  name=None, quiet=False, apply_profile=False,
                  resume_screening=None, resume_final=None,
                  clean_checker=require_clean_worktree,
                  tournament_runner=None,
                  report_generator=None, validator=None):
    commit = clean_checker()
    settings = load_competition_settings(profile_path)
    seed = settings.random_seed
    screening_games = (
        settings.tournament_games if screening_games is None else int(screening_games)
    )
    final_games = settings.tournament_final_games if final_games is None else int(final_games)
    runner = tournament_runner or tournament.run_tournament
    generator = report_generator or competition_report.generate_report
    validate = validator or _validate
    stem = name or datetime.now().strftime("selection_%Y%m%d_%H%M%S")

    if resume_screening:
        screening_path = Path(resume_screening).resolve()
        screening = _resume_document(screening_path)
        screening_validation = validate(screening, settings)
    else:
        screening, screening_paths = runner(
            list(tournament.DEFAULT_OPPONENTS),
            screening_games,
            seed=seed,
            output_dir=Path(screening_dir),
            name=f"{stem}_screening",
            quiet=quiet,
            competition_settings=settings,
            require_clean=True,
        )
        screening_path = Path(screening_paths["json"]).resolve()
        screening_validation = validate(screening, settings)
    finalists = select_finalists(screening)

    if resume_final:
        final_path = Path(resume_final).resolve()
        final = _resume_document(final_path)
        final_validation = validate(final, settings, require_recommendation=True)
    else:
        final, final_paths = runner(
            finalists,
            final_games,
            seed=seed,
            output_dir=Path(final_dir),
            name=f"{stem}_final",
            quiet=quiet,
            competition_settings=settings,
            require_clean=True,
        )
        final_path = Path(final_paths["json"]).resolve()
        final_validation = validate(final, settings, require_recommendation=True)
    final_opponents = set(final.get("config", {}).get("opponents", []))
    if final_opponents != set(finalists):
        raise ValueError("final tournament opponents do not match the selected finalists")
    winner = final.get("recommendation", {}).get("recommended_ai")
    if not winner:
        raise ValueError("final tournament has no unique recommended AI")
    if winner not in finalists:
        raise ValueError("recommended AI is not one of the finalists")

    report_paths = generator(
        tournament_path=final_path,
        ablation_path=None,
        output_dir=Path(report_dir) / stem,
        profile_path=profile_path,
    )
    profile_updated = False
    if apply_profile:
        relative_benchmark = os.path.relpath(
            final_path, PROJECT_ROOT.parent
        ).replace("\\", "/")
        update_profile_selection(
            profile_path or PROJECT_ROOT / "competition_profile.json",
            winner=winner,
            benchmark_file=relative_benchmark,
            freeze_commit=commit,
        )
        profile_updated = True

    summary = {
        "selection_schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "screening_games": screening_games,
        "final_games": final_games,
        "screening_file": str(screening_path),
        "final_file": str(final_path),
        "finalists": finalists,
        "winner": winner,
        "profile_updated": profile_updated,
        "report": {key: str(value) for key, value in report_paths.items()},
        "screening_validation": screening_validation,
        "final_validation": final_validation,
    }
    summary_path = _atomic_json(Path(report_dir) / stem / "selection_summary.json", summary)
    summary["summary_file"] = str(summary_path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Select the competition AI using validated screening and finals"
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--screening-games", type=int, default=None)
    parser.add_argument("--final-games", type=int, default=None)
    parser.add_argument("--screening-dir", default=str(DEFAULT_SCREENING_DIR))
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume-screening", default=None)
    parser.add_argument("--resume-final", default=None)
    parser.add_argument("--apply-profile", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_selection(
            profile_path=args.profile,
            screening_games=args.screening_games,
            final_games=args.final_games,
            screening_dir=args.screening_dir,
            final_dir=args.final_dir,
            report_dir=args.report_dir,
            name=args.name,
            quiet=args.quiet,
            apply_profile=args.apply_profile,
            resume_screening=args.resume_screening,
            resume_final=args.resume_final,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())