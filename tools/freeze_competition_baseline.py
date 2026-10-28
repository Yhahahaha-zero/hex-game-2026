# -*- coding: utf-8 -*-
"""Freeze a validated competition baseline into an immutable directory."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import (
    file_sha256,
    load_competition_settings,
    reproducibility_metadata,
    require_clean_worktree,
)
from tools import competition_report, verify_competition_results


DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "evaluation" / "competition_baselines"
DEFAULT_FINAL_CHECK = WORKSPACE_ROOT / "evaluation" / "test_results" / "final_check.txt"
BASELINE_SCHEMA_VERSION = 1


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


def _safe_id(value):
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("baseline_id must contain only letters, digits, dot, underscore or hyphen")
    return value


def _copy(source, destination, roles, role):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    roles[destination] = role
    return destination


def _relative_workspace(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _parse_final_check(path, expected_commit):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"final test report not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    test_count = values.get("test_count")
    if not test_count or not test_count.isdigit():
        raise ValueError("final test report has no valid test_count")
    if values.get("test_result") != "OK":
        raise ValueError("final test report is not OK")
    report_commit = values.get("git_commit", "")
    if not report_commit or not expected_commit.startswith(report_commit):
        raise ValueError("final test report commit does not match the current HEAD")
    if values.get("working_tree_modified") != "no":
        raise ValueError("final test report was generated from a dirty worktree")
    return {
        "test_count": int(test_count),
        "test_result": "OK",
        "git_commit": report_commit,
    }


def _validate_result(document, settings, *, require_recommendation):
    result = verify_competition_results.verify_document(
        document,
        max_p99_ms=settings.hard_move_limit_ms,
        allow_dirty=False,
        allow_timeouts=False,
        allow_unverified_db=False,
        require_recommendation=require_recommendation,
        expected_profile_sha256=settings.profile_sha256,
        require_minimum_pairs=True,
    )
    if not result["passed"]:
        raise ValueError("result validation failed: " + "; ".join(result["errors"]))
    return result


def _cleanup_temp(temp_dir, output_root):
    temp_path = Path(temp_dir).resolve()
    root = Path(output_root).resolve()
    if temp_path.parent == root and temp_path.exists():
        shutil.rmtree(temp_path)


def freeze_baseline(*, profile_path=None, tournament_path, ablation_path=None,
                    benchmark_path=None, final_check_path=DEFAULT_FINAL_CHECK,
                    output_dir=DEFAULT_OUTPUT_DIR, baseline_id=None,
                    copy_model=False, clean_checker=require_clean_worktree,
                    validator=None, report_generator=None):
    commit = clean_checker()
    settings = load_competition_settings(profile_path)
    validate = validator or _validate_result
    generator = report_generator or competition_report.generate_report
    source_profile_path = Path(profile_path or settings.profile_path).resolve()
    profile_document = json.loads(source_profile_path.read_text(encoding="utf-8-sig"))
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    baseline_id = _safe_id(
        baseline_id or datetime.now().strftime("competition_%Y%m%d_%H%M%S")
    )
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / baseline_id
    if final_dir.exists():
        raise ValueError(f"baseline directory already exists: {final_dir}")

    tournament_document = json.loads(Path(tournament_path).read_text(encoding="utf-8-sig"))
    tournament_validation = validate(
        tournament_document, settings, require_recommendation=True
    )
    if tournament_document.get("reproducibility", {}).get("code_commit") != commit:
        raise ValueError("tournament result commit does not match the current HEAD")
    ablation_document = None
    ablation_validation = None
    if ablation_path:
        ablation_document = json.loads(Path(ablation_path).read_text(encoding="utf-8-sig"))
        ablation_validation = validate(
            ablation_document, settings, require_recommendation=False
        )
        if ablation_document.get("reproducibility", {}).get("code_commit") != commit:
            raise ValueError("memory ablation commit does not match the current HEAD")
    benchmark_document = None
    benchmark_validation = None
    if benchmark_path:
        benchmark_document = json.loads(Path(benchmark_path).read_text(encoding="utf-8-sig"))
        benchmark_validation = validate(
            benchmark_document, settings, require_recommendation=False
        )
        if benchmark_document.get("reproducibility", {}).get("code_commit") != commit:
            raise ValueError("benchmark result commit does not match the current HEAD")
    test_summary = _parse_final_check(final_check_path, commit)
    frozen_profile = dict(profile_document)
    frozen_profile.update({
        "freeze_commit": commit,
        "frozen_at": created_at,
        "baseline_id": baseline_id,
    })

    temp_dir = Path(tempfile.mkdtemp(dir=output_root, prefix=f".{baseline_id}."))
    roles = {}
    try:
        profile_copy = _atomic_json(temp_dir / "competition_profile.frozen.json", frozen_profile)
        roles[profile_copy] = "competition_profile"
        tournament_copy = _copy(
            tournament_path, temp_dir / "tournament.json", roles, "tournament_result"
        )
        copies = [tournament_copy]
        if ablation_path:
            copies.append(_copy(
                ablation_path, temp_dir / "memory_ablation.json", roles,
                "memory_ablation_result",
            ))
        if benchmark_path:
            copies.append(_copy(
                benchmark_path, temp_dir / "benchmark.json", roles,
                "benchmark_result",
            ))
        copies.append(_copy(
            final_check_path, temp_dir / "final_check.txt", roles,
            "test_report",
        ))
        selected_manifest = WORKSPACE_ROOT / "models" / "selected_model.json"
        copies.append(_copy(
            selected_manifest, temp_dir / "selected_model.json", roles,
            "selected_model_manifest",
        ))
        model_path = settings.neural_model_path
        model_record = {
            "file": settings.neural_model_file,
            "sha256": file_sha256(model_path),
            "source_path": _relative_workspace(model_path),
            "copied": bool(copy_model),
        }
        if copy_model:
            model_copy = _copy(
                model_path, temp_dir / "models" / model_path.name, roles,
                "model_checkpoint",
            )
            copies.append(model_copy)
        for suffix, role in (
            (".sha256", "model_sidecar"),
            ("_config.json", "model_config"),
        ):
            candidate = model_path.with_name(model_path.name + suffix) if suffix == ".sha256" else model_path.with_name(model_path.stem + suffix)
            if candidate.is_file():
                copies.append(_copy(
                    candidate, temp_dir / candidate.name, roles, role
                ))
        common_config = model_path.parent / "model_config.json"
        if common_config.is_file():
            copies.append(_copy(
                common_config, temp_dir / "model_config.common.json", roles,
                "model_config_common",
            ))
        report_paths = generator(
            tournament_path=tournament_copy,
            ablation_path=temp_dir / "memory_ablation.json" if ablation_path else None,
            output_dir=temp_dir / "report",
            profile_path=profile_path,
        )
        for key, path in report_paths.items():
            roles[Path(path)] = f"competition_report_{key}"

        file_records = []
        for path in sorted(temp_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(temp_dir).as_posix()
            file_records.append({
                "path": relative,
                "role": roles.get(path, "artifact"),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            })
        manifest = {
            "baseline_schema_version": BASELINE_SCHEMA_VERSION,
            "baseline_id": baseline_id,
            "created_at": created_at,
            "git_commit": commit,
            "working_tree_dirty": False,
            "source_profile": {
                "path": str(source_profile_path),
                "sha256": settings.profile_sha256,
            },
            "frozen_profile": {
                "path": "competition_profile.frozen.json",
                "sha256": file_sha256(profile_copy),
                "ai_engine": settings.ai_engine,
                "memory_mode": settings.memory_mode,
                "search_time_limit_ms": settings.search_time_limit_ms,
                "hard_move_limit_ms": settings.hard_move_limit_ms,
            },
            "model": model_record,
            "tests": test_summary,
            "results": {
                "tournament": tournament_validation,
                "memory_ablation": ablation_validation,
                "benchmark": benchmark_validation,
                "tournament_recommendation": tournament_document.get("recommendation"),
                "memory_recommendation": (
                    ablation_document.get("recommendation")
                    if ablation_document else None
                ),
            },
            "environment": reproducibility_metadata(settings),
            "files": file_records,
        }
        _atomic_json(temp_dir / "baseline_manifest.json", manifest)
        os.replace(temp_dir, final_dir)
        manifest["baseline_dir"] = str(final_dir)
        return manifest
    except Exception:
        _cleanup_temp(temp_dir, output_root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Freeze and validate a competition baseline"
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--tournament", required=True)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--final-check", default=str(DEFAULT_FINAL_CHECK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--baseline-id", default=None)
    parser.add_argument("--copy-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = freeze_baseline(
            profile_path=args.profile,
            tournament_path=args.tournament,
            ablation_path=args.ablation,
            benchmark_path=args.benchmark,
            final_check_path=args.final_check,
            output_dir=args.output_dir,
            baseline_id=args.baseline_id,
            copy_model=args.copy_model,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())